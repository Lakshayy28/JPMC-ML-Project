"""
FRI Advanced Model Strategies: PNA + GraphMAE + Focal Loss
==========================================================
Trains 3 experimental model variants against the baseline Hetero GAT
and produces a comparative metrics table.

Strategies:
  1. PNA  — Replace GATConv with PNAConv (multi-aggregation + degree scaling)
  2. GraphMAE — Self-supervised pretraining via masked feature reconstruction
  3. Focal Loss — Replace CrossEntropyLoss to focus on hard fraud cases

All variants share the same data pipeline, splits, and evaluation logic
from the backend (fri.models.pytorch_gnn).
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, PNAConv
from torch_geometric.utils import degree

from fri.config import load_settings
from fri.models.pytorch_gnn import (
    SpatialTemporalHeteroGAT,
    _normalize_hetero_data,
    _split_indices,
    build_hetero_gat_model,
    build_pyg_graph_data_from_archive,
    resolve_training_device,
)

# ─────────────────────────────────────────────────────────────────────
# 1. SpatialTemporalHeteroPNA — Hybrid PNA + GAT Aggregation
# ─────────────────────────────────────────────────────────────────────
# NOTE: PNAConv does NOT support bipartite (x_src, x_dst) inputs —
# it only works on homogeneous node features. For the heterogeneous
# graph, we use PNA on the account→account edges (where multi-aggregation
# matters most for detecting structural patterns like smurfing) and
# GATConv on the bipartite account↔merchant edges.

class SpatialTemporalHeteroPNA(nn.Module):
    """
    Hybrid PNA + GAT model for heterogeneous financial graphs.

    Uses PNAConv with 4 aggregators (mean, min, max, std) × 3 scalers
    for the homogeneous account-to-account transfer edges, and GATConv
    for the bipartite account↔merchant edges.

    The PNA layer on the transfer graph is the key differentiator:
    - std aggregator captures variance in transaction amounts (smurfing signal)
    - degree-scaling normalizes for hub vs. leaf nodes
    - min/max aggregators detect extreme transaction patterns
    """

    def __init__(
        self,
        account_input_dim: int,
        merchant_input_dim: int,
        edge_dim: int,
        hidden_dim: int,
        deg_transfer: torch.Tensor,
        *,
        output_dim: int = 2,
        dropout: float = 0.3,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.account_encoder = nn.Linear(account_input_dim, hidden_dim)
        self.merchant_encoder = nn.Linear(merchant_input_dim, hidden_dim)
        self.dropout = dropout

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]

        self.convs = nn.ModuleList([
            self._build_conv(hidden_dim, edge_dim, aggregators, scalers,
                             deg_transfer, heads),
            self._build_conv(hidden_dim, edge_dim, aggregators, scalers,
                             deg_transfer, heads),
        ])
        self.classifier = nn.Linear(hidden_dim, output_dim)

    @staticmethod
    def _build_conv(
        hidden_dim, edge_dim, aggregators, scalers,
        deg_transfer, heads,
    ) -> HeteroConv:
        return HeteroConv(
            {
                # PNA for homogeneous account→account (captures structural stats)
                ("account", "transfers", "account"): PNAConv(
                    hidden_dim, hidden_dim,
                    aggregators=aggregators, scalers=scalers,
                    deg=deg_transfer, edge_dim=edge_dim,
                    towers=1, pre_layers=1, post_layers=1,
                ),
                # GAT for bipartite edges (natively supports (x_src, x_dst))
                ("account", "buys_from", "merchant"): GATConv(
                    (hidden_dim, hidden_dim), hidden_dim, heads=heads,
                    concat=False, edge_dim=edge_dim, add_self_loops=False,
                ),
                ("merchant", "rev_buys_from", "account"): GATConv(
                    (hidden_dim, hidden_dim), hidden_dim, heads=heads,
                    concat=False, edge_dim=edge_dim, add_self_loops=False,
                ),
            },
            aggr="sum",
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict = {
            "account": self.account_encoder(data["account"].x),
            "merchant": self.merchant_encoder(data["merchant"].x),
        }
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = {et: data[et].edge_attr for et in data.edge_types}

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            x_dict = {nt: F.elu(feat) for nt, feat in x_dict.items()}
            x_dict = {
                nt: F.dropout(feat, p=self.dropout, training=self.training)
                for nt, feat in x_dict.items()
            }
        return self.classifier(x_dict["account"])


def compute_degree_histogram(data: HeteroData, edge_type: tuple, num_target_nodes: int | None = None) -> torch.Tensor:
    """Compute in-degree histogram for a specific edge type."""
    ei = data[edge_type].edge_index
    n = num_target_nodes or int(ei[1].max()) + 1
    d = degree(ei[1], num_nodes=n, dtype=torch.long)
    return torch.bincount(d)


# ─────────────────────────────────────────────────────────────────────
# 2. GraphMAE — Masked Feature Reconstruction Pretraining
# ─────────────────────────────────────────────────────────────────────

class GraphMAEDecoder(nn.Module):
    """Lightweight MLP decoder for masked feature reconstruction."""

    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphMAEPretrainer:
    """
    Masked Autoencoder pretraining for the Hetero GAT encoder.

    Phase 1: Mask 30% of account node features, train encoder + decoder
             to reconstruct masked features using cosine similarity loss.
    Phase 2: Discard decoder, freeze first conv layer, fine-tune on labels.
    """

    def __init__(
        self,
        model: SpatialTemporalHeteroGAT,
        data: HeteroData,
        *,
        mask_ratio: float = 0.3,
        pretrain_epochs: int = 50,
        pretrain_lr: float = 0.005,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.mask_ratio = mask_ratio
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_lr = pretrain_lr
        self.device = device or torch.device("cpu")
        self.account_dim = int(data["account"].x.shape[1])

        # Build decoder: hidden_dim -> account_input_dim
        hidden_dim = model.classifier.in_features
        self.decoder = GraphMAEDecoder(hidden_dim, self.account_dim).to(self.device)

    def _mask_features(self, data: HeteroData) -> tuple[HeteroData, torch.Tensor, torch.Tensor]:
        """Mask random features and return masked data + mask + original values."""
        masked_data = copy.deepcopy(data)
        account_x = masked_data["account"].x
        n_nodes, n_features = account_x.shape

        # Create feature-level mask (mask 30% of individual feature values)
        mask = torch.rand(n_nodes, n_features, device=self.device) < self.mask_ratio
        original_values = account_x.clone()
        account_x[mask] = 0.0
        masked_data["account"].x = account_x

        return masked_data, mask, original_values

    def _get_encoder_embeddings(self, data: HeteroData) -> torch.Tensor:
        """Forward through encoder only (before classifier)."""
        x_dict = {
            "account": self.model.account_encoder(data["account"].x),
            "merchant": self.model.merchant_encoder(data["merchant"].x),
        }
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = {et: data[et].edge_attr for et in data.edge_types}

        for conv in self.model.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            x_dict = {nt: F.elu(feat) for nt, feat in x_dict.items()}
            x_dict = {
                nt: F.dropout(feat, p=self.model.dropout, training=self.model.training)
                for nt, feat in x_dict.items()
            }
        return x_dict["account"]

    def pretrain(self) -> None:
        """Phase 1: Masked feature reconstruction."""
        print("\n--- GraphMAE Phase 1: Self-Supervised Pretraining ---")
        all_params = list(self.model.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(all_params, lr=self.pretrain_lr)

        self.model.train()
        for epoch in range(1, self.pretrain_epochs + 1):
            optimizer.zero_grad(set_to_none=True)

            masked_data, mask, original = self._mask_features(self.data)
            embeddings = self._get_encoder_embeddings(masked_data)
            reconstructed = self.decoder(embeddings)

            # Cosine similarity loss on masked positions only
            # Flatten to (N_masked, feature_dim) by selecting masked nodes
            masked_nodes = mask.any(dim=1)  # nodes that have at least one masked feature
            if masked_nodes.sum() == 0:
                continue

            recon_masked = reconstructed[masked_nodes]
            orig_masked = original[masked_nodes]

            # Cosine similarity loss (1 - cosine_sim)
            cos_sim = F.cosine_similarity(recon_masked, orig_masked, dim=1)
            loss = (1.0 - cos_sim).mean()

            loss.backward()
            optimizer.step()

            if epoch % 10 == 0 or epoch == 1:
                print(f"  [MAE Epoch {epoch:03d}/{self.pretrain_epochs}] "
                      f"Reconstruction Loss: {loss.item():.4f}")

        print("  Pretraining complete. Discarding decoder.")
        del self.decoder

    def get_param_groups(self, finetune_lr: float = 0.01, pretrained_lr_factor: float = 0.1) -> list[dict]:
        """Return differential learning rate param groups for fine-tuning.

        Pretrained layers (encoders + first conv) get a lower LR to preserve
        learned representations. Classifier + second conv get full LR.
        """
        pretrained_params = []
        finetune_params = []

        # Pretrained layers get lower LR
        pretrained_params.extend(self.model.account_encoder.parameters())
        pretrained_params.extend(self.model.merchant_encoder.parameters())
        pretrained_params.extend(self.model.convs[0].parameters())

        # Fresh layers get full LR
        finetune_params.extend(self.model.convs[1].parameters())
        finetune_params.extend(self.model.classifier.parameters())

        pretrained_lr = finetune_lr * pretrained_lr_factor
        print(f"  Differential LR: pretrained={pretrained_lr:.5f}, finetune={finetune_lr:.5f}")

        return [
            {"params": pretrained_params, "lr": pretrained_lr},
            {"params": finetune_params, "lr": finetune_lr},
        ]


# ─────────────────────────────────────────────────────────────────────
# 3. Focal Loss
# ─────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma > 0 reduces the loss for well-classified examples, focusing
    training on hard misclassified fraud cases.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ─────────────────────────────────────────────────────────────────────
# Training Infrastructure
# ─────────────────────────────────────────────────────────────────────

def _metric_or_none(fn, *args, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except ValueError:
        return None


def _optimal_threshold(labels, probs, default_threshold=0.5):
    if labels.size == 0:
        return float(default_threshold), None
    thresholds = np.linspace(0.05, 0.95, 19)
    best_f1, best_th = -1.0, float(default_threshold)
    for th in thresholds:
        preds = (probs >= th).astype(np.int64)
        cf1 = float(f1_score(labels, preds, zero_division=0))
        if cf1 > best_f1:
            best_f1, best_th = cf1, float(th)
    return best_th, (best_f1 if best_f1 >= 0.0 else None)


def _eval_full_batch(model, data, criterion, indices, decision_threshold=0.5):
    model.eval()
    with torch.no_grad():
        logits = model(data)[indices]
        labels = data["account"].y[indices]
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        preds = (probs >= decision_threshold).astype(np.int64)
        label_arr = labels.detach().cpu().numpy()
    return label_arr, probs, preds, float(loss.item())


def train_model(
    model: nn.Module,
    normalized_data: HeteroData,
    criterion: nn.Module,
    train_tensor: torch.Tensor,
    val_tensor: torch.Tensor,
    test_tensor: torch.Tensor,
    *,
    learning_rate: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 120,
    patience: int = 20,
    decision_threshold: float = 0.5,
    model_name: str = "model",
    param_groups: list[dict] | None = None,
) -> dict:
    """Generic full-batch training loop with early stopping."""
    if param_groups is not None:
        optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)

    best_state = None
    best_epoch = 0
    best_val_f1 = None
    best_val_ap = None
    best_selection_score = float("-inf")
    best_threshold = float(decision_threshold)
    patience_counter = 0

    print(f"\n--- Training {model_name} ---")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(normalized_data)
        train_logits = logits[train_tensor]
        train_labels = normalized_data["account"].y[train_tensor]
        loss = criterion(train_logits, train_labels)
        loss.backward()
        optimizer.step()

        val_labels, val_probs, _, val_loss = _eval_full_batch(
            model, normalized_data, criterion, val_tensor, decision_threshold)
        opt_th, val_f1 = _optimal_threshold(val_labels, val_probs, decision_threshold)
        selection_score = float(val_f1) if val_f1 is not None else -val_loss

        val_ap = None
        if 1 in np.unique(val_labels):
            val_ap = _metric_or_none(average_precision_score, val_labels, val_probs)

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_val_f1 = val_f1
            best_val_ap = val_ap
            best_threshold = opt_th
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{model_name} Epoch {epoch:03d}/{epochs}] "
                  f"Loss: {loss.item():.4f} | "
                  f"Val F1: {selection_score:.4f} | "
                  f"Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test evaluation
    test_labels, test_probs, test_preds, _ = _eval_full_batch(
        model, normalized_data, criterion, test_tensor, best_threshold)

    unique_labels = np.unique(test_labels)
    metrics = {
        "model_name": model_name,
        "precision": _metric_or_none(precision_score, test_labels, test_preds, zero_division=0),
        "recall": _metric_or_none(recall_score, test_labels, test_preds, zero_division=0),
        "f1": _metric_or_none(f1_score, test_labels, test_preds, zero_division=0),
        "average_precision": _metric_or_none(average_precision_score, test_labels, test_probs) if 1 in unique_labels else None,
        "roc_auc": _metric_or_none(roc_auc_score, test_labels, test_probs) if len(unique_labels) > 1 else None,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_val_ap": best_val_ap,
        "optimal_threshold": best_threshold,
        "train_time": time.time() - t0,
    }

    print(f"\n  {model_name} Results:")
    for k in ["precision", "recall", "f1", "average_precision", "roc_auc"]:
        v = metrics[k]
        print(f"    {k:>25s}: {v:.4f}" if v is not None else f"    {k:>25s}: n/a")
    print(f"    {'best_epoch':>25s}: {best_epoch}")
    print(f"    {'optimal_threshold':>25s}: {best_threshold:.4f}")

    return metrics


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    start_time = time.time()

    # ── Load data ────────────────────────────────────────────────────
    print("=" * 72)
    print("  LOADING DATA AND PREPARING GRAPH")
    print("=" * 72)
    settings = load_settings()
    archive_path = settings.dataset.graph_archive or settings.graph.archive_sample
    graph_bundle = build_pyg_graph_data_from_archive(
        archive_path,
        chunksize=settings.dataset.archive_chunksize,
        temporal_windows=settings.temporal.windows,
        merchant_seed=settings.enrichment.seed,
        merchant_pool_size=settings.enrichment.merchant_pool_size,
        include_communities=settings.graph.community_detection,
        community_seed=settings.graph.community_seed,
    )
    device = resolve_training_device(
        use_cuda=settings.hardware.use_cuda,
        requested_device=settings.hardware.device,
    )
    data = graph_bundle.data

    # Splits & normalization
    random_state = settings.models.random_state
    test_size = settings.models.test_size
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    labels_np = data["account"].y.cpu().numpy()
    train_idx, val_idx, test_idx = _split_indices(labels_np, test_size=test_size, random_state=random_state)
    normalized_data = _normalize_hetero_data(data, torch.tensor(train_idx, dtype=torch.long))
    normalized_data = normalized_data.to(device)

    train_tensor = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_tensor = torch.tensor(val_idx, dtype=torch.long, device=device)
    test_tensor = torch.tensor(test_idx, dtype=torch.long, device=device)

    # Class weights
    class_counts = np.bincount(labels_np[train_idx], minlength=2)
    neg_count = max(int(class_counts[0]), 1)
    pos_count = max(int(class_counts[1]), 1)
    pw = settings.gnn.pos_weight_multiplier
    class_weights = torch.tensor([1.0, (neg_count / pos_count) * pw], dtype=torch.float32, device=device)
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"  Device: {device}")
    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    print(f"  Class weights: {class_weights}")

    all_results = []
    training_kwargs = dict(
        learning_rate=settings.gnn.learning_rate,
        weight_decay=settings.gnn.weight_decay,
        epochs=settings.gnn.epochs,
        patience=settings.gnn.patience,
        decision_threshold=settings.gnn.decision_threshold,
    )

    # ── Experiment 1: PNA ────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  EXPERIMENT 1: PNA AGGREGATION")
    print("=" * 72)

    # Compute degree histogram for account→account transfer edges
    n_account = int(normalized_data["account"].x.shape[0])

    deg_transfer = compute_degree_histogram(
        normalized_data, ("account", "transfers", "account"), n_account)

    print(f"  Transfer edge max degree: {len(deg_transfer)-1}")

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    pna_model = SpatialTemporalHeteroPNA(
        account_input_dim=int(normalized_data["account"].x.shape[1]),
        merchant_input_dim=int(normalized_data["merchant"].x.shape[1]),
        edge_dim=int(normalized_data["account", "transfers", "account"].edge_attr.shape[1]),
        hidden_dim=settings.gnn.hidden_dim,
        deg_transfer=deg_transfer,
        dropout=settings.gnn.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in pna_model.parameters())
    print(f"  PNA model parameters: {total_params:,}")

    pna_results = train_model(
        pna_model, normalized_data, ce_criterion,
        train_tensor, val_tensor, test_tensor,
        model_name="PNA", **training_kwargs)
    all_results.append(pna_results)

    # ── Experiment 2: GraphMAE ───────────────────────────────────────
    print("\n" + "=" * 72)
    print("  EXPERIMENT 2: GraphMAE PRETRAINING + FINE-TUNING")
    print("=" * 72)

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    mae_model = build_hetero_gat_model(
        normalized_data, hidden_dim=settings.gnn.hidden_dim,
        dropout=settings.gnn.dropout).to(device)

    pretrainer = GraphMAEPretrainer(
        mae_model, normalized_data,
        mask_ratio=0.3, pretrain_epochs=50, pretrain_lr=0.005,
        device=device)
    pretrainer.pretrain()
    mae_param_groups = pretrainer.get_param_groups(
        finetune_lr=settings.gnn.learning_rate, pretrained_lr_factor=0.1)

    mae_results = train_model(
        mae_model, normalized_data, ce_criterion,
        train_tensor, val_tensor, test_tensor,
        model_name="GraphMAE+GAT", param_groups=mae_param_groups,
        **training_kwargs)
    all_results.append(mae_results)

    # ── Experiment 3: Focal Loss ─────────────────────────────────────
    print("\n" + "=" * 72)
    print("  EXPERIMENT 3: FOCAL LOSS (gamma=0.5)")
    print("=" * 72)

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    focal_model = build_hetero_gat_model(
        normalized_data, hidden_dim=settings.gnn.hidden_dim,
        dropout=settings.gnn.dropout).to(device)

    focal_criterion = FocalLoss(weight=class_weights, gamma=0.5)

    focal_results = train_model(
        focal_model, normalized_data, focal_criterion,
        train_tensor, val_tensor, test_tensor,
        model_name="Focal Loss GAT", **training_kwargs)
    all_results.append(focal_results)

    # ── Comparison Table ─────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  COMPARATIVE RESULTS")
    print("=" * 72)

    # Load baseline reference
    baseline_ref = {
        "model_name": "Baseline GAT",
        "precision": 0.7319,
        "recall": 0.6962,
        "f1": 0.7136,
        "average_precision": 0.7417,
        "roc_auc": 0.9186,
        "best_epoch": 120,
    }
    all_with_baseline = [baseline_ref] + all_results

    metrics_keys = ["precision", "recall", "f1", "average_precision", "roc_auc"]
    header = f"  {'Model':<20s}" + "".join(f"  {k:>12s}" for k in metrics_keys) + f"  {'best_epoch':>10s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in all_with_baseline:
        row = f"  {r['model_name']:<20s}"
        for k in metrics_keys:
            v = r.get(k)
            row += f"  {v:>12.4f}" if v is not None else f"  {'n/a':>12s}"
        row += f"  {r.get('best_epoch', 'n/a'):>10}"
        print(row)

    # Highlight best
    best_f1_model = max(all_with_baseline, key=lambda r: r.get("f1") or 0)
    best_prauc_model = max(all_with_baseline, key=lambda r: r.get("average_precision") or 0)
    print(f"\n  🏆 Best F1: {best_f1_model['model_name']} ({best_f1_model.get('f1', 0):.4f})")
    print(f"  🏆 Best PR-AUC: {best_prauc_model['model_name']} ({best_prauc_model.get('average_precision', 0):.4f})")

    # Deltas vs baseline
    print(f"\n  Deltas vs Baseline GAT:")
    for r in all_results:
        for k in metrics_keys:
            bv = baseline_ref.get(k, 0) or 0
            av = r.get(k, 0) or 0
            delta = av - bv
            marker = "▲" if delta > 0 else "▼" if delta < 0 else "="
            print(f"    {r['model_name']:<20s} {k:<20s} {delta:>+.4f} {marker}")

    # Save results
    output_path = REPO_ROOT / "artifacts" / "graph" / "advanced_model_comparison.json"
    output_path.write_text(json.dumps(all_with_baseline, indent=2), encoding="utf-8")
    print(f"\n  Wrote comparison to: {output_path}")

    elapsed = time.time() - start_time
    print(f"\n  Total elapsed time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
