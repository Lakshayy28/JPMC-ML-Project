from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from fri.config import Settings, load_settings
from fri.explainability.service import HeteroGraphExplainerService, NodeExplanationReport
from fri.graph.io import load_archive_graph_data
from fri.graph.service import build_graph_feature_bundle
from fri.models.pytorch_gnn import (
    build_pyg_graph_data_from_feature_bundle,
    load_trained_hetero_gat_model,
    prepare_hetero_inference_data,
    resolve_training_device,
)
from fri.temporal.drift import compute_distribution_drift_report, select_distribution_feature_columns


@dataclass(frozen=True)
class PredictionResult:
    account_id: int
    fraud_probability: float
    is_high_risk: bool
    threshold_used: float


@dataclass(frozen=True)
class DriftAnalysisResult:
    drift_detected: bool
    drift_score: float
    drifted_features: list[str]


class EngineState:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.device = resolve_training_device(
            use_cuda=self.settings.hardware.use_cuda,
            requested_device=self.settings.hardware.device,
        )
        self.metrics = self._load_metrics(self.settings.graph.output_root / "pytorch_gcn_metrics.json")
        archive_path = self.settings.dataset.graph_archive or self.settings.graph.archive_sample
        archive_data = load_archive_graph_data(archive_path)
        self.feature_bundle = build_graph_feature_bundle(
            nodes=archive_data.nodes,
            transactions=archive_data.transactions,
            temporal_windows=self.settings.temporal.windows,
            merchant_seed=self.settings.enrichment.seed,
            merchant_pool_size=self.settings.enrichment.merchant_pool_size,
            include_communities=self.settings.graph.community_detection,
            community_seed=self.settings.graph.community_seed,
            embedding_dimensions=self.settings.graph.embedding_dimensions,
            embedding_random_state=self.settings.models.random_state,
            include_embeddings=True,
        )
        self.graph_bundle = build_pyg_graph_data_from_feature_bundle(self.feature_bundle)
        self.data = prepare_hetero_inference_data(
            self.graph_bundle.data,
            random_state=self.settings.models.random_state,
            test_size=self.settings.models.test_size,
            device=self.device,
            pin_memory=self.settings.hardware.pin_memory,
        )
        checkpoint_path = self._resolve_checkpoint_path()
        self.model, self.checkpoint_payload = load_trained_hetero_gat_model(
            self.data,
            checkpoint_path,
            hidden_dim=self.settings.gnn.hidden_dim,
            dropout=self.settings.gnn.dropout,
            device=self.device,
        )
        self.account_node_ids = [int(value) for value in self.data["account"].node_id.detach().cpu().tolist()]
        self.account_id_to_index = {account_id: index for index, account_id in enumerate(self.account_node_ids)}
        self.merchant_node_ids = self._merchant_node_ids(self.feature_bundle.get("merchant_features"))
        self.threshold_used = float(
            self.metrics.get(
                "optimal_threshold",
                self.metrics.get("decision_threshold", self.settings.gnn.decision_threshold),
            )
        )
        self.drift_baseline_frame = self._drift_baseline_frame(self.feature_bundle)
        self.drift_feature_columns = select_distribution_feature_columns(self.drift_baseline_frame)
        self.explainer = HeteroGraphExplainerService(
            self.model,
            feature_columns=[str(value) for value in self.metrics.get("feature_columns", [])],
            account_node_ids=self.account_node_ids,
            merchant_node_ids=self.merchant_node_ids,
            raw_edge_records=self._raw_edge_records(self.feature_bundle),
        )

    @staticmethod
    def _load_metrics(metrics_path: Path) -> dict[str, Any]:
        if not metrics_path.exists():
            raise FileNotFoundError(f"Expected metrics file does not exist: {metrics_path}")
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    def _resolve_checkpoint_path(self) -> Path:
        candidates: list[Path] = []

        checkpoint_value = self.metrics.get("checkpoint_path")
        if checkpoint_value:
            recorded_path = Path(str(checkpoint_value))
            candidates.append(self.settings.graph.output_root / recorded_path.name)
            candidates.append(recorded_path)

        candidates.append(self.settings.graph.output_root / self.settings.gnn.checkpoint_name)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[0]

    @staticmethod
    def _merchant_node_ids(merchant_features: object) -> list[str]:
        if not isinstance(merchant_features, pd.DataFrame) or "merchant_id" not in merchant_features.columns:
            return []
        return merchant_features["merchant_id"].astype(str).tolist()

    @staticmethod
    def _drift_baseline_frame(feature_bundle: dict[str, object]) -> pd.DataFrame:
        baseline_frame = feature_bundle.get("tabular_account_features")
        if not isinstance(baseline_frame, pd.DataFrame) or baseline_frame.empty:
            raise ValueError("Expected non-empty tabular_account_features for drift baseline analysis")
        return baseline_frame.copy()

    @staticmethod
    def _raw_edge_records(feature_bundle: dict[str, object]) -> dict[str, list[dict[str, Any]]]:
        records: dict[str, list[dict[str, Any]]] = {}

        transfer_frame = feature_bundle.get("normalized_transactions")
        if isinstance(transfer_frame, pd.DataFrame) and not transfer_frame.empty:
            records["account->transfers->account"] = transfer_frame[
                ["transaction_id", "source_node_id", "target_node_id", "amount", "event_time", "transaction_type_code"]
            ].to_dict("records")

        merchant_links = feature_bundle.get("merchant_links")
        if isinstance(merchant_links, pd.DataFrame) and not merchant_links.empty:
            merchant_records = merchant_links[
                ["transaction_id", "source_node_id", "merchant_id", "amount", "event_time", "transaction_type_code"]
            ].to_dict("records")
            records["account->buys_from->merchant"] = merchant_records
            records["merchant->rev_buys_from->account"] = merchant_records

        return records

    def health_payload(self) -> dict[str, str]:
        return {"status": "healthy", "model": "hetero_gat"}

    def account_index_for_id(self, account_id: int) -> int:
        if account_id not in self.account_id_to_index:
            raise KeyError(account_id)
        return self.account_id_to_index[account_id]

    def risk_probabilities(self) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model(self.data)
            probabilities = torch.softmax(logits, dim=1)[:, 1]
        return probabilities.detach().cpu()

    def predict_account(self, account_id: int) -> PredictionResult:
        account_index = self.account_index_for_id(account_id)
        probabilities = self.risk_probabilities()
        fraud_probability = float(probabilities[account_index].item())
        return PredictionResult(
            account_id=account_id,
            fraud_probability=fraud_probability,
            is_high_risk=fraud_probability >= self.threshold_used,
            threshold_used=self.threshold_used,
        )

    @lru_cache(maxsize=128)
    def _explain_account_cached(self, account_id: int, epochs: int) -> NodeExplanationReport:
        account_index = self.account_index_for_id(account_id)
        return self.explainer.explain_account(self.data, account_index, epochs=epochs)

    def explain_account(self, account_id: int, *, epochs: int = 50) -> NodeExplanationReport:
        return self._explain_account_cached(account_id, int(epochs))

    def analyze_drift(self, recent_features: list[dict[str, object]]) -> DriftAnalysisResult:
        if not recent_features:
            raise ValueError("At least one recent feature record is required")

        report = compute_distribution_drift_report(
            self.drift_baseline_frame,
            pd.DataFrame(recent_features),
            feature_columns=self.drift_feature_columns,
        )
        return DriftAnalysisResult(
            drift_detected=bool(report["drift_detected"]),
            drift_score=float(report["drift_score"]),
            drifted_features=[str(item["feature"]) for item in report["drifted_features"]],
        )


_ENGINE_STATE: EngineState | None = None


def initialize_engine(*, force_reinitialize: bool = False) -> EngineState:
    global _ENGINE_STATE
    if _ENGINE_STATE is None or force_reinitialize:
        _ENGINE_STATE = EngineState()
    return _ENGINE_STATE


def get_engine_state() -> EngineState:
    if _ENGINE_STATE is None:
        raise RuntimeError("Engine state has not been initialized")
    return _ENGINE_STATE


def set_engine_state(state: EngineState | None) -> None:
    global _ENGINE_STATE
    _ENGINE_STATE = state