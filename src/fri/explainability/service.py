from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.explain import Explainer, GNNExplainer


@dataclass(frozen=True)
class NodeExplanationReport:
    node_id: Any
    risk_score: float
    top_node_features: dict[str, float]
    critical_edges: list[dict[str, Any]]


class HeteroGraphExplainerService:
    def __init__(
        self,
        model: nn.Module,
        *,
        feature_columns: list[str],
        account_node_ids: list[int] | None = None,
        merchant_node_ids: list[str] | None = None,
        raw_edge_records: dict[str, list[dict[str, Any]]] | None = None,
        top_k_features: int = 10,
        top_k_edges: int = 10,
        explainer_epochs: int = 200,
    ) -> None:
        self.model = model.eval()
        self.feature_columns = list(feature_columns)
        self.account_node_ids = list(account_node_ids or [])
        self.merchant_node_ids = list(merchant_node_ids or [])
        self.raw_edge_records = dict(raw_edge_records or {})
        self.top_k_features = int(top_k_features)
        self.top_k_edges = int(top_k_edges)
        self.default_explainer_epochs = int(explainer_epochs)

    def _build_explainer(self, *, epochs: int | None = None) -> Explainer:
        resolved_epochs = int(epochs if epochs is not None else self.default_explainer_epochs)
        return Explainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=resolved_epochs),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="node",
                return_type="raw",
            ),
        )

    def explain_account(self, data: HeteroData, account_index: int, *, epochs: int = 50) -> NodeExplanationReport:
        edge_attr_dict = {
            edge_type: data[edge_type].edge_attr
            for edge_type in data.edge_types
            if hasattr(data[edge_type], "edge_attr") and data[edge_type].edge_attr is not None
        }

        with torch.no_grad():
            logits = self.model(data.x_dict, data.edge_index_dict, edge_attr_dict=edge_attr_dict)
            risk_probabilities = torch.softmax(logits, dim=1)[:, 1]
            risk_score = float(risk_probabilities[account_index].detach().cpu().item())

        explanation = self._build_explainer(epochs=epochs)(
            data.x_dict,
            data.edge_index_dict,
            edge_attr_dict=edge_attr_dict,
            index=account_index,
        )

        account_store = explanation["account"]
        node_mask = getattr(account_store, "node_mask", None)
        if node_mask is None:
            raise ValueError("Explainer did not produce an account node mask")

        top_node_features = self._top_node_features(node_mask, account_index)
        critical_edges = self._critical_edges(explanation, data, account_index)
        node_id = self._account_node_id(data, account_index)
        return NodeExplanationReport(
            node_id=node_id,
            risk_score=risk_score,
            top_node_features=top_node_features,
            critical_edges=critical_edges,
        )

    def _top_node_features(self, node_mask: torch.Tensor, account_index: int) -> dict[str, float]:
        detached_mask = node_mask.detach().cpu()
        if detached_mask.dim() == 1:
            feature_scores = detached_mask
        elif detached_mask.dim() == 2 and detached_mask.size(0) == 1:
            feature_scores = detached_mask[0]
        elif detached_mask.dim() == 2 and account_index < detached_mask.size(0):
            feature_scores = detached_mask[account_index]
        else:
            feature_scores = detached_mask.reshape(-1)

        scored_features: list[tuple[str, float]] = []
        for index, score in enumerate(feature_scores.tolist()):
            feature_name = self.feature_columns[index] if index < len(self.feature_columns) else f"feature_{index}"
            scored_features.append((feature_name, float(score)))
        scored_features.sort(key=lambda item: item[1], reverse=True)
        return {name: score for name, score in scored_features[: self.top_k_features]}

    def _critical_edges(self, explanation: HeteroData, data: HeteroData, account_index: int) -> list[dict[str, Any]]:
        edge_reports: list[dict[str, Any]] = []
        for edge_type in data.edge_types:
            explanation_store = explanation[edge_type]
            edge_mask = getattr(explanation_store, "edge_mask", None)
            if edge_mask is None:
                continue

            edge_index = data[edge_type].edge_index.detach().cpu()
            edge_attr = data[edge_type].edge_attr.detach().cpu() if hasattr(data[edge_type], "edge_attr") else None
            edge_scores = edge_mask.detach().cpu()

            for edge_position in self._incident_edge_positions(edge_type, edge_index, account_index):
                report = self._edge_report(
                    edge_type=edge_type,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    edge_scores=edge_scores,
                    edge_position=edge_position,
                )
                edge_reports.append(report)

        edge_reports.sort(key=lambda item: float(item["importance"]), reverse=True)
        return edge_reports[: self.top_k_edges]

    def _incident_edge_positions(
        self,
        edge_type: tuple[str, str, str],
        edge_index: torch.Tensor,
        account_index: int,
    ) -> list[int]:
        source_type, _, target_type = edge_type
        positions: list[int] = []
        for edge_position in range(edge_index.size(1)):
            source_index = int(edge_index[0, edge_position].item())
            target_index = int(edge_index[1, edge_position].item())
            if source_type == "account" and target_type == "account":
                if source_index == account_index or target_index == account_index:
                    positions.append(edge_position)
            elif source_type == "account" and source_index == account_index:
                positions.append(edge_position)
            elif target_type == "account" and target_index == account_index:
                positions.append(edge_position)
        return positions

    def _edge_report(
        self,
        *,
        edge_type: tuple[str, str, str],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        edge_scores: torch.Tensor,
        edge_position: int,
    ) -> dict[str, Any]:
        source_type, relation, target_type = edge_type
        source_index = int(edge_index[0, edge_position].item())
        target_index = int(edge_index[1, edge_position].item())
        report: dict[str, Any] = {
            "relation": f"{source_type}->{relation}->{target_type}",
            "importance": float(edge_scores[edge_position].item()),
            "source_index": source_index,
            "target_index": target_index,
            "source_node_id": self._typed_node_id(source_type, source_index),
            "target_node_id": self._typed_node_id(target_type, target_index),
        }
        raw_edge_record = self._raw_edge_record(report["relation"], edge_position)
        if raw_edge_record is not None:
            report.update(raw_edge_record)
            return report

        if edge_attr is not None and edge_position < edge_attr.size(0) and edge_attr.size(1) >= 3:
            report.update(
                {
                    "amount": float(edge_attr[edge_position, 0].item()),
                    "event_time": float(edge_attr[edge_position, 1].item()),
                    "transaction_type_code": float(edge_attr[edge_position, 2].item()),
                }
            )
        return report

    def _raw_edge_record(self, relation: str, edge_position: int) -> dict[str, Any] | None:
        relation_records = self.raw_edge_records.get(relation, [])
        if edge_position >= len(relation_records):
            return None

        raw_record = relation_records[edge_position]
        if relation == "account->transfers->account":
            return {
                "transaction_id": int(raw_record.get("transaction_id", edge_position)),
                "source_node_id": int(raw_record.get("source_node_id", raw_record.get("source_index", 0))),
                "target_node_id": int(raw_record.get("target_node_id", raw_record.get("target_index", 0))),
                "amount": float(raw_record.get("amount", 0.0)),
                "event_time": float(raw_record.get("event_time", 0.0)),
                "transaction_type_code": float(raw_record.get("transaction_type_code", 0.0)),
            }
        if relation == "account->buys_from->merchant":
            return {
                "transaction_id": int(raw_record.get("transaction_id", edge_position)),
                "source_node_id": int(raw_record.get("source_node_id", raw_record.get("source_index", 0))),
                "target_node_id": str(raw_record.get("merchant_id", raw_record.get("target_node_id", "unknown"))),
                "amount": float(raw_record.get("amount", 0.0)),
                "event_time": float(raw_record.get("event_time", 0.0)),
                "transaction_type_code": float(raw_record.get("transaction_type_code", 0.0)),
            }
        if relation == "merchant->rev_buys_from->account":
            return {
                "transaction_id": int(raw_record.get("transaction_id", edge_position)),
                "source_node_id": str(raw_record.get("merchant_id", raw_record.get("source_node_id", "unknown"))),
                "target_node_id": int(raw_record.get("source_node_id", raw_record.get("target_node_id", 0))),
                "amount": float(raw_record.get("amount", 0.0)),
                "event_time": float(raw_record.get("event_time", 0.0)),
                "transaction_type_code": float(raw_record.get("transaction_type_code", 0.0)),
            }
        return None

    def _account_node_id(self, data: HeteroData, account_index: int) -> Any:
        if self.account_node_ids and account_index < len(self.account_node_ids):
            return self.account_node_ids[account_index]
        if hasattr(data["account"], "node_id"):
            return int(data["account"].node_id[account_index].detach().cpu().item())
        return account_index

    def _typed_node_id(self, node_type: str, node_index: int) -> Any:
        if node_type == "account" and node_index < len(self.account_node_ids):
            return self.account_node_ids[node_index]
        if node_type == "merchant" and node_index < len(self.merchant_node_ids):
            return self.merchant_node_ids[node_index]
        return node_index