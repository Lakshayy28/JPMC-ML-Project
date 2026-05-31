from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TrainingTask:
    label: str
    script_path: Path


@dataclass(frozen=True)
class MetricRow:
    track: str
    model: str
    precision: float | None
    recall: float | None
    f1: float | None
    pr_auc: float | None
    roc_auc: float | None


TRAINING_TASKS: tuple[TrainingTask, ...] = (
    TrainingTask("Tabular Baselines", REPO_ROOT / "scripts" / "train_baseline.py"),
    TrainingTask("Graph Classical Baselines", REPO_ROOT / "scripts" / "train_graph_baseline.py"),
    TrainingTask("Graph Embedding Baselines", REPO_ROOT / "scripts" / "train_graph_embedding_baseline.py"),
    TrainingTask("PyTorch Geometric GNN", REPO_ROOT / "scripts" / "train_pytorch_gcn.py"),
)

METRIC_PATHS: dict[str, Path] = {
    "tabular": REPO_ROOT / "artifacts" / "baseline_metrics.json",
    "graph_classical": REPO_ROOT / "artifacts" / "graph" / "graph_baseline_metrics.json",
    "graph_embedding": REPO_ROOT / "artifacts" / "graph" / "graph_embedding_metrics.json",
    "pytorch_gnn": REPO_ROOT / "artifacts" / "graph" / "pytorch_gcn_metrics.json",
}


def _run_training_task(task: TrainingTask) -> None:
    print(f"\n=== Running {task.label} ===", flush=True)
    subprocess.run([sys.executable, str(task.script_path)], cwd=REPO_ROOT, check=True)


def _load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Expected metrics file was not produced: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_metric(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _metric_row(track: str, model: str, metrics: Mapping[str, Any]) -> MetricRow:
    return MetricRow(
        track=track,
        model=model,
        precision=_coerce_metric(metrics.get("precision")),
        recall=_coerce_metric(metrics.get("recall")),
        f1=_coerce_metric(metrics.get("f1")),
        pr_auc=_coerce_metric(metrics.get("average_precision")),
        roc_auc=_coerce_metric(metrics.get("roc_auc")),
    )


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected mapping payload for {label}, got {type(value).__name__}")
    return value


def _collect_tabular_rows(payload: Mapping[str, Any]) -> list[MetricRow]:
    rows: list[MetricRow] = []
    for dataset_name in ("transaction", "party"):
        dataset_metrics = _require_mapping(payload.get(dataset_name), label=f"tabular/{dataset_name}")
        for model_name, model_metrics in dataset_metrics.items():
            rows.append(
                _metric_row(
                    "Tabular",
                    f"{dataset_name}/{model_name}",
                    _require_mapping(model_metrics, label=f"tabular/{dataset_name}/{model_name}"),
                )
            )
    return rows


def _collect_graph_classical_rows(payload: Mapping[str, Any]) -> list[MetricRow]:
    rows: list[MetricRow] = []
    for model_name, model_metrics in payload.items():
        rows.append(
            _metric_row(
                "Graph Classical",
                model_name,
                _require_mapping(model_metrics, label=f"graph_classical/{model_name}"),
            )
        )
    return rows


def _collect_graph_embedding_rows(payload: Mapping[str, Any]) -> list[MetricRow]:
    rows: list[MetricRow] = []
    for bundle_name in ("embedding_only", "combined_graph_features_and_embeddings"):
        bundle_metrics = _require_mapping(payload.get(bundle_name), label=f"graph_embedding/{bundle_name}")
        for model_name, model_metrics in bundle_metrics.items():
            rows.append(
                _metric_row(
                    "Graph Embedding",
                    f"{bundle_name}/{model_name}",
                    _require_mapping(model_metrics, label=f"graph_embedding/{bundle_name}/{model_name}"),
                )
            )
    return rows


def _collect_pytorch_rows(payload: Mapping[str, Any]) -> list[MetricRow]:
    model_name = str(payload.get("model_name", "pytorch_graphsage"))
    return [_metric_row("PyTorch Geometric GNN", model_name, payload)]


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _print_markdown_table(rows: Sequence[MetricRow]) -> None:
    print("\n## Consolidated Model Performance", flush=True)
    print("| Track | Model | Precision | Recall | F1-Score | PR-AUC | ROC-AUC |", flush=True)
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |", flush=True)
    for row in rows:
        print(
            f"| {row.track} | {row.model} | {_format_metric(row.precision)} | {_format_metric(row.recall)} | "
            f"{_format_metric(row.f1)} | {_format_metric(row.pr_auc)} | {_format_metric(row.roc_auc)} |",
            flush=True,
        )


def main() -> None:
    print("Starting end-to-end model retraining and evaluation...", flush=True)
    for task in TRAINING_TASKS:
        _run_training_task(task)

    print("\n=== Loading refreshed metric artifacts ===", flush=True)
    tabular_metrics = _require_mapping(_load_metrics(METRIC_PATHS["tabular"]), label="tabular")
    graph_classical_metrics = _require_mapping(_load_metrics(METRIC_PATHS["graph_classical"]), label="graph_classical")
    graph_embedding_metrics = _require_mapping(_load_metrics(METRIC_PATHS["graph_embedding"]), label="graph_embedding")
    pytorch_gnn_metrics = _require_mapping(_load_metrics(METRIC_PATHS["pytorch_gnn"]), label="pytorch_gnn")

    rows = [
        *_collect_tabular_rows(tabular_metrics),
        *_collect_graph_classical_rows(graph_classical_metrics),
        *_collect_graph_embedding_rows(graph_embedding_metrics),
        *_collect_pytorch_rows(pytorch_gnn_metrics),
    ]
    _print_markdown_table(rows)


if __name__ == "__main__":
    main()