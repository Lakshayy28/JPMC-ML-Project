from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DatasetSettings:
    raw_root: Path
    processed_root: Path
    sample_preferred: bool = True


@dataclass(frozen=True)
class EnrichmentSettings:
    seed: int = 17
    device_pool_size: int = 16
    suspicious_device_pool_size: int = 4
    ip_pool_size: int = 64
    suspicious_ip_pool_size: int = 8
    merchant_pool_size: int = 24


@dataclass(frozen=True)
class ModelSettings:
    random_state: int = 42
    test_size: float = 0.25


@dataclass(frozen=True)
class GraphSettings:
    archive_sample: Path
    output_root: Path
    community_detection: bool = True
    community_seed: int = 42
    embedding_dimensions: int = 16


@dataclass(frozen=True)
class Settings:
    dataset: DatasetSettings
    enrichment: EnrichmentSettings
    models: ModelSettings
    graph: GraphSettings


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_settings(config_path: str | Path | None = None) -> Settings:
    config_file = _resolve_path(config_path or REPO_ROOT / "configs" / "default.yaml")
    with config_file.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    dataset_payload = payload.get("dataset", {})
    enrichment_payload = payload.get("enrichment", {})
    model_payload = payload.get("models", {})
    graph_payload = payload.get("graph", {})

    return Settings(
        dataset=DatasetSettings(
            raw_root=_resolve_path(dataset_payload.get("raw_root", "data/external/AMLSim")),
            processed_root=_resolve_path(dataset_payload.get("processed_root", "data/processed/amlsim")),
            sample_preferred=bool(dataset_payload.get("sample_preferred", True)),
        ),
        enrichment=EnrichmentSettings(
            seed=int(enrichment_payload.get("seed", 17)),
            device_pool_size=int(enrichment_payload.get("device_pool_size", 16)),
            suspicious_device_pool_size=int(enrichment_payload.get("suspicious_device_pool_size", 4)),
            ip_pool_size=int(enrichment_payload.get("ip_pool_size", 64)),
            suspicious_ip_pool_size=int(enrichment_payload.get("suspicious_ip_pool_size", 8)),
            merchant_pool_size=int(enrichment_payload.get("merchant_pool_size", 24)),
        ),
        models=ModelSettings(
            random_state=int(model_payload.get("random_state", 42)),
            test_size=float(model_payload.get("test_size", 0.25)),
        ),
        graph=GraphSettings(
            archive_sample=_resolve_path(
                graph_payload.get("archive_sample", "data/external/AMLSim/sample/20K_fanin200cycle200.tgz")
            ),
            output_root=_resolve_path(graph_payload.get("output_root", "artifacts/graph")),
            community_detection=bool(graph_payload.get("community_detection", True)),
            community_seed=int(graph_payload.get("community_seed", 42)),
            embedding_dimensions=int(graph_payload.get("embedding_dimensions", 16)),
        ),
    )
