from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fri.config import load_settings
from fri.data.loaders import load_processed_tables
from fri.features.baseline import build_feature_sets
from fri.models.baseline import train_all_baselines


def main() -> None:
    print("--> Loading tabular baseline settings...", flush=True)
    settings = load_settings()

    print(f"--> Loading processed canonical tables from: {settings.dataset.processed_root}", flush=True)
    tables = load_processed_tables(settings.dataset.processed_root)

    print("--> Extracting rolling window temporal feature sets...", flush=True)
    feature_sets = build_feature_sets(tables, temporal_windows=settings.temporal.windows)

    print("--> Training tabular baseline estimators...", flush=True)
    metrics = train_all_baselines(
        feature_sets,
        random_state=settings.models.random_state,
        test_size=settings.models.test_size,
        verbose=True,
    )

    output_dir = REPO_ROOT / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "baseline_metrics.json"
    output_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[SUCCESS] Wrote execution metrics output to: {output_file}", flush=True)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
