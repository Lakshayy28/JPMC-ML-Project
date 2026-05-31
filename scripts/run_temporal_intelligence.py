from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fri.config import load_settings
from fri.data.loaders import load_processed_tables
from fri.features.baseline import build_feature_sets
from fri.temporal.drift import compute_temporal_drift_report
from fri.temporal.reporting import build_temporal_activity_summary


def main() -> None:
    settings = load_settings()
    tables = load_processed_tables(settings.dataset.processed_root)
    feature_sets = build_feature_sets(tables, temporal_windows=settings.temporal.windows)

    activity_summary = build_temporal_activity_summary(
        tables,
        feature_sets,
        windows=settings.temporal.windows,
    )
    drift_report = compute_temporal_drift_report(
        feature_sets["transaction"],
        step_column="event_step",
        baseline_window=settings.temporal.baseline_window,
        recent_window=settings.temporal.recent_window,
        top_k=settings.temporal.drift_top_k,
    )

    output_dir = REPO_ROOT / "artifacts" / "temporal"
    output_dir.mkdir(parents=True, exist_ok=True)
    activity_path = output_dir / "temporal_activity_summary.json"
    drift_path = output_dir / "temporal_drift_report.json"
    activity_path.write_text(json.dumps(activity_summary, indent=2), encoding="utf-8")
    drift_path.write_text(json.dumps(drift_report, indent=2), encoding="utf-8")

    print(f"Wrote temporal activity summary to: {activity_path}")
    print(f"Wrote temporal drift report to: {drift_path}")


if __name__ == "__main__":
    main()
