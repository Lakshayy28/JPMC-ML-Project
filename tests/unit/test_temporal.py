from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fri.config import load_settings
from fri.data.loaders import load_processed_tables
from fri.features.baseline import build_feature_sets
from fri.temporal.drift import compute_temporal_drift_report
from fri.temporal.reporting import build_temporal_activity_summary


def test_temporal_reports_build_from_processed_tables() -> None:
    settings = load_settings(REPO_ROOT / "configs" / "default.yaml")
    tables = load_processed_tables(settings.dataset.processed_root)
    feature_sets = build_feature_sets(tables, temporal_windows=settings.temporal.windows)
    activity = build_temporal_activity_summary(tables, feature_sets, windows=settings.temporal.windows)
    drift = compute_temporal_drift_report(
        feature_sets["transaction"],
        baseline_window=settings.temporal.baseline_window,
        recent_window=settings.temporal.recent_window,
    )

    assert activity["transaction_count"] == len(tables["transactions"])
    assert activity["step_range"]["max"] >= activity["step_range"]["min"]
    assert drift["cohorts"]["recent_rows"] > 0
    assert drift["feature_count_analyzed"] > 0
