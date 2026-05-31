from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fri.config import load_settings
from fri.data.loaders import load_processed_tables
from fri.features.baseline import build_feature_sets
from fri.temporal.drift import compute_distribution_drift_report, compute_temporal_drift_report
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


def test_distribution_drift_report_flags_shifted_feature_distributions() -> None:
    baseline_frame = load_processed_tables(load_settings(REPO_ROOT / "configs" / "default.yaml").dataset.processed_root)[
        "accounts"
    ].head(20).copy()
    baseline_frame["outgoing_tx_velocity_30d"] = [0.5 + (index * 0.05) for index in range(len(baseline_frame))]
    baseline_frame["total_amount"] = [100.0 + (index * 10.0) for index in range(len(baseline_frame))]

    recent_frame = baseline_frame[["outgoing_tx_velocity_30d", "total_amount"]].copy()
    recent_frame["outgoing_tx_velocity_30d"] = recent_frame["outgoing_tx_velocity_30d"] * 5.0
    recent_frame["total_amount"] = recent_frame["total_amount"] * 5.0

    report = compute_distribution_drift_report(
        baseline_frame,
        recent_frame,
        feature_columns=["outgoing_tx_velocity_30d", "total_amount"],
    )

    assert report["drift_detected"] is True
    assert report["drift_score"] >= 0.2
    assert {item["feature"] for item in report["drifted_features"]} == {
        "outgoing_tx_velocity_30d",
        "total_amount",
    }
