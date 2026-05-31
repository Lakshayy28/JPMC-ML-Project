from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def select_temporal_feature_columns(feature_frame: pd.DataFrame) -> list[str]:
    excluded = {"transaction_id", "source_account_id", "destination_account_id", "transaction_type", "label_source"}
    candidates: list[str] = []
    for column in feature_frame.columns:
        if column in excluded:
            continue
        if any(token in column for token in ("prev_", "recent_", "gap", "velocity_ratio", "amount_ratio")):
            candidates.append(column)
    return candidates


def _cohorts_by_time(
    feature_frame: pd.DataFrame,
    *,
    step_column: str,
    baseline_window: int,
    recent_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    max_step = int(feature_frame[step_column].max()) if not feature_frame.empty else 0
    min_step = int(feature_frame[step_column].min()) if not feature_frame.empty else 0

    recent_start = max_step - recent_window + 1
    baseline_end = recent_start - 1
    baseline_start = max(min_step, baseline_end - baseline_window + 1)

    recent = feature_frame[feature_frame[step_column] >= recent_start].copy()
    baseline = feature_frame[
        (feature_frame[step_column] >= baseline_start) & (feature_frame[step_column] <= baseline_end)
    ].copy()
    if baseline.empty:
        baseline = feature_frame[feature_frame[step_column] < recent_start].copy()

    cohort_meta = {
        "min_step": min_step,
        "max_step": max_step,
        "baseline_start": baseline_start,
        "baseline_end": baseline_end,
        "recent_start": recent_start,
        "recent_end": max_step,
        "baseline_rows": int(len(baseline)),
        "recent_rows": int(len(recent)),
    }
    return baseline, recent, cohort_meta


def compute_temporal_drift_report(
    feature_frame: pd.DataFrame,
    *,
    step_column: str = "event_step",
    baseline_window: int = 30,
    recent_window: int = 7,
    top_k: int = 10,
    feature_columns: Sequence[str] | None = None,
) -> dict[str, object]:
    baseline, recent, cohort_meta = _cohorts_by_time(
        feature_frame,
        step_column=step_column,
        baseline_window=baseline_window,
        recent_window=recent_window,
    )
    columns = list(feature_columns or select_temporal_feature_columns(feature_frame))

    feature_drift: list[dict[str, object]] = []
    for column in columns:
        baseline_series = pd.to_numeric(baseline[column], errors="coerce").dropna()
        recent_series = pd.to_numeric(recent[column], errors="coerce").dropna()
        if baseline_series.empty or recent_series.empty:
            continue

        statistic, pvalue = ks_2samp(baseline_series, recent_series)
        baseline_mean = float(baseline_series.mean())
        recent_mean = float(recent_series.mean())
        relative_change = 0.0 if baseline_mean == 0 else float((recent_mean - baseline_mean) / abs(baseline_mean))
        feature_drift.append(
            {
                "feature": column,
                "baseline_mean": baseline_mean,
                "recent_mean": recent_mean,
                "mean_delta": float(recent_mean - baseline_mean),
                "relative_change": relative_change,
                "ks_statistic": float(statistic),
                "ks_pvalue": float(pvalue),
            }
        )

    feature_drift.sort(key=lambda item: (item["ks_statistic"], abs(item["mean_delta"])), reverse=True)
    top_drift = feature_drift[:top_k]

    return {
        "cohorts": cohort_meta,
        "feature_count_analyzed": len(feature_drift),
        "top_drift_features": top_drift,
        "strong_drift_features": [item for item in feature_drift if item["ks_statistic"] >= 0.2],
    }
