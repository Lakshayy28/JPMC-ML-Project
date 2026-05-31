from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest


PROMETHEUS_CONTENT_TYPE = CONTENT_TYPE_LATEST


@dataclass(frozen=True)
class DriftEvent:
    timestamp: str
    drift_detected: bool
    drift_score: float
    drifted_features: list[str]
    sample_size: int
    analyzed_feature_count: int


class DriftMonitor:
    def __init__(self, events_path: Path) -> None:
        self.events_path = events_path
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self.registry = CollectorRegistry()
        self.drift_analyses_total = Counter(
            "fri_drift_analyses_total",
            "Total number of drift analysis requests processed by the API.",
            registry=self.registry,
        )
        self.drift_detected_total = Counter(
            "fri_drift_detected_total",
            "Total number of drift analyses that detected feature drift.",
            registry=self.registry,
        )
        self.drift_events_logged_total = Counter(
            "fri_drift_events_logged_total",
            "Total number of persisted drift events written to disk.",
            registry=self.registry,
        )
        self.drift_last_score = Gauge(
            "fri_drift_last_score",
            "Drift score from the most recent drift analysis.",
            registry=self.registry,
        )
        self.drift_last_sample_size = Gauge(
            "fri_drift_last_sample_size",
            "Number of recent feature rows included in the last drift analysis.",
            registry=self.registry,
        )
        self.drift_last_feature_count = Gauge(
            "fri_drift_last_feature_count_analyzed",
            "Number of feature distributions compared in the last drift analysis.",
            registry=self.registry,
        )
        self.drift_last_drifted_feature_count = Gauge(
            "fri_drift_last_drifted_feature_count",
            "Number of features flagged as drifted in the last drift analysis.",
            registry=self.registry,
        )
        self.drift_analysis_duration_seconds = Histogram(
            "fri_drift_analysis_duration_seconds",
            "Time spent running drift analysis requests.",
            registry=self.registry,
        )

    def record_drift_event(self, event: DriftEvent, *, duration_seconds: float) -> None:
        with self._lock:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")

        self.drift_analyses_total.inc()
        if event.drift_detected:
            self.drift_detected_total.inc()
        self.drift_events_logged_total.inc()
        self.drift_last_score.set(event.drift_score)
        self.drift_last_sample_size.set(float(event.sample_size))
        self.drift_last_feature_count.set(float(event.analyzed_feature_count))
        self.drift_last_drifted_feature_count.set(float(len(event.drifted_features)))
        self.drift_analysis_duration_seconds.observe(duration_seconds)

    def build_event(
        self,
        *,
        drift_detected: bool,
        drift_score: float,
        drifted_features: list[str],
        sample_size: int,
        analyzed_feature_count: int,
    ) -> DriftEvent:
        return DriftEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            drift_detected=drift_detected,
            drift_score=drift_score,
            drifted_features=list(drifted_features),
            sample_size=int(sample_size),
            analyzed_feature_count=int(analyzed_feature_count),
        )

    def render_metrics(self) -> bytes:
        return generate_latest(self.registry)