from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from fri.api import main as api_main
from fri.api import monitoring as api_monitoring
from fri.api import state as api_state
from fri.explainability.service import NodeExplanationReport


class _FakeEngineState:
    def __init__(self) -> None:
        self.explain_calls: list[tuple[int, int]] = []
        self.drift_calls: list[list[dict[str, object]]] = []

    def health_payload(self) -> dict[str, str]:
        return {"status": "healthy", "model": "hetero_gat"}

    def predict_account(self, account_id: int):
        if account_id != 19204:
            raise KeyError(account_id)

        class _Prediction:
            account_id = 19204
            fraud_probability = 0.9985
            is_high_risk = True
            threshold_used = 0.5

        return _Prediction()

    def explain_account(self, account_id: int, *, epochs: int = 50) -> NodeExplanationReport:
        if account_id != 19204:
            raise KeyError(account_id)
        self.explain_calls.append((account_id, epochs))
        return NodeExplanationReport(
            node_id=19204,
            risk_score=0.9985,
            top_node_features={"incoming_amount_velocity_1d": 0.56},
            critical_edges=[{"transaction_id": 117726, "amount": 18.69}],
        )

    def analyze_drift(self, recent_features: list[dict[str, object]]) -> api_state.DriftAnalysisResult:
        if not recent_features:
            raise ValueError("At least one recent feature record is required")
        self.drift_calls.append(recent_features)
        return api_state.DriftAnalysisResult(
            drift_detected=True,
            drift_score=0.91,
            drifted_features=["outgoing_tx_velocity_30d", "total_amount"],
            sample_size=len(recent_features),
            analyzed_feature_count=2,
        )

    def metrics_payload(self) -> bytes:
        return b"# HELP fri_drift_analyses_total Total number of drift analysis requests processed by the API.\n"


def test_api_routes_return_expected_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_state = _FakeEngineState()
    monkeypatch.setattr(api_state, "initialize_engine", lambda: fake_state)
    monkeypatch.setattr(api_state, "get_engine_state", lambda: fake_state)
    monkeypatch.setattr(api_state, "set_engine_state", lambda _: None)

    with TestClient(api_main.app) as client:
        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json() == {"status": "healthy", "model": "hetero_gat"}

        prediction_response = client.get("/predict/19204")
        assert prediction_response.status_code == 200
        assert prediction_response.json()["account_id"] == 19204
        assert prediction_response.json()["is_high_risk"] is True

        explanation_response = client.get("/explain/19204")
        assert explanation_response.status_code == 200
        assert explanation_response.json()["account_id"] == 19204
        assert "incoming_amount_velocity_1d" in explanation_response.json()["top_features"]
        assert fake_state.explain_calls == [(19204, 50)]

        drift_response = client.post(
            "/analyze-drift",
            json=[
                {
                    "outgoing_tx_velocity_30d": 12.0,
                    "total_amount": 5000.0,
                }
            ],
        )
        assert drift_response.status_code == 200
        assert drift_response.json()["drift_detected"] is True
        assert drift_response.json()["drifted_features"] == ["outgoing_tx_velocity_30d", "total_amount"]
        assert len(fake_state.drift_calls) == 1

        missing_prediction = client.get("/predict/1")
        assert missing_prediction.status_code == 404

        missing_explanation = client.get("/explain/1")
        assert missing_explanation.status_code == 404

        invalid_drift = client.post("/analyze-drift", json=[])
        assert invalid_drift.status_code == 400

        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
        assert "fri_drift_analyses_total" in metrics_response.text


def test_engine_state_caches_explanations() -> None:
    api_state.EngineState._explain_account_cached.cache_clear()

    class _CountingExplainer:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def explain_account(self, data: object, account_index: int, *, epochs: int = 50) -> NodeExplanationReport:
            self.calls.append((account_index, epochs))
            return NodeExplanationReport(
                node_id=19204,
                risk_score=0.9985,
                top_node_features={"incoming_amount_velocity_1d": 0.56},
                critical_edges=[{"transaction_id": 117726, "amount": 18.69}],
            )

    counting_explainer = _CountingExplainer()
    engine = api_state.EngineState.__new__(api_state.EngineState)
    engine.account_id_to_index = {19204: 7}
    engine.data = object()
    engine.explainer = counting_explainer

    first_report = engine.explain_account(19204, epochs=50)
    second_report = engine.explain_account(19204, epochs=50)
    third_report = engine.explain_account(19204, epochs=25)

    assert first_report.node_id == 19204
    assert second_report.node_id == 19204
    assert third_report.node_id == 19204
    assert counting_explainer.calls == [(7, 50), (7, 25)]

    api_state.EngineState._explain_account_cached.cache_clear()


def test_engine_state_resolves_checkpoint_path_portably(tmp_path: Path) -> None:
    output_root = tmp_path / "artifacts" / "graph"
    output_root.mkdir(parents=True)
    portable_checkpoint = output_root / "pytorch_hetero_gat_model.pt"
    portable_checkpoint.write_bytes(b"checkpoint")

    engine = api_state.EngineState.__new__(api_state.EngineState)
    engine.metrics = {
        "checkpoint_path": "/Users/lakshaychandra/JPMC ML Project/artifacts/graph/pytorch_hetero_gat_model.pt"
    }
    engine.settings = type(
        "_Settings",
        (),
        {
            "graph": type("_GraphSettings", (), {"output_root": output_root})(),
            "gnn": type("_GNNSettings", (), {"checkpoint_name": "fallback_model.pt"})(),
        },
    )()

    assert engine._resolve_checkpoint_path() == portable_checkpoint


def test_engine_state_analyzes_drift_from_recent_feature_payload(tmp_path: Path) -> None:
    engine = api_state.EngineState.__new__(api_state.EngineState)
    engine.drift_baseline_frame = pytest.importorskip("pandas").DataFrame(
        {
            "outgoing_tx_velocity_30d": [0.5, 0.7, 0.8, 1.0, 1.2, 1.3],
            "total_amount": [100.0, 120.0, 130.0, 150.0, 170.0, 180.0],
        }
    )
    engine.drift_feature_columns = ["outgoing_tx_velocity_30d", "total_amount"]
    engine.drift_monitor = api_monitoring.DriftMonitor(tmp_path / "artifacts" / "temporal" / "drift_events.jsonl")

    result = engine.analyze_drift(
        [
            {"outgoing_tx_velocity_30d": 6.0, "total_amount": 900.0},
            {"outgoing_tx_velocity_30d": 5.5, "total_amount": 880.0},
            {"outgoing_tx_velocity_30d": 5.8, "total_amount": 910.0},
            {"outgoing_tx_velocity_30d": 6.1, "total_amount": 920.0},
            {"outgoing_tx_velocity_30d": 5.9, "total_amount": 905.0},
            {"outgoing_tx_velocity_30d": 6.2, "total_amount": 930.0},
        ]
    )

    assert result.drift_detected is True
    assert result.drift_score > 0.2
    assert set(result.drifted_features) == {"outgoing_tx_velocity_30d", "total_amount"}


def test_drift_monitor_persists_events_and_exports_prometheus_metrics(tmp_path: Path) -> None:
    events_path = tmp_path / "artifacts" / "temporal" / "drift_events.jsonl"
    monitor = api_monitoring.DriftMonitor(events_path)

    monitor.record_drift_event(
        monitor.build_event(
            drift_detected=True,
            drift_score=0.87,
            drifted_features=["total_amount", "outgoing_tx_velocity_30d"],
            sample_size=32,
            analyzed_feature_count=12,
        ),
        duration_seconds=0.25,
    )

    persisted_lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(persisted_lines) == 1
    assert "\"drift_detected\": true" in persisted_lines[0]

    metrics_text = monitor.render_metrics().decode("utf-8")
    assert "fri_drift_analyses_total" in metrics_text
    assert "fri_drift_detected_total" in metrics_text
    assert "fri_drift_last_score" in metrics_text