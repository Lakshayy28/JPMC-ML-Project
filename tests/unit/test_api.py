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
from fri.api import state as api_state
from fri.explainability.service import NodeExplanationReport


class _FakeEngineState:
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

    def explain_account(self, account_id: int) -> NodeExplanationReport:
        if account_id != 19204:
            raise KeyError(account_id)
        return NodeExplanationReport(
            node_id=19204,
            risk_score=0.9985,
            top_node_features={"incoming_amount_velocity_1d": 0.56},
            critical_edges=[{"transaction_id": 117726, "amount": 18.69}],
        )


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

        missing_prediction = client.get("/predict/1")
        assert missing_prediction.status_code == 404

        missing_explanation = client.get("/explain/1")
        assert missing_explanation.status_code == 404