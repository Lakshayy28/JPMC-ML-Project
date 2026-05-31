from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class RiskPredictionResponse(BaseModel):
    account_id: int
    fraud_probability: float
    is_high_risk: bool
    threshold_used: float


class ExplanationResponse(BaseModel):
    account_id: int
    fraud_probability: float
    top_features: dict[str, float]
    critical_transactions: list[dict[str, Any]]


class DriftReportResponse(BaseModel):
    drift_detected: bool
    drift_score: float
    drifted_features: list[str]