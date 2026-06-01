from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware

from fri.api import monitoring as api_monitoring
from fri.api import schemas
from fri.api import state as api_state


@asynccontextmanager
async def lifespan(_: FastAPI):
    api_state.initialize_engine()
    yield


app = FastAPI(title="FRI Risk Engine", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    engine = api_state.get_engine_state()
    return engine.health_payload()


@app.get("/predict/{account_id}", response_model=schemas.RiskPredictionResponse)
def predict(account_id: int) -> schemas.RiskPredictionResponse:
    engine = api_state.get_engine_state()
    try:
        result = engine.predict_account(account_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Account id {account_id} was not found in the graph") from exc
    return schemas.RiskPredictionResponse(
        account_id=result.account_id,
        fraud_probability=result.fraud_probability,
        is_high_risk=result.is_high_risk,
        threshold_used=result.threshold_used,
    )


@app.get("/explain/{account_id}", response_model=schemas.ExplanationResponse)
def explain(
    account_id: int,
    epochs: int = Query(default=12, ge=1, le=50, description="GNNExplainer optimization epochs."),
) -> schemas.ExplanationResponse:
    engine = api_state.get_engine_state()
    try:
        report = engine.explain_account(account_id, epochs=epochs)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Account id {account_id} was not found in the graph") from exc
    return schemas.ExplanationResponse(
        account_id=int(report.node_id),
        fraud_probability=report.risk_score,
        top_features=report.top_node_features,
        critical_transactions=report.critical_edges,
    )


@app.post("/analyze-drift", response_model=schemas.DriftReportResponse)
def analyze_drift(recent_features: list[dict[str, Any]]) -> schemas.DriftReportResponse:
    engine = api_state.get_engine_state()
    try:
        report = engine.analyze_drift(recent_features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return schemas.DriftReportResponse(
        drift_detected=report.drift_detected,
        drift_score=report.drift_score,
        drifted_features=report.drifted_features,
    )


@app.get("/metrics")
def metrics() -> Response:
    engine = api_state.get_engine_state()
    return Response(content=engine.metrics_payload(), media_type=api_monitoring.PROMETHEUS_CONTENT_TYPE)
