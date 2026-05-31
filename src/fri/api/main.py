from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from fri.api import schemas
from fri.api import state as api_state


@asynccontextmanager
async def lifespan(_: FastAPI):
    api_state.initialize_engine()
    yield


app = FastAPI(title="FRI Risk Engine", version="0.1.0", lifespan=lifespan)


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
def explain(account_id: int) -> schemas.ExplanationResponse:
    engine = api_state.get_engine_state()
    try:
        report = engine.explain_account(account_id, epochs=50)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Account id {account_id} was not found in the graph") from exc
    return schemas.ExplanationResponse(
        account_id=int(report.node_id),
        fraud_probability=report.risk_score,
        top_features=report.top_node_features,
        critical_transactions=report.critical_edges,
    )