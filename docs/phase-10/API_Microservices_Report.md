# API Microservices Report

## Purpose

Document the successful implementation of the Phase 10 FastAPI microservice layer for the Financial Risk Intelligence platform, including service architecture, exposed endpoints, latency optimizations, and run instructions.

## Reporting Date

- 2026-06-01

## Executive Summary

Phase 10 introduced a production-style API layer that exposes the trained heterogenous GAT model and the graph explainability service through FastAPI.

The microservice now supports:

- engine health checks
- real-time fraud risk prediction for account nodes
- on-demand graph explanation retrieval for flagged accounts

This work converts the repository from an offline experimentation stack into a consumable model-serving surface that downstream systems can call over HTTP.

## Architecture Overview

### FastAPI Application Layer

The API is implemented in `src/fri/api/main.py` using FastAPI.

The application uses a lifespan context manager to initialize heavy model-serving state exactly once on startup instead of per request.

### Pydantic Response Schemas

Response contracts are defined in `src/fri/api/schemas.py` using `pydantic.BaseModel`.

Implemented schemas:

- `RiskPredictionResponse`
- `ExplanationResponse`

These schemas provide typed, documented payloads for consumers and appear automatically in FastAPI's interactive Swagger interface.

### Singleton State Manager

The serving engine is managed by `src/fri/api/state.py`.

`EngineState` performs one-time startup initialization for:

- repository settings
- device resolution
- archive graph loading
- unified feature-bundle construction
- heterogenous PyG graph materialization
- trained Hetero GAT checkpoint loading
- explainability service initialization
- account-id lookup indexing

This design avoids repeated graph reconstruction or checkpoint loading on individual requests.

## Available Endpoints

### `GET /health`

Purpose:

- confirm the service is up
- confirm the expected model family is loaded

Response shape:

```json
{
  "status": "healthy",
  "model": "hetero_gat"
}
```

### `GET /predict/{account_id}`

Purpose:

- return the current fraud probability for a specific account node
- indicate whether the score crosses the serving threshold

Behavior:

- resolves the requested `account_id` to the internal account index
- runs a `torch.no_grad()` forward pass on the prepared heterogenous graph
- returns probability, boolean high-risk flag, and threshold used
- returns HTTP 404 when the account does not exist in the graph

Example response:

```json
{
  "account_id": 19204,
  "fraud_probability": 0.9985247254371643,
  "is_high_risk": true,
  "threshold_used": 0.5
}
```

### `GET /explain/{account_id}`

Purpose:

- return an explanation for the requested account using the graph explainability service

Behavior:

- resolves the requested `account_id`
- invokes the cached explanation path in `EngineState`
- runs the PyG explainer with a lighter API-serving compute budget
- returns the account score, top attributed features, and critical structural transactions
- returns HTTP 404 when the account does not exist in the graph

Example response fields:

- `account_id`
- `fraud_probability`
- `top_features`
- `critical_transactions`

## Latency Optimizations

### Startup-Time State Reuse

The microservice avoids per-request model loading by initializing the graph, model, and explainability objects at startup through the singleton-style engine state manager.

This removes the largest fixed-cost latency sources from the request path.

### LRU Explanation Caching

`EngineState` now caches explanation lookups using an LRU cache keyed by:

- `account_id`
- explainer epoch budget

Operational result:

- repeated requests for the same account explanation return instantly from memory
- duplicate explainability calls do not rerun `GNNExplainer`

### Reduced API Explainer Budget

The offline explainability workflow still supports a larger explainer budget, but the API route now explicitly calls the explanation path with `epochs=50`.

This reduces compute time for synchronous CPU-serving scenarios while preserving a heavier offline mode for artifact generation and deeper investigation.

### Raw Edge Payload Reuse

The explanation engine reuses prebuilt raw transaction records instead of recomputing human-readable edge payloads after each request.

This keeps explanation outputs analyst-readable without adding unnecessary runtime postprocessing overhead.

## Run Instructions

### Start The Server

Activate the project environment and launch the API:

```bash
source .venv/bin/activate
python scripts/run_api.py
```

The service starts on:

- `http://localhost:8000`

### Open Swagger UI

FastAPI automatically exposes interactive API documentation at:

- `http://localhost:8000/docs`

From that interface, you can execute:

- `/health`
- `/predict/{account_id}`
- `/explain/{account_id}`

### Example Test Accounts

Validated example account:

- `19204`

Useful manual calls:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/predict/19204
curl -s http://127.0.0.1:8000/explain/19204
```

## Validation Notes

The FastAPI slice was validated through:

- targeted API unit tests
- syntax compilation of the API modules
- live endpoint checks against the running server

Validated live behaviors included:

- successful `/health` response
- successful `/predict/19204` response
- successful `/explain/19204` response

## Current Limitations

- the explanation endpoint is still materially slower than the prediction endpoint because graph explanation remains substantially more expensive than a forward pass
- serving is currently optimized for development and validation on CPU, not horizontal scaling or multi-worker production deployment
- merchant semantics remain derived proxies because the underlying 20K archive does not expose native merchant entities

## Recommended Next Steps

1. Add a background job or asynchronous explanation mode for long-running explanations.
2. Add response timing and request metrics for operational observability.
3. Add request authentication and model version reporting before moving beyond local development use.