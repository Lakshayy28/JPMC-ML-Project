# MLOps Drift Report

## Purpose

Document the Phase 12 MLOps completion slice for the Financial Risk Intelligence platform, centered on concept drift monitoring for the production FastAPI engine.

## Reporting Date

- 2026-06-01

## Executive Summary

Phase 12 adds an operational drift-monitoring surface to the deployed fraud engine so the system can detect when incoming behavioral feature distributions diverge from the 20K AMLSim archive used to train the current model family.

The new monitoring slice includes:

- a live `POST /analyze-drift` API endpoint
- baseline feature distributions stored in the initialized engine state
- KS-statistic-based drift scoring over incoming feature batches
- persistent drift event logging under `artifacts/temporal/drift_events.jsonl`
- a Prometheus-style `GET /metrics` export for drift monitoring counters and gauges
- a simulation script that perturbs known laundering-related features and sends them to the running Docker API

This closes the loop between model serving and model maintenance by exposing a direct signal for retraining readiness.

## MLOps Architecture

### Baseline Storage In Engine State

At API startup, `EngineState` now stores the archive-derived `tabular_account_features` frame as the baseline distribution reference for drift analysis.

This means the serving layer has direct access to the same behavioral feature population that underpins the current trained model, rather than relying on a separate external monitoring store.

### Drift Analysis Endpoint

The FastAPI application now exposes:

- `POST /analyze-drift`

The endpoint accepts a JSON list of recent feature dictionaries and compares their numeric feature distributions against the stored baseline frame.

The response contract includes:

- `drift_detected`
- `drift_score`
- `drifted_features`

### Drift Evaluation Logic

The implementation reuses the repository’s existing temporal drift approach by extending `src/fri/temporal/drift.py` with a baseline-versus-recent comparison helper.

The serving path applies the Kolmogorov-Smirnov two-sample test to overlapping numeric feature columns and computes:

- the per-feature KS statistic
- the per-feature KS p-value
- the aggregate drift score as the maximum KS statistic observed across analyzed features

Features are flagged as drifted when:

- `ks_statistic >= 0.2`
- `ks_pvalue <= 0.05`

This gives the API a statistically defensible way to surface meaningful distribution shifts while avoiding false alerts from minor noise.

## Why KS-Based Monitoring Protects Model Quality

Concept drift is one of the main failure modes for fraud models in production. Even when model code is stable and inference is healthy, the model can degrade if the underlying transaction behavior changes materially over time.

In this platform, the KS-based monitoring layer helps protect model quality by:

- detecting when new behavioral patterns no longer resemble the 20K archive distributions
- surfacing the exact features whose distributions shifted most strongly
- providing a simple scalar drift score that can be thresholded by downstream alerting systems
- enabling retraining decisions before prediction quality silently decays

This is especially important in AML and fraud monitoring, where adversarial behavior can change quickly and static training distributions are rarely permanent.

## Operational Workflow

### Persistent Drift Event Logging

Every successful drift analysis request now appends a JSONL event to `artifacts/temporal/drift_events.jsonl`.

Each persisted record includes:

- an RFC 3339 UTC timestamp
- whether drift was detected
- the drift score
- the list of drifted features
- the request sample size
- the count of analyzed features

Because the Docker Compose setup already persists the `artifacts/` directory back to the host, these monitoring events survive container restarts and can be harvested by downstream batch jobs or alerting pipelines.

### Prometheus Metrics Export

The API now exposes `GET /metrics` in Prometheus text format.

Current exported drift-monitoring metrics include:

- `fri_drift_analyses_total`
- `fri_drift_detected_total`
- `fri_drift_events_logged_total`
- `fri_drift_last_score`
- `fri_drift_last_sample_size`
- `fri_drift_last_feature_count_analyzed`
- `fri_drift_last_drifted_feature_count`
- `fri_drift_analysis_duration_seconds`

This makes the service ready for pull-based scraping by Prometheus, Grafana Agent, or other compatible observability infrastructure.

### Live API Monitoring

An external scheduler, orchestrator, or batch collector can periodically send recent feature windows to `POST /analyze-drift`.

When the endpoint reports drift:

- the system can trigger an alert
- analysts can inspect the returned `drifted_features`
- retraining or data-refresh workflows can be scheduled

### Simulation And Validation

The new `scripts/simulate_drift.py` utility validates the monitoring path end to end by:

- rebuilding the archive feature bundle
- selecting the tabular account feature population
- amplifying `outgoing_tx_velocity_30d` and `total_amount` by a configurable multiplier
- posting the manipulated batch to the running Docker API
- printing the resulting drift alert payload

This provides a repeatable smoke test for the monitoring stack and a concrete example of the endpoint’s expected behavior.

## Run Instructions

### Start The Containerized API

```bash
source .venv/bin/activate
docker compose up --build -d
```

### Call The Drift Endpoint Directly

```bash
curl -X POST http://127.0.0.1:8000/analyze-drift \
  -H 'Content-Type: application/json' \
  -d '[{"outgoing_tx_velocity_30d": 12.0, "total_amount": 5000.0}]'
```

### Run The Drift Simulation Script

```bash
source .venv/bin/activate
python scripts/simulate_drift.py
```

### Scrape The Prometheus Metrics

```bash
curl http://127.0.0.1:8000/metrics
```

## Validation Summary

Phase 12 was validated through:

- focused API unit tests for the new drift endpoint
- focused temporal tests for the baseline-versus-recent KS comparison helper
- syntax checks on the touched API and temporal modules
- live containerized API validation after rebuild
- an end-to-end drift simulation against the running Docker service

## Conclusion

Phase 12 completes the MLOps monitoring slice by giving the platform a deployable concept drift signal tied directly to the live inference service.

With this phase complete, the 12-Phase Project Master Guide has now been delivered end to end across:

- problem framing and architecture
- dataset assessment and canonicalization
- validation and exploratory analysis
- baseline modeling
- graph analytics and embeddings
- temporal intelligence
- heterogenous GNN modeling
- explainability
- microservice serving
- latency optimization
- containerization
- MLOps drift monitoring

The platform now spans the full lifecycle from raw AMLSim data to a containerized, explainable, monitored fraud-intelligence API.