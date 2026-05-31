# MLOps and Deployment

## Purpose

Document the deployment and MLOps operating model of the Financial Risk Intelligence Engine, with emphasis on containerization, runtime dependency control, concept drift detection, persistent monitoring artifacts, and developer runbook procedures.

## Executive Summary

The repository now supports a production-style deployment pattern in which the fraud engine is packaged as a Dockerized FastAPI service and started through Docker Compose.

That service is not limited to model inference. It also includes:

- container health checking
- persistent artifact storage
- live concept drift analysis
- persistent JSONL drift event logging
- Prometheus-style metrics export

The result is a single deployable unit that can score risk, explain decisions, monitor drift, and emit operational telemetry from the same runtime boundary.

## Containerization Strategy

### Docker Image Design

The runtime image is defined by [Dockerfile](Dockerfile) and built from:

- `python:3.11-slim`

The container sets the following environment variables:

- `PYTHONDONTWRITEBYTECODE=1`
- `PYTHONUNBUFFERED=1`
- `PYTHONPATH=/app/src`

This is a pragmatic serving configuration:

- Python bytecode files are suppressed to keep the container cleaner
- unbuffered output ensures logs appear immediately in container logs
- `PYTHONPATH=/app/src` allows the `fri` package to be imported without an editable install step

### CPU-Optimized PyTorch Build

The image does not install a GPU-focused PyTorch stack. Instead, it explicitly installs:

- `torch==2.12.0`

from the PyTorch CPU wheel index before installing the remaining requirements.

That choice is operationally important because it avoids pulling unnecessary CUDA and NVIDIA runtime packages into the container. For the current repository state, that is the correct deployment tradeoff because the validated serving path is CPU-first.

### Runtime Dependencies

The image installs a pinned [requirements.txt](requirements.txt), which includes the serving and monitoring stack used by the live API:

- FastAPI
- Uvicorn
- Pydantic
- PyTorch
- PyTorch Geometric
- pandas
- numpy
- scikit-learn
- networkx
- PyYAML
- httpx
- prometheus-client

This dependency pinning matters because it stabilizes inference, explanation, and drift-monitoring behavior across environments.

### Files Copied Into The Image

The Docker image copies:

- [src/](src/)
- [configs/](configs/)
- [data/](data/)
- [artifacts/](artifacts/)

This means the container is self-contained with respect to:

- source code
- runtime configuration
- the AMLSim archive input data
- trained model checkpoints and metrics

The current serving engine therefore does not depend on an external model registry or remote object store at startup.

### Health Checking

The image exposes port `8000` and defines a health check against:

- `GET /health`

This distinguishes process existence from real application readiness. In this repository, that distinction matters because startup includes archive loading, feature engineering, heterogenous graph preparation, and checkpoint restoration before the service is actually ready.

## Docker Compose Runtime

### Compose Topology

[docker-compose.yml](docker-compose.yml) defines one service:

- `fri-api-engine`

with:

- build context at repository root
- port mapping `8000:8000`
- bind mount `./artifacts:/app/artifacts`
- restart policy `unless-stopped`

This is intentionally simple. The current system is not deployed as a multi-container service mesh. It is a single containerized inference engine with host-persisted artifacts.

### Why The Artifact Volume Matters

The artifact bind mount is a critical operational design decision.

Because [artifacts/](artifacts/) is mounted as a container volume, runtime outputs generated inside the container are written back to the host. That includes:

- trained model artifacts already present in the repository
- explainability outputs
- drift monitoring event logs

Without this mount, drift events and other generated operational files would disappear whenever the container was rebuilt or replaced.

## Concept Drift Detection

### Where The Baseline Comes From

The serving engine stores its baseline drift reference in `EngineState`.

During startup, `EngineState` builds the unified archive feature bundle and captures:

- `tabular_account_features`

as the drift baseline frame.

This is important because the live drift monitor is comparing incoming payloads against the same archive-derived behavioral representation that underpins the trained model.

### Endpoint Contract

The drift interface is exposed through:

- `POST /analyze-drift`

The endpoint accepts a JSON list of recent feature dictionaries. The payload is expected to contain numeric fields that overlap with the baseline account-feature frame.

The response returns:

- `drift_detected`
- `drift_score`
- `drifted_features`

### Statistical Logic

The drift analysis implementation lives in [src/fri/temporal/drift.py](src/fri/temporal/drift.py).

The active online path uses `compute_distribution_drift_report(...)`, which:

1. excludes identifier and label-like columns from comparison
2. selects overlapping numeric feature columns between the baseline frame and incoming payload
3. applies `ks_2samp(...)` feature by feature
4. computes:
   - baseline mean
   - recent mean
   - mean delta
   - relative change
   - KS statistic
   - KS p-value
5. flags a feature as drifted when:
   - `ks_statistic >= 0.2`
   - `ks_pvalue <= 0.05`
6. defines the overall `drift_score` as the maximum KS statistic observed across analyzed features

This gives the engine a simple but statistically grounded concept-drift signal suitable for online monitoring.

### What Drift Means Operationally

In this system, concept drift does not immediately retrain the model automatically. Instead, it raises a serving-time signal that the recent feature population has diverged from the 20K archive baseline.

That signal can be used to:

- alert operators or analysts
- justify retraining
- trigger deeper dataset review
- feed external monitoring dashboards or alerting pipelines

## Persistent Drift Event Logging

### JSONL Event Store

Persistent drift monitoring is implemented in [src/fri/api/monitoring.py](src/fri/api/monitoring.py).

Every successful drift analysis appends a JSONL record to:

- [artifacts/temporal/drift_events.jsonl](artifacts/temporal/drift_events.jsonl)

Each event includes:

- UTC timestamp
- `drift_detected`
- `drift_score`
- `drifted_features`
- request sample size
- analyzed feature count

Because the [artifacts/](artifacts/) directory is volume-mounted by Docker Compose, these drift events persist on the host even when the container is restarted or rebuilt.

### Why This Matters

This is the first durable monitoring trail in the serving stack.

It means the platform now retains a time-series record of observed distribution shifts rather than only returning ephemeral HTTP responses.

## Prometheus Metrics Export

### Metrics Endpoint

The live API exports Prometheus-compatible telemetry through:

- `GET /metrics`

The payload is rendered directly from the in-process monitoring registry.

### Exported Drift Metrics

The current metrics set includes:

- `fri_drift_analyses_total`
- `fri_drift_detected_total`
- `fri_drift_events_logged_total`
- `fri_drift_last_score`
- `fri_drift_last_sample_size`
- `fri_drift_last_feature_count_analyzed`
- `fri_drift_last_drifted_feature_count`
- `fri_drift_analysis_duration_seconds`

This means external infrastructure such as Prometheus-compatible scrapers can observe:

- how often drift analysis is being called
- how often drift is actually detected
- the magnitude of the last observed drift event
- the size of the analyzed payload
- drift-analysis latency

## Deployment Readiness Assessment

The repository is now materially closer to production readiness because it has all of the following at the serving layer:

- a pinned dependency set
- a containerized runtime
- a health endpoint and image health check
- model and artifact packaging inside the image
- persisted operational artifacts through a host volume
- live drift detection
- persisted drift events
- scrapeable monitoring metrics

What it does not yet include is a broader orchestration layer such as Kubernetes manifests, service mesh policy, or automated retraining pipelines. Those would be logical next steps, but they are not implemented in the current repository state.

## Developer Runbook

### Prerequisites

Before running the containerized engine, a developer should ensure:

- Docker Desktop is installed and running
- `docker compose` is available in the shell
- the repository contains the archive data and artifacts already expected by the service

### 1. Build And Start The Container

From the repository root:

```bash
docker compose up --build -d
```

This builds the image, starts the `fri-api-engine` container, and runs the FastAPI service on port `8000`.

### 2. Confirm The Container Is Healthy

```bash
docker compose ps
curl http://127.0.0.1:8000/health
```

Expected health response:

```json
{"status":"healthy","model":"hetero_gat"}
```

### 3. Open Swagger UI

Open the API documentation in a browser:

```text
http://127.0.0.1:8000/docs
```

From Swagger UI, a developer can interactively test:

- `GET /health`
- `GET /predict/{account_id}`
- `GET /explain/{account_id}`
- `POST /analyze-drift`
- `GET /metrics`

### 4. Test A Prediction From The Command Line

```bash
curl http://127.0.0.1:8000/predict/19204
```

### 5. Test A Drift Request Directly

```bash
curl -X POST http://127.0.0.1:8000/analyze-drift \
  -H 'Content-Type: application/json' \
  -d '[{"outgoing_tx_velocity_30d": 12.0, "total_amount": 5000.0}]'
```

### 6. Run The Built-In Drift Simulation

If the local Python environment is available, the repository also provides a higher-fidelity drift smoke test:

```bash
python scripts/simulate_drift.py
```

This script rebuilds the archive feature bundle locally, amplifies selected behavioral features, and posts the resulting payload to `POST /analyze-drift` on the running Docker service.

### 7. Scrape Metrics

```bash
curl http://127.0.0.1:8000/metrics
```

### 8. Inspect Persisted Drift Events

```bash
tail -n 5 artifacts/temporal/drift_events.jsonl
```

### 9. Stop The Stack

```bash
docker compose down
```

## Summary

The Financial Risk Intelligence Engine now supports a coherent deployment and MLOps operating model.

The containerization layer freezes the runtime environment and packages the model artifacts with the serving code. Docker Compose exposes the engine as a single stable service with persisted artifacts. The MLOps layer monitors live feature distributions against archive-derived baselines using KS-based drift detection, stores drift events durably, and exports Prometheus-compatible metrics for external observability.

This makes the current repository more than a model-serving prototype. It is now a containerized inference system with integrated monitoring and an operational runbook suitable for disciplined engineering handoff.