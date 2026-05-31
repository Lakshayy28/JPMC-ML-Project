# Containerization Report

## Purpose

Document the Phase 11 containerization strategy for the Financial Risk Intelligence API engine so the validated FastAPI, Hetero GAT, and explainability runtime can be packaged and deployed consistently across local, cloud, and orchestration environments.

## Reporting Date

- 2026-06-01

## Executive Summary

Phase 11 packages the existing FastAPI microservice into a portable Docker image backed by a pinned runtime dependency set, a production-oriented Dockerfile, and a `docker-compose.yml` entry point for local orchestration.

This containerization step freezes the application runtime around:

- Python 3.11
- the validated FastAPI serving layer
- the trained Hetero GAT checkpoint and metrics artifacts
- the AMLSim 20K archive inputs required by the engine

The result is a reproducible deployment unit that can move from a local workstation to cloud compute without relying on the host Python environment.

## Docker Architecture

### Base Image

The image uses `python:3.11-slim` as the base runtime.

Python 3.11 was selected for container standardization because it is a stable deployment target for the current ML and API stack while staying leaner than a full Debian-based Python image.

### Environment Variables

The Dockerfile sets the following runtime variables:

- `PYTHONDONTWRITEBYTECODE=1` to avoid unnecessary `.pyc` generation in the container
- `PYTHONUNBUFFERED=1` to keep logs flushed immediately for container observability
- `PYTHONPATH=/app/src` so the `fri` package resolves directly without an editable install step

### Runtime Layout

The image uses `/app` as the working directory and copies the exact directories the engine needs at runtime:

- `src/`
- `configs/`
- `data/`
- `artifacts/`

This ensures the API can load:

- the FastAPI application code
- the YAML configuration
- the AMLSim archive sample
- trained model checkpoints and metrics artifacts

### Exposed Port

The container exposes port `8000`, matching the existing Uvicorn runtime and FastAPI documentation surface.

### Health Verification

The Dockerfile includes a container health check against `GET /health`, allowing orchestrators and operators to detect whether the service finished model initialization and is ready to accept traffic.

## Dependency Management

### Pinned Runtime Dependencies

The new root-level `requirements.txt` pins the core API and inference libraries to the validated local runtime versions:

- `fastapi==0.136.3`
- `uvicorn==0.48.0`
- `pydantic==2.13.4`
- `torch==2.12.0`
- `torch-geometric==2.7.0`
- `pandas==3.0.3`
- `numpy==2.4.6`
- `scikit-learn==1.8.0`
- `networkx==3.6.1`
- `pyyaml==6.0.3`
- `httpx==0.28.1`

Pinning these versions reduces drift between development and deployment environments and keeps the image aligned with the runtime that already passed API and explainability validation.

### CPU-First Torch Installation

The Docker image installs `torch==2.12.0` from the PyTorch CPU wheel index before processing the full `requirements.txt` set.

This avoids pulling unnecessary CUDA and NVIDIA runtime packages into the container when building the API on a CPU-first local workflow such as Docker Desktop on Apple Silicon.

### Build Context Hygiene

A `.dockerignore` file excludes non-runtime local content such as virtual environments, test output, and documentation from the Docker build context.

This keeps image builds smaller and avoids leaking unnecessary workstation-specific files into the container build process.

## Run Instructions

### Build The Image

```bash
docker compose build
```

### Start The Service

```bash
docker compose up
```

### Start With Rebuild

```bash
docker compose up --build
```

### Access The API

Once the container is healthy, the service is available at:

- `http://localhost:8000/health`
- `http://localhost:8000/predict/19204`
- `http://localhost:8000/explain/19204`
- `http://localhost:8000/docs`

### Persistence Behavior

`docker-compose.yml` mounts `./artifacts` into `/app/artifacts` so runtime-generated JSON outputs or refreshed artifacts can persist on the host filesystem instead of being trapped inside the container layer.

## Deployment Readiness

### Cloud Portability

The Docker image now packages the serving code, model assets, configuration, and archive data into a single deployable unit.

That makes the engine suitable for:

- AWS ECS or EKS
- Google Cloud Run or GKE
- Azure Container Apps or AKS
- self-managed Kubernetes clusters

### Kubernetes Readiness

The image is ready to be wrapped by a Kubernetes `Deployment` and `Service` because it now has:

- a deterministic startup command
- a fixed container port
- a health endpoint appropriate for readiness and liveness checks
- a reproducible dependency lock via `requirements.txt`

### Operational Benefits

Containerizing the engine improves deployment quality in several ways:

- eliminates workstation-specific Python dependency drift
- preserves compatibility between the API layer and the trained model assets
- makes local and remote environments behave the same way
- provides a stable foundation for later CI/CD, registry publishing, autoscaling, and drift-monitoring integrations

## Current Constraints

- the container currently targets CPU-oriented serving and does not yet include GPU-specific CUDA images or runtime hooks
- image size will remain meaningful because the build intentionally carries model artifacts and AMLSim archive inputs for self-contained execution
- synchronous explainability remains slower than prediction, even though the API path now benefits from explanation caching in the application layer

## Recommended Next Steps

1. Add a Kubernetes manifest or Helm chart for cluster deployment.
2. Publish the image to a registry such as Amazon ECR or GitHub Container Registry.
3. Add CI automation to build, scan, and smoke-test the container on every release candidate.