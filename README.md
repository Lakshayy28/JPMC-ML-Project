# Financial Risk Intelligence Engine

A graph-based anti–money laundering fraud detection platform: AMLSim data, heterogeneous GAT inference, FastAPI serving, and a React forensic dashboard.

## Repository layout

| Path | Description |
|------|-------------|
| [`backend/`](backend/) | Python ML pipeline, FastAPI API, Docker image, data, artifacts, scripts, tests, and technical docs |
| [`frontend/`](frontend/) | React + Vite dashboard (calls the API via `VITE_API_BASE_URL`) |

All application code lives under these two directories. There is no shared Python or Node package at the repo root.

## Quick start

### Backend (API)

```bash
cd backend
docker compose up --build -d fri-api-engine
curl -s http://localhost:8000/health
```

See [`backend/README.md`](backend/README.md) for full startup, endpoints, and troubleshooting.

### Frontend (UI)

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 (API default: http://localhost:8000).

### Full stack (Docker)

From `backend/`:

```bash
docker compose up --build -d
```

- API: http://localhost:8000  
- UI (nginx): http://localhost:3000  

## Documentation

- **Backend runbook & architecture:** [`backend/README.md`](backend/README.md)  
- **Audit-ready system docs:** [`backend/docs/final/README.md`](backend/docs/final/README.md)  
- **Phase history & specs:** [`backend/docs/`](backend/docs/)  

## Development

Use app-local environments only:

- `backend/.venv` for Python tooling and tests
- `frontend/.venv` if you want a local utility Python env alongside the UI

The repo root should not be used as the active Python environment anymore.

```bash
# Backend tests (from backend/)
cd backend
PYTHONPATH=src pytest tests/unit -q

# Local API without Docker
cd backend
PYTHONPATH=src python scripts/run_api.py
```
