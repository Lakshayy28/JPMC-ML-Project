# Financial Risk Intelligence Platform — Startup Configuration

**Date**: June 1, 2026
**Status**: ✅ Successfully Started (Both Containers)

---

## System Overview

The Financial Risk Intelligence (FRI) Platform is a microservices-based fraud detection system:

| Service | Technology | Container | Host Port |
|---------|-----------|-----------|-----------|
| **Backend API** | FastAPI + PyTorch (Hetero GAT) | `fri-api-engine` | `8000` |
| **Frontend UI** | React 18 + Vite 6 → Nginx | `fri-frontend` | `3000` |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Browser                           │
│                  http://localhost:3000                       │
└──────────────────────┬──────────────────────────────────────┘
                       │  HTTP/REST (browser → host → container)
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Docker Network                                              │
│                                                              │
│  ┌────────────────────────┐   ┌───────────────────────────┐  │
│  │  fri-frontend          │   │  fri-api-engine           │  │
│  │  nginx:alpine          │   │  python:3.11-slim         │  │
│  │  Port 80 → Host 3000   │   │  Port 8000 → Host 8000   │  │
│  │                        │   │                           │  │
│  │  Serves React SPA      │   │  FastAPI + Uvicorn        │  │
│  │  Static assets (dist/) │   │  PyTorch Hetero GAT       │  │
│  │  Client-side routing   │   │  GNNExplainer (XAI)       │  │
│  └────────────────────────┘   │  KS-test drift monitor    │  │
│                               │  Prometheus metrics        │  │
│                               └───────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

> **Note:** The frontend container serves static files. All API calls
> originate from the user's browser (`http://localhost:8000`), not
> container-to-container.

---

## Prerequisites

| Requirement | Minimum Version | Check Command |
|-------------|----------------|---------------|
| Docker Desktop | 4.x | `docker --version` |
| Docker Compose | v2 (bundled) | `docker compose version` |
| Node.js (dev only) | 18+ | `node --version` |
| Ports available | 8000, 3000 | `lsof -i :8000 :3000` |

---

## Quick Start (Docker — Full Stack)

### Step 1: Ensure Docker Desktop is running

```bash
open -a Docker          # macOS — starts Docker Desktop
sleep 10                # wait for the daemon to be ready
docker info >/dev/null  # verify daemon responds
```

### Step 2: Kill any processes on the target ports

```bash
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
lsof -ti:5173 | xargs kill -9 2>/dev/null || true
```

### Step 3: Build and start both containers

```bash
cd backend
docker compose up --build -d
```

This single command:
- Builds the backend image (`backend/Dockerfile`) — installs PyTorch CPU, copies `src/`, `configs/`, `data/`, `artifacts/`
- Builds the frontend image (`frontend/Dockerfile`) — runs `npm ci && npm run build`, serves the `dist/` via Nginx
- Starts both containers with `restart: unless-stopped`

### Step 4: Wait for the API to become healthy

The backend container has a health check with a 90-second start period (model loading takes time on first boot):

```bash
# Wait for healthy status
docker compose -f backend/docker-compose.yml ps

# Or poll the health endpoint directly
until curl -sf http://localhost:8000/health >/dev/null; do
  echo "Waiting for API..." && sleep 5
done
echo "API is ready!"
```

### Step 5: Verify both services

```bash
# Backend health
curl -s http://localhost:8000/health | python3 -m json.tool
# Expected: {"status": "healthy", "model": "hetero_gat"}

# Frontend reachability
curl -s -o /dev/null -w "Frontend HTTP: %{http_code}\n" http://localhost:3000
# Expected: Frontend HTTP: 200
```

### Step 6: Access the application

| Service | URL |
|---------|-----|
| **Frontend UI** | http://localhost:3000 |
| **API Base** | http://localhost:8000 |
| **Swagger Docs** | http://localhost:8000/docs |

---

## Starting Services Individually

### Backend Only

```bash
cd backend
docker compose up --build -d fri-api-engine
```

### Frontend Only (Docker)

```bash
cd backend
docker compose up --build -d fri-frontend
```

### Frontend Only (Local Dev — hot reload)

```bash
cd frontend
npm install
npm run dev
# Serves at http://localhost:5173 with hot module replacement
```

---

## API Endpoints Reference

### Health Check
```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```
```json
{"status": "healthy", "model": "hetero_gat"}
```

### Predict Account Risk
```bash
curl -s http://localhost:8000/predict/19204 | python3 -m json.tool
```
```json
{
    "account_id": 19204,
    "fraud_probability": 0.9985247254371643,
    "is_high_risk": true,
    "threshold_used": 0.49999999999999994
}
```

### Predict Low-Risk Account
```bash
curl -s http://localhost:8000/predict/19205 | python3 -m json.tool
```
```json
{
    "account_id": 19205,
    "fraud_probability": 0.1668752282857895,
    "is_high_risk": false,
    "threshold_used": 0.49999999999999994
}
```

### Non-existent Account (404)
```bash
curl -s http://localhost:8000/predict/99999 | python3 -m json.tool
```
```json
{"detail": "Account id 99999 was not found in the graph"}
```

### Explainability (GNNExplainer)
```bash
curl -s "http://localhost:8000/explain/19204?epochs=12" | python3 -m json.tool
```
> ⚠️ First call is slow (~37s, 50 epochs). Subsequent calls are LRU-cached (~60ms).

### MLOps Drift Analysis
```bash
curl -s -X POST http://localhost:8000/analyze-drift \
  -H "Content-Type: application/json" \
  -d '[{"outgoing_tx_velocity_30d": 12.0, "total_amount": 5000.0}]' \
  | python3 -m json.tool
```
```json
{
    "drift_detected": true,
    "drift_score": 1.0,
    "drifted_features": ["outgoing_tx_velocity_30d"]
}
```

### Prometheus Metrics
```bash
curl -s http://localhost:8000/metrics | head -20
```

---

## Testing from the Browser Console

Open http://localhost:3000 in your browser, then open **DevTools → Console** (Cmd+Option+J on macOS) and run:

```javascript
// Health check
fetch("http://localhost:8000/health").then(r => r.json()).then(console.log)

// Predict high-risk account
fetch("http://localhost:8000/predict/19204").then(r => r.json()).then(console.log)

// Predict low-risk account
fetch("http://localhost:8000/predict/19205").then(r => r.json()).then(console.log)

// Drift analysis
fetch("http://localhost:8000/analyze-drift", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify([{"outgoing_tx_velocity_30d": 12.0, "total_amount": 5000.0}])
}).then(r => r.json()).then(console.log)
```

---

## Docker File Reference

### Backend — `backend/Dockerfile`

| Stage | Base Image | Purpose |
|-------|-----------|---------|
| Single | `python:3.11-slim` | Install PyTorch CPU, FastAPI, copy source, run Uvicorn |

Key settings:
- `PYTHONPATH=/app/src` — module resolution
- Health check: `--start-period=90s` (model loading time)
- Exposes port `8000`

### Frontend — `frontend/Dockerfile`

| Stage | Base Image | Purpose |
|-------|-----------|---------|
| Build | `node:18-alpine` | `npm ci` + `npm run build` → produces `dist/` |
| Serve | `nginx:alpine` | Serves static files with client-side routing |

Key settings:
- Custom `nginx.conf` with SPA fallback (`try_files $uri /index.html`)
- Static asset caching: `Cache-Control: public, max-age=604800, immutable`
- Exposes port `80` (mapped to host `3000`)

### Compose — `backend/docker-compose.yml`

```yaml
services:
  fri-api-engine:        # Backend API
    ports: "8000:8000"
    volumes: ./artifacts:/app/artifacts
    restart: unless-stopped

  fri-frontend:          # Frontend UI (Nginx)
    context: ../frontend
    ports: "3000:80"
    depends_on: fri-api-engine
    restart: unless-stopped
```

---

## Performance Metrics

| Operation | Cold Start | Warm (Cached) | Notes |
|-----------|-----------|---------------|-------|
| Health Check | ~50ms | ~10ms | Instant response |
| Predict (Account) | ~392ms | ~50ms | CPU inference |
| Explain (GNNExplainer) | ~37s | ~60ms | 50 epochs, LRU-cached |
| Drift Analysis | ~200ms | ~50ms | KS-based distribution test |
| Metrics Export | ~30ms | ~15ms | Prometheus format |

---

## Operations Commands

### View Container Status
```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### View Logs
```bash
# Backend logs (follow)
docker logs -f fri-api-engine

# Frontend logs (follow)
docker logs -f fri-frontend

# Both via compose
cd backend && docker compose logs -f
```

### Stop All Services
```bash
cd backend && docker compose down
```

### Rebuild From Scratch (no cache)
```bash
cd backend
docker compose down
docker compose build --no-cache
docker compose up -d
```

### Restart a Single Service
```bash
cd backend
docker compose restart fri-api-engine
docker compose restart fri-frontend
```

---

## Troubleshooting

### Docker not starting
```bash
open -a Docker
docker ps  # Should return container list without errors
```

### Port already in use
```bash
lsof -i :8000    # Find what's using port 8000
lsof -i :3000    # Find what's using port 3000
kill -9 <PID>    # Kill the process
```

### Frontend can't reach API (CORS or network)
1. Verify API is running: `curl http://localhost:8000/health`
2. CORS is configured to allow all origins (`allow_origins=["*"]`) in `backend/src/fri/api/main.py`
3. The frontend uses `VITE_API_BASE_URL` env var, defaulting to `http://localhost:8000`
4. API calls go from the **browser** to the host, not container-to-container

### API container unhealthy
```bash
docker logs fri-api-engine --tail=50
# Common cause: model checkpoint missing from artifacts/
```

### API inference slow on first call
- The startup lifespan initializes the graph and model before accepting traffic
- First prediction for a new account takes ~400ms (cold); subsequent calls are ~50ms (LRU cache)
- Explainability endpoint is the most expensive (~37s first call with 50 epochs)

---

## Key Files & Directories

| Path | Purpose |
|------|---------|
| `backend/docker-compose.yml` | Docker orchestration for both services |
| `backend/Dockerfile` | Backend container image definition |
| `frontend/Dockerfile` | Frontend multi-stage build (Node → Nginx) |
| `frontend/nginx.conf` | Nginx config with SPA routing |
| `backend/src/fri/api/main.py` | FastAPI application entry point |
| `frontend/src/services/api.js` | Frontend API client (axios) |
| `frontend/src/components/ForensicDashboard.jsx` | Main UI dashboard component |
| `backend/artifacts/graph/` | Model checkpoints & metrics |
| `backend/configs/` | Application configuration files |

---

## Environment Variables

| Variable | Default | Where | Purpose |
|----------|---------|-------|---------|
| `VITE_API_BASE_URL` | `http://localhost:8000` | Frontend (build-time) | API endpoint for browser requests |
| `PYTHONPATH` | `/app/src` | Backend container | Module resolution |
| `PYTHONDONTWRITEBYTECODE` | `1` | Backend container | Prevents `.pyc` files |
| `PYTHONUNBUFFERED` | `1` | Backend container | Real-time log output |

---

**Last Updated**: June 1, 2026
**Status**: ✅ Production Ready — Both containers running
