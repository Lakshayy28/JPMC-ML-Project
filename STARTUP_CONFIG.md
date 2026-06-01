# Financial Risk Intelligence Platform - Startup Configuration

**Date**: June 1, 2026  
**Status**: ✅ Successfully Started

## System Overview

The Financial Risk Intelligence (FRI) Platform is a microservices-based fraud detection system consisting of:
- **Backend API**: FastAPI microservice running in Docker
- **Frontend UI**: React/Vite application running locally
- **ML Engine**: PyTorch-based Graph Neural Network (Hetero GAT) for fraud detection

---

## Startup Configuration

### Prerequisites

- **Docker Desktop**: Running on macOS (started automatically)
- **Node.js**: v18+ (for frontend development)
- **Python**: 3.11+ (for backend, containerized)
- **Network Connectivity**: Ports 8000 (API) and 5173 (UI) available

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    User Browser                              │
├─────────────────────────────────────────────────────────────┤
│                         UI                                   │
│                    Vite Dev Server                           │
│              http://localhost:5173                           │
│        (React + Tailwind + Recharts)                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ HTTP/REST
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   Docker Container                          │
│                   fri-api-engine                             │
│                  http://localhost:8000                       │
│              (FastAPI + PyTorch)                             │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Endpoints:                                          │  │
│  │  • GET /health - Health check                        │  │
│  │  • GET /predict/{account_id} - Account risk score    │  │
│  │  • GET /explain/{account_id} - XAI explanations      │  │
│  │  • POST /analyze-drift - MLOps drift detection       │  │
│  │  • GET /metrics - Prometheus metrics export          │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Service Details

### API Server (Docker)

**Container Name**: `fri-api-engine`  
**Port**: 8000  
**Technology Stack**: FastAPI, PyTorch, PyG, PyTorch Geometric  
**Model**: Heterogeneous Graph Attention Network (Hetero GAT)  

**Backend-only Startup Command**:
```bash
docker compose up --build -d fri-api-engine
```

**Health Check**:
```bash
curl -s http://localhost:8000/health | jq .
```

**Expected Response**:
```json
{
  "status": "healthy",
  "model": "hetero_gat"
}
```

**Key Features**:
- ✅ Account-level fraud risk scoring
- ✅ Transaction pattern analysis
- ✅ Graph-based anomaly detection
- ✅ Explainable AI (XAI) using GNNExplainer
- ✅ MLOps drift monitoring (KS test)
- ✅ Prometheus metrics export
- ✅ Cold-start inference: ~0.392s
- ✅ Warm prediction: <50ms

---

### Frontend Server (Local)

**Port**: 5173 (Vite default)  
**Technology Stack**: React 18, Vite 6, Tailwind CSS, Recharts  

**Local Development Startup Command**:
```bash
cd frontend
npm install
npm run dev
```

**Access**: http://localhost:5173

### Frontend Server (Docker)

**Container Name**: `fri-frontend`  
**Port**: 3000  
**Startup Command**:
```bash
docker compose up --build -d fri-frontend
```

**Access**: http://localhost:3000

**Components**:
- Account risk dashboard
- Transaction graph visualization
- Model prediction interface
- Drift monitoring dashboard
- Explainability viewer

---

## Complete Startup Sequence

### Step 1: Clean Up Existing Processes
```bash
pkill -f "docker.*compose" || true
pkill -f "vite" || true
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000,5173 | xargs kill -9 2>/dev/null || true
```

### Step 2: Start Docker Daemon
```bash
open -a Docker
sleep 10  # Wait for Docker to start
```

### Step 3: Start API Server
```bash
cd "/Users/lakshaychandra/JPMC ML Project"
docker compose up --build -d fri-api-engine
sleep 5   # Wait for container to initialize
```

### Step 4: Verify API Health
```bash
curl -s http://localhost:8000/health | jq .
```

### Step 5: Install Frontend Dependencies
```bash
cd frontend
npm install
```

### Step 6: Start Frontend Development Server
```bash
npm run dev
```

### Step 7: Access the Application
- **API**: http://localhost:8000
- **UI**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs (Swagger UI)

---

## API Endpoints

### Health Check
```bash
curl -X GET http://localhost:8000/health
```

### Predict Account Risk
```bash
curl -X GET http://localhost:8000/predict/19204
```

**Response Example**:
```json
{
  "account_id": 19204,
  "fraud_probability": 0.9985247254371643,
  "is_high_risk": true,
  "threshold_used": 0.49999999999999994
}
```

### Get Explainability
```bash
curl -X GET "http://localhost:8000/explain/19204?epochs=12"
```

### MLOps Drift Analysis
```bash
curl -X POST http://localhost:8000/analyze-drift \
  -H "Content-Type: application/json" \
  -d '[{"outgoing_tx_velocity_30d": 12.0, "total_amount": 5000.0}]'
```

### Prometheus Metrics
```bash
curl -s http://localhost:8000/metrics
```

---

## Verification Checklist

- ✅ Docker daemon running
- ✅ API container (fri-api-engine) up and healthy
- ✅ API health endpoint responding with 200 OK
- ✅ Frontend dependencies installed
- ✅ Vite dev server started on port 5173
- ✅ Both services accessible from localhost

---

## Performance Metrics

| Operation | Cold Start | Warm | Notes |
|-----------|-----------|------|-------|
| Health Check | ~50ms | ~10ms | Instant response |
| Predict (Account) | ~392ms | ~50ms | CPU inference |
| Explain (GNNExplainer) | ~37s | ~60ms | 50 epochs, LRU-cached |
| Drift Analysis | ~200ms | ~50ms | KS-based distribution test |
| Metrics Export | ~30ms | ~15ms | Prometheus format |

---

## Troubleshooting

### Docker not starting
```bash
open -a Docker
# Wait for the daemon to be ready
docker ps  # Should return container list
```

### Port already in use
```bash
# Find process on port 8000
lsof -i :8000

# Kill process
kill -9 <PID>
```

### Frontend not connecting to API
- Verify API is running: `curl http://localhost:8000/health`
- Check CORS settings in [src/fri/api/main.py](src/fri/api/main.py)
- Ensure frontend is configured to call `http://localhost:8000`
- Local development origin `http://localhost:5173` and Docker frontend origin `http://localhost:3000` are both allowed by the API CORS middleware.

### API inference slow
- API startup is the expensive phase because graph/model state is initialized before Uvicorn accepts traffic.
- Subsequent predictions use LRU cache (~50ms)
- Explanation is the expensive endpoint. The API defaults to `epochs=12` for interactive use and caches repeated explanation calls.

---

## Development Commands

### View API Logs
```bash
docker compose logs -f fri-api-engine
```

### Stop All Services
```bash
docker compose down
pkill -f "vite"
```

### Rebuild Docker Image
```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

### Run Frontend in Production Mode
```bash
cd frontend
npm run build
npm run preview
```

---

## Key Files & Directories

| Path | Purpose |
|------|---------|
| [docker-compose.yml](docker-compose.yml) | Docker orchestration |
| [Dockerfile](Dockerfile) | API container image |
| [src/fri/api/](src/fri/api/) | FastAPI application |
| [frontend/src/](frontend/src/) | React application |
| [artifacts/graph/](artifacts/graph/) | Model checkpoints & metrics |
| [scripts/run_api.py](scripts/run_api.py) | Local API runner |

---

## Deployment Notes

- **Production UI Port**: The docker-compose.yml specifies port 3000 for the containerized frontend, while local development uses Vite's default port 5173
- **Model Architecture**: Heterogeneous Graph Attention Network (Hetero GAT) trained on AMLSim 20K dataset
- **Model Checkpoint**: Located at `artifacts/graph/pytorch_hetero_gat_model.pt`
- **PYTHONPATH**: Set to `/app/src` in Docker container for proper module resolution

---

## Next Steps

1. Access the Dockerized UI at http://localhost:3000 or the local dev UI at http://localhost:5173
2. Navigate to the account search page
3. Enter an account ID (e.g., 19204)
4. View risk score, transactions, and explanations
5. Monitor drift detection in the MLOps dashboard

---

**Last Updated**: June 1, 2026  
**Status**: ✅ Production Ready
