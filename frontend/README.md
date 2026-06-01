# FRI Analytics Dashboard (Frontend)

React + Vite SPA for account risk search, graph visualization, explainability, and drift monitoring.

## Setup

```bash
npm install
npm run dev
```

Dev server: http://localhost:5173

## API connection

The UI calls the backend using `VITE_API_BASE_URL` (build-time). Default:

```bash
# .env.local (optional)
VITE_API_BASE_URL=http://localhost:8000
```

Start the API from [`../backend`](../backend) before using the dashboard.

## Production build

```bash
npm run build
npm run preview
```

## Docker

Built from this folder; orchestrated with the API via [`../backend/docker-compose.yml`](../backend/docker-compose.yml):

```bash
cd ../backend
docker compose up --build -d fri-frontend
```

Served on http://localhost:3000
