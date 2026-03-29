# app/ — Purpose

### Purpose
User-facing application layer. Split into two logical zones:
- **front/** — React SPA (TypeScript + Tailwind) with 6 pages
- **back/** — FastAPI server with 10 routers; bridges the UI with K8s, Airflow, S3, Iceberg, and MLflow

### When to use
- Modifying UI behavior, adding pages, changing component props → `front/`
- Adding/changing API endpoints, request/response schemas → `back/`
- All user-initiated actions (upload, train, serve, monitor) flow through `app/`

### When not to use
- Core feature engineering logic → `src/dsl/`
- Serving runtime / inference → `src/serve/`
- Orchestration / DAGs → `k3s/`
- GPU provider queries → `src/services/gpu_catalog.py`
- Training pipeline (Ray Train) → `src/pipeline/`

### Key elements
See:
- `context/app/front/key_elements.md` — pages, components, stores, API client
- `context/app/back/key_elements.md` — routers, services, endpoints

### Physical layout
```
app/
  backend/
    main.py            ← FastAPI app; mounts all routers
    routers/           ← 10 domain routers
    services/          ← orchestration_selector.py, job_builder.py
  frontend/
    src/
      pages/           ← 6 pages
      components/      ← shared components
      api/
        platformClient.ts  ← ALL backend calls
      store/           ← Zustand stores (uiStore, datasetStore, pipelineStore)
  Dockerfile           ← Multi-stage: python-builder (deps + SkyPilot venv) → spark:4.0.1 + Node
```
