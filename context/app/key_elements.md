# app/ — Key Elements

See sub-pages for detail:
- `context/app/front/key_elements.md` — 6 pages, components, API client, stores
- `context/app/back/key_elements.md` — 10 routers, 2 services, main.py

## Cross-app facts

- `app/Dockerfile` — Multi-stage: Stage 1 (`python-builder`) installs Python deps + SkyPilot venv (`/opt/sky-venv`). Stage 2 (`spark:4.0.1`) copies Python packages, adds Node.js, builds frontend. Final image runs FastAPI backend + React frontend together.
- `app/backend/main.py` — No auth; CORS `*`; mounts all 10 routers.
- `app/frontend/src/api/platformClient.ts` — Single source of truth for all API calls and TypeScript types. When backend schema changes, update here.
