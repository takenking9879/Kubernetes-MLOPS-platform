# Serving

## Status
- `partial` (important): serving components exist, but this repository does not present them as production-complete

## Implemented Building Blocks
- Serving config API for run metadata + deployment trigger paths.
- Airflow serving DAGs for RayService patching and SkyPilot-based serving routes.
- Router/canary primitives and MLflow webhook handling for alias updates.
- vLLM and tabular serving template selection in orchestration code.

## Gaps To Production
- Operational hardening and full end-to-end reliability validation.
- Clear SLO/error-budget posture for serving lifecycle.
- More complete automated rollback/runbook coverage.

## Trade-Offs
- Keeping serving in active development enabled rapid experimentation, but current maturity is intentionally labeled partial.

## Evidence Pointers
- `app/backend/routers/serving_configs.py`
- `k3s/airflow/dags/serving_dag.py`
- `k3s/airflow/dags/vllm_serving_dag.py`
- `k3s/kuberay/serving/app.py`
- `src/serve/router.py`
