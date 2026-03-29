# src/serve/ — Key Elements

## ModelRuntime (runtime.py)

Function/Class: `ModelRuntime`
File: `src/serve/runtime.py`

Does:
- Stateful predictor for one model variant (stable or canary)
- `predict(payload)` → predictions + latency + metadata
- `_build_matrix(payload)` — two paths:
  1. `{"data": [[f1…f14]]}` → direct to adapter (pre-processed)
  2. `{"raw": {timestamp, event_id, properties}}` → NumpyPipelineExecutor → adapter
- Reads DSL YAML `final_features[]` for column ordering
- Emits Prometheus metrics

Inputs: payload dict (one of two formats above)
Outputs: `{predictions, probabilities, latency_ms, model_version, framework}`
Side effects: Prometheus counters + histograms

Depends on: `src/dsl/numpy_executor.py`, `src/serve/adapters.py`, `src/serve/registry.py`

Prometheus metrics:
- `serve_infer_requests_total` (Counter)
- `serve_infer_errors_total` (Counter)
- `serve_infer_latency_ms` (Histogram, boundaries: 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000)

---

## ConfigLoader (config.py)

Function/Class: `ConfigLoader.load()`
File: `src/serve/config.py`

Does:
- Merges config from 3 sources (priority order):
  1. `kuberay.serving/canary` in params.yaml (new format)
  2. `PARAMS_SERVING_PATH` / `PARAMS_SERVING_S3_PATH` env var (overlay file)
  3. `config.yaml` `serving.*` (static infra defaults, lowest priority)
- Returns `ServingConfig` dataclass

Key fields in ServingConfig: `model.tracking_uri`, `model.registry_name`, `model.default_alias`, `model.dsl_path`, `webhook.public_base_url`, `canary.alias`, `canary.probability`, `canary_enabled`

---

## MLflowRegistry (registry.py)

Function/Class: `MLflowRegistry`
File: `src/serve/registry.py`

Does:
- `load_by_alias(registry_name, alias)` → `ModelArtifact` (loaded model + framework + version)
  - Reads `tags["framework"]` on model version (xgboost / pytorch)
  - Calls `mlflow.xgboost.load_model()` or `mlflow.pytorch.load_model()`
- `resolve_alias_version(registry_name, alias)` → version string
- `ensure_webhook(webhook_config, registry_name, alias)` → creates/updates MLflow webhook
  - Events: `model_version_alias.created/deleted` (preferred) or legacy fallback

---

## Adapters (adapters.py)

| Class | Framework | Input | Output |
|-------|-----------|-------|--------|
| `XGBoostAdapter` | XGBoost | `List[List[float]]` | `{predictions[], probabilities[]}` |
| `PyTorchAdapter` | PyTorch | `List[List[float]]` | `{predictions[], probabilities[]}` |

XGBoost: validates feature names against model's `feature_names`. Legacy models skip check.
PyTorch: auto-detects CUDA; falls back to CPU.

---

## TrafficRouter (router.py)

Function/Class: `TrafficRouter`
File: `src/serve/router.py`

Does:
- Routes requests to stable or canary handle by probability
- `route(payload)` → async call to stable (1-p) or canary (p)
- `set_canary_probability(p)` — updated on webhook event; clamped [0.0, 1.0]

---

## PipelineArtifactLoader (pipeline_loader.py)

Function/Class: `PipelineArtifactLoader`
File: `src/serve/pipeline_loader.py`

Does:
- `load_executor(registry_name, alias)` → `NumpyPipelineExecutor`

Resolution chain:
1. MLflow: read tag `artifact_set_id` from model version
2. Iceberg: query `preprocessing_artifacts` table → `pipeline_hash`
3. S3: download `{bucket}/pipelines/{pipeline_hash}/stages.json` + `config.json`
4. `NumpyPipelineExecutor.from_dir(tmpdir)`

Depends on: `src/iceberg/metadata_query_utils.py`, boto3, MLflow client
