# src/serve/ — Routing

- Change prediction payload handling → `runtime.py:ModelRuntime.predict()` + `_build_matrix()`
- Change feature vector ordering → `runtime.py:_resolve_dsl_meta()` (reads `final_features` from DSL YAML)
- Change model loading from MLflow → `registry.py:MLflowRegistry.load_by_alias()`
- Change how XGBoost inference works → `adapters.py:XGBoostAdapter`
- Change how PyTorch inference works → `adapters.py:PyTorchAdapter`
- Change canary traffic probability → `router.py:TrafficRouter.set_canary_probability()`
- Change how serving config is loaded (env vars, YAML keys) → `config.py:ConfigLoader.load()`
- Change DSL artifact resolution chain → `pipeline_loader.py:PipelineArtifactLoader`
- Change webhook creation/management → `registry.py:MLflowRegistry.ensure_webhook()`
