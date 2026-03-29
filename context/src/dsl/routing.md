# src/dsl/ — Routing

- Add stateless transform → `transformers.py` + mirror in `numpy_executor.py` + register in `state_registry.py`
- Add learned transform (estimator) → `estimators.py` (Estimator + FittedTransformer) + mirror in `numpy_executor.py` + register in `state_registry.py`
- Change feature selection (which columns go to model) → `base.py:PipelineModel.select_features()`
- Change how YAML is loaded → `pipeline.py:Pipeline.__init__()`
- Change fit orchestration order → `pipeline.py:Pipeline.fit()`
- Fix a timestamp conversion bug → both `transformers.py:TemporalExtractor` AND `numpy_executor.py` (must stay in sync)
- Fix dayofweek mapping → `numpy_executor.py` (Python must match Spark: `(isoweekday() % 7) + 1`)
- Add a custom stage type externally → `state_registry.register_transformer()` / `register_estimator()`
