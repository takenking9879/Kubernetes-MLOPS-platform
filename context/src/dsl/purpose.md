# src/dsl/ — Purpose

### Purpose
Declarative feature engineering pipeline. YAML-configured; fits on Spark DataFrames; runs deterministic inference in pure Python (NumPy). The same DSL YAML drives both batch training (Spark) and online inference (Ray Serve).

### When to use
- Adding a new feature transform or encoder
- Changing fit/transform behavior
- Debugging preprocessing output

### When not to use
- Serving runtime → `src/serve/`
- Data conversion from raw Kafka events → `src/converters/`

### Physical layout
```
src/dsl/
  base.py             ← PipelineStage, Transformer, Estimator, FittedTransformer, PipelineModel
  transformers.py     ← 11 stateless transformers (Spark SQL)
  estimators.py       ← 5 estimators + 5 FittedTransformers (Spark SQL)
  pipeline.py         ← Pipeline (YAML loader), PipelineBuilder (fluent API)
  numpy_executor.py   ← Online mirror of all stages (pure Python, no Spark)
  state_registry.py   ← Factory: create_stage(config), from_dict(config)
```
