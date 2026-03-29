# src/dsl/ — Key Elements

## Core Abstractions (base.py)

| Class | Role |
|-------|------|
| `PipelineStage` | Base: name, inputs[], outputs[], params{}, fitted flag |
| `Transformer` | Stateless; `fit()` validates, `transform()` applies |
| `Estimator` | Learns params; `fit()` → `FittedTransformer` |
| `FittedTransformer` | Holds `learned_params{}` (JSON-serializable); `transform()` applies |
| `PipelineModel` | Sequence of fitted stages; `transform()`, `select_features()`, `save()`, `load()` |

`PipelineModel.select_features()` narrows output to `final_features` columns defined in DSL YAML.

## Transformers (transformers.py) — 11 stages, all stateless

| Class | Operation |
|-------|-----------|
| `CastTransformer` | Type cast (timestamp→double, string→int, etc.) |
| `TemporalExtractor` | Extract hour/dayofweek/month/quarter from timestamp |
| `ArithmeticTransformer` | +, −, ×, ÷, power, abs, negate |
| `ConditionalTransformer` | isin, >, <, ==, is_null → 0/1 |
| `CyclicTransformer` | sin/cos encoding for periodic features |
| `LogTransformer` | log, log1p, log10, log2 |
| `ConcatTransformer` | Concatenate string columns with separator |
| `RatioTransformer` | Division with zero-guard |
| `BinningTransformer` | Bucket continuous → discrete by bin edges |
| `ClipTransformer` | Clamp to [min, max] |
| `FillNATransformer` | Fill nulls with constant |

## Estimators (estimators.py) — 5 learned stages

| Estimator | Learns | `learned_params` key |
|-----------|--------|---------------------|
| `StringIndexerEstimator` | str→int vocabulary | `mappings[col] = {str: idx, num_labels}` |
| `StandardScalerEstimator` | mean, std per column | `stats[col] = {mean, std}` |
| `MinMaxScalerEstimator` | min, max per column | `stats[col] = {min, max, range_min, range_max}` |
| `ImputerEstimator` | imputation value (mean/median/mode) | `imputation_values[col] = value` |
| `FrequencyEncoderEstimator` | frequency/count encoding + top-k | `mappings[col] = {str: freq, encoding, other_key}` |

All FittedTransformers support `from_dict(config)` for pipeline reconstruction from S3 artifacts.

## NumpyPipelineExecutor (numpy_executor.py)

Function/Class: `NumpyPipelineExecutor`
File: `src/dsl/numpy_executor.py`

Does:
- Loads stages from `stages.json` + `config.json` (downloaded from S3)
- `transform(row_dict)` → enriched row dict (all intermediate + final features)
- `transform_to_vector(row_dict)` → `List[float]` ordered by `final_features`

Inputs: `row_dict: {col: scalar}` (flat dict; can include nested raw event)
Outputs: feature vector ready for model inference

Depends on: json, math, datetime (no Spark)

Implements all 16 stage types as pure Python. Timestamp handling: Unix epoch (int/float), ISO-8601 strings, Python datetime → datetime objects.

## Pipeline (pipeline.py)

Function/Class: `Pipeline`
File: `src/dsl/pipeline.py`

Does:
- Loads YAML from local path or S3
- Instantiates stages via `state_registry.create_stage()`
- `fit(df)` → `PipelineModel` (all stages fitted sequentially)

## state_registry.py

Function/Class: `create_stage(config)`, `from_dict(config)`, `register_transformer()`, `register_estimator()`
File: `src/dsl/state_registry.py`

Does:
- Factory pattern: maps `type` key in config dict to class
- `register_*` enables extensibility for custom stages

## Spark dayofweek note
Spark `dayofweek()` returns 1=Sun, 2=Mon…7=Sat.
Python equivalent: `(isoweekday() % 7) + 1`
This must match in `numpy_executor.py`.
