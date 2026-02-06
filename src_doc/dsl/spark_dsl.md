# Declarative Feature Engineering Pipeline for PySpark

A production-ready, YAML-driven feature engineering framework for PySpark that maintains explicit columns (no VectorAssembler), supports both deterministic and learned transformations, and scales efficiently with Spark's lazy evaluation.

## 🎯 Key Features

- **📝 YAML-Driven Configuration**: Define entire pipelines declaratively
- **🔄 Fit-Transform Pattern**: Proper separation between Transformers and Estimators
- **📊 Explicit Columns**: No VectorAssembler - all features remain as explicit columns
- **💾 Serializable**: Save and load complete pipelines with learned parameters
- **⚡ Production-Ready**: Lazy evaluation, no UDFs, scales to large datasets
- **🔌 Compatible**: Works with Parquet, Ray, PyTorch, and other downstream tools
- **🏗️ Extensible**: Easy to add custom transformers and estimators

## 📦 Installation

```bash
# Clone or copy the spark_feature_pipeline directory to your project
# No external dependencies beyond PySpark and PyYAML
pip install pyspark pyyaml
```

## 🚀 Quick Start

### Option 1: YAML Configuration (Recommended)

```python
from spark_feature_pipeline import Pipeline

# Load pipeline from YAML
pipeline = Pipeline.from_yaml("pipeline_config.yaml")

# Fit on training data
model = pipeline.fit(train_df)

# Transform new data
test_transformed = model.transform(test_df)

# Select final features
final_features = model.select_features(test_transformed)

# Save for later use
model.save("./pipeline_artifacts")
```

### Option 2: Programmatic Builder

```python
from spark_feature_pipeline import PipelineBuilder

pipeline = (
    PipelineBuilder()
    .set_name("my_pipeline")
    .add_transformer("cast_transformer", "parse_timestamp",
                    inputs=["timestamp"], outputs=["timestamp_ts"],
                    target_type="timestamp")
    .add_transformer("temporal_extractor", "extract_hour",
                    inputs=["timestamp_ts"], outputs=["hour"],
                    component="hour")
    .add_estimator("string_indexer", "encode_category",
                  inputs=["category"], outputs=["category_idx"],
                  handle_invalid="keep")
    .add_estimator("standard_scaler", "scale_features",
                  inputs=["feature1", "feature2"],
                  outputs=["feature1_norm", "feature2_norm"],
                  with_mean=True, with_std=True)
    .set_final_features(
        categorical=["category_idx"],
        numerical=["feature1_norm", "feature2_norm"],
        target=["label"]
    )
    .build()
)

model = pipeline.fit(train_df)
```

### Option 3: Backward Compatible Interface

```python
from preprocessing_compat import preprocess_spark

# Same interface as original code
df_transformed, model = preprocess_spark(spark_df, train=True)
df_test, _ = preprocess_spark(spark_df_test, model=model, train=False)
```

## 🏗️ Architecture

### Core Abstractions

```
PipelineStage (ABC)
├── Transformer (deterministic, no learning)
│   ├── fit(df) → self  [just validates]
│   └── transform(df) → DataFrame
│
└── Estimator (learns from data)
    ├── fit(df) → FittedTransformer
    └── FittedTransformer
        └── transform(df) → DataFrame

Pipeline
├── fit(df) → PipelineModel
│   └── orchestrates: fit → transform → fit → transform ...
└── PipelineModel
    └── transform(df) → DataFrame
```

### Stage Types

#### Transformers (Deterministic)
- `cast_transformer`: Type conversions
- `temporal_extractor`: Hour, day, month, etc.
- `arithmetic_transformer`: Add, subtract, multiply, divide
- `conditional_transformer`: If-then logic
- `cyclic_transformer`: Sin/cos encoding
- `log_transformer`: Log, log1p, log10
- `concat_transformer`: String concatenation
- `ratio_transformer`: Calculate ratios
- `binning_transformer`: Discretization
- `clip_transformer`: Value clipping
- `fillna_transformer`: Fill missing values

#### Estimators (Learn from Data)
- `string_indexer`: Categorical encoding (learns vocabulary)
- `standard_scaler`: Z-score normalization (learns mean/std)
- `minmax_scaler`: Min-max scaling (learns min/max)
- `imputer`: Fill missing with learned statistics (mean/median/mode)

## 📋 YAML Configuration Example

```yaml
pipeline:
  name: "feature_pipeline"
  version: "1.0"
  
  stages:
    # 1. Parse timestamp
    - type: "cast_transformer"
      name: "timestamp_parser"
      inputs: ["timestamp"]
      output: "timestamp_ts"
      params:
        target_type: "timestamp"
    
    # 2. Extract hour
    - type: "temporal_extractor"
      name: "hour_extractor"
      inputs: ["timestamp_ts"]
      output: "hour"
      params:
        component: "hour"
    
    # 3. Cyclic encoding
    - type: "cyclic_transformer"
      name: "hour_sin"
      inputs: ["hour"]
      output: "hour_sin"
      params:
        function: "sin"
        period: 24.0
    
    # 4. Categorical encoding (learns vocabulary)
    - type: "string_indexer"
      name: "category_encoder"
      inputs: ["category"]
      output: "category_idx"
      params:
        handle_invalid: "keep"
    
    # 5. Numerical scaling (learns mean/std)
    - type: "standard_scaler"
      name: "scaler"
      inputs: ["feature1", "feature2"]
      outputs: ["feature1_norm", "feature2_norm"]
      params:
        with_mean: true
        with_std: true
  
  final_features:
    categorical: ["category_idx"]
    numerical: ["hour_sin", "feature1_norm", "feature2_norm"]
    target: ["label"]
    metadata: ["id"]
```

## 🔄 Pipeline Execution Flow

The pipeline follows Spark ML's fit-transform pattern:

```python
# Training Phase
pipeline = Pipeline.from_yaml("config.yaml")
model = pipeline.fit(train_df)

# What happens internally:
# 1. Stage 1 (Transformer): fit() returns self → transform()
# 2. Stage 2 (Transformer): fit() returns self → transform()
# 3. Stage 3 (Estimator): fit() learns params → returns FittedTransformer → transform()
# 4. Stage 4 (Estimator): fit() learns params → returns FittedTransformer → transform()
# ... and so on

# Each stage sees the output of the previous stage
```

## 💾 Serialization

```python
# Save fitted pipeline
model.save("./pipeline_artifacts")

# Directory structure:
# pipeline_artifacts/
# ├── config.json      # Pipeline configuration
# └── stages.json      # Learned parameters for all stages

# Load later
loaded_model = PipelineModel.load("./pipeline_artifacts")
transformed = loaded_model.transform(new_df)
```

## ⚡ Performance Considerations

### Efficient by Design
- ✅ All operations are Spark-native (no Python UDFs)
- ✅ Lazy evaluation throughout
- ✅ Efficient column operations
- ✅ No unnecessary data collection
- ✅ Scales to billions of rows

### Best Practices
```python
# ✅ Good: Apply scaler at the end (after all features are numeric)
stages:
  - encode categorical
  - engineer features
  - scale everything at once

# ❌ Bad: Multiple scaling operations
stages:
  - scale some features
  - engineer more features
  - scale again
```

## 🔌 Integration Examples

### Export to Parquet
```python
final_df = model.select_features(transformed_df)
final_df.write.parquet("features.parquet")
```

### Use with Ray for Training
```python
import ray

# Features are explicit columns - Ray can use them directly
ray_ds = ray.data.from_spark(final_df)
```

### Use with PyTorch
```python
# Convert to Pandas (if data fits in memory)
pdf = final_df.toPandas()

# Or use petastorm for large-scale
from petastorm.spark import SparkDatasetConverter
converter = SparkDatasetConverter(sc, parquet_path, "file:///tmp/cache")
```

## 🛠️ Extending the Pipeline

### Add Custom Transformer

```python
from spark_feature_pipeline.base import Transformer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

class MyCustomTransformer(Transformer):
    def get_type(self) -> str:
        return "my_custom_transformer"
    
    def transform(self, df: DataFrame) -> DataFrame:
        input_col = self.inputs[0]
        output_col = self.outputs[0]
        
        # Your transformation logic
        return df.withColumn(output_col, F.upper(F.col(input_col)))

# Register it
from spark_feature_pipeline import StageRegistry
registry = StageRegistry()
registry.register_transformer("my_custom_transformer", MyCustomTransformer)
```

### Add Custom Estimator

```python
from spark_feature_pipeline.base import Estimator, FittedTransformer

class MyEstimator(Estimator):
    def get_type(self) -> str:
        return "my_estimator"
    
    def fit(self, df: DataFrame) -> 'MyFittedTransformer':
        # Learn parameters
        learned_value = df.select(F.max(self.inputs[0])).collect()[0][0]
        
        learned_params = {"max_value": learned_value}
        self._fitted = True
        
        return MyFittedTransformer(
            name=self.name,
            inputs=self.inputs,
            outputs=self.outputs,
            params=self.params,
            learned_params=learned_params
        )

class MyFittedTransformer(FittedTransformer):
    def get_type(self) -> str:
        return "my_estimator"
    
    def transform(self, df: DataFrame) -> DataFrame:
        max_val = self.learned_params["max_value"]
        return df.withColumn(
            self.outputs[0],
            F.col(self.inputs[0]) / F.lit(max_val)
        )
    
    @classmethod
    def from_dict(cls, config):
        return cls(
            config["name"], config["inputs"], config["outputs"],
            config["params"], config["learned_params"]
        )

# Register both
registry.register_estimator("my_estimator", MyEstimator, MyFittedTransformer)
```

## 📊 Comparison with Original Approach

| Aspect | Original | This Framework |
|--------|----------|----------------|
| Configuration | Hardcoded in Python | YAML-driven |
| Reusability | Low | High |
| Testing | Difficult | Easy |
| Extensibility | Manual | Plugin-based |
| Maintainability | Low | High |
| Serialization | Custom dict | Built-in |
| Documentation | Code comments | Self-documenting YAML |
| Output Format | Explicit columns ✅ | Explicit columns ✅ |

## 🐛 Debugging

```python
# Validate pipeline before running
pipeline.validate()

# Print pipeline summary
print(pipeline.summary())

# List all stages
print(pipeline.list_stages())

# Get specific stage
stage = pipeline.get_stage("my_stage_name")
print(stage.to_dict())

# Inspect fitted model
for stage in model.stages:
    print(f"Stage: {stage.name}")
    if hasattr(stage, 'learned_params'):
        print(f"  Learned: {stage.learned_params}")
```

## 📚 Complete Example

See `example_usage.py` for a full working example that:
- Creates sample data
- Loads pipeline from YAML
- Fits and transforms
- Saves and loads model
- Exports to Parquet
- Shows programmatic builder usage

Run it:
```bash
python example_usage.py
```

## 🤝 Contributing

To add new stage types:
1. Create transformer/estimator class in appropriate module
2. Register in `StageRegistry`
3. Document in YAML schema
4. Add tests

## 📄 License

This is a reference implementation. Adapt as needed for your use case.

## 🙏 Credits

Designed to replace and improve upon manual feature engineering approaches while maintaining Spark's performance characteristics and avoiding VectorAssembler limitations.