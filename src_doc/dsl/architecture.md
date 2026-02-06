# Architecture & Best Practices

## 🏛️ System Architecture

### Design Principles

1. **Separation of Concerns**
   - Configuration (YAML) vs Implementation (Python)
   - Transformers (stateless) vs Estimators (stateful)
   - Fit logic vs Transform logic

2. **Lazy Evaluation**
   - All operations use Spark's DataFrame API
   - No premature data collection
   - Transformations are chained, not materialized

3. **Type Safety**
   - Clear interfaces (ABC classes)
   - Explicit parameter validation
   - Runtime type checking where needed

4. **Extensibility**
   - Plugin architecture via StageRegistry
   - Easy to add custom stages
   - No core modifications needed

### Component Hierarchy

```
dsl/
│
├── base.py                    # Core abstractions
│   ├── PipelineStage          # Base class for all stages
│   ├── Transformer            # Stateless transformations
│   ├── Estimator              # Learn from data
│   ├── FittedTransformer      # Learned transformation
│   └── PipelineModel          # Fitted pipeline
│
├── transformers.py            # Deterministic stages
│   ├── CastTransformer
│   ├── TemporalExtractor
│   ├── CyclicTransformer
│   └── ... (11 total)
│
├── estimators.py              # Learning stages
│   ├── StringIndexerEstimator
│   ├── StandardScalerEstimator
│   └── ImputerEstimator
│
├── pipeline.py                # Orchestration
│   ├── Pipeline               # Builder from YAML
│   └── PipelineBuilder        # Programmatic builder
│
└── stage_registry.py          # Factory pattern
    └── StageRegistry          # Maps types to classes
```

## 🎯 Design Patterns

### 1. Factory Pattern (StageRegistry)

```python
class StageRegistry:
    """Maps stage types to implementation classes."""
    
    def create_stage(self, config: Dict) -> PipelineStage:
        stage_type = config["type"]
        
        if stage_type in self.transformers:
            return self.transformers[stage_type](...)
        elif stage_type in self.estimators:
            return self.estimators[stage_type](...)
```

**Why?** Decouples configuration from implementation. New stages can be added without modifying core code.

### 2. Strategy Pattern (Transformers/Estimators)

```python
class PipelineStage(ABC):
    @abstractmethod
    def transform(self, df: DataFrame) -> DataFrame:
        pass

class ConcreteTransformer(PipelineStage):
    def transform(self, df: DataFrame) -> DataFrame:
        # Specific transformation logic
        return df.withColumn(...)
```

**Why?** Each stage encapsulates its own transformation logic. Pipeline just orchestrates.

### 3. Builder Pattern (PipelineBuilder)

```python
pipeline = (
    PipelineBuilder()
    .add_transformer(...)
    .add_estimator(...)
    .build()
)
```

**Why?** Fluent API for programmatic construction. Clear, readable code.

### 4. Template Method Pattern (Estimator.fit)

```python
class Estimator:
    def fit(self, df: DataFrame) -> FittedTransformer:
        # 1. Learn parameters (subclass implements)
        params = self._learn_parameters(df)
        
        # 2. Create fitted transformer (template)
        return FittedTransformer(params)
```

**Why?** Enforces consistent fit-transform pattern across all estimators.

## ⚡ Performance Best Practices

### 1. Stage Ordering

```yaml
# ✅ GOOD: Scaling at the end
stages:
  - cast_transformer        # Fast
  - temporal_extractor      # Fast
  - string_indexer          # Learning (one pass)
  - standard_scaler         # Learning (one pass, at end)

# ❌ BAD: Multiple scaling operations
stages:
  - standard_scaler         # Pass 1
  - temporal_extractor
  - standard_scaler         # Pass 2 (unnecessary)
```

**Rule**: Apply all learned transformations (estimators) at the end, after all deterministic transformations.

### 2. Batch Operations

```python
# ✅ GOOD: Scale multiple columns at once
.add_estimator("standard_scaler", "scale_all",
              inputs=["col1", "col2", "col3", "col4"],
              outputs=["col1_norm", "col2_norm", "col3_norm", "col4_norm"])

# ❌ BAD: Scale one at a time
.add_estimator("standard_scaler", "scale_1", inputs=["col1"], ...)
.add_estimator("standard_scaler", "scale_2", inputs=["col2"], ...)
.add_estimator("standard_scaler", "scale_3", inputs=["col3"], ...)
```

**Rule**: Combine related operations into single stages when possible.

### 3. Avoid Unnecessary Transformations

```python
# ✅ GOOD: Only create needed columns
stages:
  - extract hour
  - use hour directly

# ❌ BAD: Create intermediate columns that aren't used
stages:
  - extract hour
  - extract minute  # Not used later
  - extract second  # Not used later
```

**Rule**: Only compute what you'll use in final features.

### 4. Efficient String Operations

```python
# ✅ GOOD: Use native Spark functions
.add_transformer("concat_transformer", "combine",
                inputs=["col1", "col2"],
                params={"separator": "_"})

# ❌ BAD: Use UDFs (slow, breaks optimization)
def concat_udf(a, b):
    return f"{a}_{b}"
```

**Rule**: Always prefer native Spark operations over UDFs.

## 🔧 Production Deployment

### 1. Pipeline Versioning

```yaml
pipeline:
  name: "production_pipeline"
  version: "2.1.3"  # Track versions!
  
  # Document changes
  changelog:
    - "2.1.3: Added new ratio features"
    - "2.1.2: Fixed scaling bug"
```

### 2. Model Artifacts Structure

```
models/
├── v1.0/
│   ├── pipeline_artifacts/
│   │   ├── config.json
│   │   └── stages.json
│   └── metadata.json
│
├── v2.0/
│   └── pipeline_artifacts/
│       ├── config.json
│       └── stages.json
```

### 3. Validation Pipeline

```python
def validate_pipeline_output(df: DataFrame, expected_cols: list):
    """Validate pipeline output before deployment."""
    
    # Check all expected columns exist
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Check no nulls in critical columns
    for col in expected_cols:
        null_count = df.filter(F.col(col).isNull()).count()
        if null_count > 0:
            raise ValueError(f"Column {col} has {null_count} nulls")
    
    # Check data types
    for col in df.columns:
        if "_idx" in col:
            assert df.schema[col].dataType == IntegerType()
        elif "_norm" in col:
            assert df.schema[col].dataType == DoubleType()
```

### 4. Monitoring

```python
class PipelineMonitor:
    """Monitor pipeline health in production."""
    
    def __init__(self, model: PipelineModel):
        self.model = model
        self.metrics = {}
    
    def track_transform(self, df: DataFrame) -> DataFrame:
        """Track metrics during transformation."""
        
        # Record input size
        input_count = df.count()
        
        # Transform
        result = self.model.transform(df)
        
        # Record output size
        output_count = result.count()
        
        # Check for data loss
        if output_count != input_count:
            logging.warning(f"Row count mismatch: {input_count} -> {output_count}")
        
        # Track null rates
        for col in result.columns:
            null_rate = result.filter(F.col(col).isNull()).count() / output_count
            if null_rate > 0.01:  # 1% threshold
                logging.warning(f"High null rate in {col}: {null_rate:.2%}")
        
        return result
```

## 🚨 Common Pitfalls

### 1. Scaling Before Encoding

```yaml
# ❌ WRONG: Scaling categorical columns
stages:
  - standard_scaler         # Tries to scale "protocol" (string)
  - string_indexer          # Encodes to integers

# ✅ CORRECT: Encode first, then scale
stages:
  - string_indexer          # Encodes to integers
  - standard_scaler         # Scales numeric columns
```

### 2. Forgetting to Set Final Features

```python
# ❌ BAD: Transform returns ALL columns (including intermediates)
model = pipeline.fit(df)
result = model.transform(df)  # 50+ columns!

# ✅ GOOD: Select only final features
result = model.select_features(
    model.transform(df)
)  # Only 15 columns
```

### 3. Not Handling Unknowns

```yaml
# ❌ BAD: Fails on new categories
- type: string_indexer
  params:
    handle_invalid: "error"  # Crashes on new values

# ✅ GOOD: Gracefully handle unknowns
- type: string_indexer
  params:
    handle_invalid: "keep"   # Maps to -1
```

### 4. Materializing Too Early

```python
# ❌ BAD: Forces computation
df = df.withColumn("temp", ...)
df.cache()  # Materializes entire dataset
df = df.withColumn("temp2", ...)

# ✅ GOOD: Chain transformations
df = (df
     .withColumn("temp", ...)
     .withColumn("temp2", ...)
     .select(final_cols))  # Only materialize at the end
```

## 🧪 Testing Strategy

### 1. Unit Tests

Test individual transformers and estimators:

```python
def test_cyclic_transformer():
    """Test sin/cos encoding."""
    df = spark.createDataFrame([(0,), (6,), (12,)], ["hour"])
    
    transformer = CyclicTransformer(
        name="hour_sin",
        inputs=["hour"],
        outputs=["hour_sin"],
        params={"function": "sin", "period": 24}
    )
    
    result = transformer.fit(df).transform(df)
    
    # At hour=0 and hour=12, sin should be ~0
    values = result.select("hour_sin").collect()
    assert abs(values[0][0]) < 0.01
    assert abs(values[2][0]) < 0.01
```

### 2. Integration Tests

Test complete pipelines:

```python
def test_full_pipeline():
    """Test end-to-end pipeline."""
    pipeline = Pipeline.from_yaml("config.yaml")
    model = pipeline.fit(train_df)
    result = model.transform(test_df)
    
    # Verify all expected columns exist
    expected = ["protocol_idx", "port_norm", "attack"]
    assert all(col in result.columns for col in expected)
```

### 3. Regression Tests

Compare outputs across versions:

```python
def test_backward_compatibility():
    """Ensure new version produces same results."""
    
    # Load old model
    old_model = PipelineModel.load("models/v1.0/")
    old_result = old_model.transform(test_df)
    
    # Load new model
    new_model = PipelineModel.load("models/v2.0/")
    new_result = new_model.transform(test_df)
    
    # Compare outputs (allowing small numerical differences)
    for col in old_result.columns:
        if col in new_result.columns:
            assert_frame_equal(
                old_result.select(col).toPandas(),
                new_result.select(col).toPandas(),
                atol=1e-6
            )
```

## 📊 Scaling Considerations

### Small Data (< 1GB)
- Use single-node Spark (local mode)
- No special optimizations needed
- Can materialize intermediate results

### Medium Data (1GB - 100GB)
- Use cluster with 4-16 nodes
- Partition data appropriately
- Avoid unnecessary shuffles

### Large Data (> 100GB)
- Use larger cluster (16+ nodes)
- Optimize partitioning strategy
- Consider data skew
- Monitor stage execution times

```python
# Configure partitions based on data size
spark.conf.set("spark.sql.shuffle.partitions", "200")  # Default
spark.conf.set("spark.sql.shuffle.partitions", "1000") # For large data
```

## 🎓 Advanced Topics

### Custom Stage Development

See the "Extending the Pipeline" section in README.md for:
- Creating custom transformers
- Creating custom estimators
- Registering new stage types
- Serialization considerations

### Multi-Language Pipelines

The YAML format allows pipelines to be:
- Shared across teams
- Translated to other languages (Scala, Java)
- Version controlled independently
- Reviewed by non-developers

### MLOps Integration

Pipeline configs can be:
- Stored in feature store (e.g., Feast)
- Versioned in ML platform (e.g., MLflow)
- Deployed via CI/CD
- A/B tested in production

## 📚 Further Reading

- [Spark SQL Programming Guide](https://spark.apache.org/docs/latest/sql-programming-guide.html)
- [Feature Engineering Best Practices](https://developers.google.com/machine-learning/crash-course/representation/feature-engineering)
- [ML Pipeline Patterns](https://martinfowler.com/articles/cd4ml.html)