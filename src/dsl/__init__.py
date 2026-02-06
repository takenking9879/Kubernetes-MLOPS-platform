"""
Declarative Feature Engineering Pipeline for PySpark

A production-ready, YAML-driven feature engineering framework that:
- Supports both Transformers (deterministic) and Estimators (learned)
- Maintains explicit columns (no VectorAssembler)
- Scales efficiently with Spark's lazy evaluation
- Serializes and loads complete pipelines

Usage:
    from spark_feature_pipeline import Pipeline
    
    # From YAML
    pipeline = Pipeline.from_yaml("config.yaml")
    model = pipeline.fit(train_df)
    
    # Transform new data
    test_transformed = model.transform(test_df)
    
    # Save and load
    model.save("./pipeline_artifacts")
    loaded_model = PipelineModel.load("./pipeline_artifacts")
"""

from .base import (
    PipelineStage,
    Transformer,
    Estimator,
    FittedTransformer,
    PipelineModel
)

from .pipeline import (
    Pipeline,
    PipelineBuilder,
    PipelineModel
)

from .stage_registry import StageRegistry

from .transformers import (
    CastTransformer,
    TemporalExtractor,
    ArithmeticTransformer,
    ConditionalTransformer,
    CyclicTransformer,
    LogTransformer,
    ConcatTransformer,
    RatioTransformer,
    BinningTransformer,
    ClipTransformer,
    FillNATransformer
)

from .estimators import (
    StringIndexerEstimator,
    StandardScalerEstimator,
    MinMaxScalerEstimator,
    ImputerEstimator
)

__version__ = "1.0.0"

__all__ = [
    # Core abstractions
    "PipelineStage",
    "Transformer",
    "Estimator",
    "FittedTransformer",
    "PipelineModel",
    
    # Pipeline
    "Pipeline",
    "PipelineBuilder",
    "StageRegistry",
    
    # Transformers
    "CastTransformer",
    "TemporalExtractor",
    "ArithmeticTransformer",
    "ConditionalTransformer",
    "CyclicTransformer",
    "LogTransformer",
    "ConcatTransformer",
    "RatioTransformer",
    "BinningTransformer",
    "ClipTransformer",
    "FillNATransformer",
    
    # Estimators
    "StringIndexerEstimator",
    "StandardScalerEstimator",
    "MinMaxScalerEstimator",
    "ImputerEstimator",
]