"""Declarative Feature Engineering Pipeline package.

This module intentionally uses lazy imports so non-Spark runtimes (e.g. Ray
Serve online inference) can import ``src.dsl.numpy_executor`` without requiring
``pyspark`` at package import time.
"""

from importlib import import_module

__version__ = "1.0.0"

_EXPORTS = {
    # Core abstractions
    "PipelineStage": (".base", "PipelineStage"),
    "Transformer": (".base", "Transformer"),
    "Estimator": (".base", "Estimator"),
    "FittedTransformer": (".base", "FittedTransformer"),
    "PipelineModel": (".pipeline", "PipelineModel"),
    # Pipeline
    "Pipeline": (".pipeline", "Pipeline"),
    "PipelineBuilder": (".pipeline", "PipelineBuilder"),
    "StageRegistry": (".state_registry", "StageRegistry"),
    # Transformers
    "CastTransformer": (".transformers", "CastTransformer"),
    "TemporalExtractor": (".transformers", "TemporalExtractor"),
    "ArithmeticTransformer": (".transformers", "ArithmeticTransformer"),
    "ConditionalTransformer": (".transformers", "ConditionalTransformer"),
    "CyclicTransformer": (".transformers", "CyclicTransformer"),
    "LogTransformer": (".transformers", "LogTransformer"),
    "ConcatTransformer": (".transformers", "ConcatTransformer"),
    "RatioTransformer": (".transformers", "RatioTransformer"),
    "BinningTransformer": (".transformers", "BinningTransformer"),
    "ClipTransformer": (".transformers", "ClipTransformer"),
    "FillNATransformer": (".transformers", "FillNATransformer"),
    # Estimators
    "StringIndexerEstimator": (".estimators", "StringIndexerEstimator"),
    "StandardScalerEstimator": (".estimators", "StandardScalerEstimator"),
    "MinMaxScalerEstimator": (".estimators", "MinMaxScalerEstimator"),
    "ImputerEstimator": (".estimators", "ImputerEstimator"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value