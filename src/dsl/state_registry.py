"""
Stage Registry for mapping stage types to their implementations.

This module provides a factory for creating pipeline stages from configuration.
"""

from typing import Dict, Any
from .base import PipelineStage, Transformer, Estimator, FittedTransformer
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
    FrequencyFittedTransformer,
    StringIndexerEstimator,
    StringIndexerFittedTransformer,
    StandardScalerEstimator,
    StandardScalerFittedTransformer,
    MinMaxScalerEstimator,
    MinMaxScalerFittedTransformer,
    ImputerEstimator,
    ImputerFittedTransformer,
    FrequencyEncoderEstimator,
    FrequencyFittedTransformer
)


class StageRegistry:
    """
    Registry for all available pipeline stages.
    Maps stage types to their implementation classes.
    """
    
    def __init__(self):
        # Transformers (deterministic, no learning)
        self.transformers = {
            "cast_transformer": CastTransformer,
            "temporal_extractor": TemporalExtractor,
            "arithmetic_transformer": ArithmeticTransformer,
            "conditional_transformer": ConditionalTransformer,
            "cyclic_transformer": CyclicTransformer,
            "log_transformer": LogTransformer,
            "concat_transformer": ConcatTransformer,
            "ratio_transformer": RatioTransformer,
            "binning_transformer": BinningTransformer,
            "clip_transformer": ClipTransformer,
            "fillna_transformer": FillNATransformer,
        }
        
        # Estimators (learn from data)
        self.estimators = {
            "string_indexer": StringIndexerEstimator,
            "standard_scaler": StandardScalerEstimator,
            "minmax_scaler": MinMaxScalerEstimator,
            "imputer": ImputerEstimator,
            "frequency_encoder": FrequencyEncoderEstimator,
        }
        
        # Fitted transformers (for loading saved pipelines)
        self.fitted_transformers = {
            "string_indexer": StringIndexerFittedTransformer,
            "standard_scaler": StandardScalerFittedTransformer,
            "minmax_scaler": MinMaxScalerFittedTransformer,
            "imputer": ImputerFittedTransformer,
            "frequency_encoder": FrequencyFittedTransformer,
        }
    
    def create_stage(self, stage_config: Dict[str, Any]) -> PipelineStage:
        """
        Create a stage instance from configuration.
        
        Args:
            stage_config: Dictionary with 'type', 'name', 'inputs', 'outputs', 'params'
        
        Returns:
            PipelineStage instance (Transformer or Estimator)
        """
        stage_type = stage_config["type"]
        name = stage_config["name"]
        inputs = stage_config.get("inputs", [])
        outputs = stage_config.get("outputs", [])
        params = stage_config.get("params", {})
        
        # Ensure inputs and outputs are lists
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(outputs, str):
            outputs = [outputs]
        
        # Check if it's a transformer
        if stage_type in self.transformers:
            cls = self.transformers[stage_type]
            return cls(name, inputs, outputs, params)
        
        # Check if it's an estimator
        elif stage_type in self.estimators:
            cls = self.estimators[stage_type]
            return cls(name, inputs, outputs, params)
        
        else:
            raise ValueError(f"Unknown stage type: {stage_type}")
    
    def from_dict(self, stage_dict: Dict[str, Any]) -> PipelineStage:
        """
        Reconstruct a stage from a saved dictionary.
        
        This is used when loading a saved pipeline. If the stage has learned_params,
        it creates a FittedTransformer. Otherwise, it creates a regular stage.
        """
        stage_type = stage_dict["type"]
        
        # Check if this is a fitted transformer (has learned_params)
        if "learned_params" in stage_dict and stage_dict["learned_params"]:
            if stage_type in self.fitted_transformers:
                cls = self.fitted_transformers[stage_type]
                return cls.from_dict(stage_dict)
            else:
                raise ValueError(f"No fitted transformer registered for type: {stage_type}")
        
        # Otherwise, create a regular stage
        return self.create_stage(stage_dict)
    
    def register_transformer(self, type_name: str, cls: type):
        """Register a custom transformer class."""
        if not issubclass(cls, Transformer):
            raise TypeError(f"Class {cls} must be a subclass of Transformer")
        self.transformers[type_name] = cls
    
    def register_estimator(self, type_name: str, estimator_cls: type, fitted_cls: type):
        """Register a custom estimator and its fitted transformer."""
        if not issubclass(estimator_cls, Estimator):
            raise TypeError(f"Class {estimator_cls} must be a subclass of Estimator")
        if not issubclass(fitted_cls, FittedTransformer):
            raise TypeError(f"Class {fitted_cls} must be a subclass of FittedTransformer")
        
        self.estimators[type_name] = estimator_cls
        self.fitted_transformers[type_name] = fitted_cls
    
    def list_transformers(self):
        """List all registered transformer types."""
        return list(self.transformers.keys())
    
    def list_estimators(self):
        """List all registered estimator types."""
        return list(self.estimators.keys())