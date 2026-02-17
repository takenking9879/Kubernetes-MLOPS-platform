"""
Base abstractions for declarative feature engineering pipeline.

This module provides the core interfaces that all pipeline stages must implement.
Follows Spark ML Pipeline architecture but without VectorAssembler dependency.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pyspark.sql import DataFrame
import json


class PipelineStage(ABC):
    """Base class for all pipeline stages."""
    
    def __init__(self, name: str, inputs: list, outputs: list, params: Dict[str, Any]):
        self.name = name
        self.inputs = inputs if isinstance(inputs, list) else [inputs]
        self.outputs = outputs if isinstance(outputs, list) else [outputs]
        self.params = params or {}
        self._fitted = False
    
    @abstractmethod
    def get_type(self) -> str:
        """Return the stage type identifier."""
        pass
    
    def is_fitted(self) -> bool:
        """Check if this stage has been fitted (for estimators)."""
        return self._fitted
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize stage configuration to dictionary."""
        return {
            "type": self.get_type(),
            "name": self.name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "params": self.params,
            "fitted": self._fitted
        }


class Transformer(PipelineStage):
    """
    Transformer: Applies deterministic transformations without learning.
    
    - Does NOT learn state from data
    - fit(df) simply calls transform(df) and returns self
    - transform(df) applies the transformation
    """
    
    def fit(self, df: DataFrame) -> 'Transformer':
        """
        For transformers, fit just validates and returns self.
        The actual transformation happens in transform().
        """
        self._fitted = True
        return self
    
    @abstractmethod
    def transform(self, df: DataFrame) -> DataFrame:
        """Apply the transformation to the DataFrame."""
        pass


class Estimator(PipelineStage):
    """
    Estimator: Learns parameters from data and returns a fitted Transformer.
    
    - DOES learn state from data
    - fit(df) learns parameters and returns a FittedTransformer
    - The fitted transformer can then be used to transform new data
    """
    
    @abstractmethod
    def fit(self, df: DataFrame) -> 'FittedTransformer':
        """
        Learn parameters from the DataFrame.
        Returns a FittedTransformer that can apply the learned transformation.
        """
        pass
    
    def transform(self, df: DataFrame) -> DataFrame:
        """
        Estimators cannot transform before fitting.
        This should raise an error if called before fit().
        """
        if not self._fitted:
            raise RuntimeError(
                f"Estimator {self.name} must be fitted before transform. "
                f"Call fit() first or use a Pipeline."
            )
        raise NotImplementedError("Call fit() first to get a FittedTransformer")


class FittedTransformer(Transformer):
    """
    A Transformer that has been created by fitting an Estimator.
    Contains learned parameters.
    """
    
    def __init__(self, name: str, inputs: list, outputs: list, params: Dict[str, Any],
                 learned_params: Dict[str, Any]):
        super().__init__(name, inputs, outputs, params)
        self.learned_params = learned_params
        self._fitted = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Include learned parameters in serialization."""
        base_dict = super().to_dict()
        base_dict["learned_params"] = self.learned_params
        return base_dict
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'FittedTransformer':
        """Reconstruct a FittedTransformer from serialized config."""
        raise NotImplementedError("Subclasses must implement from_dict")


class PipelineModel:
    """
    Fitted pipeline that can transform new data.
    Contains the sequence of fitted transformers.
    """
    
    def __init__(self, stages: list, config: Dict[str, Any]):
        self.stages = stages
        self.config = config
        self.name = config.get("name", "pipeline")
        self.version = config.get("version", "1.0")
    
    def transform(self, df: DataFrame) -> DataFrame:
        """Apply all stages sequentially to transform the DataFrame."""
        for stage in self.stages:
            if not isinstance(stage, Transformer):
                raise TypeError(f"Stage {stage.name} is not a Transformer")
            df = stage.transform(df)
        return df
    
    def select_features(self, df: DataFrame) -> DataFrame:
        """
        Select only the final features specified in the config.
        Keeps target and metadata columns if present.
        """
        final_features = self.config.get("final_features", {})

        if not isinstance(final_features, dict):
            raise ValueError("Invalid final_features config: expected object with features/target/metadata")

        selected_feature_cols = list(final_features.get("features", []))
        target_cols = list(final_features.get("target", []))
        metadata_cols = list(final_features.get("metadata", []))
        passthrough_cols = list(final_features.get("passthrough", []))
        
        selected_cols = []
        selected_cols.extend(selected_feature_cols)
        
        # Add target if present in both config and DataFrame
        for col in target_cols:
            if col in df.columns:
                selected_cols.append(col)
        
        # Add metadata if present in both config and DataFrame
        for col in metadata_cols:
            if col in df.columns:
                selected_cols.append(col)
        
        # Add passthrough if present in both config and DataFrame
        for col in passthrough_cols:
            if col in df.columns:
                selected_cols.append(col)
        
        # Only select columns that actually exist
        existing_cols = [c for c in selected_cols if c in df.columns]
        
        if not existing_cols:
            raise ValueError("No final features found in DataFrame")
        
        return df.select(*existing_cols)
    
    def save(self, path: str):
        """Save the fitted pipeline to disk."""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save each stage
        stages_config = []
        for i, stage in enumerate(self.stages):
            stage_dict = stage.to_dict()
            stages_config.append(stage_dict)
        
        stages_path = os.path.join(path, "stages.json")
        with open(stages_path, 'w') as f:
            json.dump(stages_config, f, indent=2)
        
        print(f"Pipeline saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'PipelineModel':
        """Load a fitted pipeline from disk."""
        import os
        
        # Load configuration
        config_path = os.path.join(path, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load stages
        stages_path = os.path.join(path, "stages.json")
        with open(stages_path, 'r') as f:
            stages_config = json.load(f)
        
        # Reconstruct stages (will be done by the registry)
        from .state_registry import StageRegistry
        registry = StageRegistry()
        stages = []
        
        for stage_dict in stages_config:
            stage = registry.from_dict(stage_dict)
            stages.append(stage)
        
        return cls(stages, config)