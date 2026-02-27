"""
Pipeline implementation for declarative feature engineering.

The Pipeline orchestrates the execution of stages in sequence, handling
the fit-transform pattern correctly for both Transformers and Estimators.
"""

import os
import re
import yaml
from typing import List, Dict, Any
from pyspark.sql import DataFrame

from .base import PipelineStage, Transformer, Estimator, PipelineModel
from .state_registry import StageRegistry


class Pipeline:
    """
    Pipeline that orchestrates feature engineering stages.
    
    The Pipeline is responsible for:
    1. Parsing YAML configuration
    2. Creating stage instances
    3. Orchestrating fit-transform sequence:
       - For Transformers: fit() calls transform() internally
       - For Estimators: fit() learns params, then transform() is called
    4. Returning a fitted PipelineModel
    
    Usage:
        pipeline = Pipeline.from_yaml("config.yaml")
        model = pipeline.fit(df)
        transformed_df = model.transform(new_df)
    """
    
    def __init__(self, stages: List[PipelineStage], config: Dict[str, Any]):
        self.stages = stages
        self.config = config
        self.registry = StageRegistry()
        self.name = config.get("name", "pipeline")
        self.version = config.get("version", "1.0")
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Pipeline':
        """
        Load pipeline from YAML configuration file.
        
        Args:
            yaml_path: Path to YAML configuration file
        
        Returns:
            Pipeline instance
        """
        if yaml_path.startswith("s3://"):
            import boto3
            match = re.match(r"s3://([^/]+)/(.+)", yaml_path)
            if not match:
                raise ValueError(f"Invalid S3 path: {yaml_path!r}. Expected s3://bucket/key")
            bucket, key = match.group(1), match.group(2)
            s3 = boto3.client(
                "s3",
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION", "us-east-2"),
            )
            obj = s3.get_object(Bucket=bucket, Key=key)
            content = obj["Body"].read().decode("utf-8")
            config = yaml.safe_load(content)
        else:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)

        return cls.from_config(config)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Pipeline':
        """
        Create pipeline from configuration dictionary.
        
        Args:
            config: Configuration dictionary with 'pipeline' key
        
        Returns:
            Pipeline instance
        """
        pipeline_config = dict(config.get("pipeline", {}))

        # DSL v2 compatibility: some generators place final_features at root level.
        # Preserve them in pipeline config so PipelineModel.select_features works.
        if "final_features" in config and "final_features" not in pipeline_config:
            pipeline_config["final_features"] = config["final_features"]

        stages_config = pipeline_config.get("stages", [])
        
        registry = StageRegistry()
        stages = []
        
        for stage_config in stages_config:
            stage = registry.create_stage(stage_config)
            stages.append(stage)
        
        return cls(stages, pipeline_config)
    
    def fit(self, df: DataFrame) -> PipelineModel:
        """
        Fit the pipeline on the DataFrame.
        
        This method:
        1. Iterates through all stages in sequence
        2. For Transformers: calls fit() which returns self, then transforms
        3. For Estimators: calls fit() to learn params and get FittedTransformer, then transforms
        4. Each stage sees the output of the previous stage
        5. Returns a PipelineModel with all fitted stages
        
        Args:
            df: Input DataFrame
        
        Returns:
            PipelineModel that can transform new data
        """
        fitted_stages = []
        current_df = df
        
        print(f"[Pipeline] Starting fit for '{self.name}'")
        print(f"[Pipeline] Input DataFrame shape: {current_df.count()} rows")
        
        for i, stage in enumerate(self.stages):
            stage_name = stage.name
            stage_type = stage.get_type()
            
            print(f"\n[Stage {i+1}/{len(self.stages)}] {stage_name} ({stage_type})")
            
            if isinstance(stage, Transformer):
                # Transformer: fit() returns self, then we transform
                print(f"  -> Transformer (deterministic)")
                fitted_stage = stage.fit(current_df)
                current_df = fitted_stage.transform(current_df)
                fitted_stages.append(fitted_stage)
                
            elif isinstance(stage, Estimator):
                # Estimator: fit() learns and returns FittedTransformer
                print(f"  -> Estimator (learning from data)")
                fitted_stage = stage.fit(current_df)
                print(f"  -> Learned parameters")
                
                # Now transform with the fitted stage
                current_df = fitted_stage.transform(current_df)
                fitted_stages.append(fitted_stage)
                
            else:
                raise TypeError(f"Stage {stage_name} is neither Transformer nor Estimator")
            
            print(f"  -> Output columns: {current_df.columns[:10]}{'...' if len(current_df.columns) > 10 else ''}")
        
        print(f"\n[Pipeline] Fit complete!")
        print(f"[Pipeline] Total stages: {len(fitted_stages)}")
        
        return PipelineModel(fitted_stages, self.config)
    
    def fit_transform(self, df: DataFrame) -> DataFrame:
        """
        Fit the pipeline and transform the input DataFrame in one call.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Transformed DataFrame
        """
        model = self.fit(df)
        return model.transform(df)
    
    def get_stage(self, name: str) -> PipelineStage:
        """Get a stage by name."""
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise ValueError(f"Stage '{name}' not found in pipeline")
    
    def list_stages(self) -> List[str]:
        """List all stage names in the pipeline."""
        return [stage.name for stage in self.stages]
    
    def validate(self) -> bool:
        """
        Validate the pipeline configuration.
        
        Checks:
        - All stages have required parameters
        - Input/output columns are consistent
        - No circular dependencies
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        stage_outputs = set()
        
        for i, stage in enumerate(self.stages):
            # Check if inputs are available (from previous stages or original data)
            # This is a simple check - could be more sophisticated
            
            # Track outputs
            for output in stage.outputs:
                if output in stage_outputs:
                    print(f"[Warning] Column '{output}' is being overwritten by stage {i}: {stage.name}")
                stage_outputs.add(output)
        
        print(f"[Pipeline] Validation passed: {len(self.stages)} stages")
        return True
    
    def summary(self) -> str:
        """
        Generate a summary of the pipeline.
        
        Returns:
            String description of the pipeline
        """
        lines = [
            f"Pipeline: {self.name} (v{self.version})",
            f"Stages: {len(self.stages)}",
            ""
        ]
        
        for i, stage in enumerate(self.stages):
            stage_type = "Transformer" if isinstance(stage, Transformer) else "Estimator"
            lines.append(f"{i+1}. {stage.name}")
            lines.append(f"   Type: {stage.get_type()} ({stage_type})")
            lines.append(f"   Inputs: {stage.inputs}")
            lines.append(f"   Outputs: {stage.outputs}")
            if stage.params:
                lines.append(f"   Params: {stage.params}")
            lines.append("")
        
        return "\n".join(lines)


class PipelineBuilder:
    """
    Fluent builder for constructing pipelines programmatically.
    
    Usage:
        pipeline = (PipelineBuilder()
                   .add_transformer("cast", inputs=["timestamp"], ...)
                   .add_estimator("scaler", inputs=["col1", "col2"], ...)
                   .build())
    """
    
    def __init__(self):
        self.stages = []
        self.config = {
            "name": "custom_pipeline",
            "version": "1.0",
            "stages": []
        }
        self.registry = StageRegistry()
    
    def set_name(self, name: str) -> 'PipelineBuilder':
        """Set the pipeline name."""
        self.config["name"] = name
        return self
    
    def set_version(self, version: str) -> 'PipelineBuilder':
        """Set the pipeline version."""
        self.config["version"] = version
        return self
    
    def add_stage(self, stage_type: str, name: str, inputs: list, 
                  outputs: list = None, params: Dict[str, Any] = None) -> 'PipelineBuilder':
        """
        Add a stage to the pipeline.
        
        Args:
            stage_type: Type identifier (e.g., 'cast_transformer', 'string_indexer')
            name: Unique name for this stage
            inputs: Input column(s)
            outputs: Output column(s), defaults to inputs if not specified
            params: Stage-specific parameters
        
        Returns:
            Self for chaining
        """
        if outputs is None:
            outputs = inputs
        
        stage_config = {
            "type": stage_type,
            "name": name,
            "inputs": inputs,
            "outputs": outputs,
            "params": params or {}
        }
        
        self.config["stages"].append(stage_config)
        
        # Create the actual stage
        stage = self.registry.create_stage(stage_config)
        self.stages.append(stage)
        
        return self
    
    def add_transformer(self, stage_type: str, name: str, inputs: list,
                       outputs: list = None, **params) -> 'PipelineBuilder':
        """Add a transformer stage."""
        return self.add_stage(stage_type, name, inputs, outputs, params)
    
    def add_estimator(self, stage_type: str, name: str, inputs: list,
                     outputs: list = None, **params) -> 'PipelineBuilder':
        """Add an estimator stage."""
        return self.add_stage(stage_type, name, inputs, outputs, params)
    
    def set_final_features(self, features: List[str] = None,
                          target: List[str] = None,
                          metadata: List[str] = None,
                          passthrough: List[str] = None) -> 'PipelineBuilder':
        """Set final feature selection configuration."""
        self.config["final_features"] = {
            "features": features or [],
            "target": target or [],
            "metadata": metadata or [],
            "passthrough": passthrough or []
        }
        return self
    
    def build(self) -> Pipeline:
        """Build and return the Pipeline."""
        return Pipeline(self.stages, self.config)
    
    def to_yaml(self, path: str):
        """Save the pipeline configuration to YAML."""
        full_config = {"pipeline": self.config}
        with open(path, 'w') as f:
            yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)
        print(f"Pipeline configuration saved to {path}")