"""
FastAPI backend for Spark Feature Designer.

Endpoints:
  POST /api/schema/from-csv    - Load schema from CSV file
  GET  /api/schema/from-iceberg - Load schema from Iceberg table
  POST /api/dry-run             - Execute pipeline dry-run with PySpark DSL
  GET  /api/health              - Health check
"""

from __future__ import annotations

import hashlib
import time
import traceback
from pathlib import Path
from typing import Any, Optional
import io

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Spark Feature Designer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request / Response Models ───────────────────────────────────────


class ColumnMeta(BaseModel):
    name: str
    sparkType: str
    nullable: bool
    cardinality: Optional[int] = None
    nullRatio: Optional[float] = None


class SparkSchemaResponse(BaseModel):
    columns: list[ColumnMeta]
    sampleRows: Optional[list[dict[str, Any]]] = None
    schemaHash: Optional[str] = None


class SchemaFromCSVRequest(BaseModel):
    path: str
    delimiter: str = ","
    header: bool = True
    sample_rows: int = 1000


class DryRunRequest(BaseModel):
    yaml_content: str = Field(..., alias="yaml")
    datasetPath: str
    sampleLimit: int = 1000

    model_config = {"populate_by_name": True}


class ValidateYAMLRequest(BaseModel):
    yaml_content: str = Field(..., alias="yaml")

    model_config = {"populate_by_name": True}


# ─── Utility: schema hash ────────────────────────────────────────────


def compute_schema_hash(columns: list[ColumnMeta]) -> str:
    """
    Compute a deterministic hash for a schema.

    Algorithm (must match frontend computeSchemaHash):
      1. Sort columns by name
      2. Format each as 'name:sparkType:nullable'
      3. Join with '|'
      4. SHA-256, take first 16 hex chars
    """
    sorted_cols = sorted(columns, key=lambda c: c.name)
    content = "|".join(
        f"{c.name}:{c.sparkType}:{str(c.nullable).lower()}" for c in sorted_cols
    )
    hex_digest = hashlib.sha256(content.encode()).hexdigest()
    return hex_digest[:16]


# ─── Utility: pandas dtype -> Spark type ─────────────────────────────

PANDAS_TO_SPARK: dict[str, str] = {
    "int64": "long",
    "int32": "integer",
    "float64": "double",
    "float32": "double",
    "object": "string",
    "bool": "boolean",
    "datetime64[ns]": "timestamp",
    "datetime64": "timestamp",
    "category": "string",
}


def pandas_to_spark_type(dtype_str: str) -> str:
    """Map a pandas dtype string to a Spark type string."""
    for pd_type, spark_type in PANDAS_TO_SPARK.items():
        if pd_type in dtype_str:
            return spark_type
    return "string"


def generate_spark_schema_from_df(df: pd.DataFrame) -> SparkSchemaResponse:
    """Extract schema metadata from a pandas DataFrame."""
    columns: list[ColumnMeta] = []
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        spark_type = pandas_to_spark_type(dtype_str)
        null_count = int(df[col].isna().sum())
        total = len(df)

        columns.append(
            ColumnMeta(
                name=str(col),
                sparkType=spark_type,
                nullable=null_count > 0,
                cardinality=int(df[col].nunique()),
                nullRatio=round(null_count / total, 4) if total > 0 else 0.0,
            )
        )

    sample_rows = df.head(10).to_dict(orient="records")
    schema_hash = compute_schema_hash(columns)
    return SparkSchemaResponse(columns=columns, sampleRows=sample_rows, schemaHash=schema_hash)


# ─── POST /api/schema/from-csv ───────────────────────────────────────


@app.post("/api/schema/from-csv", response_model=SparkSchemaResponse)
async def get_schema_from_csv(request: SchemaFromCSVRequest):
    """
    Read a CSV file from local path, infer types, and return the schema.
    """
    csv_path = Path(request.path)
    if not csv_path.exists():
        raise HTTPException(status_code=400, detail=f"File not found: {request.path}")

    try:
        df = pd.read_csv(
            csv_path,
            delimiter=request.delimiter,
            header=0 if request.header else None,
            nrows=request.sample_rows,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {exc}") from exc

    return generate_spark_schema_from_df(df)


# ─── POST /api/schema/upload-csv ─────────────────────────────────────


@app.post("/api/schema/upload-file", response_model=SparkSchemaResponse)
async def upload_schema_file(
    file: UploadFile = File(...),
    delimiter: str = ",",
    header: bool = True,
    sample_rows: int = 1000,
):
    """
    Receive an uploaded CSV or Parquet file, infer types from a sample, and return the schema.
    """
    filename = (file.filename or '').lower()
    if not (filename.endswith('.csv') or filename.endswith('.parquet')):
        raise HTTPException(status_code=400, detail="Only CSV and Parquet files are allowed")

    try:
        content = await file.read()
        if filename.endswith('.csv'):
            df = pd.read_csv(
                io.BytesIO(content),
                delimiter=delimiter,
                header=0 if header else None,
                nrows=sample_rows,
            )
        else:
            # Parquet: read using pandas (requires pyarrow or fastparquet in environment)
            df = pd.read_parquet(io.BytesIO(content))
            if sample_rows and len(df) > sample_rows:
                df = df.head(sample_rows)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse uploaded file: {exc}") from exc

    return generate_spark_schema_from_df(df)


# ─── GET /api/schema/from-iceberg ────────────────────────────────────


@app.get("/api/schema/from-iceberg", response_model=SparkSchemaResponse)
async def get_schema_from_iceberg(table: str):
    """Load schema from an Iceberg table. Placeholder — requires catalog setup."""
    raise HTTPException(
        status_code=501,
        detail=f"Iceberg schema loading not yet implemented for table: {table}",
    )


# ─── POST /api/dry-run ───────────────────────────────────────────────


@app.post("/api/dry-run")
async def dry_run(request: DryRunRequest) -> dict[str, Any]:
    """
    Execute a pipeline dry-run:
    1. Parse the YAML config
    2. Load a CSV sample
    3. Execute pipeline.fit() + transform()
    4. Return output schema, preview rows, and metrics
    5. On error: return the raw Spark/Python traceback
    """
    try:
        config = yaml.safe_load(request.yaml_content)
    except yaml.YAMLError as exc:
        return {
            "success": False,
            "error": {"message": f"Invalid YAML: {exc}"},
        }

    csv_path = Path(request.datasetPath)
    if not csv_path.exists():
        return {
            "success": False,
            "error": {"message": f"Dataset file not found: {request.datasetPath}"},
        }

    try:
        import sys

        # Add project root to sys.path so we can import src.dsl
        project_root = str(Path(__file__).resolve().parent.parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from pyspark.sql import SparkSession
        from src.dsl.pipeline import Pipeline

        spark = (
            SparkSession.builder
            .appName("FeatureDesigner-DryRun")
            .master("local[*]")
            .config("spark.ui.enabled", "false")
            .config("spark.driver.memory", "1g")
            .getOrCreate()
        )

        df = spark.read.csv(
            str(csv_path), header=True, inferSchema=True
        ).limit(request.sampleLimit)

        start_time = time.time()
        pipeline = Pipeline.from_config(config)
        model = pipeline.fit(df)
        result_df = model.transform(df)
        elapsed = time.time() - start_time

        output_columns: list[dict[str, Any]] = []
        for field in result_df.schema.fields:
            output_columns.append({
                "name": field.name,
                "sparkType": field.dataType.simpleString(),
                "nullable": field.nullable,
            })

        preview_rows = [row.asDict() for row in result_df.head(20)]
        stage_count = len(model.stages) if hasattr(model, "stages") else 0

        return {
            "success": True,
            "schema": {"columns": output_columns},
            "preview": preview_rows,
            "metrics": {
                "executionTime": round(elapsed, 3),
                "stageCount": stage_count,
            },
        }

    except Exception:
        return {
            "success": False,
            "error": {
                "message": traceback.format_exc().splitlines()[-1],
                "sparkTrace": traceback.format_exc(),
            },
        }


# ─── POST /api/validate-yaml ─────────────────────────────────────────


@app.post("/api/validate-yaml")
async def validate_yaml(request: ValidateYAMLRequest) -> dict[str, Any]:
    """
    Validate YAML structure and stage configs without executing PySpark.
    Returns structural validation errors/warnings.
    """
    try:
        config = yaml.safe_load(request.yaml_content)
    except yaml.YAMLError as exc:
        return {
            "valid": False,
            "errors": [f"Invalid YAML: {exc}"],
            "warnings": [],
        }

    errors: list[str] = []
    warnings: list[str] = []

    pipeline = config.get("pipeline")
    if not pipeline:
        return {
            "valid": False,
            "errors": ["Missing top-level 'pipeline' key"],
            "warnings": [],
        }

    stages = pipeline.get("stages")
    if not stages or not isinstance(stages, list):
        return {
            "valid": False,
            "errors": ["'pipeline.stages' must be a non-empty list"],
            "warnings": [],
        }

    # Lazy import StageRegistry for validation
    try:
        import sys
        project_root = str(Path(__file__).resolve().parent.parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from src.dsl.state_registry import StageRegistry
        registry = StageRegistry()

        for i, stage in enumerate(stages):
            stage_type = stage.get("type")
            stage_name = stage.get("name", f"stage_{i}")

            if not stage_type:
                errors.append(f"Stage {i} ({stage_name}): missing 'type' field")
                continue

            if stage_type not in registry.transformers and stage_type not in registry.estimators:
                errors.append(f"Stage {i} ({stage_name}): unknown type '{stage_type}'")
                continue

            inputs = stage.get("inputs")
            outputs = stage.get("outputs")

            if not inputs:
                errors.append(f"Stage {i} ({stage_name}): missing 'inputs'")

            if not outputs:
                errors.append(f"Stage {i} ({stage_name}): missing 'outputs'")

            # Validate N→N length constraints for one_to_one stages
            if inputs and outputs:
                inputs_list = [inputs] if isinstance(inputs, str) else inputs
                outputs_list = [outputs] if isinstance(outputs, str) else outputs
                one_to_one_types = {
                    "cast_transformer", "temporal_extractor",
                    "arithmetic_transformer", "conditional_transformer",
                    "cyclic_transformer", "log_transformer",
                    "binning_transformer", "clip_transformer",
                    "fillna_transformer", "string_indexer",
                    "frequency_encoder", "standard_scaler",
                    "minmax_scaler", "imputer",
                }
                if stage_type in one_to_one_types and len(inputs_list) != len(outputs_list):
                    errors.append(
                        f"Stage {i} ({stage_name}): one_to_one stage requires equal "
                        f"number of inputs ({len(inputs_list)}) and outputs ({len(outputs_list)})"
                    )

    except ImportError:
        warnings.append(
            "StageRegistry not available — skipping stage type validation"
        )

    # Check meta block
    meta = config.get("meta")
    if not meta:
        warnings.append("No 'meta' block found — treating as legacy v1 format")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stageCount": len(stages),
    }


# ─── Health check ─────────────────────────────────────────────────────


@app.get("/api/health")
async def health():
    return {"status": "ok"}
