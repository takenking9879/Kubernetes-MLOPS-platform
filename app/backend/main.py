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
import csv
import os

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Spark Feature Designer API", version="0.1.0")

# ─── Dataset-oriented platform routers (v2) ──────────────────────────────────
from routers import datasets as _datasets_router
from routers import dsls as _dsls_router
from routers import runs as _runs_router

app.include_router(_datasets_router.router)
app.include_router(_dsls_router.router)
app.include_router(_runs_router.router)
# ─────────────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Detect project root (K3s git-sync or local)
if Path("/git-data/repo").exists():
    PROJECT_ROOT = Path("/git-data/repo")
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Default to /tmp for broad compatibility, but allow override (e.g., .uploaded_datasets)
UPLOAD_CACHE_DIR = Path(os.getenv("UPLOAD_CACHE_DIR", "/tmp/dsl_uploaded_datasets"))

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
    uploadedPath: Optional[str] = None


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


def ensure_upload_cache_dir() -> Path:
    UPLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_CACHE_DIR


def persist_uploaded_dataset(file_name: str, content: bytes) -> Path:
    cache_dir = ensure_upload_cache_dir()
    base_name = Path(file_name).name or "uploaded_dataset"
    digest = hashlib.sha256(content).hexdigest()[:12]
    stored_name = f"{digest}__{base_name}"
    target = cache_dir / stored_name
    target.write_bytes(content)
    return target.resolve()


def normalize_spark_columns(df):
    seen: dict[str, int] = {}
    renamed_df = df

    for index, original in enumerate(df.columns):
        candidate = str(original).lstrip("\ufeff").strip().strip('"').strip("'").strip("`").strip()
        if not candidate:
            candidate = f"col_{index}"

        duplicate_count = seen.get(candidate, 0)
        final_name = candidate if duplicate_count == 0 else f"{candidate}_{duplicate_count}"
        seen[candidate] = duplicate_count + 1

        if final_name != original:
            renamed_df = renamed_df.withColumnRenamed(original, final_name)

    return renamed_df


def detect_csv_delimiter(csv_path: Path) -> str:
    candidates = [",", ";", "\t", "|"]
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            sample = handle.read(8192)
            if not sample.strip():
                return ","

            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=";,\t|")
                if dialect.delimiter in candidates:
                    return dialect.delimiter
            except csv.Error:
                pass

            first_line = next((line for line in sample.splitlines() if line.strip()), "")
            if not first_line:
                return ","

            best = max(candidates, key=lambda delimiter: first_line.count(delimiter))
            return best if first_line.count(best) > 0 else ","
    except Exception:
        return ","


def is_parquet_file(file_path: Path) -> bool:
    """
    Detect Parquet by magic bytes, independent of file extension.
    Parquet files start and end with the bytes: PAR1
    """
    try:
        if not file_path.exists() or not file_path.is_file() or file_path.stat().st_size < 8:
            return False

        with file_path.open("rb") as handle:
            header = handle.read(4)
            handle.seek(-4, 2)
            footer = handle.read(4)

        return header == b"PAR1" and footer == b"PAR1"
    except Exception:
        return False


def detect_dataset_format(dataset_path: Path) -> Optional[str]:
    """
    Detect dataset format robustly.

    Priority:
    1. Known extension
    2. Parquet magic bytes
    3. Text-like file => CSV/TXT
    """
    suffix = dataset_path.suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if suffix in {".csv", ".txt", ".tsv"}:
        return "csv"

    if is_parquet_file(dataset_path):
        return "parquet"

    try:
        with dataset_path.open("rb") as handle:
            sample = handle.read(8192)
        if not sample:
            return "csv"

        sample.decode("utf-8-sig")
        return "csv"
    except Exception:
        return None


def resolve_dataset_path(dataset_ref: str) -> Optional[Path]:
    if not dataset_ref or not dataset_ref.strip():
        return None
    
    requested = Path(dataset_ref.strip())

    candidates: list[Path] = []
    if requested.is_absolute():
        candidates.append(requested)
    else:
        candidates.extend([
            Path.cwd() / requested,
            Path(__file__).resolve().parent / requested,
            PROJECT_ROOT / requested,
        ])

    cache_dir = ensure_upload_cache_dir()
    candidates.append(cache_dir / requested.name)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    if requested.name:
        uploaded_matches = sorted(
            cache_dir.glob(f"*__{requested.name}"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if uploaded_matches:
            return uploaded_matches[0].resolve()

    return None


# ─── POST /api/schema/from-csv ───────────────────────────────────────


@app.post("/api/schema/from-csv", response_model=SparkSchemaResponse)
async def get_schema_from_csv(request: SchemaFromCSVRequest):
    """
    Read a CSV file from local path, infer types, and return the schema.
    """
    resolved_path = resolve_dataset_path(request.path)
    if not resolved_path:
        raise HTTPException(status_code=400, detail=f"File not found: {request.path}")

    try:
        df = pd.read_csv(
            resolved_path,
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
        stored_path = persist_uploaded_dataset(file.filename or "uploaded_dataset", content)
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

    schema = generate_spark_schema_from_df(df)
    return SparkSchemaResponse(
        columns=schema.columns,
        sampleRows=schema.sampleRows,
        schemaHash=schema.schemaHash,
        uploadedPath=str(stored_path),
    )


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
    2. Load a dataset sample (CSV/Parquet/Iceberg)
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

    dataset_path = request.datasetPath
    if not dataset_path:
        return {
            "success": False,
            "error": {"message": "No dataset path provided. Please select a table or upload a file."},
        }

    try:
        import sys
        # Add project root to sys.path so we can import src.dsl
        project_root = str(PROJECT_ROOT)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from pyspark.sql import SparkSession
        from src.dsl.pipeline import Pipeline

        # Initialize Spark with Iceberg support if needed
        builder = (
            SparkSession.builder
            .appName("FeatureDesigner-DryRun")
            .master("local[*]")
            .config("spark.ui.enabled", "false")
            .config("spark.driver.memory", "1g")
        )

        if dataset_path.startswith("iceberg:"):
            from routers.datasets import AWS_REGION, S3_BUCKET
            builder = (
                builder
                .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
                .config("spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog")
                .config("spark.sql.catalog.iceberg.type", "glue")
                .config("spark.sql.catalog.iceberg.warehouse", f"s3a://{S3_BUCKET}/warehouse/")
                .config("spark.hadoop.fs.s3a.endpoint", os.getenv("S3_ENDPOINT", "http://minio-service.magda:9000"))
                .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID", ""))
                .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY", ""))
                .config("spark.hadoop.fs.s3a.path.style.access", "true")
            )

        spark = builder.getOrCreate()

        if dataset_path.startswith("iceberg:"):
            table_name = dataset_path.replace("iceberg:", "")
            # Assume local dry-run uses the same catalog name as production if possible
            # or just use the table name if it's qualified
            if "." not in table_name:
                table_name = f"iceberg.raw.{table_name}"
            df = spark.table(table_name).limit(request.sampleLimit)
        else:
            resolved_path = resolve_dataset_path(dataset_path)
            if not resolved_path:
                 return {
                    "success": False,
                    "error": {"message": f"Dataset file not found: {dataset_path}"},
                }
            
            detected_format = detect_dataset_format(resolved_path)
            if detected_format == "parquet":
                df = spark.read.parquet(str(resolved_path)).limit(request.sampleLimit)
            elif detected_format == "csv":
                detected_delimiter = detect_csv_delimiter(resolved_path)
                df = spark.read.option("sep", detected_delimiter).csv(
                    str(resolved_path), header=True, inferSchema=True
                ).limit(request.sampleLimit)
            else:
                return {
                    "success": False,
                    "error": {"message": f"Unsupported format for path: {dataset_path}"},
                }
                "success": False,
                "error": {
                    "message": (
                        "Could not detect dataset format. "
                        "Provide a valid CSV or Parquet file (or use .csv/.parquet extension)."
                    )
                },
            }

        df = normalize_spark_columns(df)

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
