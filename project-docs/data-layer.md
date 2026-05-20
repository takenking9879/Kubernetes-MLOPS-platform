# Data Layer

## Status
- `working`: S3-backed raw uploads, Spark ingestion to Iceberg, schema/version references in run APIs
- `partial`: broader storage backends and lifecycle governance beyond current S3-centric setup

## Design
- Dataset APIs create/manage `raw/{dataset}/` storage prefixes.
- Ingestion Spark job writes to Iceberg raw tables.
- Processing runs read raw tables + DSL and produce processed artifacts and metadata.
- Config anchors object storage and warehouse paths in static infra config.

## Why This Design
- Keeps ingestion and transformation decoupled.
- Enables reproducible lineage via run IDs and S3 parameter snapshots.
- Supports schema/version checks before expensive jobs.

## Trade-Offs
- S3/Glue assumptions are deeply wired in current code paths.
- Additional object stores (e.g., R2) require explicit adaptation and validation.

## Evidence Pointers
- `app/backend/routers/datasets.py`
- `k3s/spark/ingestion/ingestion_main.py`
- `app/backend/routers/processing_runs.py`
- `k3s/config.yaml`
