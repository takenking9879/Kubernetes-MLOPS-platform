# Decommission Checklist

Items to remove once all consumers have migrated to the new `runs/{stage}/{id}/` S3 structure
and the new DAGs (`preprocessing_pipeline`, `training_pipeline`, `full_ml_pipeline`) are stable.

**Do NOT execute these steps until the new architecture has been validated end-to-end.**

---

## S3 — Legacy `params/` folder

The old system wrote all params into `s3://k8s-mlops-platform-bucket/params/{execution_id}/`.
No new runs write there. Once all consumers read from `runs/preprocessing/` and `runs/training/`,
delete the legacy folder:

```bash
# DRY RUN first — review output before deleting
aws s3 rm s3://k8s-mlops-platform-bucket/params/ --recursive --dryrun

# Then delete for real
aws s3 rm s3://k8s-mlops-platform-bucket/params/ --recursive
```

---

## Airflow — Legacy `ml_pipeline` DAG

The monolithic `ml_pipeline` DAG has been superseded by three separate DAGs:
- `preprocessing_pipeline` (Spark only)
- `training_pipeline` (Ray only)
- `full_ml_pipeline` (Spark + Ray chained)

Steps:
1. In the Airflow UI, **pause** `ml_pipeline` (already done in step 5 of the migration plan).
2. After confirming no active runs reference it, delete the file:
   ```bash
   rm k3s/airflow/dags/ml_pipeline_dag.py
   ```
3. Trigger a DAG refresh in Airflow to deregister it.

---

## Backend — Legacy fallback in `GET .../params` endpoints

### `app/backend/routers/processing_runs.py`

The `GET /api/v2/processing-runs/{execution_id}/params` endpoint has a backward-compat fallback
that tries `params/{execution_id}/params_preprocess.yaml` if the new path
`runs/preprocessing/{execution_id}/params_preprocess.yaml` is not found.

Once all active runs use the new path, remove the fallback block:

```python
# REMOVE this fallback block from processing_runs.py:
# legacy path: params/{execution_id}/params_preprocess.yaml
legacy_key = f"params/{execution_id}/params_preprocess.yaml"
try:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=legacy_key)
    ...
except ClientError:
    pass
```

### `app/backend/routers/runs.py`

Same pattern — remove fallback from `GET /api/v2/runs/{execution_id}/params` if it exists.

---

## Backend — `pipeline_mode` field

The `RunRequest` used to have a `pipeline_mode: "full" | "preprocessing_only" | "training_only"`
field. This has been replaced by separate DAGs. Remove any remaining references to this field
in `runs.py` and `processing_runs.py`.

Also remove the constant `DAG_ID = "ml_pipeline"` from both routers once all runs target
the new DAG IDs (`preprocessing_pipeline`, `training_pipeline`, `full_ml_pipeline`).

---

## Frontend — `RunPage.tsx` (already cleaned)

The `RunPage.tsx` no longer contains `SchemaBuilderSection`, `rawDatasetFilter`, `processedTable`,
or the legacy `processed_table` field in `RunRequest`. No further cleanup needed here.

---

## Frontend — `SchemaBuilderSection` component

`SchemaBuilderSection` (full 3-tab version) is no longer used in `RunPage.tsx`.
It may still be referenced by other pages. Once `ProcessingPage.tsx` uses `FullYamlEditor`
directly and no page imports `SchemaBuilderSection`, remove:

```
app/frontend/src/components/schema/SchemaBuilderSection.tsx
app/frontend/src/lib/schemaYaml.ts  # only if all generators are inlined
```

---

## `k3s/spark/main.py` — Implicit schema derivation (already replaced)

The old code derived the schema S3 path implicitly:
```python
# OLD — REMOVE if still present as a comment:
# raw_table_short = self.raw_table.split(".")[-1]
# schema_s3 = f"s3://{bucket}/schemas/datasets/{raw_table_short}/full.yaml"
```

The new code reads `params.schema_ref.full` explicitly. The `SCHEMA_S3_PATH` env var override
can also be removed once all Spark jobs consume `schema_ref` from params.

---

## Verification before cleanup

```bash
# 1. No active runs in params/ (should return empty or only old runs)
aws s3 ls s3://k8s-mlops-platform-bucket/params/ | sort -k1

# 2. All recent runs in new paths
aws s3 ls s3://k8s-mlops-platform-bucket/runs/preprocessing/ | tail -5
aws s3 ls s3://k8s-mlops-platform-bucket/runs/training/ | tail -5
aws s3 ls s3://k8s-mlops-platform-bucket/runs/serving/ | tail -5

# 3. No Airflow runs scheduled on ml_pipeline (should be paused)
# Check via Airflow UI or CLI: airflow dags list-runs --dag-id ml_pipeline

# 4. Schema references in params files are explicit
aws s3 cp s3://k8s-mlops-platform-bucket/runs/preprocessing/$(aws s3 ls s3://k8s-mlops-platform-bucket/runs/preprocessing/ | tail -1 | awk '{print $2}')params_preprocess.yaml /tmp/check.yaml
grep schema_ref /tmp/check.yaml  # must show version + full path
```
