# Dataset Schemas

Each dataset folder contains YAML schema definitions for the S3 schema registry.

## S3 Layout

```
s3://k8s-mlops-platform-bucket/schemas/
  datasets/
    network_traffic_raw/
      raw.yaml           ← raw Kafka input schema (kafka_schema_features)
      full.yaml          ← features + target label (schema_full)
      preprocessed.yaml  ← 14 DSL output features (schema_preprocessed)
    <new_dataset>/
      raw.yaml
      full.yaml
      preprocessed.yaml
  models/
    prediction.yaml      ← event_id + label (dataset-agnostic)
```

## Upload Command

```bash
# Upload all schemas for a dataset
BUCKET=k8s-mlops-platform-bucket
DATASET=network_traffic_raw

aws s3 sync k3s/schemas/datasets/${DATASET}/ \
    s3://${BUCKET}/schemas/datasets/${DATASET}/

# Or upload a single file
aws s3 cp k3s/schemas/datasets/${DATASET}/full.yaml \
    s3://${BUCKET}/schemas/datasets/${DATASET}/full.yaml
```

## Adding a New Dataset

1. Create a new folder under `k3s/schemas/datasets/<dataset_name>/`
2. Define `raw.yaml`, `full.yaml`, and `preprocessed.yaml`
3. Upload to S3 using the command above
4. When running the Spark job, set `RAW_TABLE=iceberg.raw.<dataset_name>` —
   the schema is auto-derived from the table name (no code changes required)

## YAML Format

```yaml
fields:
  - {name: field_name, type: long, nullable: true}
  - name: nested_struct
    type: struct
    nullable: false
    fields:
      - {name: inner_field, type: string, nullable: true}
```

Supported types: `string`, `long`, `int`, `integer`, `double`, `float`,
`boolean`, `timestamp`, `date`, `binary`.
