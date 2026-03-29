# src/converters/ — Purpose

### Purpose
Converts raw Kafka network traffic events to flat feature dicts. Two implementations: Spark (batch) and Python (online, zero Spark dependency).

### When to use
- Changing raw event schema (new fields from Kafka)
- Fixing type casting (timestamp, int fields)
- Adding new input fields to the feature pipeline

### When not to use
- DSL feature transforms → `src/dsl/`
- Online inference logic → `src/serve/`

### Physical layout
```
src/converters/
  spark_kafka_helper.py   ← Spark DataFrame converter (batch, training/preprocessing)
  raw_to_features.py      ← Pure-Python converter (online, Ray Serve)
```

### Key Elements

**spark_kafka_helper.py**
Function: `kafka_to_schema_features(df)`
Input: Spark DataFrame with Kafka event schema (event_id, properties.{src_port, dst_port, protocol, packet_count, conn_state, bytes_transferred}, timestamp)
Output: Spark DataFrame with flattened, typed features (Long, Double) + epoch timestamp (Long)
File: `src/converters/spark_kafka_helper.py`

Registry: `get_converter(name)`, `register_converter(name, func)` — extensibility for custom converters

---

**raw_to_features.py**
Function: `raw_event_to_features(raw_dict)`
Input: `{timestamp, event_id, properties: {src_port, ...}}`
Output: flat feature dict (same schema as Spark output)
File: `src/converters/raw_to_features.py`

Timestamp normalization `_to_epoch_seconds(ts)`:
- Handles: Unix epoch (int/float), Python datetime, ISO-8601 with Z, ISO-8601 with offset, naive (→ UTC)
- Must match Spark `to_timestamp().cast("long")` behavior exactly

### When to update both files
If raw event schema changes (new field, type change) — update BOTH `spark_kafka_helper.py` AND `raw_to_features.py`. They must produce identical feature dicts for the same input.
