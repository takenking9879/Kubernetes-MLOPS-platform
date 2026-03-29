# producer/ — Key Elements

## syntheticgenerator.py

Function/Class: `SyntheticTrafficGenerator` (or module-level functions)
File: `producer/syntheticgenerator.py`

Does:
- Generates synthetic network traffic events in 6 attack classes:
  - Normal (94.2%), DoS (2%), Probe (1%), R2L (1.5%), U2R (0.8%), Worm (0.5%)
- 3 drift modes:
  - `normal` — baseline distributions; label = sampled attack class
  - `data_drift` — feature distributions shift (beta-weighted); label semantics unchanged
  - `concept_drift` — features sampled normally; label assigned by fX_concept oracle (different boundary)
- O(1) per sample (Alias method for categorical sampling)
- Batch API: `generate_dataset(n, trend)` → pandas DataFrame

Output event schema:
```json
{
  "timestamp": <ISO-8601 string>,
  "event_id": <UUID string>,
  "properties": {
    "src_port": int,
    "dst_port": int,
    "protocol": str,
    "packet_count": int,
    "conn_state": str,
    "bytes_transferred": float
  }
}
```

Attack semantics: DoS → high packet_count + ICMP/UDP proto; Probe → many ports; etc.

---

## producer.py

Function/Class: `main()`
File: `producer/producer.py`

Does:
- Kafka producer (Confluent Kafka); publishes at 100 msgs/sec (interval=0.01s)
- Supports SASL/SCRAM authentication; defaults PLAINTEXT
- Graceful shutdown on SIGINT/SIGTERM

Env vars:
- `KAFKA_BOOTSTRAP_SERVERS` (default: localhost:9092)
- `KAFKA_TOPIC` (default: topic-traffic)
- `KAFKA_USERNAME`, `KAFKA_PASSWORD` (optional SASL)
- `KAFKA_SASL_MECHANISM` (default: PLAIN)
- `KAFKA_SECURITY_PROTOCOL` (default: PLAINTEXT)

Side effects: publishes to Kafka topic `topic-traffic`
Depends on: `syntheticgenerator.py`, confluent-kafka

---

## Data Flow

```
syntheticgenerator.produce(trend)
  ↓
{timestamp, event_id, properties: {src_port, dst_port, protocol, packet_count, conn_state, bytes_transferred}}
  ↓
Confluent Kafka producer → topic: topic-traffic
  ↓
Spark Kafka connector (k3s/spark/inference/spark-kafka-application.yaml) consumes
  ↓
src/converters/spark_kafka_helper.py:kafka_to_schema_features()
  ↓
Iceberg / Ray Serve
```
