# producer/ — Purpose

### Purpose
Standalone synthetic Kafka data producer for testing and model training. Generates network traffic events with configurable distribution (normal, data drift, concept drift) and publishes to a Kafka topic.

### When to use
- Generating test data for the ML pipeline
- Testing Kafka consumer / Spark Kafka connector
- Simulating data drift or concept drift scenarios

### When not to use
- Production data ingestion (this is test/synthetic only)
- Feature engineering — `src/dsl/`
- Kafka consumer / inference — `src/converters/spark_kafka_helper.py`

### Physical layout
```
producer/
  producer.py              ← Kafka producer (Confluent Kafka, SASL support)
  syntheticgenerator.py    ← Synthetic traffic generator (rule-based, drift modes)
```
