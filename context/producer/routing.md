# producer/ — Routing

- Change event schema (add/rename fields) → `producer/syntheticgenerator.py` AND `src/converters/spark_kafka_helper.py` AND `src/converters/raw_to_features.py` (all must stay in sync)
- Change attack class distributions → `producer/syntheticgenerator.py` (class probabilities array)
- Change drift behavior → `producer/syntheticgenerator.py` (drift mode logic)
- Change Kafka connection / authentication → `producer/producer.py` (env vars)
- Change publish rate → `producer/producer.py` (interval constant)
