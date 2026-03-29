# producer/ — Mismatches

- **Event schema coupling**: `syntheticgenerator.py` produces `properties.{src_port, dst_port, protocol, packet_count, conn_state, bytes_transferred}`. `src/converters/spark_kafka_helper.py` and `raw_to_features.py` must exactly mirror this schema. Any change in producer must be reflected in both converters.

- **`bytes_transferred` type**: Generator produces `float`; Spark converter casts to `Double`; Python converter returns float. Consistent, but verify if model expects integer counts.

- **Drift modes not integrated into platform**: The generator supports `data_drift` and `concept_drift` but the serving runtime (`src/serve/runtime.py`) has no drift detection. The drift simulation is test-only infrastructure.
