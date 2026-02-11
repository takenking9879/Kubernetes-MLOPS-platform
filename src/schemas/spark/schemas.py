"""Schemas usados por Spark y por utilidades que solo necesitan los nombres
   de columnas. Para entornos donde `pyspark` no está instalado (ej. jobs Ray
   que solo inspeccionan schemas), importamos `pyspark` de forma segura y
   proporcionamos sustitutos ligeros cuando no esté disponible.
"""
try:
    from pyspark.sql.types import StructType, StructField, LongType, DoubleType, StringType, TimestampType
except Exception:
    # Minimal lightweight stand-ins so the module can be imported without pyspark.
    class StructField:
        def __init__(self, name, dtype, nullable=True):
            self.name = name
            self.dataType = dtype
            self.nullable = nullable

        def __repr__(self):
            return f"StructField({self.name}, {self.dataType}, {self.nullable})"

    class StructType(list):
        def __init__(self, fields=None):
            super().__init__(fields or [])
            self.fields = list(self)

        def __repr__(self):
            return f"StructType({self.fields})"

    # Simple dtype placeholders
    class LongType: pass
    class DoubleType: pass
    class StringType: pass
    class TimestampType: pass

schema_features = StructType([
    StructField("src_port", LongType(), True),
    StructField("dst_port", LongType(), True),
    StructField("protocol", StringType(), True),
    StructField("packet_count", LongType(), True),
    StructField("conn_state", StringType(), True),
    StructField("bytes_transferred", DoubleType(), True),
    StructField("timestamp", TimestampType(), True),
])

schema_preprocessed = [
    "protocol_idx",
    "conn_state_idx",
    "protocol_conn_idx",
    "src_port_norm",
    "dst_port_norm",
    "packet_count_norm",
    "bytes_transferred_norm",
    "bytes_log_norm",
    "packet_log_norm",
    "hour_norm",
    "dayofweek_norm",
    "is_weekend_norm",
    "hour_sin_norm",
    "hour_cos_norm",
]

schema_full = StructType([
    StructField("src_port", LongType(), True),
    StructField("dst_port", LongType(), True),
    StructField("protocol", StringType(), True),
    StructField("packet_count", LongType(), True),
    StructField("conn_state", StringType(), True),
    StructField("bytes_transferred", DoubleType(), True),
    StructField("timestamp", TimestampType(), True),
    StructField("attack", LongType(), True)
])

kafka_schema_features = StructType([
    StructField("timestamp", StringType(), True),
    StructField("event_id", StringType(), True),
    StructField(
        "properties",
        StructType([
            StructField("src_port", LongType(), True),
            StructField("dst_port", LongType(), True),
            StructField("protocol", StringType(), True),
            StructField("packet_count", LongType(), True),
            StructField("conn_state", StringType(), True),
            StructField("bytes_transferred", DoubleType(), True),
        ]),
        True
    )
])

prediction_schema = StructType([
                StructField("event_id", StringType(), False),
                StructField("label", LongType(), False)
            ])