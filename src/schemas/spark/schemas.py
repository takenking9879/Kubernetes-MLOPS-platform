from pyspark.sql.types import StructType, StructField, LongType, DoubleType, StringType

schema_features = StructType([
    StructField("src_port", LongType(), True),
    StructField("dst_port", LongType(), True),
    StructField("protocol", StringType(), True),
    StructField("packet_count", LongType(), True),
    StructField("conn_state", StringType(), True),
    StructField("bytes_transferred", DoubleType(), True),
    StructField("timestamp", LongType(), True),
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
    StructField("timestamp", LongType(), True),
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