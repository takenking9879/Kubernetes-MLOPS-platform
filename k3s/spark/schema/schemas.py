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