from pyspark.sql import DataFrame
from pyspark.sql.functions import col, to_timestamp


def kafka_to_schema_features(df: DataFrame) -> DataFrame:
    """
    Transforma un DataFrame con el esquema de Kafka al esquema de características esperado.

    Contrato fuerte: `df` contiene las columnas anidadas `properties.*` y `timestamp`.
    Si faltan columnas, Spark lanzará una excepción (fail-fast).
    """
    return (
        df
        .select(
            col("properties.src_port").cast("long").alias("src_port"),
            col("properties.dst_port").cast("long").alias("dst_port"),
            col("properties.protocol").alias("protocol"),
            col("properties.packet_count").cast("long").alias("packet_count"),
            col("properties.conn_state").alias("conn_state"),
            col("properties.bytes_transferred").cast("double").alias("bytes_transferred"),
            to_timestamp(col("timestamp")).cast("long").alias("timestamp"),
        )
    )
