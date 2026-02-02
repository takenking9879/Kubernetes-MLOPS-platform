"""
Spark Kafka Helper - Funciones de conversión de schemas.
Mantiene conversiones específicas y provee acceso dinámico.
"""
from typing import Callable
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, to_timestamp


# ===== CONVERSORES EXISTENTES =====

def kafka_to_schema_features(df: DataFrame) -> DataFrame:
    """
    Transforma un DataFrame con el esquema de Kafka al esquema de características esperado.

    Contrato fuerte: `df` contiene las columnas anidadas `properties.*` y `timestamp`.
    Si faltan columnas, Spark lanzará una excepción (fail-fast).
    """
    return (
        df
        .select(
            col("event_id"),
            col("properties.src_port").cast("long").alias("src_port"),
            col("properties.dst_port").cast("long").alias("dst_port"),
            col("properties.protocol").alias("protocol"),
            col("properties.packet_count").cast("long").alias("packet_count"),
            col("properties.conn_state").alias("conn_state"),
            col("properties.bytes_transferred").cast("double").alias("bytes_transferred"),
            to_timestamp(col("timestamp")).cast("long").alias("timestamp"),
        )
    )


# ===== REGISTRO DINÁMICO DE CONVERSORES =====

# Registry de conversores disponibles
_CONVERTERS_REGISTRY = {
    'kafka_to_schema_features': kafka_to_schema_features,
    # Agrega más conversores aquí según los necesites:
    # 'features_to_model': features_to_model_format,
    # 'model_to_output': model_to_output_format,
}


def get_converter(converter_name: str) -> Callable[[DataFrame], DataFrame]:
    """
    Obtiene una función conversora por nombre.
    
    Args:
        converter_name: Nombre del conversor (ej: "kafka_to_schema_features")
    
    Returns:
        Función conversora que acepta y retorna DataFrame
    
    Raises:
        KeyError: Si el conversor no existe
    
    Example:
        >>> converter = get_converter("kafka_to_schema_features")
        >>> df_converted = converter(df_kafka)
    """
    if converter_name not in _CONVERTERS_REGISTRY:
        available = ', '.join(_CONVERTERS_REGISTRY.keys())
        raise KeyError(
            f"Converter '{converter_name}' not found. "
            f"Available converters: {available}"
        )
    return _CONVERTERS_REGISTRY[converter_name]


def register_converter(name: str, converter_func: Callable[[DataFrame], DataFrame]):
    """
    Registra un conversor personalizado.
    
    Args:
        name: Nombre del conversor
        converter_func: Función que acepta DataFrame y retorna DataFrame
    
    Example:
        >>> def my_converter(df):
        ...     return df.select("col1", "col2")
        >>> register_converter("my_custom_converter", my_converter)
    """
    _CONVERTERS_REGISTRY[name] = converter_func


def list_converters() -> list:
    """Lista todos los conversores disponibles."""
    return list(_CONVERTERS_REGISTRY.keys())