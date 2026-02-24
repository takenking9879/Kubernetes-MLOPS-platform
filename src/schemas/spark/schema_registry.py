"""
Schema Registry - Acceso dinámico a schemas por nombre.

Static schemas (for the current datasets) are defined in schemas.py.
Dynamic schemas for new datasets can be loaded from S3 YAML files using
SchemaRegistry.load_from_s3(). The YAML format is:

    fields:
      - {name: src_port, type: long, nullable: true}
      - {name: protocol, type: string, nullable: true}
      - name: properties
        type: struct
        nullable: false
        fields:
          - {name: nested_field, type: long, nullable: true}

Supported type strings: string, long, int, integer, double, float,
boolean, timestamp, date, binary.
"""
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import os

# Import pyspark types lazily/safely so this module can be imported
# in non-Spark runtimes (e.g., Ray jobs without pyspark installed).
# Use TYPE_CHECKING so static checkers (pylance) see the real types,
# while at runtime we still handle pyspark absence gracefully.
if TYPE_CHECKING:
    from pyspark.sql.types import (
        StructType, StructField,
        StringType, LongType, IntegerType, DoubleType, FloatType,
        BooleanType, TimestampType, DateType, BinaryType,
    )
    _SPARK_AVAILABLE = True
else:
    try:
        from pyspark.sql.types import (
            StructType, StructField,
            StringType, LongType, IntegerType, DoubleType, FloatType,
            BooleanType, TimestampType, DateType, BinaryType,
        )
        _SPARK_AVAILABLE = True
    except Exception:
        StructType = None  # type: ignore
        _SPARK_AVAILABLE = False

from . import schemas


# ---------------------------------------------------------------------------
# S3 Schema Loader helpers
# ---------------------------------------------------------------------------

def _parse_spark_type(type_str: str):
    """Map a YAML type string to a Spark DataType instance."""
    _map = {
        "string":    StringType(),
        "long":      LongType(),
        "int":       IntegerType(),
        "integer":   IntegerType(),
        "double":    DoubleType(),
        "float":     FloatType(),
        "boolean":   BooleanType(),
        "timestamp": TimestampType(),
        "date":      DateType(),
        "binary":    BinaryType(),
    }
    t = type_str.lower()
    if t in _map:
        return _map[t]
    raise ValueError(f"Unsupported schema type: {type_str!r}")


def _build_struct_type(fields: List[Dict[str, Any]]) -> "StructType":
    """Recursively build a StructType from a list of field dicts."""
    if not _SPARK_AVAILABLE:
        raise RuntimeError("pyspark is not installed; cannot build StructType from S3 schema.")

    struct_fields = []
    for f in fields:
        name = f["name"]
        type_str = f.get("type", "string")
        nullable = bool(f.get("nullable", True))

        if type_str.lower() == "struct":
            nested_fields = f.get("fields", [])
            data_type = _build_struct_type(nested_fields)
        else:
            data_type = _parse_spark_type(type_str)

        struct_fields.append(StructField(name, data_type, nullable))

    return StructType(struct_fields)


class SchemaRegistry:
    """
    Registro centralizado de schemas.
    Permite obtener schemas por nombre definido en params.yaml
    """
    
    def __init__(self):
        # _schemas maps name -> schema object (StructType or list[str] or any)
        self._schemas: Dict[str, Any] = {}
        self._register_default_schemas()
    
    def _register_default_schemas(self):
        """Registra todos los schemas disponibles en schemas.py

        El registro acepta tanto objetos tipo StructType como listas de strings
        (p. ej. `schema_preprocessed`). Se usa duck-typing para ser robusto
        cuando `pyspark` no está instalado.
        """

        def is_struct_like(obj) -> bool:
            if StructType is not None:
                # At runtime StructType is the real pyspark StructType class;
                # statically, TYPE_CHECKING provides the type for pylance.
                # Use a narrow ignore to satisfy the type checker about the
                # variable-in-type-expression situation.
                return isinstance(obj, StructType)  # type: ignore[arg-type]
            # Duck-typing: StructType suele tener atributo 'fields'
            return hasattr(obj, 'fields') and not isinstance(obj, (str, list, dict))

        def is_list_of_str(obj) -> bool:
            return isinstance(obj, list) and all(isinstance(x, str) for x in obj)

        for attr_name in dir(schemas):
            if attr_name.startswith("__"):
                continue
            attr = getattr(schemas, attr_name)

            if is_struct_like(attr) or is_list_of_str(attr):
                self._schemas[attr_name] = attr
    
    def get_schema(self, schema_name: str):
        """
        Obtiene un schema por nombre.
        
        Args:
            schema_name: Nombre del schema (ej: "kafka_schema_features")
        
        Returns:
            Schema solicitado (StructType o list)
        
        Raises:
            KeyError: Si el schema no existe
        """
        if schema_name not in self._schemas:
            available = ', '.join(self._schemas.keys())
            raise KeyError(
                f"Schema '{schema_name}' not found. "
                f"Available schemas: {available}"
            )
        return self._schemas[schema_name]
    
    def register_schema(self, name: str, schema):
        """Registra un schema personalizado."""
        self._schemas[name] = schema
    
    def list_schemas(self) -> list:
        """Lista todos los schemas disponibles."""
        return list(self._schemas.keys())

    @classmethod
    def load_from_s3(cls, s3_path: str) -> "StructType":
        """Download a schema YAML from S3 and return as a Spark StructType.

        The s3_path follows the convention:
            s3://bucket/schemas/datasets/{dataset}/full.yaml

        The YAML must have a top-level ``fields`` key — see module docstring
        for the expected format.

        Raises:
            RuntimeError: if pyspark is not installed.
            ValueError: if the YAML is missing the ``fields`` key.
            botocore.exceptions.ClientError: if the S3 object does not exist.
        """
        import boto3
        import yaml as _yaml
        import re

        # Parse s3://bucket/key
        match = re.match(r"s3://([^/]+)/(.+)", s3_path)
        if not match:
            raise ValueError(f"Invalid S3 path: {s3_path!r}. Expected s3://bucket/key")
        bucket, key = match.group(1), match.group(2)

        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-2"),
        )
        obj = s3.get_object(Bucket=bucket, Key=key)
        content = obj["Body"].read().decode("utf-8")
        schema_dict = _yaml.safe_load(content)

        fields = schema_dict.get("fields")
        if not isinstance(fields, list):
            raise ValueError(
                f"Schema YAML at {s3_path!r} must have a top-level 'fields' list."
            )

        return _build_struct_type(fields)


# Instancia global para acceso fácil
_registry = SchemaRegistry()


def get_schema(schema_name: str):
    """Función helper para obtener schemas."""
    return _registry.get_schema(schema_name)


def register_schema(name: str, schema):
    """Función helper para registrar schemas."""
    _registry.register_schema(name, schema)


def list_available_schemas() -> list:
    """Lista schemas disponibles."""
    return _registry.list_schemas()