"""
Schema Registry - Acceso dinámico a schemas por nombre.
Permite referenciar schemas desde params.yaml sin hardcodear imports.
"""
from typing import Dict, Any

# Import pyspark types lazily/safely so this module can be imported
# in non-Spark runtimes (e.g., Ray jobs without pyspark installed).
try:
    from pyspark.sql.types import StructType
except Exception:
    StructType = None  # type: ignore

from . import schemas


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
                return isinstance(obj, StructType)
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