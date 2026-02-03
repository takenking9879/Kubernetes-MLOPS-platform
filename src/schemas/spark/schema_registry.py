"""
Schema Registry - Acceso dinámico a schemas por nombre.
Permite referenciar schemas desde params.yaml sin hardcodear imports.
"""
from typing import Dict
from pyspark.sql.types import StructType
import schemas


class SchemaRegistry:
    """
    Registro centralizado de schemas.
    Permite obtener schemas por nombre definido en params.yaml
    """
    
    def __init__(self):
        self._schemas: Dict[str, StructType] = {}
        self._register_default_schemas()
    
    def _register_default_schemas(self):
        """Registra todos los schemas disponibles en schemas.py"""
        # Registrar StructType schemas automáticamente
        for attr_name in dir(schemas):
            attr = getattr(schemas, attr_name)
            if isinstance(attr, StructType):
                self._schemas[attr_name] = attr
        
        # Registrar también schema_preprocessed como lista (caso especial)
        if hasattr(schemas, 'schema_preprocessed'):
            self._schemas['schema_preprocessed'] = schemas.schema_preprocessed
    
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