"""
metadata_query_utils.py

Utilidades para consultar la tabla de metadata de preprocessing.
Permite:
- Listar todos los artifact sets disponibles
- Filtrar por DSL, fecha, snapshot
- Obtener detalles de un artifact set específico
- Cargar pipeline asociado a un artifact set
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import os
from typing import Optional, List, Dict
import json


class MetadataQueryUtils:
    """
    Utilidades para consultar metadata de preprocessing.
    """
    
    def __init__(self, spark: SparkSession, metadata_table: str = "iceberg.metadata.preprocessing_artifacts"):
        self.spark = spark
        self.metadata_table = metadata_table
    
    def list_all_artifact_sets(self, limit: int = 50):
        """
        Lista todos los artifact sets disponibles, ordenados por fecha de creación.
        
        Args:
            limit: número máximo de resultados
            
        Returns:
            DataFrame con todos los artifact sets
        """
        df = self.spark.sql(f"""
            SELECT 
                artifact_set_id,
                created_at,
                raw_snapshot,
                processed_table_name,
                dsl_name,
                pipeline_hash,
                train_start,
                train_end,
                val_start,
                val_end,
                test_start,
                test_end
            FROM {self.metadata_table}
            ORDER BY created_at DESC
            LIMIT {limit}
        """)
        
        return df
    
    def filter_by_dsl(self, dsl_name: str):
        """
        Filtra artifact sets por nombre de DSL.
        
        Args:
            dsl_name: nombre del DSL (ej: 'dsl_001.yaml')
            
        Returns:
            DataFrame filtrado
        """
        df = self.spark.sql(f"""
            SELECT *
            FROM {self.metadata_table}
            WHERE dsl_name = '{dsl_name}'
            ORDER BY created_at DESC
        """)
        
        return df
    
    def filter_by_date_range(self, start_date: str, end_date: str):
        """
        Filtra artifact sets creados en un rango de fechas.
        
        Args:
            start_date: fecha inicio (YYYY-MM-DD)
            end_date: fecha fin (YYYY-MM-DD)
            
        Returns:
            DataFrame filtrado
        """
        df = self.spark.sql(f"""
            SELECT *
            FROM {self.metadata_table}
            WHERE created_at >= '{start_date}'
              AND created_at <= '{end_date}'
            ORDER BY created_at DESC
        """)
        
        return df
    
    def filter_by_raw_snapshot(self, snapshot_id: int):
        """
        Filtra artifact sets que usan un snapshot específico de raw data.
        
        Args:
            snapshot_id: ID del snapshot de raw data
            
        Returns:
            DataFrame filtrado
        """
        df = self.spark.sql(f"""
            SELECT *
            FROM {self.metadata_table}
            WHERE raw_snapshot = {snapshot_id}
            ORDER BY created_at DESC
        """)
        
        return df
    
    def get_artifact_set_details(self, artifact_set_id: str) -> Optional[Dict]:
        """
        Obtiene detalles completos de un artifact set específico.
        
        Args:
            artifact_set_id: ID del artifact set
            
        Returns:
            Diccionario con los detalles o None si no existe
        """
        df = self.spark.sql(f"""
            SELECT *
            FROM {self.metadata_table}
            WHERE artifact_set_id = '{artifact_set_id}'
        """)
        
        rows = df.collect()
        
        if len(rows) == 0:
            return None
        
        row = rows[0]
        
        return {
            'artifact_set_id': row['artifact_set_id'],
            'created_at': row['created_at'],
            'raw_snapshot': row['raw_snapshot'],
            'processed_table_name': row['processed_table_name'],
            'dsl_name': row['dsl_name'],
            'pipeline_hash': row['pipeline_hash'],
            'splits': {
                'train': {
                    'start': row['train_start'],
                    'end': row['train_end']
                },
                'val': {
                    'start': row['val_start'],
                    'end': row['val_end']
                },
                'test': {
                    'start': row['test_start'],
                    'end': row['test_end']
                }
            }
        }
    
    def get_latest_by_dsl(self, dsl_name: str) -> Optional[Dict]:
        """
        Obtiene el artifact set más reciente para un DSL específico.
        
        Args:
            dsl_name: nombre del DSL
            
        Returns:
            Diccionario con los detalles o None si no existe
        """
        df = self.spark.sql(f"""
            SELECT *
            FROM {self.metadata_table}
            WHERE dsl_name = '{dsl_name}'
            ORDER BY created_at DESC
            LIMIT 1
        """)
        
        rows = df.collect()
        
        if len(rows) == 0:
            return None
        
        row = rows[0]
        
        return {
            'artifact_set_id': row['artifact_set_id'],
            'created_at': row['created_at'],
            'raw_snapshot': row['raw_snapshot'],
            'processed_table_name': row['processed_table_name'],
            'dsl_name': row['dsl_name'],
            'pipeline_hash': row['pipeline_hash']
        }
    
    def get_pipeline_hash(self, artifact_set_id: str) -> Optional[str]:
        """
        Obtiene el pipeline hash asociado a un artifact set.
        
        Args:
            artifact_set_id: ID del artifact set
            
        Returns:
            Pipeline hash o None si no existe
        """
        details = self.get_artifact_set_details(artifact_set_id)
        
        if details:
            return details['pipeline_hash']
        
        return None
    
    def get_processed_tables(self, artifact_set_id: str) -> Dict[str, str]:
        """
        Obtiene los nombres de las tablas procesadas para un artifact set.
        
        Args:
            artifact_set_id: ID del artifact set
            
        Returns:
            Diccionario con nombres de tablas {key: full_table_name}
        """
        return {
            'processed': f"iceberg.processed.{artifact_set_id}"
        }
    
    def print_artifact_set_summary(self, artifact_set_id: str):
        """
        Imprime un resumen legible de un artifact set.
        
        Args:
            artifact_set_id: ID del artifact set
        """
        details = self.get_artifact_set_details(artifact_set_id)
        
        if not details:
            print(f"Artifact set not found: {artifact_set_id}")
            return
        
        print("=" * 70)
        print(f"ARTIFACT SET: {details['artifact_set_id']}")
        print("=" * 70)
        print(f"Created At:     {details['created_at']}")
        print(f"DSL Name:       {details['dsl_name']}")
        print(f"Pipeline Hash:  {details['pipeline_hash']}")
        print(f"Raw Snapshot:   {details['raw_snapshot']}")
        print()
        print("PROCESSED TABLES:")
        tables = self.get_processed_tables(artifact_set_id)
        for key, table in tables.items():
            print(f"  {key:9s}: {table}")
        print()
        print("DATE RANGES:")
        for split, dates in details['splits'].items():
            print(f"  {split:5s}: {dates['start']} → {dates['end']}")
        print("=" * 70)
    
    def compare_artifact_sets(self, id1: str, id2: str):
        """
        Compara dos artifact sets lado a lado.
        
        Args:
            id1: primer artifact set ID
            id2: segundo artifact set ID
        """
        details1 = self.get_artifact_set_details(id1)
        details2 = self.get_artifact_set_details(id2)
        
        if not details1 or not details2:
            print("One or both artifact sets not found")
            return
        
        print("=" * 100)
        print(f"{'COMPARISON':^100}")
        print("=" * 100)
        print(f"{'Field':<25} | {'Artifact Set 1':<35} | {'Artifact Set 2':<35}")
        print("-" * 100)
        print(f"{'Artifact Set ID':<25} | {details1['artifact_set_id']:<35} | {details2['artifact_set_id']:<35}")
        print(f"{'Created At':<25} | {str(details1['created_at']):<35} | {str(details2['created_at']):<35}")
        print(f"{'DSL Name':<25} | {details1['dsl_name']:<35} | {details2['dsl_name']:<35}")
        print(f"{'Pipeline Hash':<25} | {details1['pipeline_hash']:<35} | {details2['pipeline_hash']:<35}")
        print(f"{'Raw Snapshot':<25} | {str(details1['raw_snapshot']):<35} | {str(details2['raw_snapshot']):<35}")
        print("=" * 100)


def main():
    """
    Ejemplo de uso de las utilidades.
    """
    # Crear SparkSession
    spark = (
        SparkSession.builder
        .appName("metadata-query-utils")
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config("spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.iceberg.type", "hadoop")
        .config("spark.sql.catalog.iceberg.warehouse", "s3a://k8s-mlops-platform-bucket/warehouse")
        .getOrCreate()
    )
    
    # Crear utilidad
    utils = MetadataQueryUtils(spark)
    
    # Ejemplo 1: Listar todos los artifact sets
    print("\n>>> Listing all artifact sets:")
    df_all = utils.list_all_artifact_sets(limit=10)
    df_all.show(truncate=False)
    
    # Ejemplo 2: Filtrar por DSL
    print("\n>>> Filtering by DSL 'dsl_001.yaml':")
    df_dsl = utils.filter_by_dsl('dsl_001.yaml')
    df_dsl.show(truncate=False)
    
    # Ejemplo 3: Obtener el más reciente para un DSL
    print("\n>>> Getting latest artifact set for 'dsl_001.yaml':")
    latest = utils.get_latest_by_dsl('dsl_001.yaml')
    if latest:
        print(json.dumps(latest, indent=2, default=str))
    
    # Ejemplo 4: Detalles de un artifact set específico
    if latest:
        artifact_set_id = latest['artifact_set_id']
        print(f"\n>>> Details for {artifact_set_id}:")
        utils.print_artifact_set_summary(artifact_set_id)
    
    spark.stop()


if __name__ == "__main__":
    main()