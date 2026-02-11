from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient

# Configuración de MLflow (ajustar a la URL de tu servicio en K8s)
MLFLOW_TRACKING_URI = "http://mlflow-service.mlflow.svc.cluster.local:5000"

def promote_model_to_champion(model_name: str, version: str):
    """
    Asigna el alias 'champion' a una versión específica del modelo.
    Esto disparará automáticamente el cambio en el Serving de Ray Serve.
    """
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    
    # 1. Validar que la versión existe
    mv = client.get_model_version(name=model_name, version=version)
    print(f"Promoviendo {model_name} v{version} (Run ID: {mv.run_id}) a @champion")
    
    # 2. Asignar Alias (MLflow 3.x)
    # Esto quita el alias de la versión anterior y se lo asigna a esta
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=version
    )
    
    # 3. Opcional: Etiquetar como 'challenger' la versión anterior si existía
    # O realizar lógica de comparación de métricas antes de promover

def auto_promote_base_on_metrics(model_name: str):
    """
    Ejemplo de lógica para promover automáticamente si las métricas son mejores.
    """
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    
    # Obtener el actual champion
    try:
        current_champion = client.get_model_version_by_alias(model_name, "champion")
        champion_run = client.get_run(current_champion.run_id)
        current_f1 = champion_run.data.metrics.get("f1_score", 0)
    except:
        current_f1 = 0 # No hay champion previo
        
    # Obtener la última versión registrada (New candidate)
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
    candidate_run = client.get_run(latest_version.run_id)
    candidate_f1 = candidate_run.data.metrics.get("f1_score", 0)
    
    if candidate_f1 > current_f1:
        print(f"Nueva versión {latest_version.version} es mejor ({candidate_f1} > {current_f1}). Promoviendo...")
        client.set_registered_model_alias(model_name, "champion", latest_version.version)
    else:
        print(f"La nueva versión no supera al champion actual.")
        client.set_registered_model_alias(model_name, "challenger", latest_version.version)

with DAG(
    dag_id='model_promotion_workflow',
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=['mlops', 'promotion']
) as dag:

    # Tarea 1: Promoción manual específica (vía params)
    promote_manual = PythonOperator(
        task_id='promote_to_champion',
        python_callable=promote_model_to_champion,
        op_kwargs={
            'model_name': 'ids_intrusion_model', # Tomado de params.yaml
            'version': "{{ dag_run.conf['version'] }}" # Se pasa al lanzar el DAG
        }
    )

    # Tarea 2: Opcionalmente, promoción automática por métricas
    check_and_promote = PythonOperator(
        task_id='auto_eval_and_promote',
        python_callable=auto_promote_base_on_metrics,
        op_kwargs={'model_name': 'ids_intrusion_model'}
    )
