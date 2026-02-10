# src/CM/monitoring_cm.py
import os
import shutil
import joblib
import pandas as pd
import numpy as np
from evidently import Report
from evidently import DataDefinition
from evidently import Dataset
from evidently.presets import DataDriftPreset, DataSummaryPreset, RegressionPreset
from evidently import Regression
from src.utils import create_logger
import re
import gc
from evidently.ui.workspace import CloudWorkspace
import warnings
import json
warnings.filterwarnings("ignore", category=FutureWarning, module="evidently")

logger = create_logger("continuous_monitoring", "logs/continuous_monitoring.log")

def delete_data(delete = False) -> None:
    if delete == False:
        return None
    folder = "/home/jorge/DocumentsWLS/Data_Science_Projects/MLOPS-project-NYC-Taxi-Fare-Prediction/CM_data"

    # Recorre todo el contenido y lo elimina
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # borrar archivo o symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # borrar directorio recursivamente
        except Exception as e:
            print(f"Error al borrar {file_path}: {e}")

class ContinuousMonitoringCM:
    def __init__(self, processed_dir: str, models_dir: str, target_col: str = "Fare_Amt"):
        self.processed_dir = processed_dir
        self.target_col = target_col
        self.logger = logger
        self.monitoring_file = os.path.join(self.processed_dir, "monitoring_processed_osrm.parquet")

        # ahora get_latest_version devuelve: previous_path, latest_path, stacking_path, chosen_on_val
        (self.previous_model_path,
         self.latest_model_path,
         self.stacking_model_path,
         self.chosen_on_val) = self.get_latest_version(models_dir)

        # modelos (se cargan en load_model según chosen_on_val)
        self.model = None             # modelo simple a usar si chosen != 'stacked'
        self.model_old = None         # previous model (XGB)
        self.model_latest = None      # latest model (XGB)
        self.meta_model = None        # stacking meta-model (Ridge u otro)

        # carga los modelos necesarios (según chosen_on_val)
        self.load_model()

        # datos
        self.df_new = self.load_monitoring_data()
        self.df_old = self.load_old_data()

        # token Evidently
        self.token = self.get_token()

    def get_latest_version(self, base_dir: str):
        """
        Devuelve tuple: (previous_model_path | None, latest_model_path | None,
                         stacking_model_path | None, chosen_on_val | None)

        Además lee promotion_choice.json si existe y extrae "chosen_on_val".
        """
        try:
            if not os.path.isdir(base_dir):
                return None, None, None, None
            
            dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            versions = [(d, int(re.sub(r"\D", "", d))) for d in dirs if re.match(r"v\d+", d)]
            if not versions:
                return None, None, None, None

            latest_name, latest_num = max(versions, key=lambda x: x[1])
            latest_model_path = os.path.join(base_dir, latest_name, "final_model.pkl")
            stacking_path = os.path.join(base_dir, latest_name, "stacking.pkl")
            promotion_json_path = os.path.join(base_dir, latest_name, "promotion_choice.json")

            chosen_on_val = None
            if os.path.isfile(promotion_json_path):
                try:
                    with open(promotion_json_path, "r") as f:
                        pj = json.load(f)
                    chosen_on_val = pj.get("chosen_on_val")
                    self.logger.info(f"promotion_choice.json encontrado en {latest_name} -> chosen_on_val = {chosen_on_val}")
                except Exception as e:
                    self.logger.warning(f"No se pudo leer promotion_choice.json: {e}")
                    chosen_on_val = None

            previous_model_path = None
            if latest_num > 1:
                previous_name = f"v{latest_num - 1}"
                previous_model_path = os.path.join(base_dir, previous_name, "final_model.pkl")

            # Los paths pueden existir o no; se devuelven igual para que la lógica de carga haga exist checks.
            return previous_model_path, latest_model_path, stacking_path, chosen_on_val

        except Exception as e:
            self.logger.error(f"Hubo un error obteniendo la última versión {e}")
            raise

    def load_model(self):
        """
        Carga los modelos según chosen_on_val:
         - si chosen_on_val == 'stacked' y stacking existe: carga previous, latest y stacking (meta).
         - si chosen_on_val == 'new'/'latest' o None: intenta cargar latest.
         - si chosen_on_val == 'old': intenta cargar previous; si falla, fallback a latest.
        """
        try:
            # Caso stacked: intentamos cargar meta + ambos modelos base
            if self.chosen_on_val == "stacked":
                # load stacking meta-model
                if self.stacking_model_path and os.path.exists(self.stacking_model_path):
                    try:
                        self.meta_model = joblib.load(self.stacking_model_path)
                        self.logger.info(f"Cargado stacking meta-model desde {self.stacking_model_path}")
                    except Exception as e:
                        self.logger.warning(f"No se pudo cargar stacking meta-model: {e}; fallback a latest.")
                        self.meta_model = None
                else:
                    self.logger.warning("chosen_on_val == 'stacked' pero stacking.pkl no existe; fallback a latest.")
                    self.meta_model = None

                # load previous and latest base models (si existen)
                if self.previous_model_path and os.path.exists(self.previous_model_path):
                    try:
                        self.model_old = joblib.load(self.previous_model_path)
                        self.logger.info(f"Cargado modelo anterior desde {self.previous_model_path}")
                    except Exception as e:
                        self.logger.warning(f"No se pudo cargar previous model: {e}")
                        self.model_old = None
                else:
                    self.logger.warning("previous model no encontrado (se esperaba para stacking).")

                if self.latest_model_path and os.path.exists(self.latest_model_path):
                    try:
                        self.model_latest = joblib.load(self.latest_model_path)
                        self.logger.info(f"Cargado modelo latest desde {self.latest_model_path}")
                    except Exception as e:
                        self.logger.warning(f"No se pudo cargar latest model: {e}")
                        self.model_latest = None
                else:
                    self.logger.warning("latest model no encontrado (se esperaba).")

                # Si por algún motivo no se pudieron cargar las piezas necesarias, dejamos model=None
                # y la función predict hará fallback al latest que pueda cargarse posteriormente.
                return None

            # Caso chosen != 'stacked' (usar single model)
            # Si explicitly 'old'
            if self.chosen_on_val == "old":
                if self.previous_model_path and os.path.exists(self.previous_model_path):
                    try:
                        self.model = joblib.load(self.previous_model_path)
                        self.logger.info(f"Cargado modelo 'old' desde {self.previous_model_path}")
                        return self.model
                    except Exception as e:
                        self.logger.warning(f"No se pudo cargar previous model: {e}; fallback a latest.")

            # Default / 'new' / fallback: cargar latest
            if self.latest_model_path and os.path.exists(self.latest_model_path):
                try:
                    self.model = joblib.load(self.latest_model_path)
                    self.logger.info(f"Cargado modelo 'latest' desde {self.latest_model_path}")
                    return self.model
                except Exception as e:
                    self.logger.error(f"No se pudo cargar latest model: {e}")
                    self.model = None
                    return None

            # Si llegamos aquí, no hay modelos cargados
            self.logger.error("No se pudo cargar ningún modelo (ni previous ni latest).")
            return None

        except Exception as e:
            self.logger.error(f"Error en load_model: {e}")
            raise

    def load_monitoring_data(self):
        if not os.path.exists(self.monitoring_file):
            raise FileNotFoundError(f"No se encontró {self.monitoring_file}")
        logger.info(f"Cargando datos de monitoring desde {self.monitoring_file}")
        df = pd.read_parquet(self.monitoring_file)
        if self.target_col not in df.columns:
            raise ValueError(f"La columna target '{self.target_col}' no está en el dataset")
        for c in ['Real_time', 'Real_distance', 'Total_Amt']:
                if c in df.columns:
                    del df[c]
        return df

    def load_old_data(self):
        old_file = "new_data/processed_osrm/val_processed_osrm.parquet"
        if not os.path.exists(old_file):
            raise FileNotFoundError(f"No se encontró el archivo {old_file}")
        df_old = pd.read_parquet(old_file)
        if self.target_col not in df_old.columns:
            raise ValueError(f"La columna target '{self.target_col}' no está en el dataset de referencia")
        # Subsample estratificado
        for c in ['Real_time', 'Real_distance', 'Total_Amt']:
                if c in df_old.columns:
                    del df_old[c]
        df_old = self.stratified_sample_skewed(df_old, self.target_col, n_samples=500000)
        gc.collect()
        return df_old

    def predict(self):
        """
        Método principal de predicción:
         - si chosen_on_val == 'stacked' y meta_model + ambos modelos están disponibles:
             -> genera preds_old y preds_new y luego preds_stack = meta_model.predict([old,new])
         - si no, usa self.model (latest o old ya cargado)
         - si falta self.model, intenta cargar latest on-the-fly como fallback
        """

        # Preparar X para new
        X_new = self.df_new.drop(columns=[self.target_col])
        self.logger.info("Generando predicciones sobre el dataset de monitoring")

        # Caso ideal: stacking elegido y todas las piezas cargadas
        if self.chosen_on_val == "stacked" and self.meta_model is not None and self.model_old is not None and self.model_latest is not None:
            try:
                preds_old_new = self.model_old.predict(X_new)
            except Exception as e:
                self.logger.warning(f"Error prediciendo con previous model sobre df_new: {e}")
                preds_old_new = None

            try:
                preds_new_new = self.model_latest.predict(X_new)
            except Exception as e:
                self.logger.warning(f"Error prediciendo con latest model sobre df_new: {e}")
                preds_new_new = None

            preds_stack_new = None
            if preds_old_new is not None and preds_new_new is not None:
                try:
                    stack_input_new = np.column_stack([preds_old_new, preds_new_new])
                    preds_stack_new = self.meta_model.predict(stack_input_new)
                except Exception as e:
                    self.logger.warning(f"Error aplicando meta_model sobre df_new: {e}")
                    preds_stack_new = None

            # si stacked OK, se usa, si no, fallback a latest
            if preds_stack_new is not None:
                self.df_new["prediction"] = preds_stack_new
                self.logger.info("Predicciones stacked generadas correctamente para df_new")
            elif preds_new_new is not None:
                self.df_new["prediction"] = preds_new_new
                self.logger.info("Stacking falló; usado latest para df_new")
            else:
                # intentar fallback: cargar latest on-the-fly
                if self.latest_model_path and os.path.exists(self.latest_model_path):
                    try:
                        lm = joblib.load(self.latest_model_path)
                        self.df_new["prediction"] = lm.predict(X_new)
                        self.logger.info("Fallback: cargado latest on-the-fly y predicho para df_new")
                    except Exception as e:
                        self.logger.error(f"Fallback failed (latest on-the-fly) para df_new: {e}")
                        raise
                else:
                    raise RuntimeError("No hay modelo disponible para predecir df_new.")

        else:
            # No stacking elegido o no disponible -> usar self.model (old o latest) si está cargado
            if self.model is None:
                # intentar cargar latest como fallback
                if self.latest_model_path and os.path.exists(self.latest_model_path):
                    try:
                        self.model = joblib.load(self.latest_model_path)
                        self.logger.info("Cargado latest on-the-fly para predicción (fallback).")
                    except Exception as e:
                        self.logger.error(f"No se pudo cargar latest on-the-fly: {e}")
                        raise
                else:
                    raise RuntimeError("No hay modelo cargado y no existe latest para fallback.")

            self.df_new["prediction"] = self.model.predict(X_new)
            self.logger.info("Predicciones generadas correctamente para df_new (modelo seleccionado)")

        del X_new

        # Ahora generar predicciones para df_old (misma lógica)
        X_old = self.df_old.drop(columns=[self.target_col])
        self.logger.info("Generando predicciones sobre el dataset anterior")

        if self.chosen_on_val == "stacked" and self.meta_model is not None and self.model_old is not None and self.model_latest is not None:
            try:
                preds_old_old = self.model_old.predict(X_old)
            except Exception as e:
                self.logger.warning(f"Error prediciendo con previous model sobre df_old: {e}")
                preds_old_old = None

            try:
                preds_new_old = self.model_latest.predict(X_old)
            except Exception as e:
                self.logger.warning(f"Error prediciendo con latest model sobre df_old: {e}")
                preds_new_old = None

            preds_stack_old = None
            if preds_old_old is not None and preds_new_old is not None:
                try:
                    stack_input_old = np.column_stack([preds_old_old, preds_new_old])
                    preds_stack_old = self.meta_model.predict(stack_input_old)
                except Exception as e:
                    self.logger.warning(f"Error aplicando meta_model sobre df_old: {e}")
                    preds_stack_old = None

            if preds_stack_old is not None:
                self.df_old["prediction"] = preds_stack_old
                self.logger.info("Predicciones stacked generadas correctamente para df_old")
            elif preds_new_old is not None:
                self.df_old["prediction"] = preds_new_old
                self.logger.info("Stacking falló; usado latest para df_old")
            else:
                if self.latest_model_path and os.path.exists(self.latest_model_path):
                    try:
                        lm = joblib.load(self.latest_model_path)
                        self.df_old["prediction"] = lm.predict(X_old)
                        self.logger.info("Fallback: cargado latest on-the-fly y predicho para df_old")
                    except Exception as e:
                        self.logger.error(f"Fallback failed (latest on-the-fly) para df_old: {e}")
                        raise
                else:
                    raise RuntimeError("No hay modelo disponible para predecir df_old.")

        else:
            if self.model is None:
                if self.latest_model_path and os.path.exists(self.latest_model_path):
                    try:
                        self.model = joblib.load(self.latest_model_path)
                        self.logger.info("Cargado latest on-the-fly para predicción (fallback) - df_old.")
                    except Exception as e:
                        self.logger.error(f"No se pudo cargar latest on-the-fly: {e}")
                        raise
                else:
                    raise RuntimeError("No hay modelo cargado y no existe latest para fallback (df_old).")

            self.df_old["prediction"] = self.model.predict(X_old)
            self.logger.info("Predicciones generadas correctamente para df_old (modelo seleccionado)")

        del X_old

    def stratified_sample_skewed(self, df, target_col, n_samples, n_bins=50, random_state=42):
        df = df.copy()
        df["_log_target"] = np.log1p(df[target_col])
        df["_bin"] = pd.qcut(df["_log_target"], q=n_bins, duplicates="drop")
        bin_counts = df["_bin"].value_counts()
        total = len(df)
        n_per_bin = (bin_counts / total * n_samples).round().astype(int)
        df_sampled = df.groupby("_bin", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), n_per_bin[x.name]), random_state=random_state)
        )
        return df_sampled.drop(columns=["_log_target", "_bin"])

    def get_token(self):
        token_file = "Evidentlytoken.txt"
        if not os.path.exists(token_file):
            raise FileNotFoundError(f"No se encontró {token_file}")
        with open(token_file, "r") as f:
            token = f.read().strip()
        return token

    def generate_evidently_report(self, output_dir="reports", html_name="monitoring_report.html"):
        # Subsample estratificado para df_old

        # Crear workspace de Evidently Cloud
        ws = CloudWorkspace(token=self.token, url="https://app.evidently.cloud")
        project = ws.get_project("0199a389-3a83-78d1-b971-9c255efdba61")

        # Definir esquema: todas las features numéricas excepto target/prediction
        feature_columns = [c for c in self.df_new.columns if c not in ["Fare_Amt", "prediction"]]

        schema = DataDefinition(
            numerical_columns=feature_columns,
            regression=[Regression(target="Fare_Amt", prediction="prediction")]
        )

        # Crear datasets de Evidently
        eval_data_new = Dataset.from_pandas(self.df_new, data_definition=schema)
        eval_data_old = Dataset.from_pandas(self.df_old, data_definition=schema)

        # Crear reporte
        report = Report([
            DataDriftPreset(), 
            RegressionPreset(),
        ], include_tests=True)
        self.logger.info("Subiendo el reporte a la nube...")
        my_eval = report.run(eval_data_new, eval_data_old)

        # Subir a Evidently Cloud
        ws.add_run(project.id, my_eval, include_data=False)
        self.logger.info("El reporte ha sido subido a la nube")


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    processed_dir = os.path.join(root_dir, "CM_data/processed_osrm")
    models_dir = os.path.join(root_dir, "flask_app/models")

    cm = ContinuousMonitoringCM(processed_dir, models_dir)
    cm.predict()
    cm.generate_evidently_report()
    delete_data(delete=True)

if __name__ == "__main__":
    main()
