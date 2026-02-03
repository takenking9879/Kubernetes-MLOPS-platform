import pickle
import pandas as pd
import numpy as np
import xgboost as xgb


class XGBoostHandler:
    def __init__(self, model_path):
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load XGBoost model from {model_path}: {e}")

    def predict(self, input_data):
        try:
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            elif isinstance(input_data, list):
                # If list of lists, column names will be 0, 1, 2...
                df = pd.DataFrame(input_data)
            else:
                df = pd.DataFrame(input_data)

            # Fix for feature names mismatch when input is list of lists
            # Ray Serve payloads often strip execution headers, resulting in integer columns.
            # We map the model's expected feature names to the DataFrame if dimensions match.
            if hasattr(self.model, "feature_names") and self.model.feature_names:
                if len(df.columns) == len(self.model.feature_names):
                    # Check if columns are likely auto-generated integers (int or string "0", "1")
                    first_col = df.columns[0]
                    is_int_col = isinstance(first_col, int) or (isinstance(first_col, str) and first_col.isdigit())
                    
                    if is_int_col:
                        df.columns = self.model.feature_names

            dmatrix = xgb.DMatrix(df)
            probs = self.model.predict(dmatrix)

            if len(probs.shape) > 1 and probs.shape[1] > 1:
                predictions = np.argmax(probs, axis=1)
            else:
                predictions = (probs > 0.5).astype(int)

            return {"predictions": predictions.tolist(), "probabilities": probs.tolist()}
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
