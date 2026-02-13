from typing import Any, Dict, List, Protocol

import pandas as pd

from src.serve.config import Framework


def normalize_payload(payload: Any) -> List[List[Any]]:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be JSON object")

    if "data" not in payload:
        raise ValueError("Missing 'data' field")

    data = payload["data"]
    if isinstance(data, list) and (len(data) == 0 or not isinstance(data[0], (list, tuple))):
        data = [data]

    if not isinstance(data, list) or any(not isinstance(r, (list, tuple)) for r in data):
        raise ValueError("'data' must be matrix (list of lists) or single row (list)")

    lengths = {len(r) for r in data}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent row lengths: {sorted(lengths)}")

    return [list(r) for r in data]


class ModelAdapter(Protocol):
    def predict(self, data: List[List[Any]]) -> Dict[str, Any]:
        ...


class XGBoostAdapter:
    def __init__(self, model):
        import numpy as np
        import xgboost as xgb

        self._model = model
        self._np = np
        self._xgb = xgb

    def predict(self, data: List[List[Any]]) -> Dict[str, Any]:
        df = pd.DataFrame(data)

        model_feature_names = getattr(self._model, "feature_names", None)
        if model_feature_names:
            if len(df.columns) != len(model_feature_names):
                raise ValueError(
                    f"Feature dimension mismatch. Model expects {len(model_feature_names)} features, "
                    f"got {len(df.columns)}"
                )
            dmatrix = self._xgb.DMatrix(df.values, feature_names=list(model_feature_names))
        else:
            dmatrix = self._xgb.DMatrix(df)

        probs = self._model.predict(dmatrix)

        if len(probs.shape) > 1 and probs.shape[1] > 1:
            predictions = self._np.argmax(probs, axis=1)
        else:
            predictions = (probs > 0.5).astype(int)

        return {
            "predictions": predictions.tolist(),
            "probabilities": probs.tolist(),
        }


class PyTorchAdapter:
    def __init__(self, model):
        import torch

        self._model = model
        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()

    def predict(self, data: List[List[Any]]) -> Dict[str, Any]:
        tensor = self._torch.tensor(data, dtype=self._torch.float32).to(self._device)
        with self._torch.no_grad():
            outputs = self._model(tensor)
            probs = self._torch.softmax(outputs, dim=1)
            predictions = self._torch.argmax(probs, dim=1)

        return {
            "predictions": predictions.cpu().numpy().tolist(),
            "probabilities": probs.cpu().numpy().tolist(),
        }


class AdapterFactory:
    @staticmethod
    def create(framework: Framework, model: Any) -> ModelAdapter:
        if framework == Framework.XGBOOST:
            return XGBoostAdapter(model)
        if framework == Framework.PYTORCH:
            return PyTorchAdapter(model)
        raise ValueError(f"Unsupported framework: {framework}")
