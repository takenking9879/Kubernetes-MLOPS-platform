"""PyTorch parameters schema.

Plain dict with common parameters for a basic supervised classifier.
Extend as needed.

NOTA: Selección de épocas según modo de ejecución:
------------------------------------------------------
- Entrenamiento normal (tune=False):
  → Se usa PYTORCH_PARAMS["max_epochs"] (valor por defecto: 10)
  
- Pipeline con tuning (tune=True):
  1. Fase de tuning: usa PYTORCH_TUNE_SETTINGS["max_epochs"] para ASHA scheduler (10 épocas)
     → Busca mejores hiperparámetros definidos en SEARCH_SPACE_PYTORCH_PARAMS
     → Retorna best_params con: batch_size, lr, weight_decay (SIN max_epochs)
  
  2. Entrenamiento final distribuido: usa PYTORCH_PARAMS["max_epochs"] (10 épocas)
     → Toma los hiperparámetros optimizados del tuning (batch_size, lr, weight_decay)
     → Usa max_epochs por defecto de PYTORCH_PARAMS porque no viene en best_params
"""

from __future__ import annotations
from ray import tune
from typing import Any, Dict

# Parámetros para entrenamiento normal (sin tuning)
PYTORCH_PARAMS: Dict[str, Any] = {
    # batch size
    "batch_size": 256,
    # Training
    "max_epochs": 10,  # ← Usado cuando tune=False
    "lr": 1e-3,
    "weight_decay": 0.0,
}

# Configuración del scheduler ASHA para tuning
PYTORCH_TUNE_SETTINGS: Dict[str, int] = {
    "grace_period": 5,
    "reduction_factor": 2,
    "max_epochs": 10,  # ← Usado cuando tune=True (límite para ASHA)
}

# Espacio de búsqueda para hyperparameter tuning
SEARCH_SPACE_PYTORCH_PARAMS: Dict[str, Any] = {
    # Hyperparameters to tune
    "batch_size": tune.choice([64, 128, 256]),
    "lr": tune.loguniform(1e-5, 1e-1),
    "weight_decay": tune.loguniform(1e-6, 1e-2)
}
