# Guía: Añadir un nuevo framework al pipeline Ray (PyTorch / XGBoost style)

Esta guía describe, paso a paso, cómo integrar un nuevo framework en la plataforma MLOps basada en Ray. Sigue el contrato y las plantillas para mantener compatibilidad con `KubeRayTraining` (main.py), el sistema de tuning y las métricas.

---

## Resumen rápido

Para añadir un framework nuevo debes crear/editar:

- `src/pipeline/utils/<framework>_utils.py` — worker-side `train_func(config: Dict)`
- `src/pipeline/train/<framework>.py` — `BaseTrainer` subclase
- `src/pipeline/tuning/<framework>.py` — `BaseTuner` subclase (si soportas tuning)
- `src/schemas/model/<framework>_params.py` — `DEFAULTS`, `SEARCH_SPACE` (opcional)
- Registrar en `src/pipeline/train/__init__.py` y `src/pipeline/tuning/__init__.py`
- (Opcional) `k3s/params.yaml` — valores por defecto globales

---

## Reglas importantes (Ray-safe)

1. Carga de datasets en tuning: dentro de la función `_trainable` (no pasar Ray Dataset vía `tune.with_parameters`).
2. `train_func` debe residir en `src/pipeline/utils/...` (worker-side). Evita closures que capturen objetos no picklables.
3. Evita importar dependencias pesadas a nivel de módulo en archivos importados por el driver (`main.py`); importa localmente dentro de `train_func` si sólo se necesitan en workers.
4. `train_loop_config` debe contener tipos simples (dict, str, int, float) para evitar problemas de serialización.

---

## Plantilla: `utils` (worker)

Archivo: `src/pipeline/utils/<framework>_utils.py`

```python
from typing import Dict, Optional, List
import os
import time
import ray

from ray.train import Checkpoint

# Opcional: prometheus per-worker (registries separados)
try:
    from src.prometheus import create_worker_registry
    _WORKER_REGISTRY, _METRICS = create_worker_registry("<framework>")
except Exception:
    _WORKER_REGISTRY = None

# Params por defecto importados desde schemas
from schemas.model.<framework>_params import <FRAMEWORK>_PARAMS


def train_func(config: Dict):
    """Entrenamiento distribuido que corre en cada worker de Ray.

    contract:
      - recibe `train` y `val` datasets via `ray.train.get_dataset_shard()`
      - no devuelve el modelo; usa checkpoints / callbacks para persistir
    """
    # Imports locales para reducir serialización del driver
    import <heavy_lib> as hl

    params = dict(config.get("<framework>_params", <FRAMEWORK>_PARAMS))

    cpus_per_worker = int(config.get("cpus_per_worker", os.getenv("CPUS_PER_WORKER", "1")))
    cpus_per_worker = max(cpus_per_worker, 1)

    # Leer shards (cada worker recibe su partición)
    train_shard = ray.train.get_dataset_shard("train")
    val_shard = ray.train.get_dataset_shard("val")

    # Materializar/convertir según necesidad
    train_df = train_shard.materialize().to_pandas()
    val_df = val_shard.materialize().to_pandas()

    # Construir objetos de entrenamiento (ej: DMatrix, DataLoader, etc.)
    # ... código específico del framework ...

    # Entrenar y reportar
    start = time.perf_counter()
    # fit(...) o loop de epochs

    # Reportar métricas periódicas y crear checkpoints con callback compatible Ray
    # Ejemplo: usar RayTrainReportCallback o ray.train.report()

    duration = time.perf_counter() - start
    print(f"[{<framework>}] worker finished in {duration:.2f}s")
```

---

## Plantilla: `train` (driver)

Archivo: `src/pipeline/train/<framework>.py`

```python
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import ray
from ray.train import Checkpoint
from ray.train.<lib> import <TrainerClass>
from pipeline.base_trainer import BaseTrainer
from schemas.model.<framework>_params import <FRAMEWORK>_PARAMS
from pipeline.utils.<framework>_utils import train_func

class <Framework>ModelTrainer(BaseTrainer):
    @property
    def framework_name(self) -> str:
        return "<framework>"

    @property
    def params_key(self) -> str:
        return "<framework>_params"

    @property
    def default_params(self) -> Dict[str, Any]:
        return <FRAMEWORK>_PARAMS

    def _get_ray_trainer_cls(self):
        return <TrainerClass>

    def _get_train_func(self):
        return train_func

    def _build_train_loop_config(...):
        return {
            "target": target,
            "feature_columns": feature_columns,
            self.params_key: params,
            "input_dim": input_dim,
            "num_classes": num_classes,
            "cpus_per_worker": cpus_per_worker,
            "is_tuning": False,
        }

    def _preprocess_datasets(...):
        # sólo si el driver debe filtrar columnas (PyTorch)
        return train_ds, val_ds, test_ds

    def _evaluate_split(...):
        # usar result.checkpoint para evaluar en driver
        return {}

# export convenience
_TRAINER = <Framework>ModelTrainer()

def train(...):
    return _TRAINER.train(...)
```

---

## Plantilla: `tuning` (driver)

Archivo: `src/pipeline/tuning/<framework>.py`

```python
from pipeline.base_tuner import BaseTuner
from schemas.model.<framework>_params import SEARCH_SPACE_<FRAMEWORK>_PARAMS, <FRAMEWORK>_TUNE_SETTINGS
from pipeline.utils.<framework>_utils import train_func
from ray.train.<lib> import <TrainerClass>

class <Framework>ModelTuner(BaseTuner):
    @property
    def framework_name(self) -> str:
        return "<framework>"

    @property
    def params_key(self) -> str:
        return "<framework>_params"

    @property
    def search_space(self) -> Dict[str, Any]:
        return SEARCH_SPACE_<FRAMEWORK>_PARAMS

    @property
    def tune_settings(self) -> Dict[str, Any]:
        return <FRAMEWORK>_TUNE_SETTINGS

    @property
    def tune_metric(self) -> str:
        return "<metric_name>"

    @property
    def tune_mode(self) -> str:
        return "min"  # o "max"

    @property
    def default_num_samples(self) -> int:
        return 3

    def _get_ray_trainer_cls(self):
        return <TrainerClass>

    def _get_train_func(self):
        return train_func

    def _get_asha_max_t_key(self) -> str:
        return "<max_t_key>"

    def _build_trial_train_loop_config(...):
        return {
            "target": target,
            "feature_columns": feature_columns,
            self.params_key: trial_config[self.params_key],
            "input_dim": input_dim,
            "num_classes": num_classes,
            "cpus_per_worker": cpus_per_worker,
            "is_tuning": True,
        }

# export convenience
_TUNER = <Framework>ModelTuner()

def tune_model(...):
    return _TUNER.tune_model(...)
```

---

## Registrar el framework

Editar `src/pipeline/train/__init__.py` y `src/pipeline/tuning/__init__.py` para añadir la rama en `get_trainer()` y `get_tuner()`:

```py
if framework == "<framework>":
    from pipeline.train.<framework> import <Framework>ModelTrainer
    return <Framework>ModelTrainer()
```

y similar para `get_tuner()`.

---

## Esquemas y defaults

Crear `src/schemas/model/<framework>_params.py` con:

```py
# defaults
<FRAMEWORK>_PARAMS = {
    # parámetros por defecto (ej: learning_rate, num_boost_round, batch_size...)
}

# search-space para tuning (Ray Tune)
SEARCH_SPACE_<FRAMEWORK>_PARAMS = {
    # ej: "<framework>_params": {"learning_rate": tune.loguniform(1e-4, 1e-1), ...}
}

# settings para ASHA
<FRAMEWORK>_TUNE_SETTINGS = {
    "max_epochs": 20,
    "grace_period": 1,
    "reduction_factor": 2,
}
```

---

## Checklist de prueba rápida

1. Lint & smoke import:

```bash
flake8 src/pipeline || true
python -c "from src.pipeline.train import get_trainer; print(get_trainer('<framework>'))"
```

2. Test local con Ray head (datasets pequeños):

```bash
ray start --head
python -c "from k3s.kuberay.main import KubeRayTraining; m=KubeRayTraining(params_path='k3s/params.yaml', output_dir='s3://...'); m.train()"
```

3. Tuning smoke test (1-2 trials, small shards)

---

## Errores comunes y consejos

- No pases `ray.data.Dataset` a `tune.with_parameters`. Carga dentro del trial.
- Evita variables globales no serializables en `train_func`.
- Asegura que `train_loop_config` y `trial_config` incluyan la key `params_key` que tus utilidades esperan.
- Testea con `NUM_WORKERS=1` y datasets muy pequeños antes de escalar.

---

## ¿Quieres que genere plantillas concretas para `lightgbm` (archivos listos para pegar)?
Si quieres, puedo crear los archivos base (`utils`, `train`, `tuning`, `schemas`) y modificar los registries automáticamente.
