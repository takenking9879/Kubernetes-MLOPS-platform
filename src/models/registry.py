"""Model registry — maps model_type strings to factory functions.

Usage::

    from models.registry import get_model

    model = get_model("mlp", {"input_dim": 14, "num_classes": 6})

To register a new model architecture::

    from models.registry import register_model

    register_model("rnn", lambda cfg: MyRNN(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg.get("hidden_dim", 128),
        num_classes=cfg["num_classes"],
    ))
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from torch import nn

ModelFactory = Callable[[Dict[str, Any]], nn.Module]

_REGISTRY: Dict[str, ModelFactory] = {}


def register_model(name: str, factory: ModelFactory) -> None:
    """Register a model factory under *name*."""
    _REGISTRY[name] = factory


def get_model(model_type: str, config: Dict[str, Any]) -> nn.Module:
    """Instantiate and return the model registered as *model_type*.

    Raises ``ValueError`` for unknown model types.
    """
    if model_type not in _REGISTRY:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            f"Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[model_type](config)


def list_models() -> List[str]:
    """Return sorted list of registered model type names."""
    return sorted(_REGISTRY)


# ── Built-in models ──────────────────────────────────────────────────────────


def _register_builtins() -> None:
    from models.pytorch import NeuralNetwork
    from models.ssm import SSMTabular
    from models.bae import BAETabular

    register_model(
        "mlp",
        lambda cfg: NeuralNetwork(
            input_dim=cfg.get("input_dim", 14),
            num_classes=cfg.get("num_classes", 6),
        ),
    )
    register_model(
        "ssm",
        lambda cfg: SSMTabular(
            input_dim=cfg.get("input_dim", 14),
            num_classes=cfg.get("num_classes", 6),
            d_state=cfg.get("d_state", 16),
        ),
    )
    register_model(
        "bae",
        lambda cfg: BAETabular(
            input_dim=cfg.get("input_dim", 14),
            num_classes=cfg.get("num_classes", 6),
            n_estimators=cfg.get("n_estimators", 5),
            latent_dim=cfg.get("latent_dim", 32),
        ),
    )


_register_builtins()
