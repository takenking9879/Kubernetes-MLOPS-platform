"""Bootstrap Aggregating Ensemble (BAE) of autoencoders for tabular data.

Each estimator is an autoencoder whose bottleneck representation feeds a
classification head.  Forward returns the *mean* of all estimators' logits
so the standard cross-entropy loss applies unchanged.

Architecture per estimator:
    encoder:     input_dim → latent_dim*2 → latent_dim  (ReLU)
    classifier:  latent_dim → num_classes
    decoder:     latent_dim → latent_dim*2 → input_dim  (used only for
                 optional regularisation; ignored in inference mode)

Usage::

    from models.bae import BAETabular

    model = BAETabular(input_dim=14, num_classes=6, n_estimators=5, latent_dim=32)
    logits = model(x)   # x: (batch, input_dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _AEEstimator(nn.Module):
    """Single autoencoder + classification head."""

    def __init__(self, input_dim: int, latent_dim: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(latent_dim, num_classes)
        # Decoder kept for possible future reconstruction regularisation
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits from the bottleneck representation."""
        return self.classifier(self.encoder(x))


class BAETabular(nn.Module):
    """Ensemble of autoencoder classifiers for tabular inputs.

    The forward pass returns mean logits across all estimators.
    Compatible with standard cross-entropy loss — no changes to
    the training loop required.

    Args:
        input_dim:    Number of input features.
        num_classes:  Number of output classes.
        n_estimators: Number of ensemble members (default 5).
        latent_dim:   Bottleneck dimension per estimator (default 32).
    """

    def __init__(
        self,
        input_dim: int = 14,
        num_classes: int = 6,
        n_estimators: int = 5,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()
        self.estimators = nn.ModuleList(
            [_AEEstimator(input_dim, latent_dim, num_classes) for _ in range(n_estimators)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stack logits: (n_estimators, batch, num_classes) → mean over dim 0
        return torch.stack([est(x) for est in self.estimators], dim=0).mean(dim=0)
