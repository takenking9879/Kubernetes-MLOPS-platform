"""State Space Model (SSM) for tabular classification / regression.

Architecture:
    Each input feature is treated as a position in a 1-D sequence.
    A lightweight scalar-embedding projects each feature value to d_model,
    then n_layers of SSM blocks (simplified diagonal-A S4 variant) refine
    the representations.  Global mean-pooling collapses the sequence, and a
    linear head produces the final class logits.

Usage::

    from models.ssm import SSMTabular

    model = SSMTabular(input_dim=14, num_classes=6, d_state=16)
    logits = model(x)   # x: (batch, input_dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _SSMBlock(nn.Module):
    """Single SSM layer with diagonal-A state transitions (S4-lite).

    Processes a sequence (batch, L, d_model) and returns (batch, L, d_model).
    Uses a recurrent scan over L for efficiency at typical tabular widths
    (L = input_dim ≤ ~200).
    """

    def __init__(self, d_model: int, d_state: int) -> None:
        super().__init__()
        # Learnable log of |A| diagonal — negated to keep |A| < 1 (stable)
        self.A_log = nn.Parameter(torch.randn(d_state))
        # Input projection: d_model → d_state
        self.B = nn.Parameter(torch.randn(d_state, d_model) * 0.01)
        # Output projection: d_state → d_model
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        # Direct pass-through (skip connection D)
        self.D = nn.Parameter(torch.ones(d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, L, d_model)
        batch, L, _ = x.shape
        A_diag = -torch.exp(self.A_log)                        # (d_state,) — stable

        h = x.new_zeros(batch, self.A_log.shape[0])           # hidden state
        outs: list[torch.Tensor] = []
        for t in range(L):
            xt = x[:, t, :]                                    # (batch, d_model)
            h = torch.exp(A_diag) * h + xt @ self.B.T         # (batch, d_state)
            y = h @ self.C.T + xt * self.D                    # (batch, d_model)
            outs.append(y)

        return self.norm(torch.stack(outs, dim=1))             # (batch, L, d_model)


class SSMTabular(nn.Module):
    """SSM-based model for tabular inputs.

    Args:
        input_dim:   Number of input features.
        num_classes: Number of output classes (or 1 for regression).
        d_model:     Internal hidden dimension (default 64).
        d_state:     SSM state dimension (default 16).
        n_layers:    Number of stacked SSM blocks (default 2).
    """

    def __init__(
        self,
        input_dim: int = 14,
        num_classes: int = 6,
        d_model: int = 64,
        d_state: int = 16,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        # Embed each scalar feature to d_model
        self.embed = nn.Linear(1, d_model)
        self.layers = nn.ModuleList(
            [_SSMBlock(d_model, d_state) for _ in range(n_layers)]
        )
        self.pool_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        x = x.unsqueeze(-1)          # (batch, input_dim, 1)
        x = self.embed(x)            # (batch, input_dim, d_model)
        for layer in self.layers:
            x = x + layer(x)         # residual connection
        x = self.pool_norm(x)
        x = x.mean(dim=1)            # global average pool → (batch, d_model)
        return self.head(x)
