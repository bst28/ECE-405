from __future__ import annotations

import torch
from torch import nn

# Try importing Linear from the package,
# otherwise import locally if running this file directly
try:
    from .linear import Linear
except ImportError:  # pragma: no cover
    from linear import Linear  # type: ignore


def _round_up_to_multiple(x: int, m: int) -> int:
    # Round x up to the nearest multiple of m
    return ((x + m - 1) // m) * m


class SwiGLU(nn.Module):
    """
    Feed-forward network using SwiGLU.

    gate = x W_gate
    up   = x W_up
    hidden = SiLU(gate) * up
    out = hidden W_down
    """

    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None) -> None:
        # Initialize parent PyTorch module
        super().__init__()

        # Model dimension
        self.d_model = int(d_model)

        # If feed-forward dimension is not given, compute a default value
        if d_ff is None:
            # Common SwiGLU rule: ~8/3 * d_model
            d_ff_calc = int((8 * self.d_model) / 3)

            # Round up to a multiple of 64 for efficiency
            d_ff_calc = _round_up_to_multiple(d_ff_calc, 64)

            # Ensure minimum size
            d_ff_calc = max(64, d_ff_calc)
            d_ff = d_ff_calc

        # Store feed-forward dimension
        self.d_ff = int(d_ff)

        # Linear layers for gate, up, and down projections
        self.w_gate = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w_up = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w_down = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute gate and up projections
        gate = self.w_gate(x)  # (..., d_ff)
        up = self.w_up(x)      # (..., d_ff)

        # Apply SiLU activation: z * sigmoid(z)
        silu_gate = gate * torch.sigmoid(gate)

        # Combine gate and up paths
        hidden = silu_gate * up

        # Project back to model dimension
        return self.w_down(hidden)