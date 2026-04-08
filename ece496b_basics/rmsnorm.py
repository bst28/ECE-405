from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        # Initialize parent PyTorch module
        super().__init__()

        # Model dimension
        self.d_model = int(d_model)

        # Small value to avoid division by zero
        self.eps = float(eps)

        # Store device and dtype settings
        factory_kwargs = {"device": device, "dtype": dtype}

        # Learnable scale parameter (gamma)
        # One value per feature in the model dimension
        self.weight = nn.Parameter(torch.ones(self.d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d_model) typically (batch_size, sequence_length, d_model)
        returns: same shape as x
        """

        # Save original data type
        orig_dtype = x.dtype

        # Convert to float32 for stable math
        x_f = x.to(torch.float32)

        # Compute RMS (root mean square) across the last dimension
        rms = torch.sqrt(torch.mean(x_f * x_f, dim=-1, keepdim=True) + self.eps)

        # Normalize the input and apply the learnable scale
        y = (x_f / rms) * self.weight

        # Convert back to the original data type
        return y.to(orig_dtype)