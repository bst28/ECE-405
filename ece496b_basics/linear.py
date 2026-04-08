from __future__ import annotations

import math
import torch
from torch import nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
    ) -> None:
        # Initialize the parent PyTorch module
        super().__init__()

        # Number of input features
        self.in_features = int(in_features)

        # Number of output features
        self.out_features = int(out_features)

        # Store device and data type settings
        factory_kwargs = {"device": device, "dtype": dtype}

        # Create the weight matrix W
        # Shape: (in_features, out_features)
        self.W = nn.Parameter(
            torch.empty(self.in_features, self.out_features, **factory_kwargs)
        )

        # Set the standard deviation for initialization
        # Smaller values help keep training stable
        std = 1.0 / math.sqrt(self.in_features)

        # Initialize weights using a truncated normal distribution
        nn.init.trunc_normal_(
            self.W,
            mean=0.0,
            std=std,
            a=-2.0 * std,
            b=2.0 * std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is the input tensor with shape (..., in_features)
        # Matrix multiply input with weights to get output
        # Output shape: (..., out_features)
        return x @ self.W