from __future__ import annotations

import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) -> None:
        # Initialize parent PyTorch module
        super().__init__()

        # Base frequency for rotation
        self.theta = float(theta)

        # Dimension of each attention head
        self.d_k = int(d_k)

        # Maximum sequence length supported
        self.max_seq_len = int(max_seq_len)

        # RoPE requires an even dimension to pair values
        if self.d_k % 2 != 0:
            raise ValueError(f"d_k must be even for RoPE, got {self.d_k}")

        # Choose device if provided
        dev = device if device is not None else None

        # Build inverse frequencies for half the dimensions
        # These control how fast each dimension rotates
        half = self.d_k // 2
        idx = torch.arange(half, device=dev, dtype=torch.float32)
        inv_freq = self.theta ** (-2.0 * idx / self.d_k)

        # Create token positions from 0 to max_seq_len - 1
        t = torch.arange(self.max_seq_len, device=dev, dtype=torch.float32)

        # Compute rotation angles for each position
        freqs = t[:, None] * inv_freq[None, :]

        # Precompute cosine and sine values
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        # Store cos and sin as buffers so they move with the model
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: (..., seq_len, d_k)
        token_positions: (..., seq_len)
        returns: same shape as x
        """

        # Check that input has the expected dimension
        if x.shape[-1] != self.d_k:
            raise ValueError(f"Expected x last dim {self.d_k}, got {x.shape[-1]}")

        # Ensure positions are integers for indexing
        pos = token_positions.long()

        # Select the correct cos and sin values for each token position
        cos_pos = self.cos_cached[pos]
        sin_pos = self.sin_cached[pos]

        # Split input into even and odd dimensions
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # Apply the rotary transformation
        out_even = x_even * cos_pos - x_odd * sin_pos
        out_odd = x_even * sin_pos + x_odd * cos_pos

        # Combine even and odd dimensions back together
        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd

        return out