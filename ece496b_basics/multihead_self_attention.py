from __future__ import annotations

import torch
from torch import nn

# Try importing from the package (for pytest),
# otherwise import locally if running this file directly
try:
    from .linear import Linear
    from .scaled_dot_product_attention import scaled_dot_product_attention
except ImportError:  # running as a script
    from linear import Linear  # type: ignore
    from scaled_dot_product_attention import scaled_dot_product_attention  # type: ignore


class CausalMultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None) -> None:
        # Initialize parent PyTorch module
        super().__init__()

        # Total model dimension
        self.d_model = int(d_model)

        # Number of attention heads
        self.num_heads = int(num_heads)

        # Each head must get an equal chunk of the model dimension
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        # Size of each head (key, query, value dimension)
        self.d_k = self.d_model // self.num_heads

        # Linear layers to create queries, keys, values, and final output
        self.w_q = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.w_k = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.w_v = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.w_o = Linear(self.d_model, self.d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, sequence_length, d_model)
        returns: (batch_size, sequence_length, d_model)
        """

        # Get batch size (b) and sequence length (t)
        b, t, _ = x.shape

        # Number of heads and dimension per head
        h = self.num_heads
        d_k = self.d_k

        # Create queries, keys, and values from the input
        q = self.w_q(x)  # (b, t, d_model)
        k = self.w_k(x)
        v = self.w_v(x)

        # Split into multiple heads
        # (b, t, d_model) -> (b, h, t, d_k)
        q = q.view(b, t, h, d_k).transpose(1, 2)
        k = k.view(b, t, h, d_k).transpose(1, 2)
        v = v.view(b, t, h, d_k).transpose(1, 2)

        # Create a causal mask so tokens cannot attend to future tokens
        # Shape: (t, t), lower triangle is allowed
        mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))

        # Apply scaled dot-product attention for each head
        y = scaled_dot_product_attention(q, k, v, mask=mask)

        # Combine all heads back together
        # (b, h, t, d_k) -> (b, t, d_model)
        y = y.transpose(1, 2).contiguous().view(b, t, h * d_k)

        # Final linear projection
        return self.w_o(y)