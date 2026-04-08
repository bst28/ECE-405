from __future__ import annotations

import torch
from torch import nn

# Try importing from the package (works in pytest),
# otherwise import locally if running this file directly
try:
    from .rmsnorm import RMSNorm
    from .multihead_self_attention import CausalMultiheadSelfAttention
    from .positionwise_feedforward import SwiGLU
except ImportError:  # running as a script (no package context)
    from rmsnorm import RMSNorm  # type: ignore
    from multihead_self_attention import CausalMultiheadSelfAttention  # type: ignore
    from positionwise_feedforward import SwiGLU  # type: ignore


class TransformerBlock(nn.Module):
    """
    One Transformer block using pre-normalization:

        x = x + Attention(RMSNorm(x))
        x = x + FeedForward(RMSNorm(x))
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None) -> None:
        # Initialize parent PyTorch module
        super().__init__()

        # Save model sizes
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_ff = int(d_ff)

        # RMSNorm layers before attention and before feedforward
        self.ln_1 = RMSNorm(d_model=self.d_model, device=device, dtype=dtype)
        self.ln_2 = RMSNorm(d_model=self.d_model, device=device, dtype=dtype)

        # Causal self-attention (cannot look at future tokens)
        self.attn = CausalMultiheadSelfAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            device=device,
            dtype=dtype,
        )

        # Feed-forward network (SwiGLU)
        self.ffn = SwiGLU(
            d_model=self.d_model,
            d_ff=self.d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize -> attention -> add back to input (residual connection)
        x = x + self.attn(self.ln_1(x))

        # Normalize -> feedforward -> add back to input (residual connection)
        x = x + self.ffn(self.ln_2(x))

        # Return updated activations
        return x
