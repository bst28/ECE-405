from __future__ import annotations

import math
import torch

# Try importing softmax from the package (pytest),
# otherwise import locally if running this file directly
try:
    from .softmax import softmax
except ImportError:  # running as a script
    from softmax import softmax  # type: ignore


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention.

    q, k: (batch_size, ..., seq_len, d_k)
    v:    (batch_size, ..., seq_len, d_v)
    mask: optional boolean tensor of shape (seq_len, seq_len)
          True  = allowed
          False = disallowed

    returns: (batch_size, ..., seq_len, d_v)
    """

    # d_k is the size of each query/key vector
    d_k = q.shape[-1]

    # Make sure q and k have the same feature size
    if k.shape[-1] != d_k:
        raise ValueError("q and k must have the same last dimension d_k")

    # Make sure q, k, v all have the same sequence length
    if q.shape[-2] != k.shape[-2] or q.shape[-2] != v.shape[-2]:
        raise ValueError("q, k, v must have the same seq_len on the second-to-last dimension")

    # Compute attention "scores" by dot product of q with k^T
    # Then scale by sqrt(d_k) to keep values from getting too large
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)

    # If we have a mask, block out disallowed positions
    if mask is not None:
        # Ensure mask is boolean
        if mask.dtype != torch.bool:
            mask = mask.bool()

        # Put -inf where mask is False so softmax makes it 0 probability
        scores = scores.masked_fill(~mask, float("-inf"))

    # Convert scores into probabilities (each row sums to 1)
    attn = softmax(scores, dim=-1)

    # If masked, force exact zeros and clean up any NaNs (from fully-masked rows)
    if mask is not None:
        attn = attn.masked_fill(~mask, 0.0)
        attn = torch.nan_to_num(attn, nan=0.0)

    # Weighted sum of values using attention probabilities
    return attn @ v