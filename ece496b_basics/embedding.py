from __future__ import annotations

import math
import torch
from torch import nn


# This class creates an embedding layer.
# It converts token IDs (numbers) into vectors.
class Embedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,   # total number of tokens in vocabulary
        embedding_dim: int,    # size of each embedding vector
        device=None,
        dtype=None,
    ) -> None:

        # Initialize parent PyTorch module
        super().__init__()

        # Store vocabulary size
        self.num_embeddings = int(num_embeddings)

        # Store vector size
        self.embedding_dim = int(embedding_dim)

        # Optional device and dtype settings
        factory_kwargs = {"device": device, "dtype": dtype}

        # --------------------------------------------------
        # Create embedding weight matrix
        # --------------------------------------------------
        # Shape: (vocab_size, embedding_dim)
        # Each row corresponds to one token’s vector.
        self.W = nn.Parameter(
            torch.empty(self.num_embeddings, self.embedding_dim, **factory_kwargs)
        )

        # --------------------------------------------------
        # Initialize weights using truncated normal distribution
        # --------------------------------------------------
        # We initialize with small random values
        # so training starts stable.
        std = 1.0 / math.sqrt(self.embedding_dim)

        nn.init.trunc_normal_(
            self.W,
            mean=0.0,
            std=std,
            a=-2.0 * std,
            b=2.0 * std,
        )

    # --------------------------------------------------
    # Forward pass (runs when we call the layer)
    # --------------------------------------------------
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:

        # Make sure token IDs are integers
        token_ids = token_ids.long()

        # --------------------------------------------------
        # Look up vectors for each token
        # --------------------------------------------------
        # If token_ids shape is (batch, seq_len),
        # output shape becomes (batch, seq_len, embedding_dim)
        return self.W[token_ids]