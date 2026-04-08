from __future__ import annotations

import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Numerically stable softmax along dimension `dim`.
    """

    # Find the maximum value along the chosen dimension
    # This helps prevent very large numbers
    x_max = torch.max(x, dim=dim, keepdim=True).values

    # Subtract the max from the input for numerical stability
    x_stable = x - x_max

    # Exponentiate the stabilized values
    exp_x = torch.exp(x_stable)

    # Sum the exponentials along the same dimension
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)

    # Divide to get probabilities that sum to 1
    return exp_x / sum_exp