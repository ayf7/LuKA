from __future__ import annotations

from abc import ABC
import torch

class Rule(ABC):
    """
    A dummy class that standardizes converting attention scores into binary
    scores. Defines a process() function that takes as input an attention_score
    matrix, and outputs a binary matrix output.

    Conventions
    ----------
    - Input:  attention_score with shape (B, H, T_q, T_k) or (H, T_q, T_k).
              It should already be non-negative and (typically) row-normalized
              over the key dimension (per query token).
    - Output: mask with the same trailing shape (..., T_q, T_k) between [0, 1].
              What these values mean is up to the user.
    """

    def process(self, attention_score: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Unimplemented rule")

class ThresholdRule(Rule):
    """
    Marks keys with attention below a scalar threshold as LOW-importance (1).

    mask[b,h,t,k] = 1  if attention_score[b,h,t,k] < tau
                    0  otherwise
    """

    def __init__(self, tau: float = 1e-3):
        self.tau = float(tau)

    def process(self, attention_score: torch.Tensor) -> torch.Tensor:
        mask = (attention_score < self.tau).to(attention_score.dtype)
        return mask