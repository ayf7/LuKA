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
    - Input:  attention_score with shape (T_q, T_k) or (T_q, T_k).
              It should already be non-negative and (typically) row-normalized
              over the key dimension (per query token).
    - Output: mask with the same trailing shape (..., T_q, T_k) between [0, 1].
              What these values mean is up to the user.
    """

    def process(self, attention_score: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Unimplemented rule")

class SimpleThresholdRule(Rule):
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

class MedianThresholdRule(Rule):
    """
    Marks keys with attention below the median as LOW-importance (1).

    For each attention map:
        mask[q, k] = 1  if attention_score[q, k] < median(attention_score)
                        0  otherwise
    """

    def process(self, attention_score: torch.Tensor) -> torch.Tensor:
        # Flatten to compute global median for this attention map
        Tq, Tk = attention_score.shape
        tril = torch.tril(torch.ones((Tq, Tk), dtype=torch.bool, device=attention_score.device))
        vals = attention_score[tril]
        median_val = vals.median()

        mask = (attention_score < median_val).to(attention_score.dtype)
        return mask
    
class PercentileDecileRule(Rule):
    """
    Assigns each attention score a decile indicator in {0.00, 0.10, ..., 0.90}
    computed from the *lower-triangular* distribution (including the diagonal).

    Semantics:
      - 0.00 -> among the highest scores
      - 0.90 -> among the lowest scores

    Upper-triangular entries are left as 0.00 by default (can be changed to NaN).
    """

    def process(self, attention_score: torch.Tensor) -> torch.Tensor:
        Tq, Tk = attention_score.shape
        device = attention_score.device
        dtype = attention_score.dtype

        # Lower-triangular mask (including diagonal)
        tril_mask = torch.tril(torch.ones((Tq, Tk), dtype=torch.bool, device=device))

        # Extract lower-triangular values
        vals = attention_score[tril_mask]
        M = vals.numel()

        if M == 0:
            return torch.zeros_like(attention_score)

        # Rank ascending
        order = torch.argsort(vals, dim=0, stable=True)
        ranks = torch.empty_like(order, dtype=torch.long)
        ranks[order] = torch.arange(M, device=device)

        # Convert to [0,1] percentile (ascending)
        p = (ranks + 1).to(torch.float32) / float(M)

        # Map directly to deciles (0.00 for lowest, 0.90 for highest)
        decile_index = torch.floor(p * 10.0).to(torch.long)
        decile_index = torch.clamp(decile_index, 0, 9)

        decile_values = decile_index.to(dtype) / 10.0  # 0.00, 0.10, ..., 0.90

        # Scatter back into full matrix
        out = torch.zeros_like(attention_score, dtype=dtype)
        out[tril_mask] = decile_values

        # Uncomment to mark upper-triangular region explicitly:
        # out[~tril_mask] = torch.nan
        return out