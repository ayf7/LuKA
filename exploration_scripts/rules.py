from __future__ import annotations

from abc import ABC
import torch
import torch.nn.functional as F

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

    def name(self) -> str:
        raise NotImplementedError("Unimplemented rule")

class SimpleThresholdRule(Rule):
    """
    Marks keys with attention below a scalar threshold as LOW-importance (1).

    mask[b,h,t,k] = 1  if attention_score[b,h,t,k] > tau
                    0  otherwise
    """

    def __init__(self, tau: float = 1e-3):
        self.tau = float(tau)

    def process(self, attention_score: torch.Tensor) -> torch.Tensor:
        mask = (attention_score > self.tau).to(attention_score.dtype)
        return mask

    def name(self):
        return f"threshold_{self.tau}"

class MaxPoolThresholdRule(Rule):
    """
    Applies 1D max pooling over the key dimension followed by a scalar threshold.

    Each query row is smoothed independently by taking the maximum value within a
    sliding window across the key axis. The pooled attention map is then converted
    into a binary mask using the same scalar threshold as SimpleThresholdRule.
    """

    def __init__(
        self,
        tau: float = 1e-3,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ):
        self.tau = float(tau)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        if padding is None:
            # emulate "same" padding for odd-sized kernels
            self.padding = self.kernel_size // 2
        else:
            self.padding = int(padding)

    def process(self, attention_score: torch.Tensor) -> torch.Tensor:
        pooled = F.max_pool1d(
            attention_score.unsqueeze(1),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        ).squeeze(1)
        mask = (pooled > self.tau).to(attention_score.dtype)
        return mask

    def name(self) -> str:
        return f"maxpool1d{self.kernel_size}_stride{self.stride}_tau{self.tau}"

class MaxPoolEmaThresholdRule(Rule):
    """
    Applies 1D max pooling followed by an exponential moving average and
    thresholds the smoothed activations.
    """

    def __init__(
        self,
        tau: float = 1e-3,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        alpha: float = 0.3,
    ):
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")

        self.tau = float(tau)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        if padding is None:
            self.padding = self.kernel_size // 2
        else:
            self.padding = int(padding)
        self.alpha = float(alpha)

    def process(self, attention_score: torch.Tensor) -> torch.Tensor:
        pooled = F.max_pool1d(
            attention_score.unsqueeze(1),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        ).squeeze(1)

        ema = pooled.clone()
        alpha = self.alpha
        one_minus_alpha = 1.0 - alpha

        for t in range(1, ema.size(-1)):
            ema[:, t] = alpha * pooled[:, t] + one_minus_alpha * ema[:, t - 1]

        mask = (ema > self.tau).to(attention_score.dtype)
        return mask

    def name(self) -> str:
        return (
            f"maxpool1d{self.kernel_size}_stride{self.stride}_"
            f"ema{self.alpha}_tau{self.tau}"
        )

class MedianThresholdRule(Rule):
    """
    Marks keys with attention below the median as LOW-importance (1).

    For each attention map:
        mask[q, k] = 1  if attention_score[q, k] > median(attention_score)
                        0  otherwise
    """

    def process(self, attention_score: torch.Tensor) -> torch.Tensor:
        # Flatten to compute global median for this attention map
        Tq, Tk = attention_score.shape
        tril = torch.tril(torch.ones((Tq, Tk), dtype=torch.bool, device=attention_score.device))
        vals = attention_score[tril]
        median_val = vals.median()

        mask = (attention_score > median_val).to(attention_score.dtype)
        return mask

    def name(self):
        return "binary_median"
    
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
    
    def name(self):
        return "10_percentiles"
