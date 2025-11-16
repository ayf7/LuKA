from __future__ import annotations

from abc import ABC
import torch
import torch.nn.functional as F
import matplotlib.colors as mcolors

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

    def color(self):
        """Default grayscale color map for binary masks."""

        return "gray"

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

class MagnitudeOrderRule(Rule):
    """Maps each attention score to its base-10 order-of-magnitude bucket."""

    _colormap = None

    def process(self, attention_score: torch.Tensor) -> torch.Tensor:
        x = attention_score
        # Only compute log10 on positive entries; mark zeros as NaN so they render distinctly.
        pos = x > 0
        out = torch.full_like(x, float('nan'))  # NaNs will show as gaps or a special color if you set one.
        out[pos] = torch.floor(torch.log10(x[pos]))
        # If you prefer to keep everything numeric, comment the two lines above and use this instead:
        # out = torch.where(pos, torch.floor(torch.log10(x)), torch.tensor(float('-inf'), device=x.device, dtype=x.dtype))
        return out  # returns buckets like ..., -5, -4, ..., 0, 1, ...

    def name(self) -> str:
        return "order_of_magnitude"

    def color(self):
        if self.__class__._colormap is None:
            self.__class__._colormap = mcolors.LinearSegmentedColormap.from_list(
                "purple_to_yellow",
                ["#5b2c83", "#f9f871"],
            )
        return self.__class__._colormap


class LaggedKLDivergenceRule(Rule):
    """
    Detects boundaries by comparing a row with averaged context windows before and after it.

    For each candidate boundary row r (with at least L rows before and after), we compute the mean
    attention distribution over the previous L rows and the next L rows. We then measure two KL
    divergences:
      - KL(current row || average of next L rows)
      - KL(current row || average of previous L rows)
    The final score is the average of these two divergences, written onto the causal portion of row r.
    First/last L rows remain zero because there is insufficient context. High scores highlight rows
    whose immediate past and future contexts differ substantially.
    """

    def __init__(
        self,
        lag: int = 32,
        eps: float = 1e-8,
        threshold: float | None = None,
    ):
        if lag <= 0:
            raise ValueError("lag must be positive")
        self.lag = int(lag)
        self.eps = float(eps)
        self.threshold = float(threshold) if threshold is not None else None

    def process(self, attention_score: torch.Tensor) -> torch.Tensor:
        if attention_score.ndim != 2:
            raise ValueError("LaggedKLDivergenceRule expects a 2D attention matrix")

        Tq, Tk = attention_score.shape
        output = torch.zeros_like(attention_score)
        if self.lag == 0 or Tq < 2 * self.lag:
            return output

        def kl_div(p_raw: torch.Tensor, q_raw: torch.Tensor) -> torch.Tensor:
            n = min(p_raw.shape[-1], q_raw.shape[-1])
            if n <= 0:
                return torch.tensor(0.0, device=attention_score.device, dtype=attention_score.dtype)
            p = p_raw[..., :n] + self.eps
            q = q_raw[..., :n] + self.eps
            p = p / p.sum()
            q = q / q.sum()
            return torch.sum(p * (torch.log(p) - torch.log(q)))

        start_idx = self.lag
        end_idx = Tq - self.lag
        for center_idx in range(start_idx, end_idx):
            current_len = min(Tk, center_idx + 1)
            if current_len <= 0:
                continue

            ahead_rows = attention_score[
                center_idx + 1 : center_idx + 1 + self.lag, :current_len
            ]
            behind_len = min(current_len, center_idx - self.lag + 1)
            if behind_len <= 0:
                continue
            behind_rows = attention_score[
                center_idx - self.lag : center_idx, :behind_len
            ]

            ahead_mean = ahead_rows.mean(dim=0)
            behind_mean = behind_rows.mean(dim=0)

            current_row = attention_score[center_idx]
            kl_ahead = kl_div(current_row[:current_len], ahead_mean)
            kl_behind = kl_div(current_row[:behind_len], behind_mean)

            score = kl_ahead / kl_behind
            score = torch.clamp(score, max=20.0)
            if self.threshold is not None:
                fill_value = 1.0 if score > self.threshold else 0.0
                output[center_idx, :current_len] = fill_value
            else:
                output[center_idx, :current_len] = score

        return output

    def name(self) -> str:
        suffix = "bin" if self.threshold is not None else "avg"
        return f"lagged_kl_l{self.lag}_{suffix}"
