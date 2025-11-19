import torch
from abc import ABC, abstractmethod
class Rule(ABC):
    
    @abstractmethod
    def process(self, attention_score: torch.Tensor) -> torch.Tensor:
        """
        Example sketch:
          - attention_score: [T, T]
          - Compute a simple feature per row, e.g., entropy or mean distance.
          - Run forward and backward EMA and output a score per row.
          - For compatibility with LukaKVCache, expand to [T, T] by broadcasting.
        """
        pass

import torch
from typing import Optional


class SimpleLaggedKLDivergenceRule(Rule):
    """
    For each query row t, compare that row's attention distribution to the
    average of the previous `lag` rows (on the causal slice).

    - Input:  attention_score: [Tq, Tk]
    - Output: scores:          [Tq, Tk] where each row t has a constant
      score on its causal part (0..t), and zeros elsewhere.

    Score at row t (for t >= lag) is:
        KL( p_t || mean_{i in [t-lag, t-1]} p_i )

    where p_t is the normalized attention over keys up to t.
    """

    def __init__(
        self,
        lag: int = 32,
        eps: float = 1e-8,
        threshold: Optional[float] = None,
        clamp_max: Optional[float] = 20.0,
    ) -> None:
        if lag <= 0:
            raise ValueError("lag must be positive")
        self.lag = int(lag)
        self.eps = float(eps)
        self.threshold = float(threshold) if threshold is not None else None
        self.clamp_max = clamp_max

    def _kl_div(
        self,
        p_raw: torch.Tensor,  # [N]
        q_raw: torch.Tensor,  # [N]
        eps: float,
    ) -> torch.Tensor:
        """
        KL(p || q) where p and q are non-negative scores on the same support.
        """
        # Ensure same length
        n = min(p_raw.shape[-1], q_raw.shape[-1])
        if n <= 0:
            return torch.tensor(
                0.0, device=p_raw.device, dtype=p_raw.dtype
            )

        p = p_raw[..., :n] + eps
        q = q_raw[..., :n] + eps
        p = p / p.sum()
        q = q / q.sum()
        return torch.sum(p * (torch.log(p) - torch.log(q)))

    def process(self, attention_score: torch.Tensor) -> torch.Tensor:
        if attention_score.ndim != 2:
            raise ValueError(
                "SimpleLaggedKLDivergenceRule expects a 2D attention matrix"
            )

        Tq, Tk = attention_score.shape
        device = attention_score.device
        dtype = attention_score.dtype

        # Output scores (same shape as attention_score).
        output = torch.zeros_like(attention_score)

        # Not enough rows to form a lag window.
        if Tq <= self.lag:
            return output

        # Normalize each row to a probability distribution over keys.
        # We always normalize on the full key axis first, but for causal
        # comparisons we later truncate to keys [0..t].
        probs = attention_score + self.eps
        row_sums = probs.sum(dim=-1, keepdim=True)
        # Avoid division by zero for degenerate rows.
        row_sums = torch.clamp(row_sums, min=self.eps)
        probs = probs / row_sums  # [Tq, Tk]

        for t in range(self.lag, Tq):
            # Causal slice: only attend to keys up to time t.
            current_len = min(Tk, t + 1)
            if current_len <= 0:
                continue

            # Current row distribution on causal slice.
            p_t = probs[t, :current_len]  # [current_len]

            # Previous `lag` rows on the same causal slice.
            behind_rows = probs[t - self.lag : t, :current_len]  # [lag, current_len]
            if behind_rows.numel() == 0:
                continue

            behind_mean = behind_rows.mean(dim=0)  # [current_len]

            # KL(p_t || behind_mean)
            kl_val = self._kl_div(p_t, behind_mean, self.eps)

            # Optional clamp
            if self.clamp_max is not None:
                kl_val = torch.clamp(kl_val, max=self.clamp_max)

            if self.threshold is not None:
                # Binary mask: boundary if KL > threshold.
                score_row = torch.where(
                    kl_val > self.threshold,
                    torch.tensor(1.0, device=device, dtype=dtype),
                    torch.tensor(0.0, device=device, dtype=dtype),
                )
            else:
                # Use raw KL as the score.
                score_row = kl_val

            # Broadcast the scalar score across the causal part of row t.
            # Non-causal positions (keys > t) remain zero.
            output[t, :current_len] = score_row

        return output