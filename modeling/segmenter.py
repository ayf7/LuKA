import torch
from abc import ABC, abstractmethod
from typing import Optional

class Segmenter(ABC):
    """
    Abstract page segmenter.

    Implementations take attention statistics and output page end indices
    (inclusive) per batch element, padded with -1 where pages are absent.
    """

    @abstractmethod
    def process(
        self,
        attention_scores: torch.Tensor,   # [B, H, L_new, T_total]
        seq_starts: torch.Tensor,         # [B] absolute start index for each sequence
        min_chunk: int,
        tail_len: int,
        max_pages: int,
    ) -> torch.LongTensor:
        """
        Returns:
            page_ends: torch.LongTensor of shape [B, max_pages]
                Each entry is an inclusive end index of a page in raw coordinates,
                or -1 where no further pages remain.
        """
        raise NotImplementedError


class DummySegmenter(Segmenter):
    """
    Fixed-length segmenter. Builds pages of length `min_chunk` until the
    remaining tokens fall into the uncompressed tail.
    """

    def process(
        self,
        attention_scores: torch.Tensor,
        seq_starts: torch.Tensor,
        min_chunk: int,
        tail_len: int,
        max_pages: int,
    ) -> torch.LongTensor:
        device = attention_scores.device
        B = attention_scores.shape[0]

        page_size = min_chunk
        total_len = attention_scores.shape[-1]
        last_valid = total_len - tail_len - 1  # last index allowed for compression

        if last_valid < 0:
            return torch.full((B, max_pages), -1, device=device, dtype=torch.long)

        offsets = torch.arange(max_pages, device=device).unsqueeze(0)  # [1, max_pages]
        page_starts = seq_starts.unsqueeze(1) + offsets * page_size    # [B, max_pages]

        page_ends = page_starts + (page_size - 1)
        page_ends = torch.minimum(page_ends, torch.tensor(last_valid, device=device))

        exceed = page_starts > last_valid
        first_hit = torch.where(
            exceed.any(dim=1),
            exceed.float().argmax(dim=1),
            torch.full((B,), max_pages, device=device, dtype=torch.long)
        )

        col_ids = torch.arange(max_pages, device=device).unsqueeze(0)  # [1, max_pages]
        mask_after = col_ids > first_hit.unsqueeze(1)                  # [B, max_pages]

        page_ends = page_ends.masked_fill(mask_after, -1)
        return page_ends


class KLDivergenceSegmenter(Segmenter):
    """
    Boundary detector using a lagged KL-divergence heuristic (see PROJECT.md).

    For each query row r (causal length r+1), we compare the current attention
    distribution against the mean of the previous `lag` rows and the next `lag`
    rows, then average the two KL divergences. Positions whose score exceeds
    `threshold` and are at least `min_chunk` apart become page boundaries.
    """

    def __init__(
        self,
        lag: int = 8,
        threshold: Optional[float] = 0.5,
        eps: float = 1e-8,
        top_k: int | None = None,
    ):
        super().__init__()
        if lag <= 0:
            raise ValueError("lag must be positive")
        self.lag = int(lag)
        self.threshold = None if threshold is None else float(threshold)
        self.eps = float(eps)
        self.top_k = None if top_k is None else int(top_k)

    def _kl_scores(self, attn: torch.Tensor) -> torch.Tensor:
        """
        attn: [L, T] (head/query-aggregated, causal rows)
        Returns: [L] scores (0 for rows without enough context)
        """
        attn = torch.nan_to_num(attn.float(), nan=0.0, posinf=0.0, neginf=0.0)
        L, T = attn.shape
        scores = attn.new_zeros(L)
        if L < 2 * self.lag:
            return scores

        for r in range(self.lag, L):
            curr_len = min(r + 1, T)
            if curr_len <= 0:
                continue

            curr = attn[r, :curr_len].clamp_min(0) + self.eps
            curr = curr / curr.sum().clamp_min(self.eps * curr_len)

            prev_rows = attn[r - self.lag : r, :curr_len].clamp_min(0) + self.eps
            next_rows = attn[r + 1 : r + 1 + self.lag, :curr_len].clamp_min(0) + self.eps

            if prev_rows.numel() == 0 or next_rows.numel() == 0:
                continue

            prev_mean = prev_rows / prev_rows.sum(dim=1, keepdim=True).clamp_min(self.eps * curr_len)
            prev_mean = prev_mean.mean(dim=0).clamp_min(self.eps)
            next_mean = next_rows / next_rows.sum(dim=1, keepdim=True).clamp_min(self.eps * curr_len)
            next_mean = next_mean.mean(dim=0).clamp_min(self.eps)

            log_curr = torch.log(curr)
            log_prev = torch.log(prev_mean)
            log_next = torch.log(next_mean)
            kl_prev = (curr * (log_curr - log_prev)).sum()
            kl_next = (curr * (log_curr - log_next)).sum()
            scores[r] = 0.5 * (kl_prev + kl_next)

        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        return scores

    def process(
        self,
        attention_scores: torch.Tensor,
        seq_starts: torch.Tensor,
        min_chunk: int,
        tail_len: int,
        max_pages: int,
    ) -> torch.LongTensor:
        """
        attention_scores: [B, H, L_new, T_total] (causal rows)
        seq_starts: [B] absolute index of the first row in attention_scores
        """
        device = attention_scores.device
        B, H, L, T_total = attention_scores.shape
        last_valid = T_total - tail_len - 1  # last token eligible for compression

        # Aggregate over heads and (optionally) recent queries
        attn_mean = attention_scores.mean(dim=1)  # [B, L, T_total]
        page_ends = torch.full((B, max_pages), -1, device=device, dtype=torch.long)

        for b in range(B):
            # Absolute positions for each row
            row_positions = seq_starts[b] + torch.arange(L, device=device)
            valid_rows = row_positions <= last_valid
            if not valid_rows.any():
                continue

            attn_b = attn_mean[b][valid_rows]  # [L_valid, T_total]
            row_pos_b = row_positions[valid_rows]  # [L_valid]
            scores = self._kl_scores(attn_b)

            if self.top_k is None and self.threshold is None:
                raise ValueError("KLDivergenceSegmenter requires either top_k or threshold to be set.")

            selected_positions = row_pos_b.new_tensor([], dtype=row_pos_b.dtype)

            if self.top_k is not None and self.top_k > 0:
                k = min(self.top_k, scores.numel())
                scores_top, top_idx = torch.topk(scores, k, largest=True)
                selected_positions = row_pos_b[top_idx]
                if self.threshold is not None:
                    mask = scores_top > self.threshold
                    selected_positions = selected_positions[mask]

            if self.threshold is not None and (self.top_k is None or self.top_k <= 0):
                selected_positions = row_pos_b[scores > self.threshold]

            if selected_positions.numel() == 0:
                continue

            selected_positions = selected_positions.sort().values

            ends = []
            last_boundary = -min_chunk  # so first boundary can be near 0
            for pos in selected_positions.tolist():
                if pos - last_boundary < min_chunk:
                    continue
                if pos > last_valid:
                    break
                ends.append(pos)
                last_boundary = pos
                if len(ends) >= max_pages:
                    break

            if ends:
                page_ends[b, : len(ends)] = torch.tensor(ends, device=device)

        return page_ends
