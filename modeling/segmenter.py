import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class _LayerBuffer:
    attention_scores: Optional[torch.Tensor] = None
    seq_starts: Optional[torch.Tensor] = None
    row_counts: Optional[torch.Tensor] = None
    last_processed: Optional[torch.Tensor] = None

class Segmenter(ABC):
    """
    Abstract page segmenter.

    Implementations take attention statistics and output page end indices
    (inclusive) per batch element, padded with -1 where pages are absent.
    """

    @abstractmethod
    def process(
        self,
        layer_idx: int,
    ) -> Optional[torch.LongTensor]:
        """
        Returns:
            page_ends: torch.LongTensor of shape [B, max_pages]
                Each entry is an inclusive end index of a page in raw coordinates,
                or -1 where no further pages remain.
        """
        raise NotImplementedError


class BufferedSegmenter(Segmenter):
    """
    Stateful segmenter that buffers attention weights per layer and exposes a
    push/flush interface for callers (e.g., KV caches).
    """

    def __init__(
        self,
        min_chunk: int = 16,
        tail_len: int = 16,
        max_pages: int = 15,
    ):
        super().__init__()
        self.min_chunk = int(min_chunk)
        self.tail_len = int(tail_len)
        self.max_pages = int(max_pages)
        self._layer_state: dict[int, _LayerBuffer] = {}

    def _get_layer_state(self, layer_idx: int) -> _LayerBuffer:
        if layer_idx not in self._layer_state:
            self._layer_state[layer_idx] = _LayerBuffer()
        return self._layer_state[layer_idx]

    @staticmethod
    def _get_seq_starts(attn_mask: torch.Tensor) -> torch.Tensor:
        last_rows = attn_mask[:, 0, -1, :]
        is_zero = (last_rows == 0)
        indices = is_zero.float().argmax(dim=1)
        no_zero = (~is_zero).all(dim=1)
        indices[no_zero] = -1
        return indices

    def push(
        self,
        layer_idx: int,
        attention_scores: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> None:
        state = self._get_layer_state(layer_idx)
        if state.seq_starts is None:
            state.seq_starts = self._get_seq_starts(attn_mask)

        if state.attention_scores is None:
            state.attention_scores = attention_scores
            b, _, rows, _ = attention_scores.shape
            state.row_counts = torch.full(
                (b,),
                rows,
                dtype=torch.long,
                device=attention_scores.device,
            )
            return

        b, h, new, total_cols = attention_scores.shape
        b0, h0, old_rows, old_cols = state.attention_scores.shape

        if b != b0 or h != h0:
            raise ValueError(
                f"push: batch/head mismatch (buffer {b0},{h0} vs new {b},{h})"
            )
        if new + old_cols != total_cols:
            raise ValueError(
                f"push: inconsistent lengths new={new}, old_cols={old_cols}, total={total_cols}"
            )

        new_buffer = attention_scores.new_zeros(b, h, old_rows + new, total_cols)
        new_buffer[:, :, :old_rows, :old_cols] = state.attention_scores
        new_buffer[:, :, -new:, :total_cols] = attention_scores

        state.attention_scores = new_buffer
        state.row_counts = (
            state.row_counts.to(device=attention_scores.device) + new
            if state.row_counts is not None
            else torch.full((b,), old_rows + new, device=attention_scores.device, dtype=torch.long)
        )

    def return_page_boundaries(self, layer_idx: int) -> Optional[torch.LongTensor]:
        state = self._layer_state.get(layer_idx)
        if state is None or state.attention_scores is None or state.seq_starts is None:
            return None
        page_ends = self.process(layer_idx)
        if page_ends is None:
            return None
        self._prune_processed(layer_idx, page_ends)
        return page_ends

    def _prune_processed(
        self,
        layer_idx: int,
        page_ends: torch.Tensor,
    ) -> None:
        state = self._layer_state[layer_idx]
        attn = state.attention_scores
        if attn is None:
            return

        b, h, _, cols = attn.shape
        new_counts: list[int] = []
        drops: list[int] = []
        max_keep = 0
        new_seq_starts = state.seq_starts.clone()
        last_processed = torch.full((b,), -1, device=attn.device, dtype=torch.long)

        for idx in range(b):
            count = int(state.row_counts[idx].item()) if state.row_counts is not None else attn.shape[2]
            valid = page_ends[idx][page_ends[idx] >= 0]
            drop = 0
            if valid.numel() > 0 and count > 0:
                last_processed[idx] = valid.max()
                drop = min(
                    count,
                    max(0, int(valid.max().item()) - int(state.seq_starts[idx].item()) + 1),
                )
            keep = max(count - drop, 0)
            new_counts.append(keep)
            drops.append(drop)
            max_keep = max(max_keep, keep)
            if drop > 0:
                new_seq_starts[idx] = state.seq_starts[idx] + drop

        if max_keep == 0:
            state.attention_scores = None
            state.seq_starts = None
            state.row_counts = None
            state.last_processed = last_processed
            return

        new_buffer = attn.new_zeros(b, h, max_keep, cols)
        for idx, keep in enumerate(new_counts):
            if keep <= 0:
                continue
            start = drops[idx]
            new_buffer[idx, :, :keep, :] = attn[idx, :, start : start + keep, :]

        state.attention_scores = new_buffer
        state.seq_starts = new_seq_starts
        state.row_counts = torch.tensor(new_counts, device=attn.device, dtype=torch.long)
        state.last_processed = last_processed


class DummySegmenter(BufferedSegmenter):
    """
    Fixed-length segmenter. Builds pages of length `min_chunk` until the
    remaining tokens fall into the uncompressed tail.
    """

    def __init__(
        self,
        min_chunk: int = 16,
        tail_len: int = 16,
        max_pages: int = 15,
    ):
        super().__init__(min_chunk=min_chunk, tail_len=tail_len, max_pages=max_pages)

    def process(
        self,
        layer_idx: int,
    ) -> Optional[torch.LongTensor]:
        state = self._layer_state.get(layer_idx)
        if state is None or state.attention_scores is None or state.seq_starts is None:
            return None

        attention_scores = state.attention_scores
        seq_starts = state.seq_starts
        device = attention_scores.device
        B = attention_scores.shape[0]
        rows_per_batch = (
            state.row_counts
            if state.row_counts is not None
            else torch.full((B,), attention_scores.shape[2], device=device, dtype=torch.long)
        )

        page_size = self.min_chunk
        total_len = attention_scores.shape[-1]
        last_valid = total_len - self.tail_len - 1  # last index allowed for compression

        if last_valid < 0:
            return torch.full((B, self.max_pages), -1, device=device, dtype=torch.long)

        page_ends = torch.full((B, self.max_pages), -1, device=device, dtype=torch.long)
        offsets = torch.arange(self.max_pages, device=device)

        for b in range(B):
            available_rows = int(rows_per_batch[b].item())
            if available_rows <= 0:
                continue

            max_row_pos = seq_starts[b] + available_rows - 1
            boundary_cap = min(last_valid, max_row_pos)
            if boundary_cap < seq_starts[b]:
                continue

            page_starts = seq_starts[b] + offsets * page_size
            ends = page_starts + (page_size - 1)
            ends = torch.minimum(ends, torch.tensor(boundary_cap, device=device))

            exceed = page_starts > boundary_cap
            first_hit = self.max_pages
            if exceed.any():
                first_hit = int(exceed.float().argmax().item())
            mask_after = torch.arange(self.max_pages, device=device) >= first_hit
            ends = ends.masked_fill(mask_after, -1)
            page_ends[b] = ends

        return page_ends


class KLDivergenceSegmenter(BufferedSegmenter):
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
        min_chunk: int = 16,
        tail_len: int = 16,
        max_pages: int = 15,
    ):
        super().__init__(min_chunk=min_chunk, tail_len=tail_len, max_pages=max_pages)
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
        layer_idx: int,
    ) -> Optional[torch.LongTensor]:
        """
        attention_scores: [B, H, L_new, T_total] (causal rows)
        seq_starts: [B] absolute index of the first row in attention_scores
        """
        state = self._layer_state.get(layer_idx)
        if state is None or state.attention_scores is None or state.seq_starts is None:
            return None

        attention_scores = state.attention_scores
        seq_starts = state.seq_starts
        device = attention_scores.device
        B, H, L, T_total = attention_scores.shape
        last_valid = T_total - self.tail_len - 1  # last token eligible for compression
        rows_per_batch = (
            state.row_counts
            if state.row_counts is not None
            else torch.full((B,), L, device=device, dtype=torch.long)
        )

        # Aggregate over heads and (optionally) recent queries
        attn_mean = attention_scores.mean(dim=1)  # [B, L, T_total]
        page_ends = torch.full((B, self.max_pages), -1, device=device, dtype=torch.long)

        for b in range(B):
            rows_available = int(min(rows_per_batch[b].item(), L))
            if rows_available <= 0:
                continue

            # Absolute positions for each row
            row_positions = seq_starts[b] + torch.arange(rows_available, device=device)
            valid_rows = row_positions <= last_valid
            if not valid_rows.any():
                continue

            attn_b = attn_mean[b, :rows_available][valid_rows]  # [L_valid, T_total]
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
            last_boundary = -self.min_chunk  # so first boundary can be near 0
            for pos in selected_positions.tolist():
                if pos - last_boundary < self.min_chunk:
                    continue
                if pos > last_valid:
                    break
                ends.append(pos)
                last_boundary = pos
                if len(ends) >= self.max_pages:
                    break

            if ends:
                page_ends[b, : len(ends)] = torch.tensor(ends, device=device)

        return page_ends
