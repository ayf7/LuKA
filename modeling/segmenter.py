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
        attn_weights: torch.Tensor,
        cover_indices: torch.Tensor,
        cover_is_summary: torch.Tensor,
        layer_idx: Optional[int] = None
    ) -> Optional[torch.LongTensor]:
        """
        Args:
            attn_weights: [B, H, L_accum, T_current]
                Accumulated attention weights.
            cover_indices: [B, T_current]
                Indices mapping columns of attn_weights to raw/summary tokens.
            cover_is_summary: [B, T_current]
                Boolean/Long mask indicating if a token is a summary (1) or raw (0).
                
        Returns:
            page_ends: torch.LongTensor of shape [B, max_pages]
                Each entry is an inclusive end index of a page in raw coordinates,
                or -1 where no further pages remain.
        """
        raise NotImplementedError

    def _validate_output(self, page_ends: torch.LongTensor, cover_indices: torch.Tensor):
        """
        Validate that generated page boundaries are within valid raw token ranges.

        Args:
            page_ends: [B, max_pages]
                Inclusive raw indices of page endings. -1 indicates padding.
            cover_indices: [B, T]
                Mapping from cover view columns to raw indices (or page indices).
                Used to determine the maximum valid raw index available.
        """
        if page_ends is None: return
        
        # Check bounds: Page ends must be valid raw indices.
        # Since cover_indices contains the latest raw tokens at the tail,
        # the max value in cover_indices represents the furthest raw token we know about.
        if cover_indices.numel() > 0:
            max_idx = cover_indices.max().item()
            assert (page_ends == -1).all() or (page_ends.max() <= max_idx), \
                f"Invariant Violation: Page end index {page_ends.max()} exceeds max available raw index {max_idx}."




class DummySegmenter(Segmenter):
    """
    Fixed-length segmenter. Builds pages of length `min_chunk` until the
    remaining tokens fall into the uncompressed tail.

    Optimized with vectorized operations and early exit.
    """

    def __init__(
        self,
        min_chunk: int = 16,
        tail_len: int = 16,
        max_pages: int = 15,
    ):
        self.min_chunk = int(min_chunk)
        self.tail_len = int(tail_len)
        self.max_pages = int(max_pages)
        # Minimum raw tokens needed to create at least one page
        self._min_tokens_needed = self.tail_len + self.min_chunk

    def process(
        self,
        attn_weights: torch.Tensor,
        cover_indices: torch.Tensor,
        cover_is_summary: torch.Tensor,
        layer_idx: Optional[int] = None
    ) -> Optional[torch.LongTensor]:

        B, H, L, T = attn_weights.shape
        device = attn_weights.device

        # Early exit: check if any batch has enough raw tokens
        is_raw = (cover_is_summary == 0) & (cover_indices != -1)  # [B, T]
        raw_counts = is_raw.sum(dim=1)  # [B]

        if (raw_counts <= self._min_tokens_needed).all():
            return None  # No batch has enough tokens for even one page

        page_ends = torch.full((B, self.max_pages), -1, device=device, dtype=torch.long)

        # Process only batches that might have pages
        active_batches = (raw_counts > self._min_tokens_needed).nonzero(as_tuple=True)[0]

        for b in active_batches.tolist():
            indices = cover_indices[b]
            is_raw_b = is_raw[b]

            raw_indices = indices[is_raw_b]
            num_raw = raw_indices.numel()

            valid_count = num_raw - self.tail_len
            num_pages = min(valid_count // self.min_chunk, self.max_pages)

            if num_pages > 0:
                # Vectorized page end computation
                page_positions = torch.arange(1, num_pages + 1, device=device) * self.min_chunk - 1
                page_ends[b, :num_pages] = raw_indices[page_positions]

        return page_ends


class KLSegmenter(Segmenter):
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
        min_page_len: int = 16,
        tail_len: int = 16,
        max_pages: int = 15,
    ):
        if lag <= 0:
            raise ValueError("lag must be positive")
        self.lag = int(lag)
        self.threshold = None if threshold is None else float(threshold)
        self.eps = float(eps)
        self.top_k = None if top_k is None else int(top_k)
        self.min_page_len = int(min_page_len)
        self.tail_len = int(tail_len)
        self.max_pages = int(max_pages)

    def _kl_scores(self, attn: torch.Tensor) -> torch.Tensor:
        """
        attn: [L, T] (head/query-aggregated, causal rows)
        Returns: [L] scores (0 for rows without enough context)
        """
        if torch.isnan(attn).any():
            raise ValueError("NaN values found in attention scores during KL segmentation.")
            
        attn = attn.float()
        L, T = attn.shape
        scores = attn.new_zeros(L)
        if L < 2 * self.lag:
            return scores

        row_idx = torch.arange(L, device=attn.device).unsqueeze(1)
        col_idx = torch.arange(T, device=attn.device).unsqueeze(0)
        tril_mask = (col_idx <= row_idx).float()
        attn_eps = attn + self.eps
        attn_norm = attn_eps / attn_eps.sum(dim=1, keepdim=True)

        c_pad = torch.cat([torch.zeros(1, T, device=attn.device, dtype=attn.dtype), attn_norm], dim=0)
        c_sum = c_pad.cumsum(dim=0)
        
        sum_prev = c_sum[self.lag:] - c_sum[:-self.lag]
        
        prev_mean = torch.zeros_like(attn)
        if sum_prev.shape[0] > 0:
             prev_mean[self.lag:] = sum_prev[:-1]
             
        prev_mean = prev_mean / self.lag
        prev_mean = prev_mean.clamp_min(self.eps)
        
        next_mean = torch.zeros_like(attn)
        
        for k in range(1, self.lag + 1):
            future = attn.roll(-k, dims=0)
            future_masked = future * tril_mask
            future_norm = future_masked + self.eps
            future_norm = future_norm / future_norm.sum(dim=1, keepdim=True)
            
            next_mean += future_norm
            
        next_mean = next_mean / self.lag
        next_mean = next_mean.clamp_min(self.eps)

        curr = attn_norm.clamp_min(self.eps)
        
        log_curr = torch.log(curr)
        log_prev = torch.log(prev_mean)
        log_next = torch.log(next_mean)
        
        kl_prev = (curr * (log_curr - log_prev)).sum(dim=1)
        kl_next = (curr * (log_curr - log_next)).sum(dim=1)
        
        scores = 0.5 * (kl_prev + kl_next)
        scores[:self.lag] = 0
        scores[L - self.lag:] = 0
        
        return scores

    def process(
        self,
        attn_weights: torch.Tensor,
        cover_indices: torch.Tensor,
        cover_is_summary: torch.Tensor,
        layer_idx: Optional[int] = None
    ) -> Optional[torch.LongTensor]:
        """
        Args:
            attn_weights: [B, H, L, T]
            cover_indices: [B, T]
            cover_is_summary: [B, T]
        """
        device = attn_weights.device
        B, H, L, T_total = attn_weights.shape
        attn_mean = attn_weights.mean(dim=1)  # [B, L, T_total]
        page_ends = torch.full((B, self.max_pages), -1, device=device, dtype=torch.long)

        for b in range(B):
            indices_b = cover_indices[b] # [T]
            is_sum_b = cover_is_summary[b] # [T]
            valid_mask = indices_b != -1
            if not valid_mask.any():
                continue
                
            # Get valid indices and is_sum
            valid_indices = indices_b[valid_mask]
            valid_is_sum = is_sum_b[valid_mask]
            raw_mask = (valid_is_sum == 0)
            if not raw_mask.any():
                continue
                
            num_raw = raw_mask.sum().item()
            if num_raw <= self.tail_len:
                continue
            
            if L < num_raw:
                # Should not happen if L accumulates
                continue
            
            attn_b = attn_mean[b] # [L, T]
            
            # Compute scores with masking
            scores = self._kl_scores(attn_b)

            start_search = L - num_raw
            end_search = L - self.tail_len
            
            if start_search >= end_search:
                continue

            search_indices = torch.arange(start_search, end_search, device=device)
            search_scores = scores[search_indices]
            
            selected_indices = search_indices.new_tensor([], dtype=torch.long)

            if self.top_k is not None and self.top_k > 0:
                k = min(self.top_k, search_scores.numel())
                scores_top, top_idx = torch.topk(search_scores, k, largest=True)
                candidates = search_indices[top_idx]
                
                if self.threshold is not None:
                    mask = scores_top > self.threshold
                    candidates = candidates[mask]
                selected_indices = candidates

            if self.threshold is not None and (self.top_k is None or self.top_k <= 0):
                mask = search_scores > self.threshold
                selected_indices = search_indices[mask]

            if selected_indices.numel() == 0:
                continue

            selected_indices = selected_indices.sort().values
            
            raw_tokens = valid_indices[-num_raw:] # [num_raw]
            
            ends = []
            last_boundary = raw_tokens[0].item() - 1
            
            for r in selected_indices.tolist():
                idx_in_raw = r - (L - num_raw)
                raw_idx = raw_tokens[idx_in_raw].item()
                
                if raw_idx - last_boundary < self.min_page_len:
                    continue
                
                ends.append(raw_idx)
                last_boundary = raw_idx
                
                if len(ends) >= self.max_pages:
                    break

            if ends:
                page_ends[b, : len(ends)] = torch.tensor(ends, device=device)

        self._validate_output(page_ends, cover_indices)
        return page_ends


class GaussianSegmenter(Segmenter):
    """
    Page generator that samples a page length from a Gaussian and tiles the raw
    tail into fixed-size pages.

    - Draws page_len ~ N(mean, std), rounded to the nearest integer and clamped
      to at least 1.
    - If page_len exceeds the number of compressible raw tokens
      (num_raw - tail_len), no pages are emitted for that batch.
    - Otherwise, fills sequential pages of length page_len from the raw tail
      until the tail_len is reached or max_pages is hit.
    """

    def __init__(
        self,
        mean: float = 64.0,
        std: float = 16.0,
        tail_len: int = 16,
        max_pages: int = 15,
    ):
        self.mean = float(mean)
        self.std = float(std)
        self.tail_len = int(tail_len)
        self.max_pages = int(max_pages)

    def _sample_length(self, device) -> int:
        sample = torch.normal(
            mean=torch.tensor(self.mean, device=device),
            std=torch.tensor(self.std, device=device),
        )
        return int(torch.round(sample).item())

    def process(
        self,
        attn_weights: torch.Tensor,
        cover_indices: torch.Tensor,
        cover_is_summary: torch.Tensor,
        layer_idx: Optional[int] = None
    ) -> Optional[torch.LongTensor]:
        B, H, L, T = attn_weights.shape
        device = attn_weights.device
        page_ends = torch.full((B, self.max_pages), -1, device=device, dtype=torch.long)

        for b in range(B):
            indices = cover_indices[b]  # [T]
            is_sum = cover_is_summary[b]  # [T]

            raw_mask = (is_sum == 0) & (indices != -1)
            if not raw_mask.any():
                continue

            raw_indices = indices[raw_mask]
            num_raw = raw_indices.numel()
            compressible = num_raw - self.tail_len
            if compressible <= 0:
                continue

            compressible_indices = raw_indices[:compressible]
            ends = []
            offset = 0
            while offset < compressible and len(ends) < self.max_pages:
                page_len = self._sample_length(device)
                # Reject invalid or too-long samples; stop paging for this batch
                if page_len <= 0 or page_len > (compressible - offset):
                    break
                end_idx = offset + page_len - 1
                ends.append(int(compressible_indices[end_idx].item()))
                offset += page_len

            if ends:
                page_ends[b, : len(ends)] = torch.tensor(ends, device=device, dtype=torch.long)

        self._validate_output(page_ends, cover_indices)
        return page_ends
