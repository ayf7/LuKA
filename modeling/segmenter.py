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
        cover_is_summary: torch.Tensor
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


class DummySegmenter(Segmenter):
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
        self.min_chunk = int(min_chunk)
        self.tail_len = int(tail_len)
        self.max_pages = int(max_pages)

    def process(
        self,
        attn_weights: torch.Tensor,
        cover_indices: torch.Tensor,
        cover_is_summary: torch.Tensor
    ) -> Optional[torch.LongTensor]:
        
        B, H, L, T = attn_weights.shape
        device = attn_weights.device
        
        page_ends = torch.full((B, self.max_pages), -1, device=device, dtype=torch.long)
        
        for b in range(B):
            indices = cover_indices[b] # [T]
            is_sum = cover_is_summary[b] # [T]
            
            # Identify raw tail: not summary and not padding (-1)
            # We assume padding is -1 in indices.
            is_raw = (is_sum == 0) & (indices != -1)
            
            if not is_raw.any():
                continue
                
            # Get raw indices
            raw_indices = indices[is_raw]
            
            # We need at least min_chunk + tail_len tokens to form a page?
            # Or just min_chunk?
            # DummySegmenter usually leaves `tail_len` alone.
            
            num_raw = raw_indices.numel()
            if num_raw <= self.tail_len:
                continue
                
            # We can form pages from the first (num_raw - tail_len) tokens
            valid_count = num_raw - self.tail_len
            
            # Chunk into pages of size min_chunk
            # We take raw_indices[:valid_count]
            candidates = raw_indices[:valid_count]
            
            # Generate end indices
            # 0-based index in candidates: min_chunk-1, 2*min_chunk-1, ...
            
            num_pages = valid_count // self.min_chunk
            if num_pages == 0:
                continue
                
            num_pages = min(num_pages, self.max_pages)
            
            ends = []
            for i in range(num_pages):
                idx_in_candidates = (i + 1) * self.min_chunk - 1
                ends.append(candidates[idx_in_candidates].item())
                
            if ends:
                page_ends[b, :len(ends)] = torch.tensor(ends, device=device, dtype=torch.long)
            
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
        max_pages: int = 15,
    ):
        if lag <= 0:
            raise ValueError("lag must be positive")
        self.lag = int(lag)
        self.threshold = None if threshold is None else float(threshold)
        self.eps = float(eps)
        self.top_k = None if top_k is None else int(top_k)
        self.min_page_len = int(min_page_len)
        self.max_pages = int(max_pages)

    def _kl_scores(self, attn: torch.Tensor, start_idx: int = 0) -> torch.Tensor:
        """
        attn: [L, T] (head/query-aggregated, causal rows)
        start_idx: index of the first valid (non-padding) row/column.
        Returns: [L] scores (0 for rows without enough context)
        """
        # attn is likely [L, T].
        L, T = attn.shape
        if L < 2 * self.lag:
            return attn.new_zeros(L)

        # Normalize attention rows to ensure they sum to 1 (handling any pre-existing issues)
        # Add eps to avoid division by zero
        attn = attn.float() + self.eps
        attn = attn / attn.sum(dim=1, keepdim=True)
        
        # Zero out padding rows to avoid NaN propagation in cumsum
        if start_idx > 0:
            attn[:start_idx] = 0.0
        
        # Compute cumulative sum along L dimension for sliding window averages
        # Pad with one zero row at the beginning for easier indexing
        # c_pad: [L+1, T]
        c_pad = torch.cat([torch.zeros(1, T, device=attn.device, dtype=attn.dtype), attn.cumsum(dim=0)], dim=0)
        
        # We need to compute scores for r in [lag, L - lag - 1]
        # But we must also respect start_idx.
        # We need `r - lag >= start_idx` to ensure the previous window is valid.
        
        min_r = max(self.lag, start_idx + self.lag)
        max_r = L - self.lag
        
        if min_r >= max_r:
             return attn.new_zeros(L)
             
        valid_r = torch.arange(min_r, max_r, device=attn.device)
        
        # 1. Compute Q_prev (mean of previous `lag` rows)
        # For row r, prev window is [r-lag, r).
        # Sum is c_pad[r] - c_pad[r-lag]
        # We need this for all r in valid_r.
        # indices in c_pad:
        idx_end_prev = valid_r
        idx_start_prev = valid_r - self.lag
        
        sum_prev = c_pad[idx_end_prev] - c_pad[idx_start_prev] # [N, T]
        Q_prev = sum_prev / self.lag
        
        # 2. Compute Q_next (mean of next `lag` rows)
        # For row r, next window is [r+1, r+1+lag).
        # Sum is c_pad[r+1+lag] - c_pad[r+1]
        idx_end_next = valid_r + 1 + self.lag
        idx_start_next = valid_r + 1
        
        sum_next = c_pad[idx_end_next] - c_pad[idx_start_next] # [N, T]
        Q_next = sum_next / self.lag
        
        # 3. Get P (current rows)
        P = attn[valid_r] # [N, T]
        
        # 4. Compute KL Divergence
        # Normalize Qs to ensure they sum to 1 (they should approx, but for numerical stability)
        Q_prev = Q_prev + self.eps
        Q_prev = Q_prev / Q_prev.sum(dim=1, keepdim=True)
        
        Q_next = Q_next + self.eps
        Q_next = Q_next / Q_next.sum(dim=1, keepdim=True)
        
        log_P = torch.log(P)
        kl_prev = (P * (log_P - torch.log(Q_prev))).sum(dim=1)
        kl_next = (P * (log_P - torch.log(Q_next))).sum(dim=1)
        
        scores_valid = 0.5 * (kl_prev + kl_next)
        
        # 5. Map back to full scores tensor
        scores = attn.new_zeros(L)
        scores[valid_r] = scores_valid
        
        return scores

    def process(
        self,
        attn_weights: torch.Tensor,
        cover_indices: torch.Tensor,
        cover_is_summary: torch.Tensor
    ) -> Optional[torch.LongTensor]:
        """
        Args:
            attn_weights: [B, H, L, T]
            cover_indices: [B, T]
            cover_is_summary: [B, T]
        """
        device = attn_weights.device
        B, H, L, T_total = attn_weights.shape
        
        # We don't have seq_starts passed in, but we can infer valid rows?
        # In LuKA, attn_weights usually accumulates.
        # We assume L corresponds to the number of queries processed so far (or in this chunk).
        # If L is small (decoding), we might not have enough context for KL.
        # But let's assume L is large enough or we are in prefill/maintenance.
        
        # Aggregate over heads
        attn_mean = attn_weights.mean(dim=1)  # [B, L, T_total]
        page_ends = torch.full((B, self.max_pages), -1, device=device, dtype=torch.long)

        for b in range(B):
            # We assume we process the whole L for now.
            # In the reference, it used seq_starts to map to absolute positions.
            # Here, we just return indices relative to the current buffer?
            # Or do we return raw indices?
            # Segmenter.process returns "page end indices (inclusive) per batch element... in raw coordinates".
            # `cover_indices` maps column index -> raw index.
            
            # We compute scores for each row in L.
            # Row `r` corresponds to query `r`.
            # Does query `r` correspond to raw token `r`?
            # In `AttentionScoreBuffer`, we accumulate weights.
            # If we assume L grows with T (roughly), then row `r` is token `r`.
            indices_b = cover_indices[b] # [T]
            is_sum_b = cover_is_summary[b] # [T]
            
            # Infer start_idx from attention weights
            # Padding rows should have near-zero sum
            attn_b = attn_mean[b] # [L, T]
            row_sums = attn_b.sum(dim=1)
            valid_rows = (row_sums > 0.01) # Threshold for valid attention
            
            if not valid_rows.any():
                continue
                
            start_idx = valid_rows.nonzero()[0].item()
            
            # Compute scores with masking
            scores = self._kl_scores(attn_b, start_idx=start_idx)
            
            # print(f"Max KL score (batch {b}): {scores.max().item()}")

            if self.top_k is None and self.threshold is None:
                # Default behavior if neither set?
                # The user's code raises ValueError.
                # But we have default threshold=0.5 in __init__.
                pass

            # Identify candidate positions (indices in L)
            row_pos_b = torch.arange(L, device=device)
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

            # Map to raw indices using cover_indices
            # We assume row `r` corresponds to column `r`?
            # If so, we can use cover_indices[b][r].
            # But we must check if column `r` is a raw token.
            
            ends = []
            
            # Initialize last_boundary to enforce min_page_len for the first page
            # First page starts at seq_start (indices_b[start_idx])
            # We want raw_idx - seq_start + 1 >= min_page_len
            # So last_boundary = seq_start - 1
            seq_start = indices_b[start_idx].item()
            last_boundary = seq_start - 1
            
            for pos in selected_positions.tolist():
                if pos >= T_total:
                    continue
                
                # Check if it's a raw token
                if is_sum_b[pos] != 0:
                    continue
                    
                raw_idx = indices_b[pos].item()
                
                if raw_idx - last_boundary < self.min_page_len:
                    continue
                
                ends.append(raw_idx)
                last_boundary = raw_idx
                
                if len(ends) >= self.max_pages:
                    break

            if ends:
                page_ends[b, : len(ends)] = torch.tensor(ends, device=device)

        return page_ends

