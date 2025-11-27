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
