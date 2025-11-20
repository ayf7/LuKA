import torch
from abc import ABC, abstractmethod

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
