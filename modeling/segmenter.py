import torch
from abc import ABC, abstractmethod

class Segmenter(ABC):
    
    @abstractmethod
    def process(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """
        Example sketch:
          - attention_score: [B, H, L_new_queries, L_seq]
          - Compute a simple feature per row, e.g., entropy or mean distance.
          - Run forward and backward EMA and output a score per row.
        
        Returns:
          - List of indices [B, max_num_pages], where, starting from the left,
          indices indicate the end of the page (inclusive), with the first index
          starting the first page.
        """
        pass

class DummySegmenter(Segmenter):
    """
    Dummy segmenter that returns fixed-size pages.

    For each batch element i, it produces page boundaries
    at indices:
        page_size - 1, 2*page_size - 1, ..., < length_i

    Output is an int64 tensor of shape [B, max_pages], where each row
    contains the *inclusive* end indices for that batch element,
    padded with -1 when a batch has fewer pages than max_pages.
    """

    def process(self, attention_scores: torch.Tensor,
                starting_pos: torch.Tensor) -> torch.LongTensor:
        """
        Input:
            attention_scores: [B, H, L_new_queries, L_seq]
            starting_pos: [B]
        
        Output:
            [B, H, max_pages], where indices in the 3rd dimension correpond to
            the end of page indices (inclusive) relative to the last L_new_queries
            positions.

            For instance, if 3 is the first element in the 3rd axis, then
            [0:3+1] is the next page.
        """
        pass