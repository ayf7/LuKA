"""
Segment (page) sampling utilities.

Implements a jittered tiling strategy over valid tokens, leaving a protected
tail of uncompressed tokens. This module only defines interfaces and shape
contracts; implementations are omitted in scaffolding.
"""

from typing import List

import torch

from training.config import SamplerParams


def sample_segments(attention_mask: torch.Tensor, params: SamplerParams) -> List[List[torch.Tensor]]:
    """
    Sample non-overlapping segments per batch row.

    Args:
        attention_mask: torch.Tensor [B, L] (1=token, 0=pad); left-padded to match inputs.
        params: SamplerParams with tail_len, mean, std, max_pages.

    Returns:
        segments: Python list of length B; segments[b] is a list of 1D LongTensor
                  index ranges (monotonic, within valid tokens) for that batch row.
                  Each segment tensor has shape [page_len] with indices in [0, L).

    Side effects:
        Should rely on torch RNG for jitter; no file I/O.
    """
    raise NotImplementedError("Segment sampler is not implemented in scaffolding.")
