"""
Segment (page) sampling utilities.

This sampler builds page boundaries from page ends only: it samples cumulative
end positions around multiples of `mean` with per-page jitter, then converts
those ends into contiguous segments. The first segment starts at the first
valid token (after padding) and ends just before the first sampled end.
"""

from typing import List

import torch

from training.config import SamplerParams


def sample_segments(attention_mask: torch.Tensor, params: SamplerParams) -> List[List[torch.Tensor]]:
    """
    Sample non-overlapping segments per batch row using jittered page ends.

    Args:
        attention_mask: torch.Tensor [B, L] (1=token, 0=pad); typically left-padded.
        params: SamplerParams with tail_len, mean, std, max_pages.

    Returns:
        segments: List length B; segments[b] is a list of 1D LongTensor index ranges
                  (monotonic, within valid tokens) for that batch row. Each segment
                  tensor has shape [len] with indices in [0, L) referring to token
                  positions (including left padding).

    Side effects:
        Uses torch RNG for jitter; no file I/O.
    """
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be [B, L], got shape {attention_mask.shape}")

    tail_len = int(params.tail_len)
    base_len = max(int(round(params.mean)), 1)
    jitter_std = float(params.std)
    max_pages = int(params.max_pages)

    B, L = attention_mask.shape
    device = attention_mask.device
    segments: List[List[torch.Tensor]] = [[] for _ in range(B)]

    # Per-page jitter (can be negative) around each multiple of mean.
    if jitter_std > 0:
        jitter = torch.randn((B, max_pages), device=device) * jitter_std
        jitter = jitter.round().to(dtype=torch.long)
    else:
        jitter = torch.zeros((B, max_pages), device=device, dtype=torch.long)

    for b in range(B):
        valid_len = int(attention_mask[b].sum().item())
        if valid_len <= tail_len:
            continue

        # With left padding, tokens start at L - valid_len.
        start_offset = L - valid_len
        compressible_len = valid_len - tail_len
        compressible_end_excl = start_offset + compressible_len

        prev_end_excl = start_offset
        for page_idx in range(max_pages):
            target_end = start_offset + int(round((page_idx + 1) * base_len + jitter[b, page_idx].item()))
            target_end = max(target_end, prev_end_excl + 1)  # ensure non-empty if possible
            target_end = min(target_end, compressible_end_excl)

            if target_end <= prev_end_excl:
                break
            seg = torch.arange(prev_end_excl, target_end, device=device, dtype=torch.long)
            if seg.numel() > 0:
                segments[b].append(seg)
            prev_end_excl = target_end
            if prev_end_excl >= compressible_end_excl:
                break

    return segments


if __name__ == "__main__":
    """
    Manual sanity check for sampler.

    Builds a fake attention mask with varying lengths, samples segments, and
    prints the resulting page starts/ends per batch row.
    """
    torch.manual_seed(0)
    B = 3
    L = 128
    lengths = torch.tensor([128, 40, 20], dtype=torch.long)
    attn_mask = torch.zeros((B, L), dtype=torch.long)
    for b in range(B):
        attn_mask[b, L - lengths[b] :] = 1  # left padding

    params = SamplerParams(tail_len=8, mean=16.0, std=4.0, max_pages=4)
    segs = sample_segments(attn_mask, params)
    for b, seg_list in enumerate(segs):
        print(f"batch {b} (len={lengths[b].item()}):")
        if not seg_list:
            print("  no segments")
            continue
        for i, s in enumerate(seg_list):
            print(f"  seg {i}: start={int(s[0])}, end={int(s[-1])}, len={s.numel()}")
