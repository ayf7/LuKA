import torch
import torch.nn as nn
from modeling.segmenter import Segmenter, DummySegmenter
from modeling.compressor import Compressor, MeanCompressor, AttentionWeightedCompressor
from modeling.refinement import RefinementRule, FixedThresholdRule, NoRefinementRule, get_refinement_rule
from transformers.cache_utils import Cache, DynamicCache
from transformers import PretrainedConfig

from typing import Any, Optional, List, Tuple
from transformers.models.qwen3.modeling_qwen3 import (
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv
)

class RawCache:

    def __init__(self, config: PretrainedConfig, cache: DynamicCache = None):
        """Container for the raw (uncompressed) KV cache.

        The raw cache mirrors the Hugging Face `DynamicCache`, but keeps a few
        extra offsets needed by LuKA:
        - `seq_start`: [B] start index of non-padding tokens (left padding).
        - `raw_seq_start`: [B] start index of tokens that are *not* covered by
          a page; indices before this point are summarized.

        Args:
            config: PretrainedConfig; used to size and align tensors with the
                underlying model (e.g., num_hidden_layers, head counts, dtype).
            cache: Optional existing `DynamicCache` to wrap.
        """
        self.cache = cache
        self.config = config
        self.num_layers = getattr(config, "num_hidden_layers", None)
        if self.num_layers is None:
            raise ValueError("config.num_hidden_layers is required to size RawCache.")
        # Per-layer metadata; key/values are read directly from DynamicCache
        self.seq_start: list[Optional[torch.Tensor]] = [None] * self.num_layers     # [B] left-pad offset
        self.raw_seq_start: list[Optional[torch.Tensor]] = [None] * self.num_layers # [B] frontier of raw tokens

    def update(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: Optional[int] = None,
        cache_kwargs: Optional[dict[str, Any]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Append new raw KV to the underlying cache.

        Args:
            keys: torch.Tensor
                [B, H, L_new, D] key states for the newly generated tokens
                (L_new is usually 1 during decoding).
            values: torch.Tensor
                [B, H, L_new, D] value states for the newly generated tokens.
            layer_idx: Optional[int]
                Layer to update inside the wrapped `DynamicCache` (if present).
            cache_kwargs: Optional[dict[str, Any]]
                Extra kwargs forwarded to `DynamicCache.update` (e.g., RoPE data).
            attention_mask: Optional[torch.Tensor]
                [B, 1, L_q, T_raw] additive mask; used to infer left padding
                (seq_start) when first initializing offsets.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: concatenated
            raw keys/values plus offsets for this layer: keys/values [B, H, T_raw, D],
            seq_start [B], raw_seq_start [B].
        """
        if layer_idx is None:
            raise ValueError("layer_idx must be provided when updating RawCache.")

        if self.cache is None:
            raise ValueError("RawCache.update requires an underlying DynamicCache. Call initialize_with_cache first.")

        # Update the underlying HF cache
        k_all, v_all = self.cache.update(keys, values, layer_idx, cache_kwargs)

        # Initialize offsets if absent
        if self.seq_start[layer_idx] is None:
            self.seq_start[layer_idx] = self._infer_seq_start(attention_mask, k_all)
        if self.raw_seq_start[layer_idx] is None:
            self.raw_seq_start[layer_idx] = self.seq_start[layer_idx].clone()

        # Invariant checks
        # Ensure seq_start (left padding) is not greater than raw_seq_start (raw frontier)
        # seq_start: [B], raw_seq_start: [B]
        if self.seq_start[layer_idx] is not None and self.raw_seq_start[layer_idx] is not None:
            assert torch.all(self.seq_start[layer_idx] <= self.raw_seq_start[layer_idx]), \
                f"Invariant Violation: seq_start ({self.seq_start[layer_idx]}) > raw_seq_start ({self.raw_seq_start[layer_idx]})"
            
        return k_all, v_all, self.seq_start[layer_idx], self.raw_seq_start[layer_idx]

    def initialize_with_cache(
        self,
        cache: DynamicCache,
    ) -> None:
        """Bind an existing `DynamicCache` to this wrapper. Does not initialize
        the indices seq_start and raw_seq_start.

        Args:
            cache: DynamicCache
                The HF cache produced by the base model during prefill.
        Raises:
            ValueError: if `self.cache` has already been initialized.
        """
        if self.cache is not None:
            raise ValueError("self.cache already initialized!")
        self.cache = cache

    def get_layer(
        self,
        layer_idx: int,
        with_offsets: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve raw keys/values (and optionally offsets) for a layer.

        Args:
            layer_idx: int
                Transformer layer index.
            with_offsets: bool
                If True, also return `seq_start` and `raw_seq_start`.

        Returns:
        (keys, values, seq_start, raw_seq_start) where:
            - keys/values: [B, H, T_raw, D]
            - seq_start: [B] left padding offset (none if with_offsets = False)
            - raw_seq_start: [B] first raw token index (frontier) (none if with_offsets = False)
        """
        if self.cache is None:
            raise ValueError("RawCache.get_layer requires an underlying DynamicCache. Call initialize_with_cache first.")
        layer = self.cache.layers[layer_idx]
        k = getattr(layer, "keys", None)
        v = getattr(layer, "values", None)
        assert k is not None and v is not None
        if not with_offsets:
            return k, v, None, None
        return k, v, self.seq_start[layer_idx], self.raw_seq_start[layer_idx]

    def slice_layer(
        self,
        layer_idx: int,
        start: torch.Tensor,
        end: torch.Tensor,
        batch_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a per-batch contiguous slice of raw KV for a layer.

        Args:
            layer_idx: int
                Transformer layer index.
            start: torch.Tensor
                Inclusive start positions in raw indices (0-based, after padding). Shape [B].
            end: torch.Tensor
                Exclusive end positions. Shape [B].
            batch_mask: torch.Tensor
                [B] bool mask; False entries produce empty slices for those batches.

        Returns:
            (keys_slice, values_slice):
                keys_slice: [B, H, L_max, D]
                values_slice: [B, H, L_max, D]
                where L_max is the max slice length across batches. Positions beyond a
                batch's slice length are zeroed. This avoids Python loops while still
                supporting per-batch ranges.
        """
        k, v = self.get_layer(layer_idx)
        if k is None or v is None:
            return k, v
        B, H, T, D = k.shape
        device = k.device
        dtype = k.dtype

        start = start.to(device=device, dtype=torch.long)
        end = end.to(device=device, dtype=torch.long)
        batch_mask = batch_mask.to(device=device, dtype=torch.bool)

        if start.shape[0] != B or end.shape[0] != B or batch_mask.shape[0] != B:
            raise ValueError("start, end, and batch_mask must all be shape [B] matching the cache batch size.")

        lengths = (end - start).clamp_min(0)
        lengths = torch.where(batch_mask, lengths, torch.zeros_like(lengths))
        max_len = lengths.max()
        if max_len.item() == 0:
            empty = k[:, :, :0, :]
            return empty, empty

        arange_idx = torch.arange(max_len, device=device).view(1, 1, max_len, 1)  # [1,1,L,1]
        start_exp = start.view(B, 1, 1, 1)  # [B,1,1,1]
        gather_idx = start_exp + arange_idx  # [B,1,L,1]
        gather_idx = torch.clamp(gather_idx, 0, T - 1)
        gather_idx_expanded = gather_idx.expand(B, H, max_len, D)

        k_slice = torch.gather(k, 2, gather_idx_expanded)
        v_slice = torch.gather(v, 2, gather_idx_expanded)

        # Zero out positions beyond each batch's valid slice length.
        valid_mask = (arange_idx < lengths.view(B, 1, 1, 1)).expand(B, H, max_len, D)
        k_slice = k_slice * valid_mask
        v_slice = v_slice * valid_mask
        return k_slice, v_slice

    @staticmethod
    def _infer_seq_start(attention_mask: Optional[torch.Tensor], kv_tensor: torch.Tensor) -> torch.Tensor:
        """Infer left padding (seq_start) from attention mask, or default to zeros.

        Args:
            attention_mask: Optional[torch.Tensor]
                [B, 1, L_q, T_raw] additive mask; padding is expected to be filled
                with a large negative value. If None, defaults to zeros.
            kv_tensor: torch.Tensor
                A KV tensor to derive batch/device when mask is absent.

        Returns:
            seq_start: torch.Tensor of shape [B]
        """
        if attention_mask is None:
            B = kv_tensor.shape[0]
            device = kv_tensor.device
            return torch.zeros(B, dtype=torch.long, device=device)
        # Mirror BufferedSegmenter._get_seq_starts_left
        last_rows = attention_mask[:, 0, -1, :]  # [B, T_total]
        mask_value, _ = last_rows.min(dim=1, keepdim=True)  # most negative per batch
        has_padding = mask_value < 0
        is_pad = (last_rows == mask_value) & has_padding
        pad_prefix = is_pad.cumprod(dim=1).sum(dim=1)
        return pad_prefix.to(dtype=torch.long)

class SummaryCache:

    def __init__(self, config: PretrainedConfig):
        """Right-padded cache for page summaries.

        Summary entries are appended per-batch whenever a new page is produced
        by the compressor/segmenter. Tensors are right padded to make it cheap
        to drop in new pages for only a subset of batch rows.

        Args:
            config: PretrainedConfig; used for dtype/device alignment.
        """
        self.config = config
        self.keys: torch.Tensor = None       # [B, H, max(L_num_pages), D], right-padded
        self.values: torch.Tensor = None     # [B, H, max(L_num_pages), D], right-padded
        self.page_lens: torch.Tensor = None  # [B] count of valid pages per batch
        # Span metadata (right-padded to align with keys/values)
        self.page_start: torch.Tensor = None    # [B, max(L_num_pages)] start idx (inclusive) in raw cache
        self.page_end: torch.Tensor = None      # [B, max(L_num_pages)] end idx (inclusive) in raw cache
        self.page_frontier: torch.Tensor = None # [B] first raw index not covered by any page
        # Adaptive log bias: log(k_eff) where k_eff = exp(entropy) is effective support
        self.log_effective_support: torch.Tensor = None  # [B, max(L_num_pages)] entropy per page

    def initialize(self, B: int, H: int, D: int, device: torch.device, dtype: torch.dtype):
        """Initialize with empty state."""
        if self.keys is not None:
            return
            
        self.keys = torch.zeros(B, H, 0, D, device=device, dtype=dtype)
        self.values = torch.zeros(B, H, 0, D, device=device, dtype=dtype)
        self.page_lens = torch.zeros(B, dtype=torch.long, device=device)
        self.page_start = torch.zeros(B, 0, dtype=torch.long, device=device)
        self.page_end = torch.zeros(B, 0, dtype=torch.long, device=device)
        self.page_frontier = torch.zeros(B, dtype=torch.long, device=device)
        self.log_effective_support = torch.zeros(B, 0, device=device, dtype=torch.float32)

    def add_pages(self,
        keys: torch.Tensor,
        values: torch.Tensor,
        batch_nums: torch.Tensor,
        page_start: torch.Tensor,
        page_end: torch.Tensor,
        page_frontier: Optional[torch.Tensor] = None,
        log_effective_support: Optional[torch.Tensor] = None,
    ):
        """Insert newly-computed summary pages.

        Args:
            keys: torch.Tensor
                [B_new, H, L_new, D] summary keys to insert. B_new can be a
                subset of the global batch (as indicated by `batch_nums`).
            values: torch.Tensor
                [B_new, H, L_new, D] summary values aligned with `keys`.
            batch_nums: torch.Tensor
                [B_new] indices into the global batch dimension specifying which
                rows should receive the new pages.
            page_start: torch.Tensor
                [B_new, L_new] raw start indices (inclusive) for each new page.
            page_end: torch.Tensor
                [B_new, L_new] raw end indices (inclusive) for each new page.
            page_frontier: Optional[torch.Tensor]
                [B_new] updated frontier (first raw index not covered) for the
                affected batch rows.
            log_effective_support: Optional[torch.Tensor]
                [B_new, L_new] log of effective support (entropy) for each page.
                Used for adaptive log(k) bias. If None, falls back to log(N).

        Returns:
            None; should update `self.keys`, `self.values`, and `self.page_lens`
            by writing into the right-padded slots for each batch row. Also
            updates span metadata (`page_start`, `page_end`, `page_frontier`).
        """
        if keys.numel() == 0:
            return

        # Invariants:
        # 1. Keys and Values must match shape [B_new, H, L_new, D]
        # 2. Page start indices must be <= Page end indices
        assert keys.shape == values.shape, f"Invariant Violation: Keys/Values shape mismatch: {keys.shape} vs {values.shape}"
        assert torch.all(page_start <= page_end), "Invariant Violation: page_start > page_end"


        B_new, H, L_new, D = keys.shape
        device = keys.device
        
        # Determine required global batch size
        max_b = int(batch_nums.max().item()) + 1
        
        # Initialize if empty
        if self.keys is None:
            # We don't know the true global B, but we must accommodate at least max_b.
            # We'll start with max_b and expand if needed later.
            self.keys = torch.zeros(max_b, H, L_new, D, device=device, dtype=keys.dtype)
            self.values = torch.zeros(max_b, H, L_new, D, device=device, dtype=values.dtype)
            self.page_lens = torch.zeros(max_b, dtype=torch.long, device=device)
            self.page_start = torch.zeros(max_b, L_new, dtype=torch.long, device=device)
            self.page_end = torch.zeros(max_b, L_new, dtype=torch.long, device=device)
            self.page_frontier = torch.zeros(max_b, dtype=torch.long, device=device)
            self.log_effective_support = torch.zeros(max_b, L_new, device=device, dtype=torch.float32)
        
        # Expand batch dim if needed
        current_B = self.keys.shape[0]
        if max_b > current_B:
            # Pad batch dim
            pad_b = max_b - current_B
            self.keys = torch.cat([self.keys, torch.zeros(pad_b, H, self.keys.shape[2], D, device=device, dtype=keys.dtype)], dim=0)
            self.values = torch.cat([self.values, torch.zeros(pad_b, H, self.values.shape[2], D, device=device, dtype=values.dtype)], dim=0)
            self.page_lens = torch.cat([self.page_lens, torch.zeros(pad_b, dtype=torch.long, device=device)], dim=0)
            self.page_start = torch.cat([self.page_start, torch.zeros(pad_b, self.page_start.shape[1], dtype=torch.long, device=device)], dim=0)
            self.page_end = torch.cat([self.page_end, torch.zeros(pad_b, self.page_end.shape[1], dtype=torch.long, device=device)], dim=0)
            self.page_frontier = torch.cat([self.page_frontier, torch.zeros(pad_b, dtype=torch.long, device=device)], dim=0)
            self.log_effective_support = torch.cat([self.log_effective_support, torch.zeros(pad_b, self.log_effective_support.shape[1], device=device, dtype=torch.float32)], dim=0)

        # Expand time dim if needed
        # We need to fit `current_len + L_new` pages.
        # Since we are adding L_new pages to *some* batches, we need to check the max resulting length.
        # But here `keys` has `L_new` pages for *each* batch in `batch_nums`.
        # So we simply need to ensure `self.keys` has enough capacity for the batch with the most pages.
        # Wait, `L_new` is the number of *new* pages being added.
        # We need to append these to the existing pages.
        
        current_lens = self.page_lens[batch_nums] # [B_new]
        new_total_lens = current_lens + L_new
        max_needed = int(new_total_lens.max().item())
        
        if max_needed > self.keys.shape[2]:
            pad_t = max_needed - self.keys.shape[2]
            # Pad time dim (dim 2)
            self.keys = torch.cat([self.keys, torch.zeros(self.keys.shape[0], H, pad_t, D, device=device, dtype=keys.dtype)], dim=2)
            self.values = torch.cat([self.values, torch.zeros(self.values.shape[0], H, pad_t, D, device=device, dtype=values.dtype)], dim=2)
            self.page_start = torch.cat([self.page_start, torch.zeros(self.page_start.shape[0], pad_t, dtype=torch.long, device=device)], dim=1)
            self.page_end = torch.cat([self.page_end, torch.zeros(self.page_end.shape[0], pad_t, dtype=torch.long, device=device)], dim=1)
            self.log_effective_support = torch.cat([self.log_effective_support, torch.zeros(self.log_effective_support.shape[0], pad_t, device=device, dtype=torch.float32)], dim=1)

        # Insert new pages
        # We can't do a simple slice assignment because `current_lens` varies per batch.
        # We have to scatter or loop.
        # Since L_new is usually small (often 1), loop might be okay?
        # Or we can use `scatter`.
        # But `keys` is [B_new, H, L_new, D].
        # We want to place `keys[i]` at `self.keys[batch_nums[i], :, current_lens[i]:current_lens[i]+L_new, :]`.
        
        # Let's loop for safety and readability for now.
        # If log_effective_support not provided, compute fallback: log(page_length)
        if log_effective_support is None:
            # Fallback to log(N) for compressors without importance weights
            page_lengths = (page_end - page_start + 1).float()  # [B_new, L_new]
            log_effective_support = page_lengths.clamp(min=1).log()  # [B_new, L_new]

        for i, b_idx in enumerate(batch_nums.tolist()):
            start_col = int(current_lens[i].item())
            end_col = start_col + L_new

            self.keys[b_idx, :, start_col:end_col, :] = keys[i]
            self.values[b_idx, :, start_col:end_col, :] = values[i]
            self.page_start[b_idx, start_col:end_col] = page_start[i]
            self.page_end[b_idx, start_col:end_col] = page_end[i]
            self.log_effective_support[b_idx, start_col:end_col] = log_effective_support[i]

        # Update lengths and frontier
        self.page_lens.index_add_(0, batch_nums, torch.full((B_new,), L_new, dtype=torch.long, device=device))

        if page_frontier is not None:
            self.page_frontier[batch_nums] = page_frontier

class GridCache:
    """
    Cache for 'lined attention' tokens: raw tokens that are globally important
    and should stay available even when surrounding context is compressed.
    
    Each layer has its own GridCache. For batch b, we store:
      - keys[b, :, :lens[b], :]   : [H, G_b, D] grid token keys
      - values[b, :, :lens[b], :] : [H, G_b, D] grid token values
      - indices[b, :lens[b]]      : [G_b] raw indices (0..T_raw-1)
    
    This cache doesn't decide which tokens are grid yet; it just stores them.
    """
    def __init__(self, config: PretrainedConfig):
        self.config = config
        self.keys: torch.Tensor | None = None       # [B, H, G_max, D]
        self.values: torch.Tensor | None = None     # [B, H, G_max, D]
        self.indices: torch.Tensor | None = None    # [B, G_max]
        self.lens: torch.Tensor | None = None       # [B]

    def initialize(self, B: int, H: int, D: int, device: torch.device, dtype: torch.dtype):
        """Initialize with empty state (0 grid tokens)."""
        if self.keys is not None:
            return
        self.keys = torch.zeros(B, H, 0, D, device=device, dtype=dtype)
        self.values = torch.zeros(B, H, 0, D, device=device, dtype=dtype)
        self.indices = torch.zeros(B, 0, dtype=torch.long, device=device)
        self.lens = torch.zeros(B, dtype=torch.long, device=device)

    def set_tokens(
        self,
        keys: torch.Tensor,     # [B, H, G, D]
        values: torch.Tensor,   # [B, H, G, D]
        indices: torch.Tensor,  # [B, G] (raw indices, -1 for padding)
    ):
        """
        Overwrite the current grid tokens with the provided ones.
        
        All batches share the same G_max (time dim), but each batch b can use
        fewer tokens as indicated by indices[b] == -1 and lens[b].
        """
        assert keys.shape == values.shape, \
            f"Invariant Violation: grid keys/values shape mismatch: {keys.shape} vs {values.shape}"
        B, H, G, D = keys.shape
        device, dtype = keys.device, keys.dtype

        if self.keys is None:
            self.initialize(B, H, D, device, dtype)

        # If we need more grid capacity than before, pad time dim
        if G > self.keys.shape[2]:
            pad_g = G - self.keys.shape[2]
            self.keys = torch.cat(
                [self.keys,
                 torch.zeros(self.keys.shape[0], H, pad_g, D, device=device, dtype=dtype)],
                dim=2,
            )
            self.values = torch.cat(
                [self.values,
                 torch.zeros(self.values.shape[0], H, pad_g, D, device=device, dtype=dtype)],
                dim=2,
            )
            self.indices = torch.cat(
                [self.indices,
                 torch.zeros(self.indices.shape[0], pad_g, dtype=torch.long, device=device)],
                dim=1,
            )

        # Resize batch dim if needed (paranoid, usually B matches)
        if B > self.keys.shape[0]:
            pad_b = B - self.keys.shape[0]
            self.keys = torch.cat(
                [self.keys,
                 torch.zeros(pad_b, H, self.keys.shape[2], D, device=device, dtype=dtype)],
                dim=0,
            )
            self.values = torch.cat(
                [self.values,
                 torch.zeros(pad_b, H, self.values.shape[2], D, device=device, dtype=dtype)],
                dim=0,
            )
            self.indices = torch.cat(
                [self.indices,
                 torch.zeros(pad_b, self.indices.shape[1], dtype=torch.long, device=device)],
                dim=0,
            )
            self.lens = torch.cat(
                [self.lens,
                 torch.zeros(pad_b, dtype=torch.long, device=device)],
                dim=0,
            )

        # Overwrite active portion
        self.keys[:, :, :G, :] = keys
        self.values[:, :, :G, :] = values
        self.indices[:, :G] = indices

        # Compute lens per batch as the count of indices >= 0
        with torch.no_grad():
            valid = (indices >= 0)
            self.lens = valid.sum(dim=1).to(dtype=torch.long)

class CoverView:

    # Default capacity growth settings
    INITIAL_EXTRA_CAPACITY = 128  # Extra slots to pre-allocate beyond initial size
    GROWTH_FACTOR = 1.5           # Multiply capacity by this when expanding

    def __init__(self):
        """Hybrid view combining summary pages with trailing raw tokens.

        The cover view is what top-down attention should run against:
        [ summaries (paged region) ] + [ raw tail ]. It is left indexed to make
        incremental decoding updates cheap when no new pages are formed.

        Uses pre-allocated buffers with capacity tracking for efficient updates
        during decoding (avoids repeated torch.cat allocations).
        """
        self.cover_keys = None      # [B, H, _capacity, D] pre-allocated buffer
        self.cover_values = None    # [B, H, _capacity, D] pre-allocated buffer
        self.seq_start = None       # [B] left-padding offset, mirrors raw cache
        self.raw_seq_start = None   # [B] first raw token index within the cover
        # Metadata to map cover positions back to sources
        self.cover_indices = None   # [B, _capacity] raw index or summary idx
        self.cover_is_summary = None # [B, _capacity] 1 if summary, else 0
        # Capacity tracking for pre-allocation optimization
        self._capacity = 0          # Total allocated slots in dim 2
        self._length = 0            # Actually used slots in dim 2
    
    def initialize(self,
        layer_idx: int,
        raw_cache: RawCache
    ):
        """Initialize cover view from the raw cache (e.g. after prefill).

        Args:
            layer_idx: int
                Transformer layer index.
            raw_cache: RawCache
                Source of raw KV tensors.
        """
        k_raw, v_raw, seq_start, raw_seq_start = raw_cache.get_layer(layer_idx, with_offsets=True)
        if k_raw is None:
            raise ValueError("CoverView.initialize called before RawCache.update.")

        B, H, T_raw, D = k_raw.shape
        device = k_raw.device
        dtype = k_raw.dtype

        # Pre-allocate with extra capacity for future tokens
        self._length = T_raw
        self._capacity = T_raw + self.INITIAL_EXTRA_CAPACITY

        # Allocate buffers with extra capacity
        self.cover_keys = torch.zeros(B, H, self._capacity, D, device=device, dtype=dtype)
        self.cover_values = torch.zeros(B, H, self._capacity, D, device=device, dtype=dtype)
        self.cover_indices = torch.full((B, self._capacity), -1, device=device, dtype=torch.long)
        self.cover_is_summary = torch.zeros(B, self._capacity, dtype=torch.long, device=device)

        # Copy initial data
        self.cover_keys[:, :, :T_raw, :] = k_raw
        self.cover_values[:, :, :T_raw, :] = v_raw

        self.seq_start = seq_start
        self.raw_seq_start = raw_seq_start

        # Initialize indices: all raw
        # cover_indices[b, t] = t (absolute raw index)
        indices = torch.arange(T_raw, device=device).unsqueeze(0).expand(B, -1)
        self.cover_indices[:, :T_raw] = indices

        # Mask padding with -1
        if seq_start is not None:
            # seq_start: [B]
            # Create mask: t < seq_start[b]
            t_indices = torch.arange(T_raw, device=device).unsqueeze(0)  # [1, T]
            is_pad = t_indices < seq_start.unsqueeze(1)  # [B, T]
            self.cover_indices[:, :T_raw][is_pad] = -1

    def _ensure_capacity(self, needed: int):
        """Expand buffer capacity if needed.

        Args:
            needed: int
                Minimum required capacity.
        """
        if needed <= self._capacity:
            return

        # Calculate new capacity with growth factor
        new_capacity = max(needed, int(self._capacity * self.GROWTH_FACTOR))

        B, H, _, D = self.cover_keys.shape
        device = self.cover_keys.device
        dtype = self.cover_keys.dtype

        # Allocate new buffers
        new_keys = torch.zeros(B, H, new_capacity, D, device=device, dtype=dtype)
        new_values = torch.zeros(B, H, new_capacity, D, device=device, dtype=dtype)
        new_indices = torch.full((B, new_capacity), -1, device=device, dtype=torch.long)
        new_is_summary = torch.zeros(B, new_capacity, dtype=torch.long, device=device)

        # Copy existing data
        new_keys[:, :, :self._length, :] = self.cover_keys[:, :, :self._length, :]
        new_values[:, :, :self._length, :] = self.cover_values[:, :, :self._length, :]
        new_indices[:, :self._length] = self.cover_indices[:, :self._length]
        new_is_summary[:, :self._length] = self.cover_is_summary[:, :self._length]

        # Replace buffers
        self.cover_keys = new_keys
        self.cover_values = new_values
        self.cover_indices = new_indices
        self.cover_is_summary = new_is_summary
        self._capacity = new_capacity

    def update(self,
        keys: torch.Tensor,
        values: torch.Tensor,
        indices: torch.Tensor
    ):
        """Append raw tokens into the cover view during decoding.

        Uses pre-allocated buffers with in-place assignment for efficiency.
        Only expands capacity when the buffer is full.

        Args:
            keys: torch.Tensor
                [B, H, L_new, D] raw keys for new tokens (typically L_new == 1).
            values: torch.Tensor
                [B, H, L_new, D] raw values for new tokens.
            indices: torch.Tensor
                [B, L_new] raw indices for the new tokens.

        Returns:
            None; mutates cover_keys/cover_values in-place when possible.
        """
        if self.cover_keys is None:
            raise ValueError("CoverView.update called before initialize()")

        B, H, L_new, D = keys.shape

        # Ensure we have enough capacity
        needed = self._length + L_new
        if needed > self._capacity:
            self._ensure_capacity(needed)

        # In-place assignment (fast path - no allocation)
        start = self._length
        end = start + L_new
        self.cover_keys[:, :, start:end, :] = keys
        self.cover_values[:, :, start:end, :] = values
        self.cover_indices[:, start:end] = indices
        # cover_is_summary stays 0 (already initialized to zeros)

        self._length = end


    def update_cover_view(self,
        layer_idx: int,
        raw_cache: RawCache,
        summary_cache: SummaryCache,
        grid_cache: Optional["GridCache"] = None,
    ):
        """Rebuild cover view after new pages are materialized. This happens after
        the raw_cache nad summary_cache have initialized their new pages.

        Args:
            layer_idx: int
                Transformer layer index.
            raw_cache: RawCache
                Provides the latest raw KV tensors and offsets.
            summary_cache: SummaryCache
                Provides the latest summary KV tensors and page lengths.

        Returns:
            None; should stitch `summary_cache` and `raw_cache` into
            `self.cover_keys/cover_values`, updating the padding and frontier
            metadata (`seq_start`, `raw_seq_start`) to remain consistent. Should
            also refresh `cover_indices` and `cover_is_summary` so downstream
            attention can translate cover positions into raw spans.
        """
        k_raw, v_raw, seq_start, raw_seq_start = raw_cache.get_layer(layer_idx, with_offsets=True)
        if k_raw is None:
            return # Nothing to do

        # Summary
        sum_k = summary_cache.keys # [B, H, P_max, D]
        sum_v = summary_cache.values
        page_lens = summary_cache.page_lens # [B]
        
        # If summary is empty, just copy raw?
        # Or handle uniformly.
        
        B, H, T_raw, D = k_raw.shape
        device = k_raw.device
        
        batched_k = []
        batched_v = []
        batched_idx = []
        batched_is_sum = []
        
        for b in range(B):
            # -------------------------
            # 0) Grid (lined) tokens
            # -------------------------
            if grid_cache is not None and grid_cache.keys is not None and grid_cache.lens is not None:
                if b < grid_cache.keys.shape[0]:
                    g_len = int(grid_cache.lens[b].item())
                    if g_len > 0:
                        k_g = grid_cache.keys[b, :, :g_len, :]   # [H, G, D]
                        v_g = grid_cache.values[b, :, :g_len, :]
                        idx_g = grid_cache.indices[b, :g_len]    # [G] raw indices
                        is_sum_g = torch.zeros(g_len, dtype=torch.long, device=device)
                    else:
                        k_g = torch.empty(H, 0, D, device=device, dtype=k_raw.dtype)
                        v_g = torch.empty(H, 0, D, device=device, dtype=v_raw.dtype)
                        idx_g = torch.empty(0, dtype=torch.long, device=device)
                        is_sum_g = torch.empty(0, dtype=torch.long, device=device)
                else:
                    k_g = torch.empty(H, 0, D, device=device, dtype=k_raw.dtype)
                    v_g = torch.empty(H, 0, D, device=device, dtype=v_raw.dtype)
                    idx_g = torch.empty(0, dtype=torch.long, device=device)
                    is_sum_g = torch.empty(0, dtype=torch.long, device=device)
            else:
                # No grid tokens for this batch/layer
                k_g = torch.empty(H, 0, D, device=device, dtype=k_raw.dtype)
                v_g = torch.empty(H, 0, D, device=device, dtype=v_raw.dtype)
                idx_g = torch.empty(0, dtype=torch.long, device=device)
                is_sum_g = torch.empty(0, dtype=torch.long, device=device)

            # -------------------------
            # 1) Summary Part (existing)
            # -------------------------
            if sum_k is not None and page_lens is not None:
                if b < page_lens.shape[0]:
                    p_len = int(page_lens[b].item())
                    k_s = sum_k[b, :, :p_len, :]  # [H, P, D]
                    v_s = sum_v[b, :, :p_len, :]
                else:
                    k_s = torch.empty(H, 0, D, device=device, dtype=k_raw.dtype)
                    v_s = torch.empty(H, 0, D, device=device, dtype=v_raw.dtype)
                    p_len = 0
                
                # Indices for summary: we use the PAGE INDEX (0 to P-1)
                # But we need to distinguish summary indices from raw indices.
                # The `cover_indices` contract says: "raw index or summary idx".
                # Top-down attention uses `cover_indices` to look up `page_start`/`page_end`.
                # So for summary tokens, it should be the page index `p`.
                idx_s = torch.arange(p_len, device=device)
                is_sum_s = torch.ones(p_len, dtype=torch.long, device=device)
            else:
                k_s = torch.empty(H, 0, D, device=device, dtype=k_raw.dtype)
                v_s = torch.empty(H, 0, D, device=device, dtype=v_raw.dtype)
                idx_s = torch.empty(0, dtype=torch.long, device=device)
                is_sum_s = torch.empty(0, dtype=torch.long, device=device)
                p_len = 0
            
            # -------------------------
            # 2) Raw Tail Part (existing)
            # -------------------------
            r_start = raw_seq_start[b].item() if raw_seq_start is not None else 0
            r_start = min(r_start, T_raw)
            k_r = k_raw[b, :, r_start:, :]  # [H, T_tail, D]
            v_r = v_raw[b, :, r_start:, :]
            tail_len = k_r.shape[1]
            idx_r = torch.arange(r_start, r_start + tail_len, device=device)
            
            if seq_start is not None:
                pad_end = seq_start[b].item()
                is_pad = idx_r < pad_end
                idx_r[is_pad] = -1
                
            is_sum_r = torch.zeros(tail_len, dtype=torch.long, device=device)
            
            # -------------------------
            # 3) Concatenate: [Grid] + [Summaries] + [Raw tail]
            # -------------------------
            batched_k.append(torch.cat([k_g, k_s, k_r], dim=1))
            batched_v.append(torch.cat([v_g, v_s, v_r], dim=1))
            batched_idx.append(torch.cat([idx_g, idx_s, idx_r], dim=0))
            batched_is_sum.append(torch.cat([is_sum_g, is_sum_s, is_sum_r], dim=0))
            
        # Pad and Stack with extra capacity for future growth
        lengths = [x.shape[1] for x in batched_k]
        max_len = max(lengths) if lengths else 0
        self._length = max_len
        self._capacity = max_len + self.INITIAL_EXTRA_CAPACITY

        self.cover_keys = torch.zeros(B, H, self._capacity, D, device=device, dtype=k_raw.dtype)
        self.cover_values = torch.zeros(B, H, self._capacity, D, device=device, dtype=v_raw.dtype)
        self.cover_indices = torch.full((B, self._capacity), -1, dtype=torch.long, device=device)
        self.cover_is_summary = torch.zeros(B, self._capacity, dtype=torch.long, device=device)

        for b in range(B):
            l = lengths[b]
            if l > 0:
                self.cover_keys[b, :, :l, :] = batched_k[b]
                self.cover_values[b, :, :l, :] = batched_v[b]
                self.cover_indices[b, :l] = batched_idx[b]
                self.cover_is_summary[b, :l] = batched_is_sum[b]

        # Update metadata
        self.seq_start = seq_start
        self.raw_seq_start = raw_seq_start

    @property
    def length(self) -> int:
        """Return the valid length (excluding pre-allocated capacity)."""
        return self._length

    def get_valid_kv(self):
        """Return only the valid (non-pre-allocated) portions of the cover view.

        Returns:
            tuple: (cover_keys, cover_values, cover_indices, cover_is_summary)
                All sliced to [B, H, _length, D] or [B, _length] as appropriate.
        """
        if self.cover_keys is None:
            return None, None, None, None
        return (
            self.cover_keys[:, :, :self._length, :],
            self.cover_values[:, :, :self._length, :],
            self.cover_indices[:, :self._length],
            self.cover_is_summary[:, :self._length],
        )

class AttentionScoreBuffer:

    # Pre-allocation settings
    INITIAL_L_CAPACITY = 256   # Pre-allocate for L (query accumulation) dimension
    INITIAL_T_EXTRA = 128      # Extra capacity for T (cover width) dimension
    GROWTH_FACTOR = 1.5

    def __init__(self):
        """Initialize empty buffer with capacity tracking."""
        self.attention_weights = None  # [B, H, _l_capacity, _t_capacity]
        self.cover_indices = None
        self.cover_is_summary = None
        self.refinement_counts = None
        # Use tensors for stats to avoid GPU sync during forward pass
        # Call .item() only when stats are requested (lazy evaluation)
        self._total_summaries_seen = None  # Tensor on device, or Python int 0
        self._total_refinements_made = None  # Tensor on device, or Python int 0
        # Capacity tracking
        self._l_length = 0      # Used length in L dimension (queries)
        self._l_capacity = 0    # Allocated capacity in L dimension
        self._t_length = 0      # Used length in T dimension (cover width)
        self._t_capacity = 0    # Allocated capacity in T dimension

    def initialize(self, B: int, H: int, T_init: int, device: torch.device, dtype: torch.dtype):
        """Initialize the buffer with pre-allocated capacity.

        Args:
            B: Batch size.
            H: Number of heads.
            T_init: Initial cover length (e.g. prefill length).
            device: Tensor device.
            dtype: Tensor dtype.
        """
        self._l_length = 0
        self._l_capacity = self.INITIAL_L_CAPACITY
        self._t_length = T_init
        self._t_capacity = T_init + self.INITIAL_T_EXTRA

        # Pre-allocate with capacity
        self.attention_weights = torch.zeros(
            B, H, self._l_capacity, self._t_capacity,
            device=device, dtype=dtype
        )
        self.cover_indices = None
        self.cover_is_summary = None
        self.refinement_counts = torch.zeros(B, self._t_capacity, dtype=torch.long, device=device)
        # Initialize stats as tensors on device to avoid GPU sync during accumulation
        self._total_summaries_seen = torch.tensor(0, dtype=torch.long, device=device)
        self._total_refinements_made = torch.tensor(0, dtype=torch.long, device=device)

    def _ensure_l_capacity(self, needed_l: int):
        """Expand L (query) dimension capacity if needed."""
        if needed_l <= self._l_capacity:
            return

        new_l_cap = max(needed_l, int(self._l_capacity * self.GROWTH_FACTOR))
        B, H, _, T_cap = self.attention_weights.shape

        new_weights = torch.zeros(B, H, new_l_cap, T_cap,
                                   device=self.attention_weights.device,
                                   dtype=self.attention_weights.dtype)
        new_weights[:, :, :self._l_length, :self._t_length] = \
            self.attention_weights[:, :, :self._l_length, :self._t_length]

        self.attention_weights = new_weights
        self._l_capacity = new_l_cap

    def _ensure_t_capacity(self, needed_t: int):
        """Expand T (cover width) dimension capacity if needed."""
        if needed_t <= self._t_capacity:
            return

        new_t_cap = max(needed_t, int(self._t_capacity * self.GROWTH_FACTOR))
        B, H, L_cap, _ = self.attention_weights.shape

        # Expand attention weights
        new_weights = torch.zeros(B, H, L_cap, new_t_cap,
                                   device=self.attention_weights.device,
                                   dtype=self.attention_weights.dtype)
        new_weights[:, :, :self._l_length, :self._t_length] = \
            self.attention_weights[:, :, :self._l_length, :self._t_length]
        self.attention_weights = new_weights

        # Expand refinement counts
        new_counts = torch.zeros(B, new_t_cap,
                                  device=self.refinement_counts.device,
                                  dtype=self.refinement_counts.dtype)
        new_counts[:, :self._t_length] = self.refinement_counts[:, :self._t_length]
        self.refinement_counts = new_counts

        self._t_capacity = new_t_cap

    def push(self, attn_weights: torch.Tensor, cover_indices: torch.Tensor, cover_is_summary: torch.Tensor):
        """Append a new attention score slice using pre-allocated buffers.

        Args:
            attn_weights: [B, H, L_new, T_new]
            cover_indices: [B, T_new]
            cover_is_summary: [B, T_new]
        """
        # Detach to save memory graph
        attn_weights = attn_weights.detach()
        cover_indices = cover_indices.detach()
        cover_is_summary = cover_is_summary.detach()

        if self.attention_weights is None:
            raise RuntimeError("AttentionScoreBuffer not initialized. Call initialize() first.")

        _, _, L_new, T_new = attn_weights.shape

        # Ensure capacity in both dimensions
        needed_l = self._l_length + L_new
        if needed_l > self._l_capacity:
            self._ensure_l_capacity(needed_l)

        if T_new > self._t_capacity:
            self._ensure_t_capacity(T_new)

        # In-place assignment for L dimension (fast path)
        l_start = self._l_length
        l_end = l_start + L_new

        # Handle T dimension: new data may be wider or narrower than current
        t_write = min(T_new, self._t_capacity)
        self.attention_weights[:, :, l_start:l_end, :t_write] = attn_weights[:, :, :, :t_write]

        self._l_length = l_end
        self._t_length = max(self._t_length, T_new)

        # Update indices (keep latest)
        self.cover_indices = cover_indices
        self.cover_is_summary = cover_is_summary

    def get_data(self):
        """Return valid (non-pre-allocated) portion of the buffer."""
        if self.attention_weights is None:
            return None, None, None
        valid_weights = self.attention_weights[:, :, :self._l_length, :self._t_length]
        return valid_weights, self.cover_indices, self.cover_is_summary

    def reset(self):
        """Reset to empty state, keeping pre-allocated buffers."""
        self._l_length = 0
        # Keep _t_length as is since cover view size doesn't change on reset
        self.cover_indices = None
        self.cover_is_summary = None
        if self.refinement_counts is not None:
            self.refinement_counts.zero_()
        # Reset stats tensors in-place if they exist, otherwise set to 0
        if self._total_summaries_seen is not None and torch.is_tensor(self._total_summaries_seen):
            self._total_summaries_seen.zero_()
        else:
            self._total_summaries_seen = None
        if self._total_refinements_made is not None and torch.is_tensor(self._total_refinements_made):
            self._total_refinements_made.zero_()
        else:
            self._total_refinements_made = None

    @property
    def total_summaries_seen(self) -> int:
        """Get total summaries seen (triggers GPU sync if tensor)."""
        if self._total_summaries_seen is None:
            return 0
        if torch.is_tensor(self._total_summaries_seen):
            return self._total_summaries_seen.item()
        return self._total_summaries_seen

    @total_summaries_seen.setter
    def total_summaries_seen(self, value):
        """Set total summaries seen (for backwards compatibility)."""
        if torch.is_tensor(self._total_summaries_seen):
            self._total_summaries_seen.fill_(value)
        else:
            self._total_summaries_seen = value

    @property
    def total_refinements_made(self) -> int:
        """Get total refinements made (triggers GPU sync if tensor)."""
        if self._total_refinements_made is None:
            return 0
        if torch.is_tensor(self._total_refinements_made):
            return self._total_refinements_made.item()
        return self._total_refinements_made

    @total_refinements_made.setter
    def total_refinements_made(self, value):
        """Set total refinements made (for backwards compatibility)."""
        if torch.is_tensor(self._total_refinements_made):
            self._total_refinements_made.fill_(value)
        else:
            self._total_refinements_made = value

    def add_summaries_seen(self, count: torch.Tensor):
        """Add to summaries seen counter without GPU sync.

        Args:
            count: Tensor containing the count to add (stays on GPU).
        """
        if self._total_summaries_seen is None:
            self._total_summaries_seen = count.clone().to(dtype=torch.long)
        elif torch.is_tensor(self._total_summaries_seen):
            self._total_summaries_seen += count.to(dtype=torch.long)
        else:
            # Fallback: was set to a Python int, convert to tensor
            device = count.device
            self._total_summaries_seen = torch.tensor(
                self._total_summaries_seen, dtype=torch.long, device=device
            ) + count.to(dtype=torch.long)

    def add_refinements_made(self, count: torch.Tensor):
        """Add to refinements made counter without GPU sync.

        Args:
            count: Tensor containing the count to add (stays on GPU).
        """
        if self._total_refinements_made is None:
            self._total_refinements_made = count.clone().to(dtype=torch.long)
        elif torch.is_tensor(self._total_refinements_made):
            self._total_refinements_made += count.to(dtype=torch.long)
        else:
            # Fallback: was set to a Python int, convert to tensor
            device = count.device
            self._total_refinements_made = torch.tensor(
                self._total_refinements_made, dtype=torch.long, device=device
            ) + count.to(dtype=torch.long)

    def compress_and_trim(
        self,
        new_pages: List[List[Tuple[int, int]]],
        new_frontiers: torch.Tensor
    ):
        """Compress attention weights for new pages and trim to new frontier.

        Uses pre-allocated buffers with capacity tracking.

        Args:
            new_pages: List of length B. new_pages[b] is a list of (start, end)
                inclusive raw indices for newly created pages.
            new_frontiers: [B] raw index of the start of the remaining raw tail.
        """
        # Early return if buffer not initialized (e.g., raw attention mode with threshold < 0)
        if self.cover_indices is None or self.attention_weights is None:
            return

        B, H, _, _ = self.attention_weights.shape
        L = self._l_length  # Use valid length, not capacity
        T = self._t_length
        device = self.attention_weights.device
        dtype = self.attention_weights.dtype

        batched_weights = []
        batched_indices = []
        batched_is_sum = []
        batched_ref_counts = []

        for b in range(B):
            # Existing state (sliced to valid lengths)
            w_b = self.attention_weights[b, :, :L, :T]  # [H, L, T]
            idx_b = self.cover_indices[b]   # [T]
            is_sum_b = self.cover_is_summary[b]  # [T]
            ref_b = self.refinement_counts[b, :T]  # [T]

            # 1. Keep existing summaries
            mask_sum = (is_sum_b == 1) & (idx_b != -1)

            cols_w = [w_b[:, :, mask_sum]]
            cols_idx = [idx_b[mask_sum]]
            cols_is_sum = [is_sum_b[mask_sum]]
            cols_ref = [ref_b[mask_sum]]

            # Determine next summary index
            if mask_sum.any():
                next_sum_idx = idx_b[mask_sum].max().item() + 1
            else:
                next_sum_idx = 0

            # 2. Add new pages (compressed)
            for start, end in new_pages[b]:
                mask_page = (is_sum_b == 0) & (idx_b >= start) & (idx_b <= end)

                if mask_page.any():
                    w_sum = w_b[:, :, mask_page].sum(dim=-1, keepdim=True)
                    cols_w.append(w_sum)
                    cols_idx.append(torch.tensor([next_sum_idx], device=device, dtype=torch.long))
                    next_sum_idx += 1
                    cols_is_sum.append(torch.tensor([1], device=device, dtype=torch.long))
                    cols_ref.append(torch.tensor([0], device=device, dtype=torch.long))

            # 3. Keep remaining raw tail
            frontier = new_frontiers[b].item()
            mask_tail = (is_sum_b == 0) & (idx_b >= frontier) & (idx_b != -1)

            if mask_tail.any():
                cols_w.append(w_b[:, :, mask_tail])
                cols_idx.append(idx_b[mask_tail])
                cols_is_sum.append(is_sum_b[mask_tail])
                cols_ref.append(ref_b[mask_tail])

            # Concatenate for this batch
            if cols_w:
                batched_weights.append(torch.cat(cols_w, dim=-1))
                batched_indices.append(torch.cat(cols_idx, dim=0))
                batched_is_sum.append(torch.cat(cols_is_sum, dim=0))
                batched_ref_counts.append(torch.cat(cols_ref, dim=0))
            else:
                batched_weights.append(torch.zeros(H, L, 0, device=device, dtype=dtype))
                batched_indices.append(torch.zeros(0, device=device, dtype=torch.long))
                batched_is_sum.append(torch.zeros(0, device=device, dtype=torch.long))
                batched_ref_counts.append(torch.zeros(0, device=device, dtype=torch.long))

        # Pad and Stack with extra capacity
        lengths = [x.shape[-1] for x in batched_weights]
        new_t_length = max(lengths) if lengths else 0
        new_t_capacity = new_t_length + self.INITIAL_T_EXTRA

        new_weights = torch.zeros(B, H, self._l_capacity, new_t_capacity, device=device, dtype=dtype)
        new_indices = torch.full((B, new_t_capacity), -1, device=device, dtype=torch.long)
        new_is_sum = torch.zeros(B, new_t_capacity, device=device, dtype=torch.long)
        new_ref_counts = torch.zeros(B, new_t_capacity, device=device, dtype=torch.long)

        for b in range(B):
            l = lengths[b]
            if l > 0:
                new_weights[b, :, :L, :l] = batched_weights[b]
                new_indices[b, :l] = batched_indices[b]
                new_is_sum[b, :l] = batched_is_sum[b]
                new_ref_counts[b, :l] = batched_ref_counts[b]

        self.attention_weights = new_weights
        self.cover_indices = new_indices
        self.cover_is_summary = new_is_sum
        self.refinement_counts = new_ref_counts
        self._t_length = new_t_length
        self._t_capacity = new_t_capacity

    def get_stats(self):
        """Get refinement statistics.

        Returns:
            dict with:
                - num_summaries_current: Current number of summary positions in cover view
                - total_summaries_seen: Cumulative (summaries × heads × queries) seen
                - total_refinements_made: Cumulative refinement count
                - refinement_rate: Fraction of summary attention that triggered refinement
                - total_queries_processed: Number of query tokens processed
                - refinements_per_query: Average refinements per query token
        """
        if self.cover_is_summary is None:
            return {}

        is_sum = (self.cover_is_summary == 1)
        num_summaries = is_sum.sum().item()

        rate = self.total_refinements_made / self.total_summaries_seen if self.total_summaries_seen > 0 else 0.0

        # Estimate queries processed from attention buffer shape
        if self.attention_weights is not None:
            total_queries = self.attention_weights.shape[2]  # L_accum
        else:
            total_queries = 0

        refinements_per_query = self.total_refinements_made / total_queries if total_queries > 0 else 0.0

        return {
            "num_summaries_current": num_summaries,
            "total_summaries_seen": self.total_summaries_seen,
            "total_refinements_made": self.total_refinements_made,
            "refinement_rate": rate,
            "total_queries_processed": total_queries,
            "refinements_per_query": refinements_per_query,
        }


class AsyncPageCreator:
    """Handles asynchronous page creation using CUDA streams.

    Page compression runs on a background stream while decoding continues
    on the main stream. Completed updates are applied between decode steps.
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.stream = None  # Lazily created on first use
        # Per-layer pending updates: (compressed_pages, metadata, event)
        self.pending: dict[int, dict] = {}

    def _ensure_stream(self, device: torch.device):
        """Create CUDA stream if not already created."""
        if self.stream is None and device.type == "cuda":
            self.stream = torch.cuda.Stream(device=device)

    def start_async(
        self,
        layer_idx: int,
        controller: "LukaKVController",
        attn_weights: torch.Tensor,
        cover_indices: torch.Tensor,
        cover_is_summary: torch.Tensor,
    ) -> bool:
        """Start async page creation on background stream.

        Args:
            layer_idx: Transformer layer index
            controller: LukaKVController to get raw cache, segmenter, compressor
            attn_weights: Attention weights from buffer
            cover_indices: Cover view indices
            cover_is_summary: Cover view summary flags

        Returns:
            True if async work was started, False if nothing to do
        """
        device = attn_weights.device
        self._ensure_stream(device)

        # If no CUDA, fall back to sync
        if self.stream is None:
            return False

        # Don't start new work if previous work for this layer is pending
        if layer_idx in self.pending:
            return False

        # Run segmentation on main stream first (it's fast and needed for decisions)
        page_ends = controller.segmenter.process(
            attn_weights, cover_indices, cover_is_summary, layer_idx=layer_idx
        )
        if page_ends is None:
            return False

        # Get raw cache info
        raw_cache = controller.raw_cache
        _, _, _, raw_seq_start = raw_cache.get_layer(layer_idx, with_offsets=True)
        if raw_seq_start is None:
            return False

        k_raw, v_raw, _, _ = raw_cache.get_layer(layer_idx, with_offsets=False)
        if k_raw is None:
            return False

        B = raw_seq_start.shape[0]
        H_kv = k_raw.shape[1]
        D = k_raw.shape[3]

        # Collect page metadata (this is fast, done on main stream)
        all_k_slices = []
        all_v_slices = []
        all_importance = []
        page_metadata = []
        new_frontiers = raw_seq_start.clone()
        max_page_len = 0
        all_new_pages = [[] for _ in range(B)]
        has_updates = False

        for b in range(B):
            frontier = raw_seq_start[b].item()
            p_ends = page_ends[b]
            valid_mask = (p_ends >= frontier) & (p_ends != -1)

            if not valid_mask.any():
                continue

            valid_ends = p_ends[valid_mask].sort().values
            current_start = frontier
            cover_idx_b = cover_indices[b] if cover_indices is not None else None
            is_sum_b = cover_is_summary[b] if cover_is_summary is not None else None

            for end_idx in valid_ends.tolist():
                end_idx = int(end_idx)
                if end_idx < current_start:
                    continue

                page_len = end_idx - current_start + 1
                max_page_len = max(max_page_len, page_len)

                # Clone slices to avoid race conditions with background stream
                k_slice = k_raw[b, :, current_start:end_idx + 1, :].clone()
                v_slice = v_raw[b, :, current_start:end_idx + 1, :].clone()
                all_k_slices.append(k_slice)
                all_v_slices.append(v_slice)

                importance = None
                if attn_weights is not None and cover_idx_b is not None:
                    page_mask = (cover_idx_b >= current_start) & (cover_idx_b <= end_idx) & (is_sum_b == 0)
                    if page_mask.any():
                        page_attn = attn_weights[b, :, :, page_mask]
                        importance = page_attn.sum(dim=1).clone()  # Clone importance too
                        H_q = importance.shape[0]
                        if H_q != H_kv:
                            num_groups = H_q // H_kv
                            importance = importance.view(H_kv, num_groups, -1).mean(dim=1)
                all_importance.append(importance)

                page_metadata.append((b, current_start, end_idx))
                all_new_pages[b].append((current_start, end_idx))
                current_start = end_idx + 1

            if current_start > frontier:
                new_frontiers[b] = current_start
                has_updates = True

        if not page_metadata:
            return False

        # Record event on main stream so background stream can wait for data to be ready
        main_stream_event = torch.cuda.Event()
        main_stream_event.record()

        # Now do the heavy lifting on background stream
        N_pages = len(all_k_slices)

        with torch.cuda.stream(self.stream):
            # Wait for main stream data to be ready (non-blocking on main stream)
            main_stream_event.wait()
            # Pad and stack pages
            padded_k = torch.zeros(N_pages, H_kv, max_page_len, D, device=device, dtype=k_raw.dtype)
            padded_v = torch.zeros(N_pages, H_kv, max_page_len, D, device=device, dtype=v_raw.dtype)
            padded_importance = None

            has_any_importance = any(imp is not None for imp in all_importance)
            if has_any_importance:
                padded_importance = torch.zeros(N_pages, H_kv, max_page_len, device=device, dtype=k_raw.dtype)

            for i, (k_s, v_s, imp) in enumerate(zip(all_k_slices, all_v_slices, all_importance)):
                plen = k_s.shape[1]
                padded_k[i, :, :plen, :] = k_s
                padded_v[i, :, :plen, :] = v_s
                if imp is not None and padded_importance is not None:
                    padded_importance[i, :, :plen] = imp

            # Compress all pages (the expensive operation)
            k_compressed, v_compressed = controller.compressor(padded_k, padded_v, padded_importance)

            # Record event when compression is done
            event = torch.cuda.Event()
            event.record()

        # Store pending update
        self.pending[layer_idx] = {
            "k_compressed": k_compressed,
            "v_compressed": v_compressed,
            "page_metadata": page_metadata,
            "new_frontiers": new_frontiers,
            "all_new_pages": all_new_pages,
            "has_updates": has_updates,
            "event": event,
        }

        return True

    def try_apply(self, layer_idx: int, controller: "LukaKVController") -> bool:
        """Check if async work is done and apply updates.

        Args:
            layer_idx: Transformer layer index
            controller: LukaKVController to update

        Returns:
            True if updates were applied, False otherwise
        """
        if layer_idx not in self.pending:
            return False

        pending = self.pending[layer_idx]
        event = pending["event"]

        # Non-blocking check: is the background work done?
        if not event.query():
            return False

        # Work is done, apply updates
        k_compressed = pending["k_compressed"]
        v_compressed = pending["v_compressed"]
        page_metadata = pending["page_metadata"]
        new_frontiers = pending["new_frontiers"]
        all_new_pages = pending["all_new_pages"]
        has_updates = pending["has_updates"]

        device = k_compressed.device
        summary_cache = controller.summary_cache[layer_idx]
        raw_cache = controller.raw_cache

        # Distribute compressed pages to batches
        batch_pages = {}
        for i, (b, start, end) in enumerate(page_metadata):
            if b not in batch_pages:
                batch_pages[b] = []
            batch_pages[b].append((k_compressed[i:i+1], v_compressed[i:i+1], start, end))

        # Add pages to summary cache
        for b, pages in batch_pages.items():
            k_list = [p[0] for p in pages]
            v_list = [p[1] for p in pages]
            starts = [p[2] for p in pages]
            ends = [p[3] for p in pages]

            k_stack = torch.cat(k_list, dim=0).unsqueeze(0).transpose(1, 2)
            v_stack = torch.cat(v_list, dim=0).unsqueeze(0).transpose(1, 2)

            summary_cache.add_pages(
                keys=k_stack,
                values=v_stack,
                batch_nums=torch.tensor([b], device=device),
                page_start=torch.tensor([starts], device=device),
                page_end=torch.tensor([ends], device=device),
                # Use slice to avoid .item() GPU sync
                page_frontier=new_frontiers[b:b+1].clone()
            )

        if has_updates:
            raw_cache.raw_seq_start[layer_idx] = new_frontiers
            controller.cover_view[layer_idx].update_cover_view(layer_idx, raw_cache, summary_cache)
            controller.attn_buffer[layer_idx].compress_and_trim(all_new_pages, new_frontiers)

        # Clear pending
        del self.pending[layer_idx]
        return True

    def has_pending(self, layer_idx: int) -> bool:
        """Check if there's pending work for a layer."""
        return layer_idx in self.pending

    def flush_all(self, controller: "LukaKVController"):
        """Wait for all pending work to complete and apply updates.

        Call this at the end of generation or when you need to ensure
        all async updates are applied.
        """
        if not self.pending:
            return

        # Wait for background stream to complete
        if self.stream is not None:
            self.stream.synchronize()

        # Apply all pending updates
        for layer_idx in list(self.pending.keys()):
            self.try_apply(layer_idx, controller)

    def clear(self):
        """Clear all pending work (without applying)."""
        self.pending.clear()


class LukaKVController:

    def __init__(
        self,
        config: PretrainedConfig,
        num_layers: Optional[int] = None,
        use_lined_attention: bool = False,
        lined_layers: Optional[List[int]] = None,
        grid_top_k: int = 16,
        grid_update_interval: int = 16,
        grid_decay: float = 0.99,
    ):
        """Coordinating facade that owns raw, summary, cover, and attention buffers.

        Caches are tracked per-layer to mirror the underlying transformer stack.
        Args:
            config: PretrainedConfig supplying at least `num_hidden_layers` (or pass
                `num_layers` explicitly).
            num_layers: Optional override for layer count if not present on config.
            use_lined_attention: If True, use lined attention for layers in lined_layers.
            lined_layers: List of layer indices to use lined attention. If None and use_lined_attention=True, uses all layers.
            grid_top_k: Number of grid tokens per batch/layer for lined attention.
            grid_update_interval: How often to recompute grid tokens.
            grid_decay: Exponential decay factor for grid scores.
        """
        self.num_layers = num_layers or getattr(config, "num_hidden_layers", None)
        if self.num_layers is None:
            raise ValueError("num_layers must be provided or derivable from config.num_hidden_layers")
        # config should come from the model config, if that's a thing.
        self.raw_cache = RawCache(config)  # wraps underlying DynamicCache across layers
        # Per-layer containers
        self.summary_cache: List[SummaryCache] = [SummaryCache(config) for _ in range(self.num_layers)]
        self.cover_view: List[CoverView] = [CoverView() for _ in range(self.num_layers)]
        self.attn_buffer: List[AttentionScoreBuffer] = [AttentionScoreBuffer() for _ in range(self.num_layers)]
        # NEW: lined-attention cache (vertical-line / global tokens)
        self.grid_cache: List[GridCache] = [GridCache(config) for _ in range(self.num_layers)]
        
        # Initialize segmenter, compressor, and refinement rule
        # Using defaults for now; ideally these come from config.
        self.segmenter = DummySegmenter(min_chunk=16, tail_len=16, max_pages=15)
        self.compressor = AttentionWeightedCompressor(temperature=1.0)
        self.refinement_rule: RefinementRule = FixedThresholdRule(threshold=0.2)
        # How often to attempt segmentation/compression (every N decode steps)
        self.segment_interval = 1
        # Whether to create pages during generation (decode) or only during prefill
        self.create_pages_in_generation = True
        # Whether to add log(N) bias to summary attention logits
        # Set to False for now (user requested no bias)
        self.use_log_bias = False
        # Log bias mode: "none", "fixed_n" (log(N)), or "adaptive_k" (log(k_eff) from entropy)
        self.log_bias_mode = 'none'
        self.seg_step_counters = [0 for _ in range(self.num_layers)]
        # Track tokens since last page creation (for fast early exit)
        self.tokens_since_last_page = [0 for _ in range(self.num_layers)]
        # Production mode: skip debug checks (invariants) for better performance
        self.production_mode = True
        # Async page creation: run compression on background CUDA stream
        self.async_pages = False
        self.async_page_creator = AsyncPageCreator(self.num_layers)
        
        # H2O-like grid / heavy-hitter parameters for lined attention
        self.use_lined_attention = use_lined_attention
        self.lined_layers = set(lined_layers) if lined_layers is not None else (set(range(self.num_layers)) if use_lined_attention else set())
        self.grid_top_k = grid_top_k
        self.grid_update_interval = grid_update_interval
        self.grid_decay = grid_decay
        # Per-layer running scores over raw positions (initialized lazily)
        self.grid_scores: List[Optional[torch.Tensor]] = [None for _ in range(self.num_layers)]
        self.grid_step_counters = [0 for _ in range(self.num_layers)]
        
        # Gentle H2O-ish knobs
        self.min_lined_seq_len = 384        # don't compress before this many tokens (increased for better quality)
        self.min_lined_tail_window = 192    # minimum local tail length (increased for better context)
        self.grid_min_change_ratio = 0.3    # only refresh grid if ≥30% of top-K changed
        
        # Debug flag: set to True to enable diagnostic prints (slows down generation significantly)
        self.debug = False
    
    def reset(self):
        """Reset all cache state. Call this before each new question/example to prevent cache contamination."""
        # Get config from raw_cache (it has the config)
        config = self.raw_cache.config
        
        # Reset raw cache (clear underlying DynamicCache)
        if self.raw_cache.cache is not None:
            # Clear the underlying DynamicCache by setting it to None
            # The cache will be reinitialized on the next forward pass
            self.raw_cache.cache = None
        
        # Reset metadata
        self.raw_cache.seq_start = [None] * self.num_layers
        self.raw_cache.raw_seq_start = [None] * self.num_layers
        
        # Reset summary caches
        for layer_idx in range(self.num_layers):
            self.summary_cache[layer_idx] = SummaryCache(config)
        
        # Reset cover views
        for layer_idx in range(self.num_layers):
            self.cover_view[layer_idx] = CoverView()
        
        # Reset attention buffers
        for layer_idx in range(self.num_layers):
            self.attn_buffer[layer_idx].reset()
        
        # Reset grid caches
        for layer_idx in range(self.num_layers):
            self.grid_cache[layer_idx] = GridCache(config)
        
        # Reset grid scores and counters
        self.grid_scores = [None for _ in range(self.num_layers)]
        self.grid_step_counters = [0 for _ in range(self.num_layers)]
        self.seg_step_counters = [0 for _ in range(self.num_layers)]
    
    def initialize_views(self, layer_idx: int):
        """Initialize cover view and attention buffer for a layer (e.g. after prefill)."""
        self.cover_view[layer_idx].initialize(layer_idx, self.raw_cache)
        
        # Initialize attention buffer
        # We need B, H, T_init.
        # raw_cache has this info.
        k_raw, _, _, _ = self.raw_cache.get_layer(layer_idx, with_offsets=False)
        if k_raw is not None:
            B, H, T, D = k_raw.shape
            # H might be different if GQA?
            # AttentionScoreBuffer stores weights for Query heads (H_q).
            # k_raw has H_kv.
            # We don't know H_q here easily without config.
            # But `top_down_attention` gets `query_states` [B, H_q, ...].
            # So `push` will have H_q.
            # `initialize` needs H.
            # We can get it from `self.summary_cache[0].config`?
            # Or just pass 0/None and let `push` handle H?
            # No, `initialize` creates the tensor.
            # We need H_q.
            # `SummaryCache` has `config`.
            config = self.summary_cache[layer_idx].config
            H_q = config.num_attention_heads
            
            self.attn_buffer[layer_idx].initialize(
                B=B,
                H=H_q,
                T_init=T,
                device=k_raw.device,
                dtype=k_raw.dtype
            )
            
            # Initialize summary cache
            self.summary_cache[layer_idx].initialize(
                B=B,
                H=H, # H_k
                D=D,
                device=k_raw.device,
                dtype=k_raw.dtype
            )
            
            # Initialize grid cache
            self.grid_cache[layer_idx].initialize(
                B=B,
                H=H, # H_k
                D=D,
                device=k_raw.device,
                dtype=k_raw.dtype
            )

    def update(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Update caches for a specific layer with newly generated raw tokens.

        Args:
            layer_idx: int
                Transformer layer index.
            keys: torch.Tensor
                [B, H, L_new, D] raw keys produced at this decode step.
            values: torch.Tensor
                [B, H, L_new, D] raw values produced at this decode step.
            cache_kwargs: Optional[dict[str, Any]]
                Forwarded to DynamicCache.update (e.g., RoPE data).
            attention_mask: Optional[torch.Tensor]
                [B, 1, L_q, T_raw] additive mask to infer padding offsets.

        Side effects:
            - Extends `raw_cache` (and underlying `DynamicCache`).
            - Extends the layer's cover view so top-down attention can see the raw tail.
            - Does not create new pages; `try_new_pages` is responsible for that.
        """

        
        # Initialize views if needed (e.g. first decode step after prefill)
        # We must initialize BEFORE updating raw_cache to avoid double-counting the new token.

        k_all, v_all, seq_start, raw_seq_start = self.raw_cache.update(
            keys,
            values,
            layer_idx=layer_idx,
            cache_kwargs=cache_kwargs,
            attention_mask=attention_mask,
        )
        if self.cover_view[layer_idx].cover_keys is None:
             self.initialize_views(layer_idx)
        else:
            # Calculate indices for new tokens
            # k_all: [B, H, T_raw, D]
            # keys: [B, H, L_new, D]
            B, H, T_raw, D = k_all.shape
            L_new = keys.shape[2]
            device = keys.device
            
            # Indices are [T_raw - L_new, ..., T_raw - 1]
            new_indices = torch.arange(T_raw - L_new, T_raw, device=device).unsqueeze(0).expand(B, -1)
            
            self.cover_view[layer_idx].update(keys, values, new_indices)
        
        return k_all, v_all, seq_start, raw_seq_start

    def try_new_pages(self, layer_idx: int) -> bool:
        """Attempt to segment raw tokens into pages and emit summaries for a layer.

        Optimized with batched compression and reduced tensor allocations.
        Supports async mode where compression runs on a background CUDA stream.

        Args:
            layer_idx: int
                Transformer layer index.

        Returns:
            bool: True if new summary pages were added (i.e., `summary_cache[layer_idx]`
            grew and `cover_view[layer_idx]` should be rebuilt), False otherwise.
        """
        # First, try to apply any pending async updates
        if self.async_pages:
            applied = self.async_page_creator.try_apply(layer_idx, self)
            if applied:
                self.tokens_since_last_page[layer_idx] = 0
                return True

        # Increment token counter (tracks tokens since last page creation)
        self.tokens_since_last_page[layer_idx] += 1

        # Throttle segmentation based on configured interval
        self.seg_step_counters[layer_idx] += 1
        if self.segment_interval > 1 and (self.seg_step_counters[layer_idx] % self.segment_interval) != 0:
            return False

        # Fast early exit: skip segmenter if not enough tokens since last page
        # This avoids the ~0.11ms segmenter overhead per call
        min_chunk = getattr(self.segmenter, 'min_chunk', 16)
        tail_len = getattr(self.segmenter, 'tail_len', 16)
        if self.tokens_since_last_page[layer_idx] < min_chunk:
            return False

        # Early exit: check cover view length before expensive operations
        cover_view = self.cover_view[layer_idx]
        if cover_view.length < getattr(self.segmenter, '_min_tokens_needed', 32):
            return False  # Not enough tokens for any pages

        # 1. Segment - Get buffer data
        attn_weights, _, _ = self.attn_buffer[layer_idx].get_data()

        # Use CoverView indices directly as they are the ground truth (sliced to valid length)
        _, _, cover_indices, cover_is_summary = cover_view.get_valid_kv()
        if attn_weights is None:
            return False

        # Async path: start background work and return immediately
        if self.async_pages:
            if not self.async_page_creator.has_pending(layer_idx):
                self.async_page_creator.start_async(
                    layer_idx, self, attn_weights, cover_indices, cover_is_summary
                )
            return False  # Updates will be applied on next call

        # 2. Run segmentation
        page_ends = self.segmenter.process(attn_weights, cover_indices, cover_is_summary, layer_idx=layer_idx)
        if page_ends is None:
            return False

        summary_cache = self.summary_cache[layer_idx]
        raw_cache = self.raw_cache

        # Get current frontier per batch
        _, _, _, raw_seq_start = raw_cache.get_layer(layer_idx, with_offsets=True)
        if raw_seq_start is None:
            return False

        B = raw_seq_start.shape[0]
        device = raw_seq_start.device

        # Get full tensors once
        k_raw, v_raw, _, _ = raw_cache.get_layer(layer_idx, with_offsets=False)
        if k_raw is None:
            return False

        H_kv = k_raw.shape[1]
        new_frontiers = raw_seq_start.clone()
        has_updates = False
        all_new_pages = [[] for _ in range(B)]

        # Collect all pages across batches for batched compression
        all_k_slices = []
        all_v_slices = []
        all_importance = []
        page_metadata = []  # (batch_idx, start, end)
        max_page_len = 0

        # Batch the .item() calls - get all frontiers at once to minimize syncs
        # This is a single sync instead of B syncs in the loop
        frontiers_list = raw_seq_start.tolist()

        for b in range(B):
            frontier = frontiers_list[b]
            p_ends = page_ends[b]
            valid_mask = (p_ends >= frontier) & (p_ends != -1)

            # Get valid ends as list - single sync per batch instead of checking .any() first
            valid_ends_sorted = p_ends[valid_mask].sort().values.tolist()
            if not valid_ends_sorted:
                continue

            current_start = frontier
            cover_idx_b = cover_indices[b] if cover_indices is not None else None
            is_sum_b = cover_is_summary[b] if cover_is_summary is not None else None

            for end_idx in valid_ends_sorted:
                end_idx = int(end_idx)
                if end_idx < current_start:
                    continue

                page_len = end_idx - current_start + 1
                max_page_len = max(max_page_len, page_len)

                # Slice raw K/V
                k_slice = k_raw[b, :, current_start:end_idx + 1, :]
                v_slice = v_raw[b, :, current_start:end_idx + 1, :]
                all_k_slices.append(k_slice)
                all_v_slices.append(v_slice)

                # Extract importance weights
                importance = None
                if attn_weights is not None and cover_idx_b is not None:
                    page_mask = (cover_idx_b >= current_start) & (cover_idx_b <= end_idx) & (is_sum_b == 0)
                    if page_mask.any():
                        page_attn = attn_weights[b, :, :, page_mask]
                        importance = page_attn.sum(dim=1)  # [H_q, page_len]
                        H_q = importance.shape[0]
                        if H_q != H_kv:
                            num_groups = H_q // H_kv
                            importance = importance.view(H_kv, num_groups, -1).mean(dim=1)
                all_importance.append(importance)

                page_metadata.append((b, current_start, end_idx))
                all_new_pages[b].append((current_start, end_idx))
                current_start = end_idx + 1

            if current_start > frontier:
                new_frontiers[b] = current_start
                has_updates = True

        if not page_metadata:
            return False

        # Batched compression: pad pages to same length and compress together
        N_pages = len(all_k_slices)
        D = k_raw.shape[3]

        # Pad and stack all pages
        padded_k = torch.zeros(N_pages, H_kv, max_page_len, D, device=device, dtype=k_raw.dtype)
        padded_v = torch.zeros(N_pages, H_kv, max_page_len, D, device=device, dtype=v_raw.dtype)
        padded_importance = None

        has_any_importance = any(imp is not None for imp in all_importance)
        if has_any_importance:
            padded_importance = torch.zeros(N_pages, H_kv, max_page_len, device=device, dtype=k_raw.dtype)

        for i, (k_s, v_s, imp) in enumerate(zip(all_k_slices, all_v_slices, all_importance)):
            plen = k_s.shape[1]
            padded_k[i, :, :plen, :] = k_s
            padded_v[i, :, :plen, :] = v_s
            if imp is not None and padded_importance is not None:
                padded_importance[i, :, :plen] = imp

        # Compress all pages at once
        k_compressed, v_compressed = self.compressor(padded_k, padded_v, padded_importance)  # [N_pages, H, D]

        # Compute log effective support (entropy) for adaptive log(k) bias
        # log_k_eff[i] = H = -sum(p * log(p)) where p is normalized importance
        # For uniform distribution: H = log(N), for peaked: H -> 0
        log_k_eff = None
        if padded_importance is not None:
            # Build mask for valid (non-padded) positions
            page_lens = torch.tensor(
                [pm[2] - pm[1] + 1 for pm in page_metadata],  # end - start + 1
                device=device, dtype=torch.long
            )  # [N_pages]
            # Create mask: [N_pages, max_page_len]
            pos_indices = torch.arange(max_page_len, device=device).unsqueeze(0)  # [1, max_page_len]
            valid_mask = pos_indices < page_lens.unsqueeze(1)  # [N_pages, max_page_len]

            # Compute entropy per page, averaged across heads
            # padded_importance: [N_pages, H_kv, max_page_len]
            # Mask out invalid positions with -inf for softmax
            masked_importance = padded_importance.clone()
            masked_importance[:, :, :][~valid_mask.unsqueeze(1).expand_as(masked_importance)] = float('-inf')

            # log_softmax for numerical stability
            log_probs = torch.nn.functional.log_softmax(masked_importance, dim=-1)  # [N_pages, H_kv, max_page_len]
            probs = log_probs.exp()

            # Entropy: H = -sum(p * log(p)), handling 0*-inf = 0
            # Where p=0 (masked), log_p=-inf, but p*log_p should be 0
            entropy_terms = torch.where(
                probs > 0,
                -probs * log_probs,
                torch.zeros_like(probs)
            )
            entropy_per_head = entropy_terms.sum(dim=-1)  # [N_pages, H_kv]
            entropy_avg = entropy_per_head.mean(dim=-1)  # [N_pages]

            log_k_eff = entropy_avg  # This is log(k_eff) since k_eff = exp(H)

        # Distribute compressed pages back to batches
        batch_pages = {}  # batch_idx -> list of (k, v, start, end, log_k)
        for i, (b, start, end) in enumerate(page_metadata):
            if b not in batch_pages:
                batch_pages[b] = []
            log_k_val = log_k_eff[i].item() if log_k_eff is not None else None
            batch_pages[b].append((k_compressed[i:i+1], v_compressed[i:i+1], start, end, log_k_val))

        # Add pages to summary cache
        for b, pages in batch_pages.items():
            k_list = [p[0] for p in pages]
            v_list = [p[1] for p in pages]
            starts = [p[2] for p in pages]
            ends = [p[3] for p in pages]
            log_k_vals = [p[4] for p in pages]

            k_stack = torch.cat(k_list, dim=0).unsqueeze(0).transpose(1, 2)  # [1, H, N, D] -> need [1, H, N, D]
            v_stack = torch.cat(v_list, dim=0).unsqueeze(0).transpose(1, 2)

            # Build log_effective_support tensor if available
            log_eff_supp = None
            if log_k_vals[0] is not None:
                log_eff_supp = torch.tensor([log_k_vals], device=device, dtype=torch.float32)  # [1, N_pages]

            summary_cache.add_pages(
                keys=k_stack,
                values=v_stack,
                batch_nums=torch.tensor([b], device=device),
                page_start=torch.tensor([starts], device=device),
                page_end=torch.tensor([ends], device=device),
                # Use slice to avoid .item() GPU sync
                page_frontier=new_frontiers[b:b+1].clone(),
                log_effective_support=log_eff_supp,
            )

        if has_updates:
            # Update raw_seq_start in RawCache
            raw_cache.raw_seq_start[layer_idx] = new_frontiers
            
            # Rebuild CoverView (topdown doesn't use grid tokens)
            self.cover_view[layer_idx].update_cover_view(
                layer_idx,
                raw_cache,
                summary_cache,
                grid_cache=None,  # topdown doesn't use grid tokens
            )
            
            # Compress and trim buffer
            self.attn_buffer[layer_idx].compress_and_trim(all_new_pages, new_frontiers)
            # Reset token counter since we just created pages
            self.tokens_since_last_page[layer_idx] = 0
            return True

        return False

    def _update_grid_scores(
        self,
        layer_idx: int,
        attn_probs: torch.Tensor,      # [B, H_q, L_q, T_cover]
        cover_indices: torch.Tensor,   # [B, T_cover] (raw indices or -1)
        cover_is_summary: torch.Tensor # [B, T_cover] (all zeros for lined)
    ):
        """Update running scores for grid token selection (H2O-style), using
        only the current step's attention and cover layout.
        
        Args:
            layer_idx: int
                Transformer layer index.
            attn_probs: torch.Tensor
                Current step's attention probabilities [B, H_q, L_q, T_cover].
            cover_indices: torch.Tensor
                Current cover's raw indices [B, T_cover].
            cover_is_summary: torch.Tensor
                Current cover's summary flags [B, T_cover] (all zeros for lined).
        """
        if attn_probs is None:
            return

        B, H_q, L_q, T_cover = attn_probs.shape
        device = attn_probs.device

        # Column scores for this step: mean over heads, sum over query positions
        # This accumulates attention from ALL query positions (important for prefill with L_q > 1)
        # In decode, L_q=1, so sum is a no-op
        # attn_probs: [B, H_q, L_q, T_cover]
        scores = attn_probs.mean(dim=1)        # [B, L_q, T_cover] - mean over heads
        scores = scores.sum(dim=1)             # [B, T_cover] - sum over all query positions
        col_scores = scores

        # Map cover -> raw indices (for lined: already raw indices)
        cover_raw_indices = cover_indices  # [B, T_cover]

        # Get raw cache shape
        k_raw, _, _, _ = self.raw_cache.get_layer(layer_idx, with_offsets=False)
        if k_raw is None:
            return
        B_raw, H_k, T_raw, D = k_raw.shape

        # Handle possible batch mismatch defensively
        if B != B_raw:
            B_use = min(B, B_raw)
            col_scores = col_scores[:B_use]
            cover_raw_indices = cover_raw_indices[:B_use]
            B = B_use

        # Initialize raw-space scores if needed
        # CRITICAL: If T_raw grows, we need to resize (pad), not reinitialize (which loses accumulated scores)
        if self.grid_scores[layer_idx] is None:
            self.grid_scores[layer_idx] = torch.zeros(B_raw, T_raw, device=device, dtype=col_scores.dtype)
        elif self.grid_scores[layer_idx].shape != (B_raw, T_raw):
            # T_raw has grown - pad the scores tensor instead of reinitializing
            old_scores = self.grid_scores[layer_idx]
            old_B, old_T = old_scores.shape
            if old_T < T_raw:
                # Pad with zeros (new positions start with zero scores)
                pad_T = T_raw - old_T
                padding = torch.zeros(old_B, pad_T, device=device, dtype=old_scores.dtype)
                self.grid_scores[layer_idx] = torch.cat([old_scores, padding], dim=1)
                if self.debug and layer_idx == 0:
                    print(f"[Layer {layer_idx}] Grid scores resized: {old_scores.shape} -> {self.grid_scores[layer_idx].shape} (T_raw grew from {old_T} to {T_raw})")
            elif old_B != B_raw:
                # Batch size changed - handle gracefully
                B_use = min(old_B, B_raw)
                if B_use < B_raw:
                    # Need to pad batch dimension
                    pad_B = B_raw - B_use
                    padding = torch.zeros(pad_B, T_raw, device=device, dtype=old_scores.dtype)
                    self.grid_scores[layer_idx] = torch.cat([old_scores[:B_use], padding], dim=0)
                else:
                    self.grid_scores[layer_idx] = old_scores[:B_use]

        raw_scores = self.grid_scores[layer_idx]

        # CRITICAL FIX #1: Exponential decay FIRST (EMA: s_t = λ * s_{t-1} + x_t)
        # Decay previous scores before adding new ones
        raw_scores.mul_(self.grid_decay)

        # CRITICAL FIX #2: Use scatter_add_ for proper handling of duplicate indices
        # This ensures deterministic accumulation when multiple cover positions map to same raw position
        valid_mask = (cover_raw_indices >= 0)
        if not valid_mask.any():
            return

        # Prepare indices and scores for scatter_add_
        # scatter_add_ needs: (dim, index, src)
        # For 2D tensor [B, T_raw], we scatter along dim=1 (T_raw dimension)
        cover_indices_clamped = cover_raw_indices.clamp(min=0, max=T_raw - 1)  # [B, T_cover]
        
        # Set invalid positions to 0 (they won't contribute) and their indices to 0 (safe dummy)
        # This allows us to use scatter_add_ on the full tensors
        indices_for_scatter = cover_indices_clamped.clone()
        scores_for_scatter = col_scores.clone()
        indices_for_scatter[~valid_mask] = 0  # Dummy index for invalid positions
        scores_for_scatter[~valid_mask] = 0   # Zero score for invalid positions
        
        # Use scatter_add_ along dim=1 (sequence dimension)
        # raw_scores[b, indices_for_scatter[b, i]] += scores_for_scatter[b, i]
        raw_scores.scatter_add_(dim=1, index=indices_for_scatter, src=scores_for_scatter)

        # Debug: check shapes and score stats (Diagnostic A)
        if self.debug and layer_idx == 0 and col_scores.numel() > 0:
            max_score = col_scores[valid_mask].max().item() if valid_mask.any() else 0
            mean_score = col_scores[valid_mask].mean().item() if valid_mask.any() else 0
            score_sum = col_scores[valid_mask].sum().item() if valid_mask.any() else 0
            num_unique = torch.unique(cover_indices_clamped[valid_mask]).numel() if valid_mask.any() else 0
            if max_score > 0:
                print(f"[Layer {layer_idx}] Grid score update: attn_probs.shape={attn_probs.shape}, col_scores.shape={col_scores.shape}, max={max_score:.6f}, mean={mean_score:.6f}, sum={score_sum:.6f}, valid_tokens={valid_mask.sum().item()}, unique_positions={num_unique}")

        self.grid_scores[layer_idx] = raw_scores

    def _refresh_grid_tokens(self, layer_idx: int):
        """Refresh grid tokens by selecting top-K based on accumulated scores (H2O-style).
        
        Args:
            layer_idx: int
                Transformer layer index.
        """
        scores = self.grid_scores[layer_idx]
        if scores is None:
            return

        k_raw, v_raw, _, _ = self.raw_cache.get_layer(layer_idx, with_offsets=False)
        if k_raw is None:
            return

        B, H_k, T_raw, D = k_raw.shape
        device = k_raw.device

        # CRITICAL DIAGNOSTIC: Verify scores shape is correct
        # Scores should be [B, T_raw], NOT [B, grid_top_k] or [B, T_cover]
        if self.debug and layer_idx == 0:
            print(f"[Layer {layer_idx}] REFRESH: scores.shape={scores.shape}, T_raw={T_raw}, grid_top_k={self.grid_top_k}")
        
        # Hard assertion to catch the bug immediately
        assert scores.shape[-1] == T_raw, \
            f"CRITICAL BUG: Selecting from wrong axis! scores.shape={scores.shape}, T_raw={T_raw}. " \
            f"Expected scores to be [B, T_raw] but got shape ending in {scores.shape[-1]}. " \
            f"This means we're selecting from grid slots instead of raw positions!"

        # Ensure scores batch size matches raw cache batch size
        B_scores = scores.shape[0]
        if B != B_scores:
            # This shouldn't happen, but handle it gracefully
            B_use = min(B, B_scores)
            scores = scores[:B_use]
            k_raw = k_raw[:B_use]
            v_raw = v_raw[:B_use]
            B = B_use

        # Optional: avoid picking from very recent tail (sliding window)
        # This ensures grid focuses on older, globally useful tokens
        # while the tail already covers the most recent tokens
        window = getattr(self.segmenter, 'tail_len', 16)
        window = max(window, self.min_lined_tail_window)  # Use same minimum as lined_attention
        
        # CRITICAL: Compute tail exclusion in RAW space, not cover space
        tail_start_raw = max(0, T_raw - window)
        
        # CRITICAL FIX: If window >= T_raw, tail_start_raw = 0 means we'd exclude everything
        # In that case, we can't hard-exclude the tail (nothing would be left to select)
        # Only exclude tail if there's enough sequence left to select from
        if tail_start_raw == 0:
            # Tail covers the whole sequence; don't hard-exclude (would leave nothing)
            # Option: soft downweight or skip exclusion entirely
            if self.debug and layer_idx == 0:
                print(f"[Layer {layer_idx}] Tail exclusion SKIPPED: T_raw={T_raw}, window={window}, tail_start_raw=0 (tail covers entire sequence)")
            # Don't modify scores - allow selection from entire sequence
        elif tail_start_raw > 0 and T_raw > tail_start_raw:
            # We have enough sequence to exclude the tail
            # Ensure we leave at least grid_top_k tokens selectable (positions 0 to tail_start_raw-1)
            # If tail_start_raw < grid_top_k, we don't have enough non-tail positions to select from
            if tail_start_raw < self.grid_top_k:
                # Not enough positions outside tail; expand selectable region to include grid_top_k positions
                tail_start_raw = self.grid_top_k
                if self.debug and layer_idx == 0:
                    print(f"[Layer {layer_idx}] Tail exclusion ADJUSTED: raised tail_start_raw to {tail_start_raw} to ensure {self.grid_top_k} tokens remain selectable")
            
            # Debug: check scores before modifying tail
            if self.debug and layer_idx == 0:
                max_score_before = scores.max().item()
                mean_score_before = scores.mean().item()
                tail_max = scores[:, tail_start_raw:].max().item() if T_raw > tail_start_raw else 0
                print(f"[Layer {layer_idx}] Tail exclusion: T_raw={T_raw}, window={window}, tail_start_raw={tail_start_raw}, will exclude {T_raw - tail_start_raw} tokens")
            
            # CRITICAL FIX #3: Clone before excluding tail (don't modify original scores)
            # Exclude tail completely from grid selection (set to -inf)
            # Tail is already in the cover, so grid should focus on older tokens
            sel_scores = scores.clone()  # Work on a copy
            sel_scores[:, tail_start_raw:] = float('-inf')  # Exclude from tail_start_raw to end
            scores = sel_scores  # Use the modified copy
        else:
            # Edge case: shouldn't happen, but handle gracefully
            if self.debug and layer_idx == 0:
                print(f"[Layer {layer_idx}] Tail exclusion SKIPPED: edge case T_raw={T_raw}, tail_start_raw={tail_start_raw}")
            
            # Debug: check scores after reducing tail weight
            if self.debug and layer_idx == 0:
                max_score_after = scores.max().item()
                mean_score_after = scores.mean().item()
                print(f"[Layer {layer_idx}] Grid scores: before tail reduction - max={max_score_before:.6f}, mean={mean_score_before:.6f}, tail_max={tail_max:.6f}; after - max={max_score_after:.6f}, mean={mean_score_after:.6f}")

        # Top-K per batch
        # CRITICAL DIAGNOSTIC B: Print scores info right before topk
        if self.debug and layer_idx == 0:
            print(f"[Layer {layer_idx}] Before topk: scores.shape={scores.shape}, scores.dtype={scores.dtype}, "
                  f"scores.min()={scores.min().item():.6f}, scores.max()={scores.max().item():.6f}, "
                  f"T_raw={T_raw}, grid_top_k={self.grid_top_k}")
            # Check if scores are all -inf (would indicate everything was excluded)
            num_inf = (scores == float('-inf')).sum().item()
            num_finite = (torch.isfinite(scores)).sum().item()
            print(f"[Layer {layer_idx}] Scores stats: num_inf={num_inf}, num_finite={num_finite}, total={scores.numel()}")
        
        # Ensure we don't exceed capacity if prefix was initialized
        # For now, just select top-K (prefix will be overwritten, but that's okay for initial testing)
        K = min(self.grid_top_k, T_raw)
        top_scores, top_indices = torch.topk(scores, k=K, dim=-1)   # [B, K]
        indices = top_indices.clone()  # [B, K]
        
        # CRITICAL DIAGNOSTIC C: Verify we're selecting raw positions and gathering correct KV
        if self.debug and layer_idx == 0 and B > 0 and K > 0:
            # Sanity check: verify grid_k[0,0,0] matches k_raw[0,0,indices[0,0]]
            test_idx = indices[0, 0].item()
            if test_idx < T_raw and test_idx >= 0:
                # We'll check this after gathering, but log the index now
                print(f"[Layer {layer_idx}] Will gather KV from raw position {test_idx} (first selected index)")
        
        # Debug: verify indices are not in tail (RED FLAG B check)
        if self.debug and layer_idx == 0 and T_raw > tail_start_raw:
            indices_in_tail = (indices >= tail_start_raw).sum().item()
            max_idx = indices.max().item()
            min_idx = indices.min().item()
            if indices_in_tail > 0:
                print(f"[Layer {layer_idx}] WARNING: {indices_in_tail}/{K} grid indices are in tail (should be 0)")
            else:
                print(f"[Layer {layer_idx}] Grid selection: max_idx={max_idx}, min_idx={min_idx}, tail_start_raw={tail_start_raw}, all_indices_outside_tail=✓")
            
            # CRITICAL: Verify indices are in valid range [0, T_raw-1]
            if max_idx >= T_raw:
                print(f"[Layer {layer_idx}] ERROR: max_idx={max_idx} >= T_raw={T_raw} (out of bounds!)")
            if min_idx < 0:
                print(f"[Layer {layer_idx}] ERROR: min_idx={min_idx} < 0 (out of bounds!)")

        # Optional H2O-ish stability: only refresh if top-K actually changed enough
        prev_indices = None
        if self.grid_min_change_ratio > 0 and self.grid_cache[layer_idx].indices is not None:
            prev = self.grid_cache[layer_idx].indices  # [B_prev, G_max]
            # Align shapes: we only compare first B rows and first K cols
            B_prev, G_prev = prev.shape
            B_cmp = min(B_prev, B)
            K_cmp = min(G_prev, K)
            if B_cmp > 0 and K_cmp > 0:
                prev_indices = prev[:B_cmp, :K_cmp]
                new_indices = indices[:B_cmp, :K_cmp]
                diff = (prev_indices != new_indices)
                change_ratio = diff.float().mean().item()
                if change_ratio < self.grid_min_change_ratio:
                    # Not enough change in grid membership; keep old grid, bail out
                    if self.debug and layer_idx == 0:  # Debug print for first layer only
                        print(f"[Layer {layer_idx}] Grid refresh skipped: change_ratio={change_ratio:.3f} < {self.grid_min_change_ratio}")
                    return

        # If we're here, either there was no previous grid or it changed meaningfully.
        # Gather K/V: [B, H_k, K, D]
        # Build gather index: [B, 1, K, 1] -> expand to [B, H_k, K, D]
        # Clamp indices to valid range
        T_raw_max = k_raw.shape[2] - 1
        indices_clamped = indices.clamp(min=0, max=T_raw_max)
        gather_idx = indices_clamped.view(B, 1, K, 1).expand(-1, H_k, -1, D)
        grid_k = torch.gather(k_raw, 2, gather_idx)
        grid_v = torch.gather(v_raw, 2, gather_idx)

        # CRITICAL DIAGNOSTIC C: Verify KV gathering correctness
        # Keep this check even when debug=False (it's cheap and catches bugs)
        if layer_idx == 0 and B > 0 and K > 0:
            test_idx = indices[0, 0].item()
            if 0 <= test_idx < T_raw:
                # Check: grid_k[0,0,0] should equal k_raw[0,0,test_idx]
                gathered_kv = grid_k[0, 0, 0]
                raw_kv = k_raw[0, 0, test_idx]
                is_close = torch.allclose(gathered_kv, raw_kv, atol=1e-5)
                max_diff = (gathered_kv - raw_kv).abs().max().item()
                if not is_close:
                    # Always print errors (not just in debug mode)
                    print(f"[Layer {layer_idx}] ERROR: KV gather mismatch! grid_k[0,0,0] != k_raw[0,0,{test_idx}], max_diff={max_diff:.6f}")
                elif self.debug:
                    print(f"[Layer {layer_idx}] KV gather verified: grid_k[0,0,0] == k_raw[0,0,{test_idx}] ✓ (max_diff={max_diff:.6e})")

        # Write into GridCache
        self.grid_cache[layer_idx].set_tokens(
            keys=grid_k,
            values=grid_v,
            indices=indices,
        )
        
        # Debug print for first layer only (with all requested diagnostics)
        if self.debug and layer_idx == 0 and B > 0:
            grid_indices_b0 = indices[0].cpu().tolist()
            grid_scores_b0 = top_scores[0].cpu().tolist()
            max_idx = indices[0].max().item()
            min_idx = indices[0].min().item()
            # Check if any indices are in tail (should be 0 after fix)
            tail_start = T_raw - window if T_raw > window else T_raw
            indices_in_tail = (indices[0] >= tail_start).sum().item()
            
            print(f"[Layer {layer_idx}] Grid refreshed: indices={grid_indices_b0[:5]}... (top 5), scores={[f'{s:.3f}' for s in grid_scores_b0[:5]]}")
            print(f"[Layer {layer_idx}] Grid selection diagnostics: max_idx={max_idx}, min_idx={min_idx}, tail_start={tail_start}, indices_in_tail={indices_in_tail} (should be 0)")

    def top_down_attention(
        self,
        layer_idx: int,
        query_states: torch.Tensor,      # [B, H_q, L_q, D]
        scaling: float,
        num_kv_groups: int,
        attention_mask: Optional[torch.Tensor] = None,  # [B, 1, L_q, T_raw]
        sliding_window: Optional[int] = None,
        threshold: Optional[float] = None,  # Deprecated: use self.refinement_rule instead
        use_exact_attention: bool = False,  # If True, skip cover view and use raw attention
    ):
        """Run attention over cover view (summaries + raw tail) with optional refinement.

        Args:
            layer_idx: Transformer layer index.
            query_states: [B, H_q, L_q, D] query projections.
            scaling: Attention scaling factor (1/sqrt(D)).
            num_kv_groups: H_q // H_k for grouped-query attention.
            attention_mask: [B, 1, L_q, T_raw] causal + padding mask.
            sliding_window: Unused, kept for API compatibility.
            threshold: DEPRECATED. Use self.refinement_rule instead.
                       For backwards compatibility: if <0, use exact raw attention.
            use_exact_attention: If True, bypass cover view and use full raw attention.

        Returns:
            attn_output: [B, L_q, H_q, D] attended values.
            attn_probs: [B, H_q, L_q, T_cover] attention probabilities.
        """
        # Backwards compatibility: threshold < 0 means exact attention
        if threshold is not None and threshold < 0:
            use_exact_attention = True

        # Exact path: skip cover view and run full raw attention
        if use_exact_attention:
            k_raw, v_raw, _, _ = self.raw_cache.get_layer(layer_idx, with_offsets=False)
            if k_raw is None or v_raw is None:
                raise ValueError(f"No raw cache available for layer {layer_idx}.")

            k_full = repeat_kv(k_raw, num_kv_groups)
            v_full = repeat_kv(v_raw, num_kv_groups)

            attn_weights = torch.matmul(query_states, k_full.transpose(2, 3)) * scaling
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask[:, :, :, :k_full.shape[-2]]

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            # Push to buffer for segmentation (all tokens are raw in this path)
            B_raw, T_raw = k_raw.shape[0], k_raw.shape[2]
            cover_indices_raw = torch.arange(T_raw, device=k_raw.device).unsqueeze(0).expand(B_raw, -1)
            cover_is_summary_raw = torch.zeros(B_raw, T_raw, device=k_raw.device, dtype=torch.long)
            self.attn_buffer[layer_idx].push(attn_weights, cover_indices_raw, cover_is_summary_raw)

            attn_output = torch.matmul(attn_weights, v_full)
            return attn_output.transpose(1, 2).contiguous(), attn_weights

        # Get cover view (sliced to valid length, excluding pre-allocated capacity)
        cover_view = self.cover_view[layer_idx]
        cover_k, cover_v, cover_indices, cover_is_summary = cover_view.get_valid_kv()

        if cover_k is None or cover_v is None:
            raise ValueError(f"Cover view not initialized for layer {layer_idx}.")

        B, H_k, T_cover, D = cover_k.shape
        H_q, L_q = query_states.shape[1], query_states.shape[2]

        cover_k_full = repeat_kv(cover_k, num_kv_groups)
        cover_v_full = repeat_kv(cover_v, num_kv_groups)

        # Map summary positions to raw indices (using page_end) for mask lookup
        summary_mask = (cover_is_summary == 1)
        has_summaries = summary_mask.any()

        if has_summaries:
            page_ends = self.summary_cache[layer_idx].page_end
            if page_ends.shape[0] < B:
                pad_b = B - page_ends.shape[0]
                zeros = torch.zeros(pad_b, page_ends.shape[1], device=page_ends.device, dtype=page_ends.dtype)
                page_ends = torch.cat([page_ends, zeros], dim=0)

            clamped_idx = cover_indices.clamp(min=0, max=page_ends.shape[1] - 1)
            summary_raw_pos = page_ends.gather(1, clamped_idx)
            cover_raw_indices = torch.where(summary_mask, summary_raw_pos, cover_indices)
        else:
            # No summaries - cover_indices are already raw indices
            cover_raw_indices = cover_indices

        # Compute attention logits
        attn_logits = torch.matmul(query_states, cover_k_full.transpose(2, 3)) * scaling

        # Save unbiased logits for refinement selection (before bias is added)
        attn_logits_unbiased = attn_logits.clone()

        # Optionally add log bias for summaries to compensate for compression
        # Modes:
        #   - "none": No bias (default)
        #   - "fixed_n": log(N) where N = page length (assumes uniform attention)
        #   - "adaptive_k": log(k_eff) where k_eff = exp(entropy) is effective support
        #                   This adapts to the actual attention distribution at compression time
        log_bias_mode = getattr(self, 'log_bias_mode', 'none')
        # Backwards compat: use_log_bias=True -> "fixed_n"
        if self.use_log_bias and log_bias_mode == 'none':
            log_bias_mode = 'fixed_n'

        if log_bias_mode != 'none' and has_summaries:
            summary_cache = self.summary_cache[layer_idx]
            page_start = summary_cache.page_start
            page_end_cache = summary_cache.page_end

            if page_start.shape[0] < B:
                pad_b = B - page_start.shape[0]
                page_start = torch.cat([page_start, torch.zeros(pad_b, page_start.shape[1], device=page_start.device, dtype=page_start.dtype)], dim=0)
                page_end_cache = torch.cat([page_end_cache, torch.zeros(pad_b, page_end_cache.shape[1], device=page_end_cache.device, dtype=page_end_cache.dtype)], dim=0)

            page_idx = cover_indices.clamp(min=0, max=page_start.shape[1] - 1)

            if log_bias_mode == 'fixed_n':
                # Original: log(N) where N = page length
                page_lengths = (page_end_cache - page_start + 1).float()
                log_bias_values = page_lengths.clamp(min=1).log()
            elif log_bias_mode == 'adaptive_k':
                # Adaptive: use stored log(k_eff) = entropy computed at compression time
                log_eff_support = summary_cache.log_effective_support
                if log_eff_support.shape[0] < B:
                    pad_b = B - log_eff_support.shape[0]
                    log_eff_support = torch.cat([log_eff_support, torch.zeros(pad_b, log_eff_support.shape[1], device=log_eff_support.device, dtype=log_eff_support.dtype)], dim=0)
                log_bias_values = log_eff_support
            else:
                raise ValueError(f"Unknown log_bias_mode: {log_bias_mode}")

            gathered_log_bias = log_bias_values.gather(1, page_idx)
            log_bias = torch.where(summary_mask, gathered_log_bias, torch.zeros_like(gathered_log_bias))
            attn_logits = attn_logits + log_bias[:, None, None, :].to(attn_logits.dtype)

        # Apply causal/padding mask
        mask_value = torch.finfo(attn_logits.dtype).min
        if attention_mask is not None:
            # Clamp to valid mask bounds (attention_mask may not include newly added tokens)
            max_mask_idx = attention_mask.shape[3] - 1
            cover_raw_indices_clamped = cover_raw_indices.clamp(min=0, max=max_mask_idx)
            idx_expanded = cover_raw_indices_clamped[:, None, None, :].expand(-1, 1, L_q, -1)
            cover_mask = attention_mask.gather(3, idx_expanded)
            attn_logits = attn_logits + cover_mask

        # Mask invalid positions (padding in cover view)
        attn_logits = attn_logits.masked_fill(cover_raw_indices[:, None, None, :] < 0, mask_value)

        # Also apply masks to unbiased logits (for refinement selection)
        if attention_mask is not None:
            attn_logits_unbiased = attn_logits_unbiased + cover_mask
        attn_logits_unbiased = attn_logits_unbiased.masked_fill(cover_raw_indices[:, None, None, :] < 0, mask_value)

        # Save logsumexp for corrected refinement (the softmax denominator in log space)
        # Shape: [B, H_q, L_q]
        logsumexp_cover = torch.logsumexp(attn_logits.float(), dim=-1)

        attn_probs = torch.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Compute unbiased probs for refinement selection (EXPERIMENTAL: bias affects output but not selection)
        attn_probs_unbiased = torch.softmax(attn_logits_unbiased, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_probs, cover_v_full)

        # Track summaries seen (even when not refining, for accurate summary_frac)
        # Use tensor accumulation to avoid GPU sync during forward pass
        if has_summaries:
            num_summaries = summary_mask.sum()  # Keep as tensor, no .item()
            self.attn_buffer[layer_idx].add_summaries_seen(num_summaries * query_states.shape[1] * query_states.shape[2])

        # Refinement: descend into pages based on refinement rule
        # Skip if no summaries exist yet, or if refinement rule is NoRefinementRule
        needs_refinement = has_summaries and not isinstance(self.refinement_rule, NoRefinementRule)

        if needs_refinement:
            # Optimized refinement check: only process summary positions
            # Instead of creating full [B, H, L, T_cover] mask, index into summary positions only
            sum_indices = summary_mask.nonzero(as_tuple=True)  # (batch_indices, position_indices) - syncs here
            if len(sum_indices[0]) > 0:  # no additional sync - uses len from nonzero
                b_sum, t_sum = sum_indices
                # Extract attention only at summary positions: [N_sum, H, L]
                # EXPERIMENTAL: Use unbiased probs for selection, so log(k) bias doesn't affect which pages get refined
                attn_at_summaries = attn_probs_unbiased[b_sum, :, :, t_sum]

                # Use refinement rule to select which summaries to refine
                # Pass page_ids so rule can apply always_refine_first_n if configured
                page_ids_at_summaries = cover_indices[b_sum, t_sum]  # [N_sum]
                summary_refine_mask, detail_refine_mask = self.refinement_rule.select(
                    attn_at_summaries, page_ids=page_ids_at_summaries
                )

                # Build refine_positions [B, T_cover] from sparse results
                refine_positions = torch.zeros(B, T_cover, dtype=torch.bool, device=query_states.device)
                refine_positions[b_sum, t_sum] = summary_refine_mask

                # Build per-(head, query) mask from detail mask
                refine_mask = torch.zeros(B, H_q, L_q, T_cover, dtype=torch.bool, device=query_states.device)
                refine_mask[b_sum, :, :, t_sum] = detail_refine_mask
                # Track refinements using tensor accumulation (no GPU sync)
                self.attn_buffer[layer_idx].add_refinements_made(detail_refine_mask.sum())

                # Get positions needing refinement - nonzero syncs, then check len
                b_idx, pos_idx = refine_positions.nonzero(as_tuple=True)
                if len(b_idx) > 0:  # no additional sync - uses len from nonzero
                    summary_cache = self.summary_cache[layer_idx]
                    page_start = summary_cache.page_start
                    page_end = summary_cache.page_end
                    if page_start.shape[0] < B:
                        pad_b = B - page_start.shape[0]
                        zeros = torch.zeros(pad_b, page_start.shape[1], device=page_start.device, dtype=page_start.dtype)
                        page_start = torch.cat([page_start, zeros], dim=0)
                        page_end = torch.cat([page_end, zeros], dim=0)

                    k_raw, v_raw, _, _ = self.raw_cache.get_layer(layer_idx, with_offsets=False)
                    page_ids = cover_indices[b_idx, pos_idx]
                    starts = page_start[b_idx, page_ids]
                    ends = page_end[b_idx, page_ids]
                    lengths = ends - starts + 1

                    # Filter valid entries and check length (no additional sync)
                    valid = (page_ids >= 0) & (ends >= starts)
                    b_idx, pos_idx = b_idx[valid], pos_idx[valid]
                    starts, lengths = starts[valid], lengths[valid]
                    page_ids = page_ids[valid]

                    if len(b_idx) > 0:  # check after filtering
                        # === Corrected refinement with proper normalizer (batched) ===
                        # When we refine summaries, the softmax normalizer changes:
                        # Z_full = Z_cover - sum(exp(logit_s)) + sum(Z_page_s)
                        # All attention must be rescaled by Z_cover / Z_full
                        #
                        # Final output formula:
                        # output = scale * (cover_output - sum(attn_s * v_summary) + sum(ratio_page * raw_out))
                        # where scale = 1 / (1 - sum(attn_s) + sum(ratio_page))

                        N_refine = b_idx.shape[0]
                        # Keep max_page_len as tensor to avoid GPU sync
                        # torch.arange accepts 0-dim tensors in PyTorch 2.x
                        max_page_len = lengths.max()

                        # Gather raw KV for all refined pages (padded to max_page_len)
                        # Build indices: [N_refine, max_page_len]
                        page_offsets = torch.arange(max_page_len, device=query_states.device).unsqueeze(0)  # [1, max_page_len]
                        raw_idx = starts.unsqueeze(1) + page_offsets  # [N_refine, max_page_len]
                        # Clamp to valid range (padding will be masked out)
                        raw_idx = raw_idx.clamp(max=k_raw.shape[2] - 1)

                        # Create padding mask: True where valid, False where padded
                        valid_mask = page_offsets < lengths.unsqueeze(1)  # [N_refine, max_page_len]

                        # Gather K/V: [N_refine, H_k, max_page_len, D]
                        # Use unsqueeze/expand pattern to avoid needing explicit shape in view()
                        gather_idx = raw_idx.unsqueeze(1).unsqueeze(-1).expand(-1, k_raw.shape[1], -1, k_raw.shape[3])
                        k_slice = repeat_kv(torch.gather(k_raw[b_idx], 2, gather_idx), num_kv_groups)
                        v_slice = repeat_kv(torch.gather(v_raw[b_idx], 2, gather_idx), num_kv_groups)

                        # Compute raw attention logits: [N_refine, H_q, L_q, max_page_len]
                        q_refine = query_states[b_idx]  # [N_refine, H_q, L_q, D]
                        raw_logits = torch.matmul(q_refine, k_slice.transpose(2, 3)) * scaling

                        # Apply attention mask if present
                        if attention_mask is not None:
                            mask_gather_idx = raw_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, attention_mask.shape[2], -1)
                            raw_logits = raw_logits + torch.gather(attention_mask[b_idx], 3, mask_gather_idx)

                        # Mask out padded positions before softmax
                        mask_value = torch.finfo(raw_logits.dtype).min
                        raw_logits = raw_logits.masked_fill(~valid_mask[:, None, None, :], mask_value)

                        # Compute logsumexp for page (Z_page in log space)
                        logsumexp_page = torch.logsumexp(raw_logits.float(), dim=-1)  # [N_refine, H_q, L_q]

                        # Softmax and weighted sum
                        raw_probs = torch.softmax(raw_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
                        raw_out = torch.matmul(raw_probs, v_slice)  # [N_refine, H_q, L_q, D]

                        # Get values for summaries being refined
                        v_summary = cover_v_full[b_idx, :, pos_idx, :]  # [N_refine, H_q, D]

                        # Active mask: which (refine_idx, head, query) combos are actually refined
                        active = refine_mask[b_idx, :, :, pos_idx].float()  # [N_refine, H_q, L_q]

                        # Proceed with refinement computation - active.any() check removed to avoid GPU sync
                        # If active is all zeros, the computation still produces correct results (no change)
                        # attn_s: attention to summary
                        attn_s = attn_probs[b_idx, :, :, pos_idx].float()  # [N_refine, H_q, L_q]

                        # ratio_page = Z_page / Z_cover
                        # Clamp to avoid overflow in subsequent operations
                        logsumexp_cover_refine = logsumexp_cover[b_idx]  # [N_refine, H_q, L_q]
                        log_ratio = (logsumexp_page - logsumexp_cover_refine).clamp(min=-80, max=80)
                        ratio_page = torch.exp(log_ratio)  # [N_refine, H_q, L_q]

                        # Apply active mask BEFORE any multiplication to avoid inf * 0 = nan
                        attn_s_active = attn_s * active
                        ratio_page_active = ratio_page * active

                        # Accumulate statistics using scatter_add
                        b_expanded_3d = b_idx.view(-1, 1, 1).expand(-1, H_q, L_q)
                        b_expanded_4d = b_idx.view(-1, 1, 1, 1).expand(-1, H_q, L_q, D)

                        sum_attn_s = torch.zeros(B, H_q, L_q, device=query_states.device, dtype=torch.float32)
                        sum_ratio_page = torch.zeros(B, H_q, L_q, device=query_states.device, dtype=torch.float32)
                        sum_attn_s.scatter_add_(0, b_expanded_3d, attn_s_active)
                        sum_ratio_page.scatter_add_(0, b_expanded_3d, ratio_page_active)

                        # Accumulate weighted outputs
                        sum_ratio_raw = torch.zeros(B, H_q, L_q, D, device=query_states.device, dtype=torch.float32)
                        sum_attn_v = torch.zeros(B, H_q, L_q, D, device=query_states.device, dtype=torch.float32)
                        ratio_raw = (ratio_page_active.unsqueeze(-1) * raw_out).float()
                        attn_v = (attn_s_active.unsqueeze(-1) * v_summary.unsqueeze(2)).float()
                        sum_ratio_raw.scatter_add_(0, b_expanded_4d, ratio_raw)
                        sum_attn_v.scatter_add_(0, b_expanded_4d, attn_v)

                        # Compute scale factor: Z_cover / Z_full = 1 / (1 - sum_attn_s + sum_ratio_page)
                        # IMPORTANT: Keep all computation in float32 to avoid overflow/underflow
                        # when ratio_page is huge (Mean compressor case). The huge values cancel
                        # out mathematically: scale ≈ 1/ratio_page, adjustment ≈ ratio_page * raw_out
                        # so scale * adjustment ≈ raw_out. But float16 conversion causes inf * 0 = nan.
                        denom = 1.0 - sum_attn_s + sum_ratio_page
                        scale = 1.0 / denom.clamp(min=1e-8)  # [B, H_q, L_q], stays float32

                        # Final output: scale * (cover_output - sum_attn_v + sum_ratio_raw)
                        adjustment = sum_ratio_raw - sum_attn_v  # stays float32
                        attn_output = (scale.unsqueeze(-1) * (attn_output.float() + adjustment)).to(attn_output.dtype)

        self.attn_buffer[layer_idx].push(attn_probs, cover_indices, cover_is_summary)
        return attn_output.transpose(1, 2).contiguous(), attn_probs

    def lined_attention(
        self,
        layer_idx: int,
        query_states: torch.Tensor,      # [B, H_q, L_q, D]
        scaling: float,
        num_kv_groups: int,
        attention_mask: Optional[torch.Tensor] = None,  # [B, 1, L_q, T_raw]
        sliding_window: Optional[int] = None,
    ):
        """Run pure lined attention (H2O-style) with grid tokens + raw tail.
        
        Cover = [Grid tokens] + [Raw tail]
        Pure lined attention: no compression/pages, just globally important tokens (grid) + local tail.
        
        Args:
            layer_idx: int
                Transformer layer index.
            query_states: torch.Tensor
                [B, H_q, L_q, D] query projections for this layer.
            scaling: float
                Attention scaling factor (typically 1/sqrt(D)).
            num_kv_groups: int
                H_q // H_k; used to expand KV heads for grouped-query attention.
            attention_mask: Optional[torch.Tensor]
                [B, 1, L_q, T_raw] causal + padding mask aligned to raw indices.
            sliding_window: Optional[int]
                If using sliding window attention; retained for parity with upstream API.
        
        Returns:
            attn_output: torch.Tensor
                [B, H_q, L_q, D] attended values.
            attn_probs: torch.Tensor
                [B, H_q, L_q, T_cover] attention probabilities over cover.
        """
        B, H_q, L_q, D = query_states.shape
        device = query_states.device
        
        # --- 1. Get raw cache and apply warmup guard ---
        k_raw, v_raw, seq_start, raw_seq_start = self.raw_cache.get_layer(layer_idx, with_offsets=True)
        if k_raw is None or v_raw is None:
            raise ValueError(f"No raw cache available for layer {layer_idx}.")
        
        B_raw, H_k, T_raw, D_k = k_raw.shape
        
        # H2O-ish warmup: don't approximate until the sequence is long enough
        if T_raw < self.min_lined_seq_len:
            # Fall back to exact/top-down attention
            return self.top_down_attention(
                layer_idx=layer_idx,
                query_states=query_states,
                scaling=scaling,
                num_kv_groups=num_kv_groups,
                attention_mask=attention_mask,
                sliding_window=sliding_window,
                threshold=-1.0,   # force exact raw attention path
            )
        
        # --- 2. Get grid tokens from GridCache ---
        grid_cache = self.grid_cache[layer_idx]
        if grid_cache.keys is None or grid_cache.lens is None:
            grid_cache.initialize(B_raw, H_k, D_k, k_raw.device, k_raw.dtype)
            # Initialize grid with prefix anchor to prevent bootstrapping failure
            # Use min(32, grid_top_k) to ensure prefix fits within grid capacity
            prefix_len = min(32, self.grid_top_k, T_raw)
            if T_raw >= prefix_len and prefix_len > 0:
                prefix_indices = torch.arange(prefix_len, device=k_raw.device, dtype=torch.long)
                prefix_k = k_raw[:, :, :prefix_len, :]  # [B, H_k, prefix_len, D]
                prefix_v = v_raw[:, :, :prefix_len, :]
                # Set grid to prefix tokens (will be expanded/refreshed later)
                grid_cache.set_tokens(
                    keys=prefix_k,
                    values=prefix_v,
                    indices=prefix_indices.unsqueeze(0).expand(B_raw, -1),  # [B, prefix_len]
                )
                if self.debug and layer_idx == 0:
                    print(f"[Layer {layer_idx}] Grid initialized with prefix anchor: first {prefix_len} tokens (grid_top_k={self.grid_top_k})")
        
        # Ensure grid is properly initialized (safety check)
        if grid_cache.keys is None or grid_cache.lens is None:
            raise ValueError(f"Grid cache not initialized for layer {layer_idx}")
        
        grid_indices = grid_cache.indices  # [B, G_max]
        grid_k = grid_cache.keys           # [B, H_k, G_max, D_k]
        grid_v = grid_cache.values
        grid_lens = grid_cache.lens        # [B]
        
        # Debug: verify grid has tokens
        if self.debug and layer_idx == 0:
            grid_len_actual = grid_lens[0].item() if grid_lens.numel() > 0 else 0
            if grid_len_actual == 0:
                print(f"[Layer {layer_idx}] WARNING: Grid has 0 tokens! T_raw={T_raw}, min_lined_seq_len={self.min_lined_seq_len}")
        
        # --- 3. Build raw tail with a H2O-ish window ---
        window = getattr(self.segmenter, 'tail_len', 16)
        # Ensure the local window is reasonably large
        window = max(window, self.min_lined_tail_window)
        
        # Build tail indices per batch
        
        batched_tail_indices = []
        batched_tail_k = []
        batched_tail_v = []
        batched_tail_lens = []
        
        for b in range(B_raw):
            if raw_seq_start is not None:
                base_start = raw_seq_start[b].item()
            else:
                base_start = 0
            
            # Sliding *local* window: last `window` tokens
            # CRITICAL: Build tail from T_raw (raw cache), not T_mask (mask size)
            # We'll pad the mask later to match T_raw, so tail indices will be valid
            # The tail should ALWAYS be included in the cover (grid + tail)
            tail_start = max(base_start, T_raw - window)
            tail_start = min(tail_start, T_raw)
            tail_len = T_raw - tail_start
            tail_end = T_raw
            
            # Force tail to always include at least some recent tokens
            # This ensures the model can always attend to recent context
            if tail_len > 0:
                tail_indices = torch.arange(tail_start, tail_end, device=device, dtype=torch.long)
                # Mask padding
                if seq_start is not None:
                    pad_end = seq_start[b].item()
                    is_pad = tail_indices < pad_end
                    tail_indices[is_pad] = -1
                
                tail_k = k_raw[b, :, tail_start:tail_end, :]  # [H_k, tail_len, D]
                tail_v = v_raw[b, :, tail_start:tail_end, :]
            else:
                tail_indices = torch.empty(0, dtype=torch.long, device=device)
                tail_k = torch.empty(H_k, 0, D_k, device=device, dtype=k_raw.dtype)
                tail_v = torch.empty(H_k, 0, D_k, device=device, dtype=v_raw.dtype)
            
            batched_tail_indices.append(tail_indices)
            batched_tail_k.append(tail_k)
            batched_tail_v.append(tail_v)
            batched_tail_lens.append(len(tail_indices))
        
        # --- 3. Concatenate grid + tail per batch (with deduplication) ---
        batched_cover_k = []
        batched_cover_v = []
        batched_cover_indices = []
        batched_cover_lens = []

        for b in range(B_raw):
            # Grid part
            g_len = int(grid_lens[b].item()) if b < grid_lens.shape[0] else 0
            if g_len > 0 and b < grid_k.shape[0]:
                g_k = grid_k[b, :, :g_len, :]  # [H_k, G, D]
                g_v = grid_v[b, :, :g_len, :]
                g_idx = grid_indices[b, :g_len]  # [G]
            else:
                g_k = torch.empty(H_k, 0, D_k, device=device, dtype=k_raw.dtype)
                g_v = torch.empty(H_k, 0, D_k, device=device, dtype=v_raw.dtype)
                g_idx = torch.empty(0, dtype=torch.long, device=device)
                g_len = 0

            # Tail part - filter out indices already in grid to avoid duplicates
            tail_k_orig = batched_tail_k[b]
            tail_v_orig = batched_tail_v[b]
            tail_idx_orig = batched_tail_indices[b]

            # Deduplicate: remove tail positions that are already in grid
            if g_len > 0 and tail_idx_orig.numel() > 0:
                # Create mask for tail indices NOT in grid (also keep padding indices -1)
                # Use broadcasting: tail_idx_orig[:, None] vs g_idx[None, :]
                in_grid = (tail_idx_orig.unsqueeze(1) == g_idx.unsqueeze(0)).any(dim=1)  # [tail_len]
                keep_mask = ~in_grid | (tail_idx_orig < 0)  # Keep if not in grid OR is padding

                if not keep_mask.all():
                    # Filter tail to remove duplicates
                    keep_positions = keep_mask.nonzero(as_tuple=True)[0]
                    tail_idx = tail_idx_orig[keep_positions]
                    # Gather KV at kept positions: tail_k_orig is [H_k, tail_len, D]
                    tail_k = tail_k_orig[:, keep_positions, :]
                    tail_v = tail_v_orig[:, keep_positions, :]

                    if self.debug and layer_idx == 0:
                        num_removed = (~keep_mask).sum().item()
                        print(f"[Layer {layer_idx}] Dedup: removed {num_removed} tail tokens already in grid")
                else:
                    tail_k = tail_k_orig
                    tail_v = tail_v_orig
                    tail_idx = tail_idx_orig
            else:
                tail_k = tail_k_orig
                tail_v = tail_v_orig
                tail_idx = tail_idx_orig
            
            # CRITICAL FIX #4: Verify concat dimension is correct (unambiguous check)
            # Handle both 3D [H, S, D] and 4D [B, H, S, D] cases
            assert g_k.ndim in (3, 4), f"Unexpected grid_k ndim: {g_k.ndim}"
            assert tail_k.ndim in (3, 4), f"Unexpected tail_k ndim: {tail_k.ndim}"
            assert g_k.ndim == tail_k.ndim, f"Dimension mismatch: grid_k.ndim={g_k.ndim} vs tail_k.ndim={tail_k.ndim}"
            
            # Verify head and hidden dimensions match
            assert g_k.shape[-3] == tail_k.shape[-3] == H_k, f"Head dimension mismatch: {g_k.shape[-3]} vs {tail_k.shape[-3]} vs {H_k}"
            assert g_k.shape[-1] == tail_k.shape[-1] == D_k, f"Hidden dimension mismatch: {g_k.shape[-1]} vs {tail_k.shape[-1]} vs {D_k}"
            
            # Determine sequence dimension based on ndim
            if g_k.ndim == 4:  # [B, H, S, D] - sequence is dim=2
                seq_dim = 2
                expected_seq_len = g_k.shape[2] + tail_k.shape[2]
            else:  # [H, S, D] - sequence is dim=1
                seq_dim = 1
                expected_seq_len = g_k.shape[1] + tail_k.shape[1]
            
            # Concatenate along sequence dimension
            cover_k = torch.cat([g_k, tail_k], dim=seq_dim)
            cover_v = torch.cat([g_v, tail_v], dim=seq_dim)
            cover_idx = torch.cat([g_idx, tail_idx], dim=0)  # [G + tail_len]
            
            # Sanity check: verify the concatenated shape
            if cover_k.ndim == 4:
                assert cover_k.shape[2] == expected_seq_len, f"Cover shape mismatch: got {cover_k.shape}, expected seq_len={expected_seq_len} at dim=2"
            else:
                assert cover_k.shape[1] == expected_seq_len, f"Cover shape mismatch: got {cover_k.shape}, expected seq_len={expected_seq_len} at dim=1"
            
            batched_cover_k.append(cover_k)
            batched_cover_v.append(cover_v)
            batched_cover_indices.append(cover_idx)
            batched_cover_lens.append(cover_k.shape[1])
        
        # Pad and stack
        max_cover_len = max(batched_cover_lens) if batched_cover_lens else 0
        cover_k = torch.zeros(B_raw, H_k, max_cover_len, D_k, device=device, dtype=k_raw.dtype)
        cover_v = torch.zeros(B_raw, H_k, max_cover_len, D_k, device=device, dtype=v_raw.dtype)
        cover_indices = torch.full((B_raw, max_cover_len), -1, dtype=torch.long, device=device)
        
        for b in range(B_raw):
            l = batched_cover_lens[b]
            if l > 0:
                cover_k[b, :, :l, :] = batched_cover_k[b]
                cover_v[b, :, :l, :] = batched_cover_v[b]
                cover_indices[b, :l] = batched_cover_indices[b]
        
        # Expand KV heads for grouped-query attention
        cover_k_full = repeat_kv(cover_k, num_kv_groups)  # [B, H_q, T_cover, D]
        cover_v_full = repeat_kv(cover_v, num_kv_groups)  # [B, H_q, T_cover, D]
        
        # --- 4. Run attention on cover ---
        # CRITICAL FIX #5: Add diagnostics for unique keys and unmasked keys
        if self.debug and layer_idx == 0:
            # Count unique cover positions (excluding -1 padding)
            valid_cover_indices = cover_indices[cover_indices >= 0]
            num_unique_cover = torch.unique(valid_cover_indices).numel() if valid_cover_indices.numel() > 0 else 0
            print(f"[Layer {layer_idx}] Cover construction: T_cover={max_cover_len}, num_grid={grid_lens[0].item() if grid_lens.numel() > 0 else 0}, tail_len={batched_tail_lens[0] if batched_tail_lens else 0}, unique_positions={num_unique_cover}")
        
        attn_logits = torch.matmul(query_states, cover_k_full.transpose(2, 3)) * scaling  # [B, H_q, L_q, T_cover]
        mask_value = torch.finfo(attn_logits.dtype).min
        
        if attention_mask is not None:
            # Map cover indices to raw indices for masking
            # For grid tokens, use their stored indices; for tail, use tail indices
            cover_raw_indices = cover_indices  # [B, T_cover] raw KV positions, -1 for padding
            B = cover_raw_indices.shape[0]
            T_mask = attention_mask.shape[-1]
            
            # CRITICAL FIX: Ensure attention_mask covers the current KV cache length
            # NOTE: attention_mask last dim must index RAW KV positions.
            # During generation, T_raw grows but attention_mask may be stuck at initial size.
            if T_mask < T_raw:
                # Pad new KV positions as UNMASKED (0.0). This fixes the OOB gather.
                # For generation (L_q == 1), padding with zeros is correct.
                # For prefill (L_q > 1), we may need to rebuild/extend causal masking
                # rather than just padding, but in practice generation is the common path
                # where this bug manifests.
                pad = (0, T_raw - T_mask)  # Pad last dimension: (pad_left, pad_right)
                T_mask_old = T_mask  # Save old size for debug assert
                attention_mask = torch.nn.functional.pad(attention_mask, pad, value=0.0)
                T_mask = attention_mask.shape[-1]
                
                # Debug assert: verify padded slice is finite/zero (keep assertion, but only print in debug mode)
                padded_slice = attention_mask[:, :, :, T_mask_old:]  # Get the newly padded portion
                assert torch.isfinite(padded_slice).all(), "Padded attention_mask slice contains non-finite values"
                assert (padded_slice == 0.0).all(), f"Padded attention_mask slice is not zero: min={padded_slice.min()}, max={padded_slice.max()}"
            
            # Assert raw indices are now valid (except padding)
            valid = cover_raw_indices >= 0
            if valid.any():
                max_idx = cover_raw_indices[valid].max().item()
                if self.debug and layer_idx == 0:
                    min_idx = cover_raw_indices[valid].min().item()
                    num_valid = valid.sum().item()
                    num_padding = (~valid).sum().item()
                    print(f"[Layer {layer_idx}] Before mask: T_raw={T_raw}, T_mask={T_mask}, T_cover={max_cover_len}")
                    print(f"[Layer {layer_idx}] cover_raw_indices: min={min_idx}, max={max_idx}, num_valid={num_valid}, num_padding={num_padding}")
                
                # CRITICAL ASSERTION: After padding, indices must be valid
                assert max_idx < T_mask, f"cover_raw_indices max {max_idx} >= T_mask {T_mask} (after padding)"
                assert max_idx < T_raw, f"cover_raw_indices max {max_idx} >= T_raw {T_raw}"
            else:
                if self.debug and layer_idx == 0:
                    print(f"[Layer {layer_idx}] WARNING: All cover_raw_indices are padding (-1)!")
                raise AssertionError("All cover_raw_indices are padding (-1); no keys to attend to.")
            
            # Gather the mask using RAW indices (no clamping to T_mask!)
            # Only clamp -1 to 0 to avoid negative indices in gather; we'll mask padding separately
            idx = cover_raw_indices.clamp(min=0)  # only to avoid -1 in gather
            idx_expanded = idx[:, None, None, :].expand(-1, 1, L_q, -1)  # [B, 1, L_q, T_cover]
            cover_mask = attention_mask.gather(3, idx_expanded)
            
            # Explicitly mask padding positions (-1)
            pad_positions = (cover_raw_indices < 0)
            if pad_positions.any():
                cover_mask = cover_mask.masked_fill(pad_positions[:, None, None, :], mask_value)
            
            # Debug output
            if self.debug and layer_idx == 0:
                num_unmasked = (cover_mask > mask_value).sum().item()
                T_cover_actual = cover_mask.shape[-1]
                print(f"[Layer {layer_idx}] Attention mask: num_unmasked_keys={num_unmasked}, "
                      f"T_cover={T_cover_actual}, T_mask={T_mask}, T_raw={T_raw}")
            
            attn_logits = attn_logits + cover_mask
        
        # Mask out invalid cover indices (padding)
        attn_logits = attn_logits.masked_fill(cover_indices[:, None, None, :] < 0, mask_value)
        
        attn_probs = torch.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_probs, cover_v_full)  # [B, H_q, L_q, D]
        
        # --- 5. Update grid scores and refresh grid tokens ---
        # Push to buffer (still useful for debugging/stats if needed)
        cover_is_summary = torch.zeros_like(cover_indices)  # All raw tokens in lined attention
        self.attn_buffer[layer_idx].push(attn_probs, cover_indices, cover_is_summary)
        
        # Update grid scores *from this step's attention*
        self._update_grid_scores(layer_idx, attn_probs, cover_indices, cover_is_summary)
        
        # Refresh grid tokens periodically
        self.grid_step_counters[layer_idx] += 1
        if self.grid_update_interval > 0 and (self.grid_step_counters[layer_idx] % self.grid_update_interval) == 0:
            self._refresh_grid_tokens(layer_idx)
        
        return attn_output.transpose(1, 2).contiguous(), attn_probs

    def print_stats(self, layer_idx: int):
        """Print debug stats for raw cache, pages, cover view, and attention buffer."""
        stats = self.attn_buffer[layer_idx].get_stats()
        if stats:
            print(f"DEBUG: Layer {layer_idx} Stats: Summaries={stats['num_summaries_current']}, Total Seen={stats['total_summaries_seen']}, Refinements={stats['total_refinements_made']}, Rate={stats['refinement_rate']:.4f}")
        print(f"\n--- Layer {layer_idx} Stats ---")
        
        # RAW_KV_STATS
        k_raw, _, seq_start, raw_seq_start = self.raw_cache.get_layer(layer_idx, with_offsets=True)
        print("RAW_KV_STATS:")
        if k_raw is not None:
            print(f"- shape of kv cache: {k_raw.shape}")
            print(f"- start of sequence: {seq_start}")
            print(f"- start of raw tail: {raw_seq_start}")
        else:
            print("- shape of kv cache: None")

        # PAGES_STATS
        summary_cache = self.summary_cache[layer_idx]
        print("PAGES_STATS:")
        if summary_cache.keys is not None:
            print(f"- shape of pages: {summary_cache.keys.shape}")
            print(f"- number of pages per: {summary_cache.page_lens}")
            
            starts = summary_cache.page_start
            ends = summary_cache.page_end
            lens = summary_cache.page_lens
            
            print("- pages (start, end):")
            for b in range(starts.shape[0]):
                num_pages = lens[b].item() if b < len(lens) else 0
                pairs = []
                for p in range(num_pages):
                    pairs.append((starts[b, p].item(), ends[b, p].item()))
                print(f"  Batch {b}: {pairs}")
        else:
            print("- shape of pages: None")
            print("- number of pages per: None")

        # COVER_STATS
        cover_view = self.cover_view[layer_idx]
        print("COVER_STATS:")
        if cover_view.cover_keys is not None:
            print(f"- shape of cover: {cover_view.cover_keys.shape}")
            print(f"- start of sequence: {cover_view.seq_start}")
            print(f"- start of raw tail: {cover_view.raw_seq_start}")
        else:
            print("- shape of cover: None")

    def verify_invariants(self, layer_idx: int):
        """
        Check consistency between raw_cache, summary_cache, cover_view, and attn_buffer.
        
        This ensures that:
        1. RawCache and CoverView agree on sequence boundaries (padding, raw tail).
        2. CoverView's raw tail correctly maps to valid indices in RawCache.
        3. SummaryCache's pages are within valid bounds of RawCache.
        4. AttentionScoreBuffer's dimensions and metadata align with CoverView.
        """
        raw_cache = self.raw_cache
        summary_cache = self.summary_cache[layer_idx]
        cover_view = self.cover_view[layer_idx]
        attn_buffer = self.attn_buffer[layer_idx]

        # Skip cover view checks for lined attention layers (they build cover dynamically)
        is_lined_layer = self.use_lined_attention and layer_idx in self.lined_layers

        # 1. RawCache vs CoverView Alignment (skip for lined attention)
        k_raw, _, seq_start, raw_seq_start = raw_cache.get_layer(layer_idx, with_offsets=True)
        if k_raw is None:
            return # Not initialized yet
        
        # Get batch size (needed for both lined and top-down checks)
        B = k_raw.shape[0]
        
        if not is_lined_layer:
            if cover_view.cover_keys is None:
                return # Not initialized yet

            # Verify seq_start (left padding) matches
            if seq_start is not None and cover_view.seq_start is not None:
                assert torch.equal(seq_start, cover_view.seq_start), \
                    f"Invariant Violation: Raw seq_start {seq_start} != Cover seq_start {cover_view.seq_start}"
            
            # Verify raw_seq_start (frontier) matches
            if raw_seq_start is not None and cover_view.raw_seq_start is not None:
                assert torch.equal(raw_seq_start, cover_view.raw_seq_start), \
                    f"Invariant Violation: Raw raw_seq_start {raw_seq_start} != Cover raw_seq_start {cover_view.raw_seq_start}"

            # Verify CoverView indices map correctly to RawCache for the tail
            # The tail starts at raw_seq_start and goes to the end.
            # In CoverView, this corresponds to where cover_is_summary == 0
            for b in range(B):
                if raw_seq_start is None: continue
                
                frontier = raw_seq_start[b].item()
                cover_indices_b = cover_view.cover_indices[b]
                cover_is_sum_b = cover_view.cover_is_summary[b]
                
                # Find raw tail in cover
                # It should be where indices >= frontier (and not padding -1)
                # And is_summary should be 0
                
                raw_mask = (cover_is_sum_b == 0) & (cover_indices_b != -1)
                if raw_mask.any():
                    raw_indices = cover_indices_b[raw_mask]
                    # Check they are valid raw indices
                    assert raw_indices.min() >= 0, f"Invariant Violation: Negative raw index in cover view batch {b}"
                    assert raw_indices.max() < k_raw.shape[2], f"Invariant Violation: Raw index {raw_indices.max()} out of bounds {k_raw.shape[2]}"

        # 2. SummaryCache vs RawCache Alignment
        if summary_cache.keys is not None:
            page_start = summary_cache.page_start
            page_end = summary_cache.page_end
            page_frontier = summary_cache.page_frontier
            
            # Check page ranges are within raw bounds
            # We only check valid pages (up to page_lens)
            for b in range(min(B, summary_cache.page_lens.shape[0])):
                num_pages = summary_cache.page_lens[b].item()
                if num_pages > 0:
                    s = page_start[b, :num_pages]
                    e = page_end[b, :num_pages]
                    assert torch.all(s <= e), f"Invariant Violation: Page start > end in batch {b}"
                    assert s.min() >= 0, f"Invariant Violation: Negative page start in batch {b}"
                    assert e.max() < k_raw.shape[2], f"Invariant Violation: Page end {e.max()} out of bounds {k_raw.shape[2]}"
            
            # Verify page_frontier aligns with raw_seq_start
            # page_frontier should be <= raw_seq_start (usually equal if up to date)
            if raw_seq_start is not None and page_frontier is not None:
                # We need to handle batch size mismatch if summary cache is lazy
                min_b = min(B, page_frontier.shape[0])
                assert torch.all(page_frontier[:min_b] <= raw_seq_start[:min_b]), \
                    f"Invariant Violation: Summary frontier {page_frontier[:min_b]} > Raw frontier {raw_seq_start[:min_b]}"

        # 3. AttentionScoreBuffer Alignment
        # Skip this check for lined attention layers since they build cover dynamically
        # and don't maintain a persistent cover_view
        is_lined_layer = self.use_lined_attention and layer_idx in self.lined_layers
        
        attn_weights, attn_indices, attn_is_sum = attn_buffer.get_data()
        if attn_weights is not None and not is_lined_layer:
            # Verify width matches cover view
            # Note: attn_buffer accumulates, cover_view grows. They should match in T dimension.

            if cover_view.cover_keys is not None:
                T_cover = cover_view.length  # Use valid length, not pre-allocated capacity
                T_attn = attn_weights.shape[-1]

                assert T_attn == T_cover, f"Invariant Violation: Attn buffer width {T_attn} != Cover view length {T_cover}"

                # Verify indices match (use sliced valid views)
                if attn_indices is not None:
                     # attn_indices: [B, T]
                     # cover_view.cover_indices: [B, T]
                     _, _, valid_indices, valid_is_sum = cover_view.get_valid_kv()
                     min_b = min(attn_indices.shape[0], valid_indices.shape[0])
                     assert torch.equal(attn_indices[:min_b], valid_indices[:min_b]), \
                         "Invariant Violation: Attn buffer indices mismatch with CoverView indices"
                     assert torch.equal(attn_is_sum[:min_b], valid_is_sum[:min_b]), \
                         "Invariant Violation: Attn buffer is_summary mismatch with CoverView"

    def print_layer_summary(self, layer_idx: int):
        """
        Print a concise summary of raw, cover, page, and attention buffer state for a layer.

        Args:
            layer_idx: int
                Transformer layer index to report.
        """
        raw_k, _, seq_start, raw_seq_start = self.raw_cache.get_layer(layer_idx, with_offsets=True)
        summary_cache = self.summary_cache[layer_idx]
        cover_view = self.cover_view[layer_idx]
        attn_buf = self.attn_buffer[layer_idx]

        def fmt_pages(b: int) -> str:
            if summary_cache.keys is None:
                return "[]"
            if b >= summary_cache.page_lens.shape[0]:
                return "[]"
            num = int(summary_cache.page_lens[b].item())
            starts = summary_cache.page_start[b, :num].tolist()
            ends = summary_cache.page_end[b, :num].tolist()
            return "[" + ", ".join(f"[{s}, {e}]" for s, e in zip(starts, ends)) + "]"

        # Raw token summary
        if raw_k is not None:
            B, _, T_raw, _ = raw_k.shape
            if self.debug:
                print(f"[Layer {layer_idx}] Raw tokens:")
            for b in range(B):
                if self.debug:
                    pad = int(seq_start[b].item()) if seq_start is not None else 0
                    num_pages = int(summary_cache.page_lens[b].item()) if summary_cache.keys is not None else 0
                    segments = fmt_pages(b)
                    print(f"  Batch {b}: pad={pad}, pages={num_pages}, segments={segments}, total_tokens={T_raw}")
        else:
            if self.debug:
                print(f"[Layer {layer_idx}] Raw tokens: <empty>")

        # Cover token summary
        if cover_view.cover_indices is not None:
            cover_idx = cover_view.cover_indices
            cover_is_sum = cover_view.cover_is_summary
            B = cover_idx.shape[0]
            if self.debug:
                print(f"[Layer {layer_idx}] Cover view:")
            for b in range(B):
                if self.debug:
                    pad = int((cover_idx[b] == -1).sum().item())
                    pages = int(((cover_is_sum[b] == 1) & (cover_idx[b] >= 0)).sum().item())
                    raw_len = int(((cover_is_sum[b] == 0) & (cover_idx[b] >= 0)).sum().item())
                    print(f"  Batch {b}: pad={pad}, pages={pages}, raw_len={raw_len}")
        else:
            if self.debug:
                print(f"[Layer {layer_idx}] Cover view: <empty>")

        # Page summary
        if summary_cache.keys is not None:
            lens = summary_cache.page_lens
            if self.debug:
                print(f"[Layer {layer_idx}] Pages per batch: {[int(x) for x in lens.tolist()]}")
        else:
            if self.debug:
                print(f"[Layer {layer_idx}] Pages per batch: <empty>")

        # Attention score buffer summary
        attn_weights, buf_idx, buf_is_sum = attn_buf.get_data()
        if attn_weights is not None:
            B, H, L_accum, T = attn_weights.shape
            if self.debug:
                print(f"[Layer {layer_idx}] Attention buffer: shape={attn_weights.shape}")
            if buf_idx is not None and buf_is_sum is not None:
                for b in range(B):
                    if self.debug:
                        pad = int((buf_idx[b] == -1).sum().item())
                        pages = int(((buf_is_sum[b] == 1) & (buf_idx[b] >= 0)).sum().item())
                        raw_len = int(((buf_is_sum[b] == 0) & (buf_idx[b] >= 0)).sum().item())
                        print(f"  Batch {b}: pad={pad}, pages={pages}, raw_len={raw_len}, attn_shape={(H, L_accum, T)}")
        else:
            print(f"[Layer {layer_idx}] Attention buffer: <empty>")

    def get_refinement_stats(self, layer_idx: Optional[int] = None) -> dict:
        """Get refinement statistics for one or all layers.

        Args:
            layer_idx: If provided, get stats for this layer only.
                       If None, aggregate across all layers.

        Returns:
            dict with:
                - num_summaries_current: Total summary positions across layers
                - total_summaries_seen: Cumulative summary × head × query combinations
                - total_refinements_made: Cumulative refinement count
                - refinement_rate: Overall fraction triggering refinement
                - total_queries_processed: Query tokens processed
                - refinements_per_query: Average refinements per query
                - total_pages: Total pages across all layers
                - per_layer: (if layer_idx is None) dict of per-layer stats
        """
        if layer_idx is not None:
            stats = self.attn_buffer[layer_idx].get_stats()
            # Add page count
            summary_cache = self.summary_cache[layer_idx]
            if summary_cache.page_lens is not None:
                stats["total_pages"] = int(summary_cache.page_lens.sum().item())
            else:
                stats["total_pages"] = 0
            return stats

        # Aggregate across all layers
        total_summaries_seen = 0
        total_refinements_made = 0
        total_queries = 0
        num_summaries_current = 0
        total_pages = 0
        per_layer = {}

        for idx in range(self.num_layers):
            layer_stats = self.attn_buffer[idx].get_stats()
            if not layer_stats:
                continue

            total_summaries_seen += layer_stats.get("total_summaries_seen", 0)
            total_refinements_made += layer_stats.get("total_refinements_made", 0)
            total_queries += layer_stats.get("total_queries_processed", 0)
            num_summaries_current += layer_stats.get("num_summaries_current", 0)

            summary_cache = self.summary_cache[idx]
            if summary_cache.page_lens is not None:
                pages = int(summary_cache.page_lens.sum().item())
                total_pages += pages
                layer_stats["total_pages"] = pages

            per_layer[idx] = layer_stats

        refinement_rate = total_refinements_made / total_summaries_seen if total_summaries_seen > 0 else 0.0
        refinements_per_query = total_refinements_made / total_queries if total_queries > 0 else 0.0

        # Average across layers for more intuitive metrics
        avg_pages = total_pages / self.num_layers if self.num_layers > 0 else 0
        avg_summaries = num_summaries_current / self.num_layers if self.num_layers > 0 else 0
        avg_queries = total_queries / self.num_layers if self.num_layers > 0 else 0

        return {
            "num_summaries_current": num_summaries_current,
            "avg_summaries_per_layer": avg_summaries,
            "total_summaries_seen": total_summaries_seen,
            "total_refinements_made": total_refinements_made,
            "refinement_rate": refinement_rate,
            "total_queries_processed": total_queries,
            "avg_queries_per_layer": avg_queries,
            "refinements_per_query": refinements_per_query,
            "total_pages_all_layers": total_pages,
            "avg_pages_per_layer": avg_pages,
            "num_layers": self.num_layers,
            "per_layer": per_layer,
        }

    def reset_refinement_stats(self, layer_idx: Optional[int] = None):
        """Reset refinement counters for one or all layers.

        Args:
            layer_idx: If provided, reset only this layer. Otherwise reset all.
        """
        if layer_idx is not None:
            self.attn_buffer[layer_idx].total_summaries_seen = 0
            self.attn_buffer[layer_idx].total_refinements_made = 0
        else:
            for idx in range(self.num_layers):
                self.attn_buffer[idx].total_summaries_seen = 0
                self.attn_buffer[idx].total_refinements_made = 0
