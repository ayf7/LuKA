import torch
import torch.nn as nn
from modeling.segmenter import Segmenter, DummySegmenter
from modeling.compressor import Compressor, MeanCompressor
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

        if layer_idx == 0:
            # print(self.seq_start[layer_idx])
            # print(self.raw_seq_start[layer_idx])
            pass
            
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

    def add_pages(self,
        keys: torch.Tensor,
        values: torch.Tensor,
        batch_nums: torch.Tensor,
        page_start: torch.Tensor,
        page_end: torch.Tensor,
        page_frontier: Optional[torch.Tensor] = None,
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

        # Insert new pages
        # We can't do a simple slice assignment because `current_lens` varies per batch.
        # We have to scatter or loop.
        # Since L_new is usually small (often 1), loop might be okay?
        # Or we can use `scatter`.
        # But `keys` is [B_new, H, L_new, D].
        # We want to place `keys[i]` at `self.keys[batch_nums[i], :, current_lens[i]:current_lens[i]+L_new, :]`.
        
        # Let's loop for safety and readability for now.
        for i, b_idx in enumerate(batch_nums.tolist()):
            start_col = int(current_lens[i].item())
            end_col = start_col + L_new
            
            self.keys[b_idx, :, start_col:end_col, :] = keys[i]
            self.values[b_idx, :, start_col:end_col, :] = values[i]
            self.page_start[b_idx, start_col:end_col] = page_start[i]
            self.page_end[b_idx, start_col:end_col] = page_end[i]
            
        # Update lengths and frontier
        self.page_lens.index_add_(0, batch_nums, torch.full((B_new,), L_new, dtype=torch.long, device=device))
        
        if page_frontier is not None:
            self.page_frontier[batch_nums] = page_frontier

class CoverView:

    def __init__(self):
        """Hybrid view combining summary pages with trailing raw tokens.

        The cover view is what top-down attention should run against:
        [ summaries (paged region) ] + [ raw tail ]. It is left indexed to make
        incremental decoding updates cheap when no new pages are formed.
        """
        self.cover_keys = None      # [B, H, max(L_num_pages + L_raw_tokens), D]
        self.cover_values = None    # [B, H, max(L_num_pages + L_raw_tokens), D]
        self.seq_start = None       # [B] left-padding offset, mirrors raw cache
        self.raw_seq_start = None   # [B] first raw token index within the cover
        # Metadata to map cover positions back to sources
        self.cover_indices = None   # [B, max(L_num_pages + L_raw_tokens)] raw index or summary idx
        self.cover_is_summary = None # [B, max(L_num_pages + L_raw_tokens)] 1 if summary, else 0
    
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

        # Share references initially for efficiency
        self.cover_keys = k_raw
        self.cover_values = v_raw
        self.seq_start = seq_start
        self.raw_seq_start = raw_seq_start
        
        B, H, T_raw, D = k_raw.shape
        device = k_raw.device
        
        # Initialize indices: all raw
        # cover_indices[b, t] = t (absolute raw index)
        self.cover_indices = torch.arange(T_raw, device=device).unsqueeze(0).expand(B, -1).clone()
        
        # Mask padding with -1
        if seq_start is not None:
            # seq_start: [B]
            # Create mask: t < seq_start[b]
            t_indices = torch.arange(T_raw, device=device).unsqueeze(0) # [1, T]
            is_pad = t_indices < seq_start.unsqueeze(1) # [B, T]
            self.cover_indices[is_pad] = -1

        self.cover_is_summary = torch.zeros(B, T_raw, dtype=torch.long, device=device)

    def update(self,
        keys: torch.Tensor,
        values: torch.Tensor,
        indices: torch.Tensor
    ):
        """Append raw tokens into the cover view during decoding.

        Args:
            keys: torch.Tensor
                [B, H, L_new, D] raw keys for new tokens (typically L_new == 1).
            values: torch.Tensor
                [B, H, L_new, D] raw values for new tokens.
            indices: torch.Tensor
                [B, L_new] raw indices for the new tokens.

        Returns:
            None; should mutate `cover_keys/cover_values` by concatenating the
            raw tail and adjust `raw_seq_start` as the frontier advances.
        """
        if self.cover_keys is None:
            # Should have been initialized.
            raise ValueError("CoverView.update called before initialize()")

        # Invariants:
        # 1. Keys and Values must match shape [B, H, L_new, D]
        # 2. Indices must match the length of new tokens L_new
        assert keys.shape == values.shape, f"Invariant Violation: Keys/Values shape mismatch: {keys.shape} vs {values.shape}"
        assert indices.shape[1] == keys.shape[2], f"Invariant Violation: Indices length {indices.shape[1]} != keys length {keys.shape[2]}"


        # Append
        self.cover_keys = torch.cat([self.cover_keys, keys], dim=2)
        self.cover_values = torch.cat([self.cover_values, values], dim=2)
        
        B, H, L_new, D = keys.shape
        
        # Update metadata
        new_is_summary = torch.zeros(B, L_new, dtype=torch.long, device=keys.device)
        self.cover_is_summary = torch.cat([self.cover_is_summary, new_is_summary], dim=1)
        
        # Update indices
        self.cover_indices = torch.cat([self.cover_indices, indices], dim=1)


    def update_cover_view(self,
        layer_idx: int,
        raw_cache: RawCache,
        summary_cache: SummaryCache
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
            # Summary Part
            # Handle case where summary_cache is smaller than raw_cache (lazy init)
            if sum_k is not None and page_lens is not None:
                if b < page_lens.shape[0]:
                    p_len = page_lens[b]
                    k_s = sum_k[b, :, :p_len, :] # [H, P, D]
                    v_s = sum_v[b, :, :p_len, :]
                else:
                    # Batch index outside summary cache -> treat as 0 pages
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
                k_s = torch.empty(H, 0, D, device=device)
                v_s = torch.empty(H, 0, D, device=device)
                idx_s = torch.empty(0, dtype=torch.long, device=device)
                is_sum_s = torch.empty(0, dtype=torch.long, device=device)
                p_len = 0
            
            # Raw Tail Part
            # Starts at `raw_seq_start[b]`
            r_start = raw_seq_start[b].item() if raw_seq_start is not None else 0
            # Ensure r_start is valid
            r_start = min(r_start, T_raw)
            
            k_r = k_raw[b, :, r_start:, :] # [H, T_tail, D]
            v_r = v_raw[b, :, r_start:, :]
            tail_len = k_r.shape[1]
            idx_r = torch.arange(r_start, r_start + tail_len, device=device)
            
            # Mask padding in raw tail
            if seq_start is not None:
                pad_end = seq_start[b].item()
                is_pad = idx_r < pad_end
                idx_r[is_pad] = -1
                
            is_sum_r = torch.zeros(tail_len, dtype=torch.long, device=device)
            
            # Concatenate
            batched_k.append(torch.cat([k_s, k_r], dim=1))
            batched_v.append(torch.cat([v_s, v_r], dim=1))
            batched_idx.append(torch.cat([idx_s, idx_r], dim=0))
            batched_is_sum.append(torch.cat([is_sum_s, is_sum_r], dim=0))
            
        # Pad and Stack
        lengths = [x.shape[1] for x in batched_k]
        max_len = max(lengths) if lengths else 0
        
        self.cover_keys = torch.zeros(B, H, max_len, D, device=device, dtype=k_raw.dtype)
        self.cover_values = torch.zeros(B, H, max_len, D, device=device, dtype=v_raw.dtype)
        self.cover_indices = torch.full((B, max_len), -1, dtype=torch.long, device=device)
        self.cover_is_summary = torch.zeros(B, max_len, dtype=torch.long, device=device)
        
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

class AttentionScoreBuffer:

    refinement_counts = None # [B, T_current]
        
    def initialize(self, B: int, H: int, T_init: int, device: torch.device, dtype: torch.dtype):
        """Initialize the buffer with empty state.
        
        Args:
            B: Batch size.
            H: Number of heads.
            T_init: Initial cover length (e.g. prefill length).
            device: Tensor device.
            dtype: Tensor dtype.
        """
        # Start with 0 accumulated length
        self.attention_weights = torch.zeros(B, H, 0, T_init, device=device, dtype=dtype)
        # We don't have indices yet, or we assume they match init?
        # We'll initialize indices to a default "raw" state matching T_init?
        # Or just None and wait for first push?
        # Push updates indices.
        self.cover_indices = None
        self.cover_is_summary = None
        self.refinement_counts = torch.zeros(B, T_init, dtype=torch.long, device=device)
        self.total_summaries_seen = 0
        self.total_refinements_made = 0

    def push(self, attn_weights: torch.Tensor, cover_indices: torch.Tensor, cover_is_summary: torch.Tensor):
        """Append a new attention score slice.

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

        # Invariant:
        # Attention weights width (T_new) must match cover indices width (T_new)
        # attn_weights: [B, H, L_new, T_new]
        # cover_indices: [B, T_new]
        assert attn_weights.shape[-1] == cover_indices.shape[-1], \
            f"Invariant Violation: Attn weights width {attn_weights.shape[-1]} != cover indices width {cover_indices.shape[-1]}"


        # Pad width if needed
        B, H, L_curr, T_curr = self.attention_weights.shape
        _, _, L_new, T_new = attn_weights.shape
        
        if T_new > T_curr:
            pad_w = T_new - T_curr
            # Pad existing weights with zeros
            padding = torch.zeros(B, H, L_curr, pad_w, device=self.attention_weights.device, dtype=self.attention_weights.dtype)
            self.attention_weights = torch.cat([self.attention_weights, padding], dim=3)
            
            # Pad refinement counts
            pad_counts = torch.zeros(B, pad_w, device=self.refinement_counts.device, dtype=self.refinement_counts.dtype)
            self.refinement_counts = torch.cat([self.refinement_counts, pad_counts], dim=1)
            
        elif T_curr > T_new:
             # Pad new weights
             pad_w = T_curr - T_new
             padding = torch.zeros(B, H, L_new, pad_w, device=attn_weights.device, dtype=attn_weights.dtype)
             attn_weights = torch.cat([attn_weights, padding], dim=3)
             
        # Concatenate along L (dim 2)
        self.attention_weights = torch.cat([self.attention_weights, attn_weights], dim=2)
        
        # Update indices (keep latest)
        self.cover_indices = cover_indices
        self.cover_is_summary = cover_is_summary
        
        # Update refinement counts?
        # We assume T_new matches current cover.
        # If T_new > T_curr (new tokens added), refinement counts for new tokens start at 0.
        # This is handled by padding above.
        
        if self.refinement_counts is None or self.refinement_counts.shape[1] < T_new:
             # This should be handled by padding logic above if T_new > T_curr
             pass

    def get_data(self):
        return self.attention_weights, self.cover_indices, self.cover_is_summary

    def reset(self):
        # Reset to empty state with 0 length, preserving B/H/device/dtype
        if self.attention_weights is not None:
             B, H, _, _ = self.attention_weights.shape
             device = self.attention_weights.device
             dtype = self.attention_weights.dtype
             self.attention_weights = torch.zeros(B, H, 0, 0, device=device, dtype=dtype)
        self.cover_indices = None
        self.cover_is_summary = None
        if self.refinement_counts is not None:
             self.refinement_counts = torch.zeros_like(self.refinement_counts)
        self.total_summaries_seen = 0
        self.total_refinements_made = 0

    def compress_and_trim(
        self,
        new_pages: List[List[Tuple[int, int]]],
        new_frontiers: torch.Tensor
    ):
        """Compress attention weights for new pages and trim to new frontier.

        Args:
            new_pages: List of length B. new_pages[b] is a list of (start, end)
                inclusive raw indices for newly created pages.
            new_frontiers: [B] raw index of the start of the remaining raw tail.
        """
        if self.attention_weights is None:
            return

        B, H, L, T = self.attention_weights.shape
        device = self.attention_weights.device
        dtype = self.attention_weights.dtype
        
        batched_weights = []
        batched_indices = []
        batched_is_sum = []
        batched_ref_counts = []
        
        for b in range(B):
            # Existing state
            w_b = self.attention_weights[b] # [H, L, T]
            idx_b = self.cover_indices[b]   # [T]
            is_sum_b = self.cover_is_summary[b] # [T]
            ref_b = self.refinement_counts[b] # [T]
            
            # 1. Keep existing summaries
            # We assume summaries are at the beginning and marked with is_sum=1
            # We also need to filter out padding (-1)
            mask_sum = (is_sum_b == 1) & (idx_b != -1)
            
            cols_w = [w_b[:, :, mask_sum]]
            cols_idx = [idx_b[mask_sum]]
            cols_is_sum = [is_sum_b[mask_sum]]
            cols_ref = [ref_b[mask_sum]]
            
            # Determine next summary index
            if mask_sum.any():
                # Assuming they are sorted and 0-indexed
                next_sum_idx = idx_b[mask_sum].max().item() + 1
            else:
                next_sum_idx = 0

            # 2. Add new pages (compressed)
            # new_pages[b] contains (start, end) raw indices
            for start, end in new_pages[b]:
                # Identify columns corresponding to this page
                # These must be raw tokens (is_sum=0) and in range [start, end]
                mask_page = (is_sum_b == 0) & (idx_b >= start) & (idx_b <= end)
                
                if mask_page.any():
                    # Sum weights: [H, L, N_page] -> [H, L, 1]
                    w_sum = w_b[:, :, mask_page].sum(dim=-1, keepdim=True)
                    cols_w.append(w_sum)
                    
                    # Metadata for new summary
                    cols_idx.append(torch.tensor([next_sum_idx], device=device, dtype=torch.long))
                    next_sum_idx += 1
                    cols_is_sum.append(torch.tensor([1], device=device, dtype=torch.long))
                    cols_ref.append(torch.tensor([0], device=device, dtype=torch.long)) # New summary, 0 refinements
            
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
                # Empty batch (shouldn't happen if initialized correctly)
                batched_weights.append(torch.zeros(H, L, 0, device=device, dtype=dtype))
                batched_indices.append(torch.zeros(0, device=device, dtype=torch.long))
                batched_is_sum.append(torch.zeros(0, device=device, dtype=torch.long))
                batched_ref_counts.append(torch.zeros(0, device=device, dtype=torch.long))

        # Pad and Stack
        lengths = [x.shape[-1] for x in batched_weights]
        max_len = max(lengths) if lengths else 0
        
        new_weights = torch.zeros(B, H, L, max_len, device=device, dtype=dtype)
        new_indices = torch.full((B, max_len), -1, device=device, dtype=torch.long)
        new_is_sum = torch.zeros(B, max_len, device=device, dtype=torch.long)
        new_ref_counts = torch.zeros(B, max_len, device=device, dtype=torch.long)
        
        for b in range(B):
            l = lengths[b]
            if l > 0:
                new_weights[b, :, :, :l] = batched_weights[b]
                new_indices[b, :l] = batched_indices[b]
                new_is_sum[b, :l] = batched_is_sum[b]
                new_ref_counts[b, :l] = batched_ref_counts[b]
                
        self.attention_weights = new_weights
        self.cover_indices = new_indices
        self.cover_is_summary = new_is_sum
        self.refinement_counts = new_ref_counts

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

class LukaKVController:

    def __init__(self, config: PretrainedConfig, num_layers: Optional[int] = None):
        """Coordinating facade that owns raw, summary, cover, and attention buffers.

        Caches are tracked per-layer to mirror the underlying transformer stack.
        Args:
            config: PretrainedConfig supplying at least `num_hidden_layers` (or pass
                `num_layers` explicitly).
            num_layers: Optional override for layer count if not present on config.
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
        
        # Initialize segmenter and compressor
        # Using defaults for now; ideally these come from config.
        self.segmenter = DummySegmenter(min_chunk=16, tail_len=16, max_pages=15)
        self.compressor = MeanCompressor()
        # How often to attempt segmentation/compression (every N decode steps)
        self.segment_interval = 1
        # Whether to create pages during generation (decode) or only during prefill
        self.create_pages_in_generation = True
        # Whether to add log(N) bias to summary attention logits
        # Set to False when using MeanCompressor (arithmetic mean != log-sum-exp assumption)
        self.use_log_bias = False
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

        Args:
            layer_idx: int
                Transformer layer index.

        Returns:
            bool: True if new summary pages were added (i.e., `summary_cache[layer_idx]`
            grew and `cover_view[layer_idx]` should be rebuilt), False otherwise. The
            method should orchestrate the segmenter/compressor once they are wired in.
        """
        # Throttle segmentation based on configured interval
        self.seg_step_counters[layer_idx] += 1
        if self.segment_interval > 1 and (self.seg_step_counters[layer_idx] % self.segment_interval) != 0:
            return False

        # 1. Segment
        # Get buffer data
        # 1. Segment
        # Get buffer data
        attn_weights, _, _ = self.attn_buffer[layer_idx].get_data()
        
        # Use CoverView indices directly as they are the ground truth
        cover_indices = self.cover_view[layer_idx].cover_indices
        cover_is_summary = self.cover_view[layer_idx].cover_is_summary
        if attn_weights is None:
            return False
         # 2. Run segmentation
        # Returns [B, max_pages] of inclusive end indices (raw coordinates)
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
        
        # Identify new pages
        new_frontiers = raw_seq_start.clone()
        has_updates = False
        all_new_pages = [[] for _ in range(B)] # List of (start, end) for each batch
        
        # Optimization: Get full tensors once
        k_raw, v_raw, _, _ = raw_cache.get_layer(layer_idx, with_offsets=False)
        if k_raw is None:
            return False
            
        # Get number of heads from k_raw for importance weight shape
        H_kv = k_raw.shape[1]

        for b in range(B):
            frontier = raw_seq_start[b].item()
            p_ends = page_ends[b]
            valid_mask = (p_ends >= frontier) & (p_ends != -1)

            if not valid_mask.any():
                continue

            valid_ends = p_ends[valid_mask].sort().values

            b_keys = []
            b_values = []
            b_starts = []
            b_ends = []

            current_start = frontier

            for end_idx in valid_ends.tolist():
                end_idx = int(end_idx)
                if end_idx < current_start:
                    continue

                # Slice: [current_start, end_idx] inclusive
                # k_raw: [B, H, T, D]
                k_slice = k_raw[b, :, current_start : end_idx + 1, :]
                v_slice = v_raw[b, :, current_start : end_idx + 1, :]

                # Extract importance weights from attention buffer
                # Find cover positions that map to raw indices in [current_start, end_idx]
                importance_weights = None
                if attn_weights is not None and cover_indices is not None:
                    cover_idx_b = cover_indices[b]  # [T_cover]
                    is_sum_b = cover_is_summary[b]  # [T_cover]

                    # Mask for raw tokens in this page's range
                    page_mask = (
                        (cover_idx_b >= current_start) &
                        (cover_idx_b <= end_idx) &
                        (is_sum_b == 0)
                    )

                    if page_mask.any():
                        # attn_weights: [B, H_q, L_accum, T_cover]
                        # Get attention to these positions, aggregate over queries
                        page_attn = attn_weights[b, :, :, page_mask]  # [H_q, L_accum, page_len]

                        # Sum over queries (L_accum) to get total attention per token
                        importance = page_attn.sum(dim=1)  # [H_q, page_len]

                        # Handle GQA: H_q may differ from H_kv
                        # Average across query head groups to get H_kv importance weights
                        H_q = importance.shape[0]
                        if H_q != H_kv:
                            num_groups = H_q // H_kv
                            importance = importance.view(H_kv, num_groups, -1).mean(dim=1)  # [H_kv, page_len]

                        importance_weights = importance.unsqueeze(0)  # [1, H_kv, page_len]

                # Compress: [H, D]
                k_in = k_slice.unsqueeze(0)
                v_in = v_slice.unsqueeze(0)
                k_sum, v_sum = self.compressor(k_in, v_in, importance_weights)  # [1, H, D]

                b_keys.append(k_sum)
                b_values.append(v_sum)
                b_starts.append(current_start)
                b_ends.append(end_idx)

                # Record for buffer update
                all_new_pages[b].append((current_start, end_idx))

                current_start = end_idx + 1
            
            if b_keys:
                k_stack = torch.stack(b_keys, dim=2)
                v_stack = torch.stack(b_values, dim=2)
                
                summary_cache.add_pages(
                    keys=k_stack,
                    values=v_stack,
                    batch_nums=torch.tensor([b], device=device),
                    page_start=torch.tensor([b_starts], device=device),
                    page_end=torch.tensor([b_ends], device=device),
                    page_frontier=torch.tensor([current_start], device=device)
                )

                new_frontiers[b] = current_start
                has_updates = True

        if has_updates:
            # Update raw_seq_start in RawCache
            raw_cache.raw_seq_start[layer_idx] = new_frontiers

            # Rebuild CoverView
            self.cover_view[layer_idx].update_cover_view(layer_idx, raw_cache, summary_cache)

            # Compress and trim buffer
            self.attn_buffer[layer_idx].compress_and_trim(all_new_pages, new_frontiers)
            return True
            
        return False

    def top_down_attention(
        self,
        layer_idx: int,
        query_states: torch.Tensor,      # [B, H_q, L_q, D]
        scaling: float,
        num_kv_groups: int,
        attention_mask: Optional[torch.Tensor] = None,  # [B, 1, L_q, T_raw]
        sliding_window: Optional[int] = None,
        threshold: float = 0.2,
    ):
        """Run attention over cover view (summaries + raw tail) with optional refinement.

        Args:
            layer_idx: Transformer layer index.
            query_states: [B, H_q, L_q, D] query projections.
            scaling: Attention scaling factor (1/sqrt(D)).
            num_kv_groups: H_q // H_k for grouped-query attention.
            attention_mask: [B, 1, L_q, T_raw] causal + padding mask.
            sliding_window: Unused, kept for API compatibility.
            threshold: Refinement threshold. If <0, use exact raw attention.
                       If >=0, refine summaries with attn_prob > threshold.

        Returns:
            attn_output: [B, L_q, H_q, D] attended values.
            attn_probs: [B, H_q, L_q, T_cover] attention probabilities.
        """
        # Exact path: skip cover view and run full raw attention
        if threshold is not None and threshold < 0:
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

        # Get cover view
        cover_view = self.cover_view[layer_idx]
        cover_k = cover_view.cover_keys
        cover_v = cover_view.cover_values
        cover_indices = cover_view.cover_indices
        cover_is_summary = cover_view.cover_is_summary

        if cover_k is None or cover_v is None:
            raise ValueError(f"Cover view not initialized for layer {layer_idx}.")

        B, H_k, T_cover, D = cover_k.shape
        L_q = query_states.shape[2]

        cover_k_full = repeat_kv(cover_k, num_kv_groups)
        cover_v_full = repeat_kv(cover_v, num_kv_groups)

        # Map summary positions to raw indices (using page_end) for mask lookup
        page_ends = self.summary_cache[layer_idx].page_end
        if page_ends.shape[0] < B:
            pad_b = B - page_ends.shape[0]
            zeros = torch.zeros(pad_b, page_ends.shape[1], device=page_ends.device, dtype=page_ends.dtype)
            page_ends = torch.cat([page_ends, zeros], dim=0)

        summary_mask = (cover_is_summary == 1)
        cover_raw_indices = cover_indices.clone()
        if summary_mask.any():
            clamped_idx = cover_indices.clamp(min=0, max=page_ends.shape[1] - 1)
            summary_raw_pos = page_ends.gather(1, clamped_idx)
            cover_raw_indices = torch.where(summary_mask, summary_raw_pos, cover_indices)

        # Compute attention logits
        attn_logits = torch.matmul(query_states, cover_k_full.transpose(2, 3)) * scaling

        # Optionally add log(N) bias for summaries: makes exp(q·k_s + log(N)) ≈ sum_i exp(q·k_i)
        # Note: This assumes k_summary ≈ log-mean-exp of raw keys. MeanCompressor uses arithmetic
        # mean, so this bias may hurt more than help. Disable with use_log_bias=False.
        if self.use_log_bias and summary_mask.any():
            page_start = self.summary_cache[layer_idx].page_start
            page_end_cache = self.summary_cache[layer_idx].page_end
            if page_start.shape[0] < B:
                pad_b = B - page_start.shape[0]
                page_start = torch.cat([page_start, torch.zeros(pad_b, page_start.shape[1], device=page_start.device, dtype=page_start.dtype)], dim=0)
                page_end_cache = torch.cat([page_end_cache, torch.zeros(pad_b, page_end_cache.shape[1], device=page_end_cache.device, dtype=page_end_cache.dtype)], dim=0)

            page_lengths = (page_end_cache - page_start + 1).float()
            page_idx = cover_indices.clamp(min=0, max=page_lengths.shape[1] - 1)
            token_lengths = page_lengths.gather(1, page_idx)
            log_bias = torch.where(summary_mask, token_lengths.clamp(min=1).log(), torch.zeros_like(token_lengths))
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

        attn_probs = torch.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_probs, cover_v_full)

        # Refinement: descend into pages where attention exceeds threshold
        if threshold is not None and threshold >= 0:
            refine_mask = (attn_probs > threshold) & summary_mask.view(B, 1, 1, T_cover)

            # Track stats
            num_summaries = summary_mask.sum().item()
            self.attn_buffer[layer_idx].total_summaries_seen += num_summaries * query_states.shape[1] * query_states.shape[2]
            self.attn_buffer[layer_idx].total_refinements_made += refine_mask.sum().item()

            refine_positions = refine_mask.any(dim=(1, 2))  # [B, T_cover]

            if refine_positions.any():
                summary_cache = self.summary_cache[layer_idx]
                page_start = summary_cache.page_start
                page_end = summary_cache.page_end
                if page_start.shape[0] < B:
                    pad_b = B - page_start.shape[0]
                    zeros = torch.zeros(pad_b, page_start.shape[1], device=page_start.device, dtype=page_start.dtype)
                    page_start = torch.cat([page_start, zeros], dim=0)
                    page_end = torch.cat([page_end, zeros], dim=0)

                k_raw, v_raw, _, _ = self.raw_cache.get_layer(layer_idx, with_offsets=False)
                b_idx, pos_idx = refine_positions.nonzero(as_tuple=True)
                page_ids = cover_indices[b_idx, pos_idx]
                starts = page_start[b_idx, page_ids]
                ends = page_end[b_idx, page_ids]
                lengths = ends - starts + 1

                valid = (page_ids >= 0) & (ends >= starts)
                if valid.any():
                    b_idx, pos_idx = b_idx[valid], pos_idx[valid]
                    starts, lengths = starts[valid], lengths[valid]

                    # Process each unique page length as a batch
                    for page_len in lengths.unique().tolist():
                        page_len = int(page_len)
                        bucket_mask = (lengths == page_len)
                        b_bucket = b_idx[bucket_mask]
                        pos_bucket = pos_idx[bucket_mask]
                        start_bucket = starts[bucket_mask]

                        # Gather raw KV for this page
                        raw_idx = start_bucket.unsqueeze(1) + torch.arange(page_len, device=query_states.device)
                        gather_idx = raw_idx.view(-1, 1, page_len, 1).expand(-1, k_raw.shape[1], -1, k_raw.shape[3])
                        k_slice = repeat_kv(torch.gather(k_raw[b_bucket], 2, gather_idx), num_kv_groups)
                        v_slice = repeat_kv(torch.gather(v_raw[b_bucket], 2, gather_idx), num_kv_groups)

                        # Compute refined attention over raw tokens
                        q_bucket = query_states[b_bucket]
                        raw_logits = torch.matmul(q_bucket, k_slice.transpose(2, 3)) * scaling
                        if attention_mask is not None:
                            mask_gather_idx = raw_idx.view(-1, 1, 1, page_len).expand(-1, 1, attention_mask.shape[2], -1)
                            raw_logits = raw_logits + torch.gather(attention_mask[b_bucket], 3, mask_gather_idx)

                        raw_probs = torch.softmax(raw_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
                        raw_out = torch.matmul(raw_probs, v_slice)

                        # Delta = (refined - summary) * attention_mass
                        base_value = cover_v_full[b_bucket, :, pos_bucket, :].unsqueeze(2)
                        attn_mass = attn_probs[b_bucket, :, :, pos_bucket].unsqueeze(-1)
                        delta = (raw_out - base_value) * attn_mass

                        # Only apply where refine_mask was true
                        active_mask = refine_mask[b_bucket, :, :, pos_bucket].unsqueeze(-1)
                        attn_output.index_add_(0, b_bucket, delta * active_mask.to(delta.dtype))

        self.attn_buffer[layer_idx].push(attn_probs, cover_indices, cover_is_summary)
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

        # 1. RawCache vs CoverView Alignment
        k_raw, _, seq_start, raw_seq_start = raw_cache.get_layer(layer_idx, with_offsets=True)
        if k_raw is None or cover_view.cover_keys is None:
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
        B = k_raw.shape[0]
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
        attn_weights, attn_indices, attn_is_sum = attn_buffer.get_data()
        if attn_weights is not None:
            # Verify width matches cover view
            # Note: attn_buffer accumulates, cover_view grows. They should match in T dimension.
            
            if cover_view.cover_keys is not None:
                T_cover = cover_view.cover_keys.shape[2]
                T_attn = attn_weights.shape[-1]
                
                assert T_attn == T_cover, f"Invariant Violation: Attn buffer width {T_attn} != Cover view length {T_cover}"
                
                # Verify indices match
                if attn_indices is not None:
                     # attn_indices: [B, T]
                     # cover_view.cover_indices: [B, T]
                     min_b = min(attn_indices.shape[0], cover_view.cover_indices.shape[0])
                     assert torch.equal(attn_indices[:min_b], cover_view.cover_indices[:min_b]), \
                         "Invariant Violation: Attn buffer indices mismatch with CoverView indices"
                     assert torch.equal(attn_is_sum[:min_b], cover_view.cover_is_summary[:min_b]), \
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
            print(f"[Layer {layer_idx}] Raw tokens:")
            for b in range(B):
                pad = int(seq_start[b].item()) if seq_start is not None else 0
                num_pages = int(summary_cache.page_lens[b].item()) if summary_cache.keys is not None else 0
                segments = fmt_pages(b)
                print(f"  Batch {b}: pad={pad}, pages={num_pages}, segments={segments}, total_tokens={T_raw}")
        else:
            print(f"[Layer {layer_idx}] Raw tokens: <empty>")

        # Cover token summary
        if cover_view.cover_indices is not None:
            cover_idx = cover_view.cover_indices
            cover_is_sum = cover_view.cover_is_summary
            B = cover_idx.shape[0]
            print(f"[Layer {layer_idx}] Cover view:")
            for b in range(B):
                pad = int((cover_idx[b] == -1).sum().item())
                pages = int(((cover_is_sum[b] == 1) & (cover_idx[b] >= 0)).sum().item())
                raw_len = int(((cover_is_sum[b] == 0) & (cover_idx[b] >= 0)).sum().item())
                print(f"  Batch {b}: pad={pad}, pages={pages}, raw_len={raw_len}")
        else:
            print(f"[Layer {layer_idx}] Cover view: <empty>")

        # Page summary
        if summary_cache.keys is not None:
            lens = summary_cache.page_lens
            print(f"[Layer {layer_idx}] Pages per batch: {[int(x) for x in lens.tolist()]}")
        else:
            print(f"[Layer {layer_idx}] Pages per batch: <empty>")

        # Attention score buffer summary
        attn_weights, buf_idx, buf_is_sum = attn_buf.get_data()
        if attn_weights is not None:
            B, H, L_accum, T = attn_weights.shape
            print(f"[Layer {layer_idx}] Attention buffer: shape={attn_weights.shape}")
            if buf_idx is not None and buf_is_sum is not None:
                for b in range(B):
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
