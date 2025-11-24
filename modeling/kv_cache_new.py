import torch
import torch.nn as nn
from modeling.segmenter import Segmenter, BufferedSegmenter
from modeling.compressor import Compressor, MeanCompressor
from transformers.cache_utils import Cache, DynamicCache
from transformers import PretrainedConfig

from typing import Any, Optional, List
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
                print(self.seq_start[layer_idx])
                print(self.raw_seq_start[layer_idx])
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
        pass

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
    
    def update(self,
        keys: torch.Tensor,
        values: torch.Tensor
    ):
        """Append raw tokens into the cover view during decoding.

        Args:
            keys: torch.Tensor
                [B, H, L_new, D] raw keys for new tokens (typically L_new == 1).
            values: torch.Tensor
                [B, H, L_new, D] raw values for new tokens.

        Returns:
            None; should mutate `cover_keys/cover_values` by concatenating the
            raw tail and adjust `raw_seq_start` as the frontier advances.
        """
        pass

    def update_cover_view(self,
        raw_cache: RawCache,
        summary_cache: SummaryCache
    ):
        """Rebuild cover view after new pages are materialized. This happens after
        the raw_cache nad summary_cache have initialized their new pages.

        Args:
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
        pass

class AttentionScoreBuffer:

    def __init__(self):
        """Buffer for attention score snapshots used by the segmenter.

        Each push stores the cover-view attention matrix for a decode step so
        that the segmenter can retrospectively form pages.
        """
        self.attention_scores = None
    
    def push(self):
        """Append a new attention score slice.

        Expected to cache tensors shaped like [B, H, L_q, L_cover] (cover view
        length matches `L_num_pages + L_raw_tokens`). The implementation should
        handle padding and batching policy consistent with the segmenter.
        """
        pass

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
        k_all, v_all, seq_start, raw_seq_start = self.raw_cache.update(
            keys,
            values,
            layer_idx=layer_idx,
            cache_kwargs=cache_kwargs,
            attention_mask=attention_mask,
        )
        # TODO: extend cover_view[layer_idx] with new raw tail
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
        pass

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
        """Run attention against cover view, with optional summary refinement.

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
            threshold: float
                Refinement threshold; >=0 triggers descent into summary spans, <0 forces
                full raw attention (exact path).

        Returns:
            attn_output: torch.Tensor
                [B, H_q, L_q, D] attended values.
            attn_probs: torch.Tensor
                [B, H_q, L_q, T_cover] attention probabilities over the cover view
                (summary + raw tail). `T_cover` = max(L_num_pages + L_raw_tokens).

        Notes:
            Implementation should:
            - Pull cover keys/values + `cover_indices/cover_is_summary` from CoverView.
            - Map cover summary positions to raw spans via SummaryCache.page_start/page_end.
            - Replace summary logits with log-sum-exp over their raw spans so probability
              mass matches exact attention before refinement.
            - If `threshold` >= 0, refine pages whose summary logits dominate by attending
              directly to raw spans and mixing the delta back into `attn_output`.
        """
        pass
