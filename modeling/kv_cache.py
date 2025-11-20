import torch
import torch.nn as nn
from modeling.segmenter import Segmenter
from transformers.cache_utils import Cache, DynamicCache

from typing import Any, Optional, List
from dataclasses import dataclass
from transformers.models.qwen3.modeling_qwen3 import (
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv
)
class LukaKVCaches:
    """Drop-in replacement of the original KV-cache; wraps the original cache,
       but also holds the additional summary cache as well."""
    
    def __init__(
            self,
            raw_cache: Cache,
            segmenter: Segmenter,
            num_layers: int,
            default_tail_len: int = 16,
            min_compress_chunk: int = 32,
            batch_size: int = 0,
            max_pages: int = 15,
    ):
        # Original K/V cache, built by default from transformers package
        self.raw_cache = raw_cache

        # # Summary K/V data structures
        self.summary_keys: List[Optional[torch.Tensor]] = [None for _ in range(num_layers)] # list of [B, H_k, MAX_PAGES, D]
        self.summary_values: List[Optional[torch.Tensor]] = [None for _ in range(num_layers)] # list of [B, H_k, MAX_PAGES, D]
        self.summary_len: list[torch.LongTensor] = [torch.zeros(batch_size, dtype=torch.long) for _ in range(num_layers)]

        self.page_start: List[torch.Tensor] = [torch.zeros(batch_size, dtype=torch.long) for _ in range(num_layers)]
        self.page_end: List[torch.Tensor] = [torch.zeros(batch_size, dtype=torch.long) for _ in range(num_layers)]

        # # This is what will be used to eventually construct pages.
        self.attn_weight_buffer = None # [B, H, L, L_past] # where L_past in this case is last index that's been processed

        self.segmenter = segmenter

        self.tail_len: List[int] = [0] * num_layers
        self.default_tail_len = default_tail_len
        self.min_compress_chunk = min_compress_chunk
        self.num_layers = num_layers
        self.initialized = [False] * num_layers
        self.starts = None
        self.max_pages = max_pages
        # Simple refinement statistics per layer
        self.refine_stats = [
            {"refine": 0, "skip": 0} for _ in range(num_layers)
        ]
        # Cached cover per layer to avoid rebuilding when unchanged
        self.cover_cache: List[Optional[dict[str, torch.Tensor]]] = [None for _ in range(num_layers)]
        pass

    def update_original(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Thin wrapper around the underlying raw cache `update`.

        Args:
            key_states: torch.Tensor
                [B, H_k, L_new, D] new key states for this layer.
            value_states: torch.Tensor
                [B, H_k, L_new, D] new value states for this layer.
            layer_idx: int
                Scalar index of the transformer layer.
            cache_kwargs: Optional[dict[str, Any]]
                Extra kwargs passed through to the underlying cache (e.g. RoPE data).

        Returns:
            k_raw_all: torch.Tensor
                [B, H_k, T_raw, D] concatenated raw keys for all past tokens.
            v_raw_all: torch.Tensor
                [B, H_k, T_raw, D] concatenated raw values for all past tokens.
        """
        # k_raw_all, v_raw_all: [B, H_k, T_raw, D]
        k_raw_all, v_raw_all = self.raw_cache.update(
            key_states, value_states, layer_idx, cache_kwargs
        )
        return k_raw_all, v_raw_all

    def _get_seq_starts(
        self,
        attn_mask: torch.Tensor # [B, 1, new_len, total_seq_len]
    ) -> torch.Tensor:
        last_rows = attn_mask[:, 0, -1, :]
        is_zero = (last_rows == 0)
        indices = is_zero.float().argmax(dim=1)
        no_zero = (~is_zero).all(dim=1)
        indices[no_zero] = -1  # or any sentinel you want
        return indices

    def buffer_weights(
        self,
        attn_weight: torch.Tensor, # [B, H, new_len, total_seq_len]
        attn_mask: torch.Tensor # [B, 1, new_len, total_seq_len]
    ) -> None:
        if self.starts is None:
            self.starts = self._get_seq_starts(attn_mask) # [B]
        if self.attn_weight_buffer is None:
            self.attn_weight_buffer = attn_weight
            return
        b, h, new, old_and_new = attn_weight.shape
        b0, h0, old_rows, old_cols = self.attn_weight_buffer.shape

        assert b == b0 and h == h0, (
            f"buffer_weights: batch/head mismatch "
            f"(buffer {b0},{h0} vs new {b},{h})"
        )
        assert new + old_cols == old_and_new, (
            f"buffer_weights: inconsistent lengths "
            f"new={new}, old_cols={old_cols}, total={old_and_new}"
        )

        # Allocate enlarged buffer:
        # rows = old_rows + new (old queries + new queries)
        # cols = old_and_new     (old keys + new keys)
        new_buffer = attn_weight.new_zeros(b, h, old_rows + new, old_and_new)

        # Copy old block into the top-left corner
        new_buffer[:, :, :old_rows, :old_cols] = self.attn_weight_buffer

        # Copy new rows (new queries) into the last new rows using negative indexing
        # new_buffer[:, :, -new:, :] has shape [B, H, new, old_and_new]
        new_buffer[:, :, -new:, :old_and_new] = attn_weight

        self.attn_weight_buffer = new_buffer
        if new_buffer.shape[-1] % 10 == 0:
            self.attn_weight_buffer = self.attn_weight_buffer[:,:,-10:,:]
    
    def populate_pages(self) -> Optional[torch.LongTensor]:
        if self.attn_weight_buffer is None or self.starts is None:
            return None
        out = self.segmenter.process(
            attention_scores=self.attn_weight_buffer,
            seq_starts=self.starts,
            min_chunk=self.min_compress_chunk,
            tail_len=self.default_tail_len,
            max_pages=self.max_pages,
        )
        print(out)
        return out

    def finalize_pages_and_build_summaries(
        self,
        layer_idx: int,
        k_raw: torch.Tensor,        # [B, H, T, D]
        v_raw: torch.Tensor,        # [B, H, T, D]
        page_ends: torch.Tensor     # [B, MAX_NUM_PAGES], each row contains page end indices or -1
    ):
        """
        Given the computed page ends for this layer, update page_start/page_end
        and build summary key/value tensors by averaging raw keys/values on each page.
        """

        B, H, T, D = k_raw.shape
        MAX_PAGES = page_ends.shape[1]
        page_size = self.min_compress_chunk

        # Compute page starts = end - page_size + 1, but clamp at >= 0.
        page_starts = page_ends - (page_size - 1)
        page_starts = torch.clamp(page_starts, min=0)

        # Store these
        self.page_start[layer_idx] = page_starts
        self.page_end[layer_idx] = page_ends

        # Prepare summary KV storage for this layer if uninitialized
        if self.summary_keys[layer_idx] is None:
            self.summary_keys[layer_idx] = k_raw.new_zeros(B, H, MAX_PAGES, D)
            self.summary_values[layer_idx] = v_raw.new_zeros(B, H, MAX_PAGES, D)

        # Clear summary lengths
        valid_counts = torch.zeros(B, dtype=torch.long, device=k_raw.device)

        # Build summaries
        for b in range(B):
            for p in range(MAX_PAGES):
                end_idx = page_ends[b, p].item()
                if end_idx < 0:
                    break

                start_idx = page_starts[b, p].item()
                if start_idx > end_idx:
                    continue

                # Slice raw KV: [H, page_len, D]
                k_slice = k_raw[b, :, start_idx:end_idx+1, :]
                v_slice = v_raw[b, :, start_idx:end_idx+1, :]

                # Average across the sequence length dimension
                k_avg = k_slice.mean(dim=1)        # [H, D]
                v_avg = v_slice.mean(dim=1)        # [H, D]

                # Store them into the L-th summary cache
                self.summary_keys[layer_idx][b, :, p, :] = k_avg
                self.summary_values[layer_idx][b, :, p, :] = v_avg

                valid_counts[b] += 1

        # Update summary lengths (how many valid pages exist for each batch item)
        self.summary_len[layer_idx] = valid_counts
        # Invalidate cover cache for this layer since summaries changed
        self.cover_cache[layer_idx] = None
    
    def get_covering_kv(
        self,
        layer_idx: int
    ):
        """
        Construct the KV-cover for this layer:
        [ summary pages ] + [ last default_tail_len raw tokens ]

        Return:
            cover_keys:        [B, H, T_max, D]
            cover_values:      [B, H, T_max, D]
            cover_is_summary:  [B, T_max] (1 summary, 0 raw)
            cover_indices:     [B, T_max] (summary index or raw index, -1 for pad)
        """

        # ---- 1. RAW K/V ----
        k_raw = self.raw_cache.layers[layer_idx].keys     # [B,H,T_raw,D]
        v_raw = self.raw_cache.layers[layer_idx].values   # [B,H,T_raw,D]

        B, H, T_raw, D = k_raw.shape
        L_tail = self.default_tail_len

        # ---- 2. SUMMARY METADATA ----
        summary_k = self.summary_keys[layer_idx]          # [B,H,P_max,D]
        summary_v = self.summary_values[layer_idx]        # [B,H,P_max,D]
        summary_len = self.summary_len[layer_idx]         # [B]

        # ---- 2.1 Check cache ----
        cached = self.cover_cache[layer_idx]
        if cached is not None:
            cached_raw_len = cached["raw_len"]
            cached_sum_len = cached["summary_len"]
            if cached_raw_len == T_raw and torch.equal(cached_sum_len, summary_len):
                return (
                    cached["keys"],
                    cached["values"],
                    cached["is_summary"],
                    cached["indices"],
                )

        # These lists will contain variable-length per-batch tensors
        batched_k = []
        batched_v = []
        batched_flags = []
        batched_indices = []

        # ---- 3. Build cover for each batch element ----
        for b in range(B):
            # ----- SUMMARY -----
            num_pages = summary_len[b].item()
            if num_pages > 0:
                # [H, num_pages, D]
                k_sum = summary_k[b, :, :num_pages, :]
                v_sum = summary_v[b, :, :num_pages, :]
            else:
                # fallback zero-length
                k_sum = k_raw.new_zeros(H, 0, D)
                v_sum = v_raw.new_zeros(H, 0, D)

            flags_sum = torch.ones(num_pages, dtype=torch.long, device=k_raw.device)
            idx_sum = torch.arange(num_pages, dtype=torch.long, device=k_raw.device)

            # ----- RAW TAIL -----
            # ----- RAW REGION AFTER LAST SUMMARY PAGE -----
            # Find last summary page end
            if num_pages > 0:
                last_page_end = self.page_end[layer_idx][b, num_pages - 1].item()
            else:
                last_page_end = -1

            # Raw region begins here
            raw_start = last_page_end + 1
            raw_end = T_raw   # exclusive

            raw_start = max(0, raw_start)
            raw_end = max(raw_start, raw_end)

            # Slice everything not included in summary pages
            k_tail = k_raw[b, :, raw_start:raw_end, :]     # [H, raw_len, D]
            v_tail = v_raw[b, :, raw_start:raw_end, :]

            raw_len = raw_end - raw_start

            flags_tail = torch.zeros(raw_len, dtype=torch.long, device=k_raw.device)
            idx_tail = torch.arange(raw_start, raw_end, dtype=torch.long, device=k_raw.device)

            # ----- CONCAT SUMMARY + RAW -----
            k_cover = torch.cat([k_sum, k_tail], dim=1)   # [H, T_b, D]
            v_cover = torch.cat([v_sum, v_tail], dim=1)

            batched_k.append(k_cover.unsqueeze(0))        # [1,H,T_b,D]
            batched_v.append(v_cover.unsqueeze(0))
            batched_flags.append(torch.cat([flags_sum, flags_tail], dim=0))    # [T_b]
            batched_indices.append(torch.cat([idx_sum, idx_tail], dim=0))      # [T_b]

        # ---- 4. Compute max length T_max ----
        lengths = [x.shape[2] for x in batched_k]
        T_max = max(lengths)

        # ---- 5. Allocate padded output ----
        cover_keys = k_raw.new_zeros(B, H, T_max, D)
        cover_values = k_raw.new_zeros(B, H, T_max, D)
        cover_is_summary = k_raw.new_zeros(B, T_max, dtype=torch.long)
        cover_indices = k_raw.new_full((B, T_max), -1, dtype=torch.long)

        # ---- 6. Fill padded output ----
        for b in range(B):
            T_b = lengths[b]
            cover_keys[b, :, :T_b, :] = batched_k[b][0]
            cover_values[b, :, :T_b, :] = batched_v[b][0]
            cover_is_summary[b, :T_b] = batched_flags[b]
            cover_indices[b, :T_b] = batched_indices[b]

        # ---- 7. Cache and return ----
        self.cover_cache[layer_idx] = {
            "keys": cover_keys,
            "values": cover_values,
            "is_summary": cover_is_summary,
            "indices": cover_indices,
            "raw_len": T_raw,
            "summary_len": summary_len.clone(),
        }
        return cover_keys, cover_values, cover_is_summary, cover_indices
    
    def top_down_attention(
        self,
        layer_idx: int,
        query_states: torch.Tensor,      # [B, H_q, L_q, D]
        scaling: float,
        num_kv_groups: int,              # passed from attention module
        attention_mask: Optional[torch.Tensor] = None,  # [B, 1, L_q, T_raw]
        sliding_window: Optional[int] = None,
        threshold: float = 0.2,
    ):
        """
        Top-down attention with grouped-query attention support, optimized.

        H_q = num_attention_heads
        H_k = num_key_value_heads
        num_kv_groups = H_q // H_k

        Strategy:
        - Do a single attention pass over the KV-cover (summary + raw).
        - For each summary token that receives enough attention, descend
        into its raw page and refine the output for ALL heads/queries at once.
        """
        # Exact path: when threshold < 0, skip approximation and run full raw attention.
        # Keeping threshold >= 0 on the refined path allows testing parity through the
        # hierarchical decomposition (especially when threshold == 0).
        if threshold is not None and threshold < 0:
            k_raw = self.raw_cache.layers[layer_idx].keys     # [B, H_k, T_raw, D]
            v_raw = self.raw_cache.layers[layer_idx].values   # [B, H_k, T_raw, D]

            k_full = repeat_kv(k_raw, num_kv_groups)          # [B, H_q, T_raw, D]
            v_full = repeat_kv(v_raw, num_kv_groups)

            attn_logits = torch.matmul(query_states, k_full.transpose(2, 3)) * scaling
            if attention_mask is not None:
                attn_logits = attn_logits + attention_mask

            attn_probs = torch.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_probs, v_full)
            return attn_output, attn_probs

        # Build the cover (summary pages + raw tail)
        cover_k, cover_v, cover_is_summary, cover_indices = self.get_covering_kv(layer_idx)
        # Cache invalidation: cover depends on raw length/summary lengths. At decode time,
        # the lengths only change when raw cache grows or new summaries appear; we clear
        # the cache here if raw cache has grown since last use for this layer.
        cached = self.cover_cache[layer_idx]
        current_raw_len = self.raw_cache.layers[layer_idx].keys.shape[2]
        if cached is not None and cached["raw_len"] != current_raw_len:
            self.cover_cache[layer_idx] = None
            cover_k, cover_v, cover_is_summary, cover_indices = self.get_covering_kv(layer_idx)
        B, H_k, T_cover, D = cover_k.shape
        _, H_q, L_q, _ = query_states.shape

        # Expand K/V to match query heads when using grouped-query attention
        cover_k = repeat_kv(cover_k, num_kv_groups)          # [B, H_q, T_cover, D]
        cover_v = repeat_kv(cover_v, num_kv_groups)          # [B, H_q, T_cover, D]

        # Map cover positions back to raw indices for masking and refinement
        page_ends = self.page_end[layer_idx]                 # [B, P_max]
        summary_mask = cover_is_summary.bool()               # [B, T_cover]
        cover_raw_indices = cover_indices                    # [B, T_cover]
        if summary_mask.any():
            clamped_summary_idx = cover_indices.clamp(min=0, max=page_ends.shape[1] - 1)
            summary_raw_pos = page_ends.gather(1, clamped_summary_idx)  # [B, T_cover]
            cover_raw_indices = torch.where(summary_mask, summary_raw_pos, cover_indices)

        # Attention over the cover
        attn_logits = torch.matmul(query_states, cover_k.transpose(2, 3)) * scaling  # [B, H_q, L_q, T_cover]
        mask_value = torch.finfo(attn_logits.dtype).min
        if attention_mask is not None:
            # Gather the precomputed causal/sliding mask for the chosen cover positions
            cover_raw_indices_clamped = cover_raw_indices.clamp(min=0)
            idx_expanded = cover_raw_indices_clamped[:, None, None, :].expand(-1, 1, L_q, -1)
            cover_mask = attention_mask.gather(3, idx_expanded)
            cover_mask = cover_mask.masked_fill(cover_raw_indices[:, None, None, :] < 0, mask_value)
            attn_logits = attn_logits + cover_mask

        # Replace summary logits with log-sum-exp of their raw pages so the summary
        # probability mass matches the true raw probabilities that will be split during refinement.
        if summary_mask.any():
            k_raw_full = self.raw_cache.layers[layer_idx].keys     # [B, H_k, T_raw, D]
            k_full_raw = repeat_kv(k_raw_full, num_kv_groups)      # [B, H_q, T_raw, D]
            page_starts = self.page_start[layer_idx]               # [B, P_max]

            b_idx, pos_idx = summary_mask.nonzero(as_tuple=True)
            if b_idx.numel() > 0:
                for b in b_idx.unique().tolist():
                    mask_b = (b_idx == b)
                    pos_tensor = pos_idx[mask_b]
                    if pos_tensor.numel() == 0:
                        continue

                    page_ids = cover_indices[b, pos_tensor]                 # [S]
                    starts_b = page_starts[b, page_ids]                     # [S]
                    ends_b = page_ends[b, page_ids]                         # [S]

                    valid = (page_ids >= 0) & (ends_b >= starts_b)
                    invalid_pos = pos_tensor[~valid]
                    if invalid_pos.numel() > 0:
                        attn_logits[b, :, :, invalid_pos] = mask_value
                    if not valid.any():
                        continue

                    pos_tensor = pos_tensor[valid]
                    starts_b = starts_b[valid]
                    ends_b = ends_b[valid]

                    lengths = ends_b - starts_b + 1                         # [S_valid]
                    max_len = lengths.max().item()
                    arange = torch.arange(max_len, device=attn_logits.device)

                    raw_idx = starts_b.unsqueeze(1) + arange.unsqueeze(0)    # [S_valid, max_len]
                    pad_mask = arange.unsqueeze(0) >= lengths.unsqueeze(1)
                    raw_idx = raw_idx.masked_fill(pad_mask, -1)

                    idx_clamped = raw_idx.clamp(min=0)
                    # Gather keys for all pages at once: [1, H_q, S_valid, max_len, D]
                    k_flat = torch.index_select(k_full_raw[b], 1, idx_clamped.view(-1))
                    k_pages = k_flat.view(k_full_raw.shape[1], raw_idx.shape[0], max_len, D).unsqueeze(0)

                    # Queries for this batch
                    q_b = query_states[b:b+1]                               # [1, H_q, L_q, D]
                    page_logits = torch.einsum('bhld,bhsmd->bhslm', q_b, k_pages) * scaling  # [1, H_q, S_valid, L_q, max_len]

                    if attention_mask is not None:
                        mask_b_full = attention_mask[b:b+1]                 # [1, 1, L_q, T_raw]
                        mask_flat = torch.index_select(mask_b_full, 3, idx_clamped.view(-1))  # [1,1,L_q,S_valid*max_len]
                        mask_pages = mask_flat.view(1, 1, L_q, raw_idx.shape[0], max_len)     # [1,1,L_q,S_valid,max_len]
                        mask_pages = mask_pages.permute(0, 1, 3, 2, 4)                         # [1,1,S_valid,L_q,max_len]
                        page_logits = page_logits + mask_pages

                    # Mask out padded positions
                    bad_mask = pad_mask.view(1, 1, raw_idx.shape[0], 1, max_len)
                    page_logits = page_logits.masked_fill(bad_mask, mask_value)

                    summary_logits = torch.logsumexp(page_logits, dim=-1)[0]   # [H_q, S_valid, L_q]
                    summary_logits = summary_logits.permute(0, 2, 1)           # [H_q, L_q, S_valid]
                    attn_logits[b, :, :, pos_tensor] = summary_logits

        attn_probs = torch.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_probs, cover_v)  # [B, H_q, L_q, D]

        # Optional refinement: descend into pages whose summary tokens dominate attention.
        if threshold is not None and threshold >= 0:
            refine_mask = (attn_probs > threshold) & summary_mask[:, None, None, :]
            # Track how many summary tokens were refined vs. skipped
            refine_positions = refine_mask.any(dim=(1, 2))    # [B, T_cover]
            stats = self.refine_stats[layer_idx]
            refined_count = refine_positions.sum().item()
            summary_count = summary_mask.sum().item()
            stats["refine"] += refined_count
            stats["skip"] += max(summary_count - refined_count, 0)

            if refine_mask.any():
                k_raw = self.raw_cache.layers[layer_idx].keys     # [B, H_k, T_raw, D]
                v_raw = self.raw_cache.layers[layer_idx].values   # [B, H_k, T_raw, D]
                page_starts = self.page_start[layer_idx]          # [B, P_max]

                # Identify cover positions needing refinement (any head/query over threshold)
                b_idx, pos_idx = refine_positions.nonzero(as_tuple=True)
                if b_idx.numel() > 0:
                    page_ids = cover_indices[b_idx, pos_idx]          # [N]
                    starts = page_starts[b_idx, page_ids]             # [N]
                    ends = page_ends[b_idx, page_ids]                 # [N]
                    lengths = ends - starts + 1                       # [N]

                    valid = (page_ids >= 0) & (ends >= starts)
                    if valid.any():
                        b_idx = b_idx[valid]
                        pos_idx = pos_idx[valid]
                        starts = starts[valid]
                        lengths = lengths[valid]

                        unique_lengths = lengths.unique()
                        for L in unique_lengths.tolist():
                            bucket_mask = lengths == L
                            if not bucket_mask.any():
                                continue

                            b_bucket = b_idx[bucket_mask]             # [M]
                            pos_bucket = pos_idx[bucket_mask]         # [M]
                            start_bucket = starts[bucket_mask]        # [M]

                            L_int = int(L)
                            arange = torch.arange(L_int, device=query_states.device)
                            raw_idx = start_bucket.unsqueeze(1) + arange.unsqueeze(0)  # [M, L_int]

                            # Gather raw KV slices for this bucket
                            k_batch = k_raw[b_bucket]                 # [M, H_k, T_raw, D]
                            v_batch = v_raw[b_bucket]                 # [M, H_k, T_raw, D]
                            idx_expanded = raw_idx[:, None, :, None].expand(-1, k_batch.shape[1], -1, k_batch.shape[3])
                            k_slice = torch.gather(k_batch, 2, idx_expanded)          # [M, H_k, L_int, D]
                            v_slice = torch.gather(v_batch, 2, idx_expanded)

                            raw_k = repeat_kv(k_slice, num_kv_groups)                 # [M, H_q, L_int, D]
                            raw_v = repeat_kv(v_slice, num_kv_groups)                 # [M, H_q, L_int, D]

                            q_bucket = query_states[b_bucket]                         # [M, H_q, L_q, D]
                            raw_logits = torch.matmul(q_bucket, raw_k.transpose(2, 3)) * scaling
                            if attention_mask is not None:
                                mask_bucket = attention_mask[b_bucket]               # [M, 1, L_q, T_raw]
                                mask_slice = torch.gather(mask_bucket, 3, raw_idx[:, None, None, :])
                                raw_logits = raw_logits + mask_slice

                            raw_probs = torch.softmax(raw_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
                            raw_out = torch.matmul(raw_probs, raw_v)                  # [M, H_q, L_q, D]

                            base_value = cover_v[b_bucket, :, pos_bucket, :].unsqueeze(2)   # [M, H_q, 1, D]
                            attn_mass = attn_probs[b_bucket, :, :, pos_bucket]              # [M, H_q, L_q]
                            delta = (raw_out - base_value) * attn_mass.unsqueeze(-1)        # [M, H_q, L_q, D]

                            active_mask = refine_mask[b_bucket, :, :, pos_bucket].unsqueeze(-1)  # [M, H_q, L_q, 1]
                            update = delta * active_mask
                            attn_output.index_add_(0, b_bucket, update)

        return attn_output, attn_probs
