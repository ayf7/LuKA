"""
Losses for compressor training.

Implements key (attention score) and output alignment losses as described in
TRAINING_FORMULATION.md, operating on captured q/k/v tensors and sampled page
segments. Exposes separate functions for each component plus a top-level
orchestrator.
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import repeat_kv


def _validate_qkv_and_masks(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    num_kv_groups: int,
) -> None:
    """
    Validate q/k/v shapes and mask coverage for grouped-query attention.

    Args:
        q: [B, H_q, L_q, D] queries after RMSNorm + RoPE.
        k: [B, H_k, L_k, D] keys after RMSNorm + RoPE (pre-repeat_kv).
        v: [B, H_k, L_k, D] values (pre-repeat_kv).
        attention_mask: Optional [B, 1, L_q, L_k_total] additive mask; may be None.
        num_kv_groups: Integer such that H_q == H_k * num_kv_groups.

    Raises:
        ValueError on any shape or consistency mismatch.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"q/k/v must be rank-4; got shapes {q.shape}, {k.shape}, {v.shape}")
    Bq, Hq, Lq, Dq = q.shape
    Bk, Hk, Lk, Dk = k.shape
    Bv, Hv, Lv, Dv = v.shape
    if not (Bq == Bk == Bv):
        raise ValueError("Batch dim mismatch among q/k/v")
    if not (Dq == Dk == Dv):
        raise ValueError("Head dim mismatch among q/k/v")
    if not (Hk == Hv and Lk == Lv):
        raise ValueError("k/v head or length mismatch")
    if Hq != Hk * num_kv_groups:
        raise ValueError(f"H_q ({Hq}) must equal H_k * num_kv_groups ({Hk} * {num_kv_groups})")
    if attention_mask is not None:
        if attention_mask.shape[:3] != (Bq, 1, Lq):
            raise ValueError(
                f"attention_mask leading dims must be [B,1,L_q]; got {attention_mask.shape}"
            )
        if attention_mask.shape[-1] < Lk:
            raise ValueError(
                f"attention_mask length {attention_mask.shape[-1]} shorter than keys {Lk}"
            )


def _segments_to_ranges(
    seg_list: List[torch.Tensor],
    L: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize a list of segments into start/end pairs.

    Accepts either explicit index tensors (must be contiguous and increasing)
    or two-element ranges [start, end) for each page (pass as Python pairs to
    disambiguate from length-2 index tensors).

    Args:
        seg_list: List for one batch row; each item is a 1D tensor or sequence.
        L: Sequence length for bounds checking.
        device: Target device for the returned tensors.

    Returns:
        starts: [P] long tensor of inclusive start positions.
        ends: [P] long tensor of exclusive end positions.
    """
    starts: List[int] = []
    ends: List[int] = []
    for seg in seg_list:
        seg_is_tensor = torch.is_tensor(seg)
        seg_t = torch.as_tensor(seg, device=device, dtype=torch.long)
        if seg_t.ndim != 1:
            raise ValueError(f"segment must be 1D; got shape {seg_t.shape}")
        if seg_t.numel() == 0:
            continue

        contiguous = torch.all(seg_t[1:] == seg_t[:-1] + 1) if seg_t.numel() > 1 else True
        is_range_like = (not seg_is_tensor and seg_t.numel() == 2) or (
            seg_t.numel() == 2 and seg_t[1] > seg_t[0] and not contiguous
        )

        if is_range_like:
            start = int(seg_t[0].item())
            end = int(seg_t[1].item())
        else:
            if int(seg_t.min()) < 0 or int(seg_t.max()) >= L:
                raise IndexError(f"segment indices out of bounds for length {L}: {seg_t}")
            if not torch.all(seg_t[1:] > seg_t[:-1]):
                raise ValueError("segment indices must be strictly increasing")
            if not contiguous:
                raise ValueError("segments must be contiguous ranges")
            start = int(seg_t[0].item())
            end = int(seg_t[-1].item()) + 1
        if start < 0 or end > L or start >= end:
            raise ValueError(f"invalid segment bounds [{start}, {end}) for length {L}")
        starts.append(start)
        ends.append(end)

    if not starts:
        empty = torch.empty((0,), device=device, dtype=torch.long)
        return empty, empty
    return (
        torch.tensor(starts, device=device, dtype=torch.long),
        torch.tensor(ends, device=device, dtype=torch.long),
    )


def attention_score_loss(
    q: torch.Tensor,                         # [B, H_q, L, D]
    k: torch.Tensor,                         # [B, H_k, L, D]
    v: torch.Tensor,                         # [B, H_k, L, D]
    segments: List[List[torch.Tensor]],      # segments[b] is list of [start,end) or contiguous index tensors
    compressor: torch.nn.Module,             # maps [B, H_k, N, D] -> (k_sum [B,H_k,D], v_sum [B,H_k,D])
    scaling: float,
    num_kv_groups: int,
    token_mask: torch.Tensor,                # [B, L]
    attention_mask: Optional[torch.Tensor],  # [B, 1, L, L] additive mask or None
) -> Dict[str, torch.Tensor]:
    """
    Key alignment loss: MSE between student summary logit and teacher logsumexp over page keys.

    Args:
        q: [B, H_q, L, D] queries (post-norm, post-RoPE).
        k: [B, H_k, L, D] keys (pre-repeat_kv).
        segments: Per-batch list of page specs; each item is either [start, end)
            for a contiguous page or a 1D LongTensor of contiguous indices.
        compressor: Module producing summary key/value for each page (per head).
        scaling: Scalar head_dim**-0.5 used in attention logits.
        num_kv_groups: Repeat factor for grouped-query attention (H_q = H_k * num_kv_groups).
        token_mask: [B, L] mask of valid query positions (1=token, 0=pad).
        attention_mask: [B, 1, L, L_total] additive mask from the model; includes causal/pad; None allowed.

    Returns:
        Dict with:
            loss_key: scalar key alignment loss (zero if no pages).
            num_pages: integer tensor count of pages processed.

    Raises:
        ValueError/IndexError on shape or index mismatches.
    """
    _validate_qkv_and_masks(q, k, v, attention_mask, num_kv_groups)
    B, H_q, L, D = q.shape
    if token_mask.shape != (B, L):
        raise ValueError(f"token_mask shape {token_mask.shape} incompatible with q shape {q.shape}")

    device = q.device
    dtype = q.dtype
    eps = torch.finfo(dtype).tiny
    k_full = repeat_kv(k, num_kv_groups)  # [B, H_q, L, D]

    logits_full = torch.matmul(q, k_full.transpose(2, 3)) * scaling  # [B, H_q, L, L]
    if attention_mask is not None:
        logits_full = logits_full + attention_mask[:, :, :, : L]
    query_mask = token_mask[:, None, :, None].to(dtype=torch.bool)  # [B,1,L,1]
    logits_full = torch.where(query_mask, logits_full, torch.zeros_like(logits_full))

    loss_terms: List[torch.Tensor] = []
    page_count = 0
    rep_factor = H_q // k.shape[1]
    positions = torch.arange(L, device=device)

    for b in range(B):
        starts, ends = _segments_to_ranges(segments[b], L, device)
        num_pages = int(starts.numel())
        if num_pages == 0:
            continue
        lengths = ends - starts  # [P]
        page_count += num_pages

        page_mask = (positions[None, :] >= starts[:, None]) & (positions[None, :] < ends[:, None])  # [P, L]
        in_page = page_mask.any(dim=0)  # [L]
        if not in_page.any():
            continue
        page_index = torch.argmax(page_mask.to(dtype=torch.int64), dim=0)  # [L]

        logits_b = logits_full[b]  # [H_q, L, L]
        exp_logits_flat = logits_b.exp().reshape(H_q * L, L)  # [H_q*L, L]
        exp_logits_flat = exp_logits_flat * in_page.to(exp_logits_flat.dtype)

        page_sums = exp_logits_flat.new_zeros((H_q * L, num_pages))  # [H_q*L, P]
        page_sums.index_add_(1, page_index, exp_logits_flat)
        z_teacher = page_sums.view(H_q, L, num_pages).permute(2, 0, 1)  # [P, H_q, L]

        # Mask out (query, page) pairs where query has no visible keys in page (due to causal masking)
        # If page_sum is very small, it means all keys in the page are masked for that query
        has_visible_keys = page_sums.view(H_q, L, num_pages).permute(2, 0, 1) > (eps * 10)  # [P, H_q, L]
        z_teacher = z_teacher.clamp_min(eps).log()

        k_b = k[b]  # [H_k, L, D]
        v_b = v[b]  # [H_k, L, D]
        k_summaries: List[torch.Tensor] = []
        for start, end in zip(starts.tolist(), ends.tolist()):
            k_seg = k_b[:, start:end, :]  # [H_k, page_len, D]
            v_seg = v_b[:, start:end, :]  # [H_k, page_len, D]
            k_sum = _compress_k_only(k_seg, v_seg, compressor, D)
            k_summaries.append(k_sum)
        if not k_summaries:
            continue
        k_summaries_t = torch.stack(k_summaries, dim=0)  # [P, H_k, D]
        k_sum_exp = k_summaries_t.repeat_interleave(rep_factor, dim=1)  # [P, H_q, D]

        z_student = torch.einsum("hld,phd->phl", q[b], k_sum_exp) * scaling  # [P, H_q, L]
        z_student = z_student + lengths.to(dtype=dtype).view(num_pages, 1, 1).log()

        key_diff = (z_student - z_teacher) ** 2  # [P, H_q, L]
        key_mask_q = token_mask[b][None, None, :].to(dtype=key_diff.dtype)  # [1,1,L]
        # Combine token mask with visible keys mask
        valid_mask = key_mask_q * has_visible_keys.to(dtype=key_diff.dtype)  # [P, H_q, L]
        valid_count = valid_mask.sum()
        if valid_count <= 0:
            continue
        loss_terms.append((key_diff * valid_mask).sum(dim=(1, 2)) / valid_count)

    # print("> loss_terms", loss_terms)
    loss_key = (
        torch.cat(loss_terms).mean()
        if loss_terms
        else torch.tensor(0.0, device=device, dtype=dtype)
    )

    return {
        "loss_key": loss_key,
        "num_pages": torch.tensor(page_count, device=device, dtype=torch.long),
    }


def output_loss(
    q: torch.Tensor,                         # [B, H_q, L, D]
    k: torch.Tensor,                         # [B, H_k, L, D]
    v: torch.Tensor,                         # [B, H_k, L, D]
    segments: List[List[torch.Tensor]],
    compressor: torch.nn.Module,
    scaling: float,
    num_kv_groups: int,
    token_mask: torch.Tensor,                # [B, L]
    attention_mask: Optional[torch.Tensor],  # [B, 1, L, L] additive mask or None
) -> torch.Tensor:
    """
    Output alignment loss: MSE between student head output (with page summary)
    and teacher output.

    Args:
        q: [B, H_q, L, D] queries (post-norm, post-RoPE).
        k: [B, H_k, L, D] keys (pre-repeat_kv).
        v: [B, H_k, L, D] values (pre-repeat_kv).
        segments: Per-batch list of page specs ([start, end) or contiguous indices).
        compressor: Module producing summary key/value for each page (per head).
        scaling: Scalar head_dim**-0.5 used in attention logits.
        num_kv_groups: Repeat factor for grouped-query attention (H_q = H_k * num_kv_groups).
        token_mask: [B, L] mask of valid query positions (1=token, 0=pad).
        attention_mask: [B, 1, L, L_total] additive mask from the model; includes causal/pad; None allowed.

    Returns:
        loss_out: scalar output alignment loss (zero if no pages or no segments).

    Raises:
        ValueError/IndexError on shape or index mismatches.
    """
    _validate_qkv_and_masks(q, k, v, attention_mask, num_kv_groups)
    B, H_q, L, D = q.shape
    if token_mask.shape != (B, L):
        raise ValueError(f"token_mask shape {token_mask.shape} incompatible with q shape {q.shape}")

    device = q.device
    dtype = q.dtype

    k_full = repeat_kv(k, num_kv_groups)  # [B, H_q, L, D]
    v_full = repeat_kv(v, num_kv_groups)  # [B, H_q, L, D]

    logits_full = torch.matmul(q, k_full.transpose(2, 3)) * scaling  # [B, H_q, L, L]
    if attention_mask is not None:
        logits_full = logits_full + attention_mask[:, :, :, : L]
    # Avoid softmax on padded query rows (all keys masked) to prevent NaNs.
    query_mask = token_mask[:, None, :, None].to(dtype=torch.bool)  # [B,1,L,1]
    logits_full = torch.where(query_mask, logits_full, torch.zeros_like(logits_full))
    p_full = F.softmax(logits_full, dim=-1)  # [B, H_q, L, L]
    assert not torch.isnan(p_full).any(), "p_full discovered nan values"

    loss_terms: List[torch.Tensor] = []
    mask_value = torch.finfo(dtype).min
    rep_factor = H_q // k.shape[1]
    positions = torch.arange(L, device=device)
    eps = torch.finfo(dtype).tiny

    for b in range(B):
        starts, ends = _segments_to_ranges(segments[b], L, device)
        num_pages = int(starts.numel())
        if num_pages == 0:
            continue

        q_b = q[b]            # [H_q, L, D]
        k_b = k[b]            # [H_k, L, D]
        v_b = v[b]            # [H_k, L, D]
        logits_b = logits_full[b]  # [H_q, L, L]
        p_b = p_full[b]            # [H_q, L, L]
        v_full_b = v_full[b]       # [H_q, L, D]
        token_mask_b = token_mask[b]  # [L]

        page_lengths = ends - starts  # [P]
        page_mask = (positions[None, :] >= starts[:, None]) & (positions[None, :] < ends[:, None])  # [P, L]

        # Compute which (query, page) pairs have visible keys (for causal masking)
        # For each page, check which keys are in the page and compute visibility per query
        in_page = page_mask.any(dim=0)  # [L]
        page_index = torch.argmax(page_mask.to(dtype=torch.int64), dim=0)  # [L]
        exp_logits_flat = logits_b.exp().reshape(H_q * L, L)  # [H_q*L, L]
        exp_logits_flat_masked = exp_logits_flat * in_page.to(exp_logits_flat.dtype)
        page_sums = exp_logits_flat.new_zeros((H_q * L, num_pages))
        page_sums.index_add_(1, page_index, exp_logits_flat_masked)
        has_visible_keys = page_sums.view(H_q, L, num_pages).permute(2, 0, 1) > (eps * 10)  # [P, H_q, L]

        # Compress all pages once
        summaries_k: List[torch.Tensor] = []
        summaries_v: List[torch.Tensor] = []
        for start, end in zip(starts.tolist(), ends.tolist()):
            k_seg = k_b[:, start:end, :]  # [H_k, page_len, D]
            v_seg = v_b[:, start:end, :]  # [H_k, page_len, D]
            k_sum, v_sum = compressor(k_seg.unsqueeze(0), v_seg.unsqueeze(0))  # [1,H_k,D]
            summaries_k.append(k_sum.squeeze(0))
            summaries_v.append(v_sum.squeeze(0))
        if not summaries_k:
            continue
        k_sum_all = torch.stack(summaries_k, dim=0)  # [P, H_k, D]
        v_sum_all = torch.stack(summaries_v, dim=0)  # [P, H_k, D]
        k_sum_exp = k_sum_all.repeat_interleave(rep_factor, dim=1)  # [P, H_q, D]
        v_sum_exp = v_sum_all.repeat_interleave(rep_factor, dim=1)  # [P, H_q, D]

        # Random compression: sample which pages to compress with p=0.5
        use_summary = torch.rand(num_pages, device=device) > 0.5  # [P]

        # Build summary logits and values for selected pages
        summary_indices = torch.nonzero(use_summary, as_tuple=False).squeeze(-1)
        num_summaries = int(summary_indices.numel())

        if num_summaries > 0:
            k_selected = k_sum_exp[summary_indices]  # [num_summaries, H_q, D]
            v_selected = v_sum_exp[summary_indices]  # [num_summaries, H_q, D]
            selected_lengths = page_lengths[summary_indices]  # [num_summaries]

            # Compute summary logits
            logits_summaries = torch.einsum("hld,phd->phl", q_b, k_selected) * scaling  # [num_summaries, H_q, L]
            logits_summaries = logits_summaries + selected_lengths.to(dtype).view(-1, 1, 1).log()
        else:
            logits_summaries = None
            v_selected = None

        # Build raw logits, masking out compressed pages
        logits_raw = logits_b.clone()  # [H_q, L, L]
        for p_idx in range(num_pages):
            if use_summary[p_idx]:
                # Mask out raw tokens for compressed pages
                logits_raw[:, :, starts[p_idx]:ends[p_idx]] = mask_value

        # Concatenate summary and raw logits
        if logits_summaries is not None:
            # [H_q, L, num_summaries + L]
            logits_student = torch.cat([logits_summaries.permute(1, 2, 0), logits_raw], dim=-1)
        else:
            logits_student = logits_raw  # [H_q, L, L]

        # Compute student attention
        p_student = F.softmax(logits_student, dim=-1)

        # Build student output: summary contributions + raw contributions
        if num_summaries > 0:
            p_summaries = p_student[..., :num_summaries]  # [H_q, L, num_summaries]
            p_raw = p_student[..., num_summaries:]  # [H_q, L, L]
            # Summary contribution: [H_q, L, num_summaries] @ [num_summaries, H_q, D] -> [H_q, L, D]
            summary_contrib = torch.einsum("hlp,phd->hld", p_summaries, v_selected)
        else:
            p_raw = p_student  # [H_q, L, L]
            summary_contrib = torch.zeros_like(v_full_b)  # [H_q, L, D]

        raw_contrib = torch.matmul(p_raw, v_full_b)  # [H_q, L, D]
        student_out = summary_contrib + raw_contrib  # [H_q, L, D]

        # Teacher output
        teacher_out = torch.matmul(p_b, v_full_b)  # [H_q, L, D]

        # Compute loss
        out_diff = (student_out - teacher_out) ** 2  # [H_q, L, D]
        mask_q = token_mask_b[None, :, None].to(dtype=out_diff.dtype)  # [1, L, 1]

        # Visible keys mask: need to check which queries have at least one visible key
        # With random compression, a query needs visibility to either summaries or raw tokens
        query_has_visible = torch.zeros(L, dtype=torch.bool, device=device)
        for h in range(H_q):
            for q_pos in range(L):
                if token_mask_b[q_pos] == 0:
                    continue  # Padded query
                # Check if this query has any visible keys
                visible = False
                # Check summaries
                for p_idx in summary_indices:
                    if has_visible_keys[p_idx, h, q_pos]:
                        visible = True
                        break
                # Check raw tokens (non-compressed pages)
                if not visible:
                    for key_pos in range(L):
                        if logits_b[h, q_pos, key_pos] > mask_value + 1:  # Not masked
                            # Check if this key is not in a compressed page
                            in_compressed_page = False
                            for p_idx in range(num_pages):
                                if use_summary[p_idx] and starts[p_idx] <= key_pos < ends[p_idx]:
                                    in_compressed_page = True
                                    break
                            if not in_compressed_page:
                                visible = True
                                break
                query_has_visible[q_pos] = visible

        valid_mask = mask_q * query_has_visible[None, :, None].to(dtype=out_diff.dtype)  # [1, L, 1]
        valid_count = valid_mask.sum()
        if valid_count == 0:
            continue
        # Normalize by number of elements: valid_count * H_q * D
        H_q, L, D = out_diff.shape
        num_elements = valid_count * H_q * D
        loss_terms.append((out_diff * valid_mask).sum() / num_elements)

    loss_out = (
        torch.stack(loss_terms).mean()
        if loss_terms
        else torch.tensor(0.0, device=device, dtype=dtype)
    )
    return loss_out


def compute_losses(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    segments: List[List[torch.Tensor]],
    compressor: torch.nn.Module,
    token_mask: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    num_kv_groups: int,
    scaling: float,
    lambda_key: float,
) -> Dict[str, torch.Tensor]:
    """
    Orchestrate output (main) loss and key (delta) loss.

    Args:
        q, k, v: Captured tensors (see attention_score_loss/output_loss).
        segments: Per-batch page specs ([start, end) or contiguous indices).
        compressor: Summary generator module.
        token_mask: [B, L] mask for valid queries.
        attention_mask: Additive mask from the model (or None).
        num_kv_groups: Repeat factor for GQA.
        scaling: Head-dim scaling.
        lambda_key: Weight on key loss (delta term added to main loss).

    Returns:
        Dict with:
            loss: total loss = loss_out + lambda_key * loss_key.
            loss_key: key loss.
            loss_out: output loss (main term).
            num_pages: pages processed.
    """
    key_dict = attention_score_loss(
        q=q,
        k=k,
        v=v,
        segments=segments,
        compressor=compressor,
        scaling=scaling,
        num_kv_groups=num_kv_groups,
        token_mask=token_mask,
        attention_mask=attention_mask,
    )
    loss_out = output_loss(
        q=q,
        k=k,
        v=v,
        segments=segments,
        compressor=compressor,
        scaling=scaling,
        num_kv_groups=num_kv_groups,
        token_mask=token_mask,
        attention_mask=attention_mask,
    )

    loss_key = key_dict["loss_key"]
    num_pages = key_dict["num_pages"]
    total = loss_out + (lambda_key * loss_key)

    return {
        "loss": total,
        "loss_key": loss_key,
        "loss_out": loss_out,
        "num_pages": num_pages,
    }


def _compress_k_only(
    k_seg: torch.Tensor, v_seg: torch.Tensor, compressor: torch.nn.Module, dim_expected: int
):
    """
    Helper to reuse the compressor for key logits only.

    Args:
        k_seg: [H_k, page_len, D] key slice.
        v_seg: [H_k, page_len, D] value slice (passed to compressor even though values are unused here).
        compressor: Module that accepts (k, v) with leading batch dim.
        dim_expected: Expected hidden size D.

    Returns:
        k_sum: [H_k, D] compressed keys.

    Raises:
        ValueError if compressor output shape mismatches expectations.
    """
    if k_seg.shape != v_seg.shape:
        raise ValueError("k_seg and v_seg must have identical shapes in _compress_k_only")
    k_seg_in = k_seg.unsqueeze(0)
    v_seg_in = v_seg.unsqueeze(0)
    k_sum, _ = compressor(k_seg_in, v_seg_in)  # [1, H_k, D]
    k_sum = k_sum.squeeze(0)
    if k_sum.shape[0] != k_seg.shape[0] or k_sum.shape[-1] != dim_expected:
        raise ValueError("compressor output shape mismatch in _compress_k_only")
    return k_sum


def _generate_random_pages(
    valid_start: int, valid_end: int, min_page_len: int = 2, max_page_len: int = 10
) -> List[torch.Tensor]:
    """
    Generate random contiguous pages within [valid_start, valid_end).

    Args:
        valid_start: Inclusive start of valid token region.
        valid_end: Exclusive end of valid token region.
        min_page_len: Minimum page length.
        max_page_len: Maximum page length.

    Returns:
        List of 1D LongTensors representing contiguous page indices.
    """
    pages: List[torch.Tensor] = []
    pos = valid_start
    valid_len = valid_end - valid_start

    if valid_len < min_page_len:
        return pages

    # Random number of pages (up to valid_len // min_page_len)
    max_possible_pages = max(1, valid_len // min_page_len)
    num_pages = torch.randint(1, min(max_possible_pages + 1, 8), size=(1,)).item()

    for _ in range(num_pages):
        remaining = valid_end - pos
        if remaining < min_page_len:
            break
        # Random page length
        page_len = torch.randint(
            min_page_len, min(max_page_len, remaining) + 1, size=(1,)
        ).item()
        page = torch.arange(pos, pos + page_len, dtype=torch.long)
        pages.append(page)
        pos += page_len

    return pages


if __name__ == "__main__":
    """
    Manual smoke test for loss computation using dummy tensors.

    Constructs random q/k/v tensors with random page segmentation and runs
    compute_losses to verify shape handling and masking logic.
    """
    torch.manual_seed(42)
    from modeling.compressor import EncoderCompressor

    # More realistic dimensions
    B = 4
    H_k = 2
    num_kv_groups = 2
    H_q = H_k * num_kv_groups
    L = 100
    D = 64

    q = torch.randn(B, H_q, L, D)
    k = torch.randn(B, H_k, L, D)
    v = torch.randn(B, H_k, L, D)

    # Random lengths per batch with left padding
    token_mask = torch.zeros(B, L, dtype=torch.long)
    lengths = torch.randint(low=20, high=L + 1, size=(B,))
    for b in range(B):
        token_mask[b, L - lengths[b] :] = 1  # left padding, tokens on the right

    # Causal + padding attention mask
    causal = torch.zeros(L, L, dtype=q.dtype)
    causal = causal.masked_fill(
        torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1), float("-inf")
    )
    pad_key = torch.where(
        token_mask == 1,
        torch.zeros_like(token_mask, dtype=q.dtype),
        torch.full_like(token_mask, float("-inf"), dtype=q.dtype),
    )  # [B, L]
    attention_mask = causal.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B,1,L,L]
    attention_mask = attention_mask + pad_key[:, None, None, :]  # mask padded keys

    print(f"Batch size: {B}, Seq len: {L}, Heads: {H_q}, Head dim: {D}")
    print(f"Lengths: {lengths.tolist()}")

    # Generate random pages per batch within valid token boundaries
    segments: List[List[torch.Tensor]] = []
    for b in range(B):
        valid_start = L - int(lengths[b].item())
        valid_end = L
        pages = _generate_random_pages(valid_start, valid_end, min_page_len=3, max_page_len=12)
        segments.append(pages)
        print(
            f"Batch {b}: {len(pages)} pages, "
            f"valid range [{valid_start}, {valid_end}), "
            f"page ranges: {[(int(p[0]), int(p[-1])+1) for p in pages]}"
        )
    print()
    compressor = EncoderCompressor()
    scaling = D ** -0.5  # Standard attention scaling
    lambda_key = 0.1

    print("Running compute_losses...")
    losses = compute_losses(
        q=q,
        k=k,
        v=v,
        segments=segments,
        compressor=compressor,
        token_mask=token_mask,
        attention_mask=attention_mask,
        num_kv_groups=num_kv_groups,
        scaling=scaling,
        lambda_key=lambda_key,
    )
    print({k: (v.item() if torch.is_tensor(v) and v.numel() == 1 else v) for k, v in losses.items()})
