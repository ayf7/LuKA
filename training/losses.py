"""
Loss interfaces for compressor training.

Defines the key/output alignment objective described in TRAINING_FORMULATION.md.
Implementations are omitted in this scaffolding; only function signatures and
shape/behavior contracts are specified to guide a clean rewrite.
"""

from typing import Dict, List, Optional

import torch


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

    Returns:
        None.

    Side effects:
        Should raise a ValueError on mismatch; no state changes otherwise.
    """
    raise NotImplementedError("Shape validation is not implemented in scaffolding.")


def alignment_losses(
    q: torch.Tensor,                         # [B, H_q, L, D]
    k: torch.Tensor,                         # [B, H_k, L, D]
    v: torch.Tensor,                         # [B, H_k, L, D]
    segments: List[List[torch.Tensor]],      # segments[b] is list of 1D LongTensor indices
    compressor: torch.nn.Module,             # maps [B, H_k, N, D] -> (k_sum [B,H_k,D], v_sum [B,H_k,D])
    token_mask: torch.Tensor,                # [B, L] 1=token, 0=pad (query positions to include)
    attention_mask: Optional[torch.Tensor],  # [B, 1, L, L] additive mask (causal/pad); may be None
    num_kv_groups: int,
    scaling: float,
    lambda_key: float,
    lambda_out: float,
) -> Dict[str, torch.Tensor]:
    """
    Compute key/output alignment losses for a batch.

    Implements:
        - Key alignment: MSE between student summary logit and teacher logsumexp over page keys.
        - Output alignment: MSE between student head output (with page summaries) and teacher output.

    Args:
        q: [B, H_q, L, D] queries (post-norm, post-RoPE).
        k: [B, H_k, L, D] keys (pre-repeat_kv).
        v: [B, H_k, L, D] values (pre-repeat_kv).
        segments: Per-batch list of page index tensors; indices refer to the last dim of k/v.
        compressor: Module producing summary key/value for each page (per head).
        token_mask: [B, L] mask of valid query positions (1=token, 0=pad).
        attention_mask: [B, 1, L, L_total] additive mask from the model; includes causal/pad; None allowed.
        num_kv_groups: Repeat factor for grouped-query attention (H_q = H_k * num_kv_groups).
        scaling: Scalar head_dim**-0.5 used in attention logits.
        lambda_key: Weight on key alignment loss term.
        lambda_out: Weight on output alignment loss term.

    Returns:
        Dict with keys:
            loss: scalar total loss (lambda_key * L_key + lambda_out * L_out).
            loss_key: scalar key alignment loss (zero if no pages).
            loss_out: scalar output alignment loss (zero if lambda_out == 0 or no pages).
            num_pages: integer tensor count of pages processed.

    Side effects:
        None beyond autograd graph construction; no mutation of inputs.
    """
    raise NotImplementedError("Alignment losses are not implemented in scaffolding.")
