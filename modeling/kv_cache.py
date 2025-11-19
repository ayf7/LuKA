import torch
import torch.nn as nn
from modeling.segmenter import Rule

from typing import Optional

class LukaKVCache(nn.Module):
    """
    HuggingFace-compatible KV cache wrapper for LuKA. Manages boundary detection
    and future KV compression during prefill/decode.
    """

    def __init__(self,
                 boundary_rule: Optional[Rule] = None):
        super().__init__()
        self.boundary_rule = boundary_rule
        self.is_prefill = True

    def prefill(
        self,
        attn_weights: torch.Tensor,  # [batch, heads, seq_len, seq_len]
        layer_idx: int,
    ) -> Optional[torch.Tensor]:
        """
        Prefill path: detect boundaries during initial context processing.

        Args:
            attn_weights: Attention weights from prefill
            layer_idx: Current layer index

        Returns:
            Boundary scores tensor or None
        """
        if self.boundary_rule is None:
            return None

        # Analyze first batch, average across heads
        attn_avg = attn_weights[0].mean(dim=0)  # [seq_len, seq_len]

        # Apply boundary detection rule
        boundary_scores = self.boundary_rule.process(attn_avg)

        return boundary_scores

    def decode(
        self,
        attn_weights: torch.Tensor,  # [batch, heads, 1, seq_len]
        layer_idx: int,
    ) -> Optional[torch.Tensor]:
        """
        Decode path: process single-token generation.

        For now, we skip boundary detection during decode since we're
        generating one token at a time.

        Args:
            attn_weights: Attention weights from decode step
            layer_idx: Current layer index

        Returns:
            None (no boundary detection during decode)
        """
        # Skip boundary detection during decode for now
        return None

    def forward(
        self,
        attn_weights: torch.Tensor,
        layer_idx: int,
    ) -> Optional[torch.Tensor]:
        """
        Main entry point: route to prefill or decode based on sequence length.

        Args:
            attn_weights: Attention weights [batch, heads, q_len, kv_len]
            layer_idx: Current layer index

        Returns:
            Boundary scores or None
        """
        q_len = attn_weights.shape[2]

        # Prefill if processing multiple tokens, decode if single token
        if q_len > 1:
            return self.prefill(attn_weights, layer_idx)
        else:
            return self.decode(attn_weights, layer_idx)
