import torch
import torch.nn as nn
from vllm.attention import Attention

from typing import Optional

class LukaKVCache(nn.Module):
    def __init__(self, base_attn: Attention):
        super().__init__()
        """
        Constructs the LukaKVCache. Note that one LukaKVCache is created per
        layer, as the heads are parallelized. See the Attention class usage.
        """
        self.base_attn = base_attn
    
    @torch.no_grad()
    def prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prefill path. For now, just calls the underlying Attention.

        Args:
            q, k, v: same shapes as Qwen3Attention currently passes to self.attn
            positions: optional, unused for now

        Returns:
            attn_output: same as base_attn(q, k, v)
        """
        return self.base_attn(q, k, v)

    @torch.no_grad()
    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode path. For now, just calls the underlying Attention.

        Args:
            q, k, v: same shapes as prefill()
            positions: optional, unused for now

        Returns:
            attn_output: same as base_attn(q, k, v)
        """
        return self.base_attn(q, k, v)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        is_prefill: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Default call path used by Qwen3Attention today: self.attn(q, k, v).

        Signature is backward-compatible:
          - Qwen3Attention currently calls self.attn(q, k, v) with no
            positions/is_prefill.
          - Those extra args are optional and ignored for now.

        Later you can:
          - detect prefill/decode here,
          - or have Qwen3Attention call prefill()/decode() explicitly.
        """
        # For now we completely ignore positions/is_prefill and just behave
        # exactly like the original Attention.
        return self.base_attn(q, k, v)