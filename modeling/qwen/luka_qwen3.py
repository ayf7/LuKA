import torch
import torch.nn as nn
from torch.types import Tensor

import vllm.model_executor.models.qwen3 as qwen3_mod
from vllm.attention import Attention, AttentionType
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig

from typing import Iterable, Tuple

from modeling.luka_cache import LukaKVCache

# Keep a handle to the original attention for inheritance.
BaseQwen3Attention = qwen3_mod.Qwen3Attention
BaseQwen3ForCausalLM = qwen3_mod.Qwen3ForCausalLM


class LukaQwenAttention(BaseQwen3Attention):
    """
    Drop-in replacement for vLLM's Qwen3Attention.

    - Reuses the original __init__ via super().__init__(...).
    - Adds one extra Attention instance (summary_attn) with its own KV cache.
    - Overrides forward() to combine outputs from both caches.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        rope_scaling: tuple | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: dict[str, object] | None = None,
    ) -> None:

        # Build the original attention stack (qkv_proj, self.attn, etc.)
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_position=max_position,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            qkv_bias=qkv_bias,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            prefix=prefix,
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )

        # Add an additional KV cache group for LuKA summaries.
        # Different prefix => different layer_name => different kv_cache & block table.
        self.summary_attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.summary_attn",
            attn_type=attn_type,
        )

        base_attn = self.attn
        self.attn = LukaKVCache(base_attn) # overwrites this attribute

    # def forward(
    #     self,
    #     positions: torch.Tensor,
    #     hidden_states: torch.Tensor,
    # ) -> torch.Tensor:
    #     # ---- 1. Same QKV / norms / RoPE as base class ----
    #     qkv, _ = self.qkv_proj(hidden_states)
    #     q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    #     # reshape to per-head for qk norm
    #     q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
    #     q_by_head = self.q_norm(q_by_head)
    #     q = q_by_head.view(q.shape)

    #     k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
    #     k_by_head = self.k_norm(k_by_head)
    #     k = k_by_head.view(k.shape)

    #     q, k = self.rotary_emb(positions, q, k)

    #     # ---- 2. Main KV cache: full Qwen3 self-attention ----
    #     main_out = self.attn(q, k, v)

    #     # ---- 3. LuKA summary KV cache: second Attention instance ----
    #     # Design idea:
    #     #   - During prefill, you can decide which tokens become "summary" pages
    #     #     and call summary_attn with their K/V to populate its KV cache.
    #     #   - During decode, you can call summary_attn with (q, None, None)
    #     #     to read from that summary cache only.
    #     #
    #     # For now, as a sketch:
    #     summary_out = self.summary_attn(q, None, None)

    #     # ---- 4. Combine both views of memory ----
    #     combined = main_out + self.summary_alpha * summary_out

    #     output, _ = self.o_proj(combined)
    #     return output


class LukaQwenForCausalLM(BaseQwen3ForCausalLM):
    """
    Identical to vLLM's Qwen3ForCausalLM, but uses LukaQwenAttention because
    we monkey-patched Qwen3Attention above.
    """
    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        # Let the base class do its normal mapping / loading.
        loaded = super().load_weights(weights)

        # Mark LuKA-only parameters as logically 'loaded', even though no
        # checkpoint entries exist for them.
        for name, _ in self.named_parameters():
            if "summary_alpha" in name:
                loaded.add(name)

        print(self.model)
        return loaded

# Global hooks: replace Qwen3Attention before any Qwen3Model is created.

def initialize_luka_hook():
    qwen3_mod.Qwen3Attention = LukaQwenAttention
    qwen3_mod.Qwen3ForCausalLM = LukaQwenForCausalLM

    from vllm.model_executor.models.registry import ModelRegistry
    ModelRegistry.register_model("LukaQwenForCausalLM", LukaQwenForCausalLM)

# Saves the path.
