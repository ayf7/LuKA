"""
LuKA integration for HuggingFace Transformers Qwen3 models.

This module provides a monkey-patch approach to integrate LuKA's KV cache
compression into HuggingFace's transformers library.
"""

from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache
import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3Config,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
    rotate_half,
)

from modeling.compressor import EncoderCompressor, MeanCompressor
from modeling.kv_cache import LukaKVController
from modeling.segmenter import DummySegmenter, KLSegmenter

# Optional global overrides for KV cache params, settable by callers (e.g., scripts/test.py)
_segmenter_override = None
_kv_params_override: dict[str, float | int | object] = {}

SEGMENTER_REGISTRY = {
    "dummy": DummySegmenter,
    "kl": KLSegmenter,
}

COMPRESSOR_REGISTRY = {
    "mean": MeanCompressor,
    "encoder": EncoderCompressor,
}

def set_luka_kv_params(
    *,
    default_tail_len: int = 16,
    min_compress_chunk: int = 16,
    max_pages: int = 15,
    refine_threshold: float = 0.05,
    compressor: object = "mean",
    compressor_kwargs: dict | None = None,
    segmenter: object = "dummy",
    segmenter_kwargs: dict | None = None,
) -> None:
    global _kv_params_override
    if default_tail_len is not None:
        _kv_params_override["default_tail_len"] = int(default_tail_len)
    if min_compress_chunk is not None:
        _kv_params_override["min_compress_chunk"] = int(min_compress_chunk)
    if max_pages is not None:
        _kv_params_override["max_pages"] = int(max_pages)
    if refine_threshold is not None:
        _kv_params_override["refine_threshold"] = float(refine_threshold)
    
    if compressor is not None:
        if isinstance(compressor, str):
            if compressor not in COMPRESSOR_REGISTRY:
                raise ValueError(f"Compressor {compressor} not found in registry. Available: {list(COMPRESSOR_REGISTRY.keys())}")
            kwargs = compressor_kwargs or {}
            print(COMPRESSOR_REGISTRY[compressor], flush=True)
            _kv_params_override["compressor"] = COMPRESSOR_REGISTRY[compressor](**kwargs)
        else:
            if compressor_kwargs:
                raise ValueError("Cannot provide compressor_kwargs when passing a compressor instance.")
            _kv_params_override["compressor"] = compressor

    if segmenter is not None:
        if isinstance(segmenter, str):
            if segmenter not in SEGMENTER_REGISTRY:
                raise ValueError(f"Segmenter {segmenter} not found in registry. Available: {list(SEGMENTER_REGISTRY.keys())}")
            kwargs = segmenter_kwargs or {}
            print(SEGMENTER_REGISTRY[segmenter], flush=True)
            _kv_params_override["segmenter"] = SEGMENTER_REGISTRY[segmenter](**kwargs)
        else:
            if segmenter_kwargs:
                raise ValueError("Cannot provide segmenter_kwargs when passing a segmenter instance.")
            _kv_params_override["segmenter"] = segmenter

class LukaQwenAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = modeling_qwen3.Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = modeling_qwen3.Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        # Luka KV controller reference; injected by LukaQwen3Model so all layers share one.
        self.luka_kv: LukaKVController | None = None

    def forward(
        self,
        hidden_states: torch.Tensor, # [B, L, D]
        position_embeddings: tuple[torch.Tensor, torch.Tensor], # [B, L, D]
        attention_mask: Optional[torch.Tensor], # [B, 1, L, L]
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None, # [L]
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        if self.layer_idx == 0:
            print(cache_position, flush=True)
            pass
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        #### CACHE MANAGEMENT ####
        cos, sin = position_embeddings
        # [B, H_q, L, D], [B, H_k, L, D]        L = 1 in decoding, typically
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_value is not None:
            if self.luka_kv is None:
                raise RuntimeError("LukaKVController not initialized; ensure LukaQwen3Model injected it.")
            # Bind the DynamicCache to Luka's RawCache if not already done.
            if self.luka_kv.raw_cache.cache is None:
                self.luka_kv.raw_cache.initialize_with_cache(
                    past_key_value,
                )
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # Update raw cache and retrieve concatenated KV (same as past_key_value.update)
            key_states, value_states, _, _ = self.luka_kv.update(
                self.layer_idx,
                key_states,
                value_states,
                cache_kwargs=cache_kwargs,
                attention_mask=attention_mask,
            )

        ####  ATTENTION
        # Replace eager_attention_forward with top_down_attention
        
        # attn_output: [B, H_q, L, D]
        # attn_probs: [B, H_q, L, T_raw]
        
        if False:
            # Eager attention (Vanilla)
            attention_interface: Callable = eager_attention_forward
            if self.config._attn_implementation != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            attn_output, attn_weights = attention_interface(
                self,
                query_states, # [B, H_q, L, D]
                key_states, # [B, H_k, L + L_past, D]
                value_states, # [B, H_k, L + L_past, D]
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,  # diff with Llama
                **kwargs,
            )
            
        else:
            # Top-down attention (LuKA)
            attn_output, attn_weights = self.luka_kv.top_down_attention(
                layer_idx=self.layer_idx,
                query_states=query_states,
                scaling=self.scaling,
                num_kv_groups=self.num_key_value_groups,
                attention_mask=attention_mask,
                sliding_window=self.sliding_window,
                threshold=_kv_params_override.get("refine_threshold", 0.05)
            )

        if self.layer_idx == 0:
            # self.luka_kv.print_stats(0)
            pass
        
        # Try to create new pages (segmentation & compression)
        # This uses the buffer populated by top_down_attention
        self.luka_kv.try_new_pages(self.layer_idx)

        # Match HF Qwen3: reshape back to [B, L, H*D] and apply o_proj.
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        # Verify invariants
        if self.luka_kv is not None:
            self.luka_kv.verify_invariants(self.layer_idx)
            
        return attn_output, attn_weights


class LukaQwen3Model(modeling_qwen3.Qwen3Model):
    """
    Qwen3Model that owns a single LukaKVController shared across all attention layers.
    """

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        # Create one controller and share it with every attention module.
        self.luka_kv_controller = LukaKVController(config)
        
        # Apply overrides from registry
        if "segmenter" in _kv_params_override:
            self.luka_kv_controller.segmenter = _kv_params_override["segmenter"]
        if "compressor" in _kv_params_override:
            self.luka_kv_controller.compressor = _kv_params_override["compressor"]
            
        for layer in self.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.luka_kv = self.luka_kv_controller


class LukaQwen3ForCausalLM(modeling_qwen3.Qwen3ForCausalLM):
    """
    Qwen3ForCausalLM with LuKA-specific parameter handling.

    This allows loading pretrained checkpoints while ignoring LuKA-specific
    parameters that don't exist in the base model.
    """

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        Override to mark LuKA-only parameters as loaded, even though they
        don't exist in the checkpoint.

        Keeps strict loading enabled for all other parameters, only exempting
        LuKA-specific additions.
        """
        # Call parent's load with strict mode
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        # Only remove LuKA-specific parameters from missing_keys
        # Keep strict checking for everything else
        missing_keys[:] = [
            k for k in missing_keys
            if not ("luka_cache" in k)
        ]


def initialize_luka_hook():
    """
    Monkey-patch HuggingFace's Qwen3 model to use LuKA attention.

    Call this before loading any Qwen3 models if you want the LuKA effects.
    """
    modeling_qwen3.Qwen3Model = LukaQwen3Model
    modeling_qwen3.Qwen3Attention = LukaQwenAttention
    modeling_qwen3.Qwen3ForCausalLM = LukaQwen3ForCausalLM


def load_luka_model(model_name: str, use_eager_attention: bool = True, **kwargs):
    """
    Load a Qwen3 model with LuKA integration.

    Args:
        model_name: HuggingFace model identifier
        use_eager_attention: If True, force eager attention mode to enable output_attentions.
                           Required for boundary detection. Default: True
        **kwargs: Additional arguments to pass to AutoModelForCausalLM.from_pretrained

    Returns:
        Model with LuKA-enabled attention
    """
    from transformers import AutoModelForCausalLM

    # Initialize the hook before loading
    initialize_luka_hook()

    # Force eager attention if requested (needed for output_attentions=True)
    if use_eager_attention:
        kwargs['attn_implementation'] = 'eager'

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    return model
