"""
LuKA integration for HuggingFace Transformers Qwen3 models.

This module provides a monkey-patch approach to integrate LuKA's KV cache
compression into HuggingFace's transformers library.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable

import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3
from transformers.models.qwen3.modeling_qwen3 import (
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
    Qwen3Config
)
from transformers.cache_utils import Cache, DynamicCache

from modeling.segmenter import DummySegmenter, KLDivergenceSegmenter
from modeling.kv_cache import LukaKVCaches
from modeling.compressor import MeanCompressor, EncoderCompressor

# Optional global overrides for segmenter and KV cache params, settable by callers (e.g., scripts/test.py)
_segmenter_override = None
_kv_params_override: dict[str, float | int | object] = {}

def set_luka_segmenter(segmenter: DummySegmenter) -> None:
    global _segmenter_override
    _segmenter_override = segmenter

def set_luka_kv_params(
    *,
    default_tail_len: int | None = None,
    min_compress_chunk: int | None = None,
    max_pages: int | None = None,
    refine_threshold: float | None = None,
    compressor: object | None = None,
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
        _kv_params_override["compressor"] = compressor

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
        kv_defaults = {
            "default_tail_len": 16,
            "min_compress_chunk": 16,
            "max_pages": 15,
            "compressor": None,
        }
        kv_defaults.update(_kv_params_override)
        segmenter = _segmenter_override if _segmenter_override is not None else KLDivergenceSegmenter(
            threshold=1.0,
            min_chunk=kv_defaults["min_compress_chunk"],
            tail_len=kv_defaults["default_tail_len"],
            max_pages=kv_defaults["max_pages"],
        )
        self.luka_kv_caches = LukaKVCaches(
            None,
            segmenter,
            config.num_hidden_layers,
            default_tail_len=kv_defaults["default_tail_len"],
            min_compress_chunk=kv_defaults["min_compress_chunk"],
            max_pages=kv_defaults["max_pages"],
            compressor=kv_defaults["compressor"],
        ) # TODO: make this configurable.

    def forward(
        self,
        hidden_states: torch.Tensor, # [B, L, D]
        position_embeddings: tuple[torch.Tensor, torch.Tensor], # [B, L, D]
        attention_mask: Optional[torch.Tensor], # [B, 1, L, L]
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None, # [L]
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
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
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            
            #### THIS RETURNS ALL PAST KEYS AND VALUES
            # Note: H_k refers to the number of K/V heads. Can be different from Q.
            # [B, H_k, L + L_past, D], [B, H_k, L + L_past, D]
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        ####  ATTENTION
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # attn_output: [B, L, H_q, D]
        # attn_weights: [B, H, L, L+L_past]

        if self.layer_idx == 0:
            print("tick:", cache_position)

        if not self.luka_kv_caches.initialized[self.layer_idx]:

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
            if self.luka_kv_caches.raw_cache is None:
                self.luka_kv_caches.raw_cache = past_key_value
            self.luka_kv_caches.buffer_weights(self.layer_idx, attn_weights, attention_mask)
            page_indices = self.luka_kv_caches.return_page_boundaries(self.layer_idx)
            k, v = past_key_value.layers[self.layer_idx].keys, past_key_value.layers[self.layer_idx].values
            self.luka_kv_caches.finalize_pages_and_build_summaries(self.layer_idx, k, v, page_indices)
            self.luka_kv_caches.initialized[self.layer_idx] = True
            print(f"LuKA layer {self.layer_idx} page_ends:", page_indices[0].tolist())
                

        else:
            # Decode-time: run LuKA top-down attention using the summary pages
            # For now we assume query_states heads already match KV heads (HF has
            # already handled repeat_kv inside attention_interface in prefill).
            # query_states: [B, H_q, L, D], with L typically 1 in decoding.

            thresh = _kv_params_override.get("refine_threshold", 0.15)
            attn_output, attn_weights = self.luka_kv_caches.top_down_attention(
                layer_idx=self.layer_idx,
                query_states=query_states,
                scaling=self.scaling,
                num_kv_groups=self.num_key_value_groups,
                attention_mask=attention_mask,
                sliding_window=self.sliding_window,
                threshold=thresh,
            )
            attn_output = attn_output.transpose(1, 2)
            if self.layer_idx == 0:
                print(self.luka_kv_caches.refine_stats[0])
                
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


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
