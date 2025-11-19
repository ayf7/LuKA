"""
LuKA integration for HuggingFace Transformers Qwen3 models.

This module provides a monkey-patch approach to integrate LuKA's KV cache
compression into HuggingFace's transformers library.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3
QwenAttention = modeling_qwen3.Qwen3Attention

from modeling.segmenter import Rule, SimpleLaggedKLDivergenceRule
from modeling.kv_cache import LukaKVCache

class LukaQwenAttention(QwenAttention):
    """
    Drop-in replacement for HuggingFace's Qwen3Attention (or Qwen2Attention) that integrates LuKA.

    This wraps the standard attention mechanism and adds:
    1. Boundary detection during forward passes
    2. Optional KV cache compression (to be implemented)
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

        # Initialize LuKA KV cache manager
        boundary_rule = SimpleLaggedKLDivergenceRule(
            lag=32,
            eps=1e-8,
            threshold=2.0,
        )
        self.luka_cache = LukaKVCache(boundary_rule=boundary_rule)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass with LuKA boundary detection.

        When using eager attention mode, this enables output_attentions to extract
        attention weights for boundary detection.
        """
        # Get outputs from parent
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )


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
