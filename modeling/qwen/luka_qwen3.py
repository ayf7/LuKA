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

from modeling.compressor import AttentionWeightedCompressor, EncoderCompressor, MeanCompressor
from modeling.kv_cache import LukaKVController
from modeling.segmenter import DummySegmenter, KLSegmenter, GaussianSegmenter
from modeling.refinement import (
    RefinementRule,
    NoRefinementRule,
    FixedThresholdRule,
    TopKRule,
    TopPRule,
    TopFracRule,
    TopKTopPRule,
    get_refinement_rule,
)

# Optional global overrides for KV cache params, settable by callers (e.g., scripts/test.py)
_kv_params_override: dict[str, float | int | object] = {}

SEGMENTER_REGISTRY = {
    "dummy": DummySegmenter,
    "kl": KLSegmenter,
    "gaussian": GaussianSegmenter,
}

COMPRESSOR_REGISTRY = {
    "mean": MeanCompressor,
    "encoder": EncoderCompressor,
    "attention_weighted": AttentionWeightedCompressor,
}

REFINEMENT_RULE_REGISTRY = {
    "none": NoRefinementRule,
    "threshold": FixedThresholdRule,
    "top_k": TopKRule,
    "top_p": TopPRule,
    "top_frac": TopFracRule,
    "top_k_top_p": TopKTopPRule,
}

def set_luka_kv_params(
    *,
    default_tail_len: int = 16,
    min_compress_chunk: int = 16,
    max_pages: int = 15,
    refinement_rule: object = "threshold",  # str, RefinementRule, or None
    refinement_rule_kwargs: dict | None = None,  # includes always_refine_first_n
    segment_interval: int = 1,
    create_pages_in_generation: bool = True,
    use_exact_attention: bool = False,
    use_log_bias: bool = False,
    log_bias_mode: str | None = None,  # "none", "fixed_n", or "adaptive_k"
    print_stats_after_generate: bool = False,
    production_mode: bool = True,
    async_pages: bool = False,
    compressor: object = "mean",
    compressor_kwargs: dict | None = None,
    segmenter: object = "dummy",
    segmenter_kwargs: dict | None = None,
) -> None:
    """Configure LuKA KV cache parameters.

    Args:
        default_tail_len: Number of raw tokens to keep uncompressed at the end.
        min_compress_chunk: Minimum number of tokens per page.
        max_pages: Maximum number of summary pages to create.
        refinement_rule: Refinement rule for selecting which summaries to expand.
            Can be:
            - str: Name from registry ("none", "threshold", "top_k", "top_p", "top_frac", "top_k_top_p")
            - RefinementRule instance
            - None: Disables refinement (NoRefinementRule)
        refinement_rule_kwargs: Kwargs for refinement rule if using string name.
            Can include always_refine_first_n for attention sinks (e.g. {"k": 3, "always_refine_first_n": 1}).
        segment_interval: How often to attempt page creation (every N decode steps).
        create_pages_in_generation: Whether to create pages during generation.
        use_exact_attention: If True, bypass cover view entirely and use raw attention.
            Use this for baseline evaluation. Ignores refinement_rule and pages.
        use_log_bias: If True, sets log_bias_mode="fixed_n" for backwards compatibility.
        log_bias_mode: Log bias mode for summary attention logits:
            - "none": No bias (default)
            - "fixed_n": log(N) where N = page length (assumes uniform attention)
            - "adaptive_k": log(k_eff) where k_eff = exp(entropy) is effective support
                           This adapts to the actual attention distribution at compression time.
        print_stats_after_generate: Print refinement stats after generate().
        production_mode: Skip debug checks for better performance.
        async_pages: Run compression on background CUDA stream.
        compressor: Compressor for page creation ("mean", "attention_weighted", etc.).
        compressor_kwargs: Kwargs for compressor if using string name.
        segmenter: Segmenter for page boundaries ("dummy", "kl", "gaussian").
        segmenter_kwargs: Kwargs for segmenter if using string name.
    """
    global _kv_params_override
    if default_tail_len is not None:
        _kv_params_override["default_tail_len"] = int(default_tail_len)
    if min_compress_chunk is not None:
        _kv_params_override["min_compress_chunk"] = int(min_compress_chunk)
    if max_pages is not None:
        _kv_params_override["max_pages"] = int(max_pages)
    if segment_interval is not None:
        _kv_params_override["segment_interval"] = int(segment_interval)
    if create_pages_in_generation is not None:
        _kv_params_override["create_pages_in_generation"] = bool(create_pages_in_generation)
    if use_exact_attention is not None:
        _kv_params_override["use_exact_attention"] = bool(use_exact_attention)
    if use_log_bias is not None:
        _kv_params_override["use_log_bias"] = bool(use_log_bias)
    if log_bias_mode is not None:
        if log_bias_mode not in ("none", "fixed_n", "adaptive_k"):
            raise ValueError(f"Invalid log_bias_mode: {log_bias_mode}. Must be 'none', 'fixed_n', or 'adaptive_k'.")
        _kv_params_override["log_bias_mode"] = log_bias_mode
    if print_stats_after_generate is not None:
        _kv_params_override["print_stats_after_generate"] = bool(print_stats_after_generate)
    if production_mode is not None:
        _kv_params_override["production_mode"] = bool(production_mode)
    if async_pages is not None:
        _kv_params_override["async_pages"] = bool(async_pages)

    # Handle refinement rule configuration
    if refinement_rule is not None:
        if isinstance(refinement_rule, RefinementRule):
            if refinement_rule_kwargs:
                raise ValueError("Cannot provide refinement_rule_kwargs when passing a RefinementRule instance.")
            _kv_params_override["refinement_rule"] = refinement_rule
        elif isinstance(refinement_rule, str):
            if refinement_rule.lower() not in REFINEMENT_RULE_REGISTRY:
                raise ValueError(
                    f"Refinement rule '{refinement_rule}' not found in registry. "
                    f"Available: {list(REFINEMENT_RULE_REGISTRY.keys())}"
                )
            kwargs = refinement_rule_kwargs or {}
            _kv_params_override["refinement_rule"] = REFINEMENT_RULE_REGISTRY[refinement_rule.lower()](**kwargs)
        elif refinement_rule is None:
            _kv_params_override["refinement_rule"] = NoRefinementRule()
        else:
            raise TypeError(
                f"refinement_rule must be str, RefinementRule, or None, "
                f"got {type(refinement_rule).__name__}"
            )

    if compressor is not None:
        if isinstance(compressor, str):
            if compressor not in COMPRESSOR_REGISTRY:
                raise ValueError(f"Compressor {compressor} not found in registry. Available: {list(COMPRESSOR_REGISTRY.keys())}")
            kwargs = compressor_kwargs or {}
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
        # Choose between lined attention (H2O-style) and top-down attention (LuKA paging)
        # attn_output: [B, H_q, L, D]
        # attn_weights: [B, H_q, L, T_raw]

        use_lined = (
            self.luka_kv.use_lined_attention and
            self.layer_idx in self.luka_kv.lined_layers
        )

        if use_lined:
            # Lined attention (H2O-style: grid tokens + raw tail)
            attn_output, attn_weights = self.luka_kv.lined_attention(
                layer_idx=self.layer_idx,
                query_states=query_states,
                scaling=self.scaling,
                num_kv_groups=self.num_key_value_groups,
                attention_mask=attention_mask,
                sliding_window=self.sliding_window,
            )
            # Lined attention doesn't use paging, so skip try_new_pages
        else:
            # Top-down attention (LuKA: pages + raw tail)
            use_exact = _kv_params_override.get("use_exact_attention", False)
            attn_output, attn_weights = self.luka_kv.top_down_attention(
                layer_idx=self.layer_idx,
                query_states=query_states,
                scaling=self.scaling,
                num_kv_groups=self.num_key_value_groups,
                attention_mask=attention_mask,
                sliding_window=self.sliding_window,
                use_exact_attention=use_exact,
            )

            # Try to create new pages (segmentation & compression)
            # Skip if using exact attention (baseline mode)
            if not use_exact and self.luka_kv.create_pages_in_generation:
                self.luka_kv.try_new_pages(self.layer_idx)

        # Match HF Qwen3: reshape back to [B, L, H*D] and apply o_proj.
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        # Verify invariants (only in debug mode)
        if self.luka_kv is not None and not self.luka_kv.production_mode:
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
        if "refinement_rule" in _kv_params_override:
            self.luka_kv_controller.refinement_rule = _kv_params_override["refinement_rule"]
        if "segment_interval" in _kv_params_override:
            self.luka_kv_controller.segment_interval = _kv_params_override["segment_interval"]
        if "create_pages_in_generation" in _kv_params_override:
            self.luka_kv_controller.create_pages_in_generation = _kv_params_override["create_pages_in_generation"]
        if "use_log_bias" in _kv_params_override:
            self.luka_kv_controller.use_log_bias = _kv_params_override["use_log_bias"]
        if "log_bias_mode" in _kv_params_override:
            self.luka_kv_controller.log_bias_mode = _kv_params_override["log_bias_mode"]
        if "production_mode" in _kv_params_override:
            self.luka_kv_controller.production_mode = _kv_params_override["production_mode"]
        if "async_pages" in _kv_params_override:
            self.luka_kv_controller.async_pages = _kv_params_override["async_pages"]

        for layer in self.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.luka_kv = self.luka_kv_controller


class LukaQwen3ForCausalLM(modeling_qwen3.Qwen3ForCausalLM):
    """
    Qwen3ForCausalLM with LuKA-specific parameter handling.

    This allows loading pretrained checkpoints while ignoring LuKA-specific
    parameters that don't exist in the base model.
    """

    def __init__(self, config):
        super().__init__(config)
        self.print_stats_after_generate = _kv_params_override.get("print_stats_after_generate", False)

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

    def generate(self, *args, **kwargs):
        """Override generate to optionally print refinement stats after completion."""
        # Reset stats before generation
        if hasattr(self.model, 'luka_kv_controller'):
            self.model.luka_kv_controller.reset_refinement_stats()

        # Run generation
        output = super().generate(*args, **kwargs)

        # Print stats if enabled
        if self.print_stats_after_generate and hasattr(self.model, 'luka_kv_controller'):
            self._print_luka_stats()

        return output

    def _print_luka_stats(self):
        """Print LuKA refinement statistics."""
        controller = self.model.luka_kv_controller
        stats = controller.get_refinement_stats()

        print("\n" + "=" * 60)
        print("LuKA Refinement Statistics")
        print("=" * 60)
        print(f"  Layers:                   {stats['num_layers']}")
        print(f"  Pages (per layer avg):    {stats['avg_pages_per_layer']:.1f}")
        print(f"  Summary tokens (per layer): {stats['avg_summaries_per_layer']:.1f}")
        print(f"  Queries processed:        {stats['avg_queries_per_layer']:.0f}")
        print(f"  Total refinements:        {stats['total_refinements_made']}")
        print(f"  Refinement rate:          {stats['refinement_rate']:.4f} ({stats['refinement_rate']*100:.2f}%)")
        print(f"  Refinements per query:    {stats['refinements_per_query']:.2f}")
        print("=" * 60 + "\n")


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
