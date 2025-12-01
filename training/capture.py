"""
Attention capture utilities for training.

Provides a monkey patch for Qwen3 attention to record q/k/v tensors, attention
masks, scaling, and grouped-query metadata per layer, plus helpers to restore
the original attention and a small manual test driver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional
import types

import torch
import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


@dataclass
class LayerCapture:
    """
    Container for a single layer's captured tensors.

    Fields:
        q: [B, H_q, L_q, D] Query states after RMSNorm and RoPE.
        k: [B, H_k, L_k, D] Key states after RMSNorm and RoPE (pre-repeat_kv).
        v: [B, H_k, L_k, D] Value states (pre-repeat_kv).
        attention_mask: [B, 1, L_q, L_k_total] Additive mask provided to attention; can be None.
        scaling: float head_dim**-0.5 used inside attention logits.
        num_kv_groups: int number of KV groups for repeat_kv (H_q == H_k * num_kv_groups).
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    scaling: float
    num_kv_groups: int


class QKVCapture:
    """
    Recorder for q/k/v captures.

    Args:
        layers: Optional iterable of layer indices to record. None captures all layers.

    Attributes:
        layers: Optional set of allowed layer indices.
        records: Dict[int, LayerCapture] keyed by layer index.
    """

    def __init__(self, layers: Optional[Iterable[int]] = None):
        self.layers = None if layers is None else set(int(i) for i in layers)
        self.records: Dict[int, LayerCapture] = {}

    def clear(self) -> None:
        """
        Drop all stored captures.

        Args:
            None.

        Returns:
            None.

        Side effects:
            Clears internal records dict.
        """
        self.records.clear()

    def add(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        num_kv_groups: int,
    ) -> None:
        """
        Store a capture for one layer if it passes the layer filter.

        Args:
            layer_idx: Integer layer index.
            q: [B, H_q, L_q, D] queries after RMSNorm + RoPE.
            k: [B, H_k, L_k, D] keys after RMSNorm + RoPE.
            v: [B, H_k, L_k, D] values.
            attention_mask: [B, 1, L_q, L_k_total] additive mask (causal/pad); can be None.
            scaling: Scalar head_dim**-0.5 used in attention.
            num_kv_groups: Repeat factor for grouped-query attention (H_q = H_k * num_kv_groups).

        Returns:
            None.

        Side effects:
            Should detach tensors and write a LayerCapture into records.
        """
        if self.layers is not None and layer_idx not in self.layers:
            return
        self.records[layer_idx] = LayerCapture(
            q=q.detach(),
            k=k.detach(),
            v=v.detach(),
            attention_mask=attention_mask.detach() if attention_mask is not None else None,
            scaling=scaling,
            num_kv_groups=num_kv_groups,
        )


def patch_qwen3_for_training(
    model: torch.nn.Module,
    layers: Optional[Iterable[int]] = None,
) -> QKVCapture:
    """
    Monkey-patch Qwen3Attention modules to capture q/k/v during forward.

    Args:
        model: Qwen3ForCausalLM (or compatible) instance.
        layers: Optional iterable of layer indices to capture; None captures all.

    Returns:
        recorder: QKVCapture that will hold the latest captures per layer.

    Side effects:
        Replaces target self_attn.forward with an instrumented wrapper and stores
        the original forward on the module for later restoration.
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("patch_qwen3_for_training expects a Qwen3* model with .model.layers")

    recorder = QKVCapture(layers)
    for idx, layer in enumerate(model.model.layers):
        if recorder.layers is not None and idx not in recorder.layers:
            continue
        attn = layer.self_attn
        if hasattr(attn, "_luka_training_orig_forward"):
            continue
        attn._luka_training_orig_forward = attn.forward
        attn.forward = types.MethodType(
            lambda self, *args, **kwargs: _instrumented_forward(self, recorder, *args, **kwargs),
            attn,
        )
    return recorder


def restore_qwen3_attention(model: torch.nn.Module) -> None:
    """
    Restore original Qwen3Attention.forward for a model patched by patch_qwen3_for_training.

    Args:
        model: Qwen3ForCausalLM (or compatible) instance previously patched.

    Returns:
        None.

    Side effects:
        Restores and deletes saved original forward methods when present.
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        return
    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, "_luka_training_orig_forward"):
            attn.forward = attn._luka_training_orig_forward
            delattr(attn, "_luka_training_orig_forward")


def _instrumented_forward(
    self: modeling_qwen3.Qwen3Attention,
    recorder: QKVCapture,
    hidden_states: torch.Tensor,  # [B, L, D_model]
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """
    Drop-in replacement for Qwen3Attention.forward that records q/k/v and mask.

    Args:
        self: Patched Qwen3Attention module.
        recorder: QKVCapture to store tensors into.
        hidden_states: [B, L, D_model] input to attention block.
        position_embeddings: Tuple (cos, sin) for RoPE, shapes [B, L, D].
        attention_mask: [B, 1, L_q, L_k_total] additive mask; can be None.
        past_key_value: HF cache (Cache or None).
        cache_position: [L] positions for cache update during decoding.
        **kwargs: Passed through to attention implementation.

    Returns:
        attn_output: [B, L, D_model]
        attn_weights: attention weights per underlying implementation (shape varies).

    Side effects:
        Updates past_key_value if provided; writes capture into recorder.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    recorder.add(
        layer_idx=self.layer_idx,
        q=query_states,
        k=key_states,
        v=value_states,
        attention_mask=attention_mask,
        scaling=self.scaling,
        num_kv_groups=self.num_key_value_groups,
    )

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


if __name__ == "__main__":
    """
    Lightweight driver to verify capture wiring on a single forward pass.

    Loads the base model, patches layer 0 for capture, runs a synthetic batch,
    and prints captured tensor shapes.

    Note: this will download the model if not cached and may use significant
    memory; intended for manual checking, not automated tests.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B-Base").to(device)
    recorder = patch_qwen3_for_training(model, layers=[0])

    seq_len = 16
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids, device=device)

    recorder.clear()
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    capture = recorder.records.get(0)
    if capture is None:
        print("No capture for layer 0.")
    else:
        print(f"q shape: {capture.q.shape}")
        print(f"k shape: {capture.k.shape}")
        print(f"v shape: {capture.v.shape}")
        if capture.attention_mask is None:
            print("attention_mask: None")
        else:
            print(f"attention_mask shape: {capture.attention_mask.shape}")
