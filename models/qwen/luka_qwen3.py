import torch
import torch.nn as nn

import vllm.model_executor.models.qwen3 as qwen3_mod

# Keep a handle to the original attention for inheritance.
BaseQwen3Attention = qwen3_mod.Qwen3Attention


class LukaQwenAttention(BaseQwen3Attention):
    """
    Drop-in replacement for vLLM's Qwen3Attention.

    - Reuses the original __init__ (QKV projections, ROPE, backend Attention, etc.)
    - Only overrides forward(), where you can plug in the LuKA attention logic.
    - Because we don't touch __init__, we don't create extra Attention() instances,
      so there are no duplicate layer-name issues.
    """

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: WRITE CUSTOM LUKA QWEN ATTENTION AFTER THIS.
        return super().forward(positions=positions, hidden_states=hidden_states)


# ---------------------------------------------------------------------
# Global hook: replace Qwen3Attention before any Qwen3Model is created.
# ---------------------------------------------------------------------

def initialize_luka_hook():
    qwen3_mod.Qwen3Attention = LukaQwenAttention

    from vllm.model_executor.models.registry import ModelRegistry
    ModelRegistry.register_model("LukaQwenForCausalLM", LukaQwenForCausalLM)

# ---------------------------------------------------------------------
# Optional convenience alias for your registry:
# ---------------------------------------------------------------------

class LukaQwenForCausalLM(qwen3_mod.Qwen3ForCausalLM):
    """
    Identical to vLLM's Qwen3ForCausalLM, but uses LukaQwenAttention because
    we monkey-patched Qwen3Attention above.
    """
    pass