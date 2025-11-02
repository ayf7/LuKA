"""
This script provides visualizations on the attention weight matrix given some
sequence. Specifically, it does the following:

1. Feed the model a sequence
2. For each layer and attention head, plot attention scores.
3. Given a binary rule, plot positive/negative.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from exploration_scripts.rules import (
  Rule,
  SimpleThresholdRule,
  MedianThresholdRule,
  PercentileDecileRule
)
import matplotlib.colors as mcolors

TEXT = "In a future scenario where artificial general intelligence (AGI) systems are integrated into the global economy, education, and governance, how should policymakers design adaptive legal frameworks that balance innovation with ethical responsibility? Specifically, discuss how dynamic regulation could be structured to respond to rapidly evolving AI capabilities without stifling creativity, while also ensuring transparency, fairness, and accountability across international jurisdictions. Additionally, consider the potential socioeconomic consequences of AGI-driven automation on employment and inequality, and propose policy interventions that could mitigate these effects while maintaining long-term economic stability and public trust in technological institutions."
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
RULE: Rule = SimpleThresholdRule(tau=0.0005)
RULE: Rule = MedianThresholdRule()
RULE: Rule = PercentileDecileRule()

MESSAGES = [
  {"role": "system",    "content": "You are a helpful assistant."},
  {"role": "user",      "content": f"{TEXT}"}
]

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- LOADING THINGS ---

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# model
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,    trust_remote_code=True,
    attn_implementation = "eager" # needed to obtain attentions
).to(device)
model.eval()

# token encodings
prompt_str = tokenizer.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
enc = tokenizer(prompt_str, return_tensors="pt").to(device)

# generation
out : CausalLMOutputWithPast = model(
    **enc,
    output_attentions=True,
    return_dict=True,
    use_cache=False
)

# Grab first layer & first head for the simplest view
# out.attentions: list of length L; each tensor (B, H, Tq, Tk)
def plot_attention_layers(out: CausalLMOutputWithPast,
                          rule: Rule,
                          model_name: str,
                          layer_indices: List[int],
                          head_indices: List[int]):
    os.makedirs("attn_simple", exist_ok=True)

    for L in layer_indices:
        attn_layer = out.attentions[L][0]  # (H, Tq, Tk)
        fig, axes = plt.subplots(len(head_indices), 2, figsize=(10, 3 * len(head_indices)))

        if len(head_indices) == 1:
            axes = [axes]

        for i, H in enumerate(head_indices):
            attn_h = attn_layer[H].detach().float().cpu()
            mask_h = rule.process(attn_h).to(torch.float32).cpu()

            # Use logarithmic normalization for better dynamic range visualization
            eps = 1e-6  # prevent log(0)
            norm = mcolors.LogNorm(vmin=eps, vmax=float(attn_h.max()) + eps)

            im0 = axes[i][0].imshow(
                attn_h.numpy(),
                aspect="auto",
                interpolation="nearest",
                norm=norm,
                cmap="viridis"
            )
            im1 = axes[i][1].imshow(
                mask_h.numpy(),
                aspect="auto",
                interpolation="nearest",
                vmin=0.0,
                vmax=1.0,
                cmap="gray"
            )

            axes[i][0].set_title(f"Layer {L}, Head {H} Attention (log scale)")
            axes[i][1].set_title(f"Layer {L}, Head {H} Rule Mask")
            axes[i][0].set_xlabel("Key (k)")
            axes[i][1].set_xlabel("Key (k)")
            axes[i][0].set_ylabel("Query (q)")
            axes[i][1].set_ylabel("Query (q)")

            fig.colorbar(im0, ax=axes[i][0], fraction=0.046, pad=0.04)
            fig.colorbar(im1, ax=axes[i][1], fraction=0.046, pad=0.04)

        fig.suptitle(f"{model_name} — Layer {L}", fontsize=11)
        fig.tight_layout()
        out_path = os.path.join("attn_simple", f"layer{L}.png")
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"Saved {out_path}")

plot_attention_layers(out, RULE, MODEL, layer_indices=[0, 3, 6, 9, 12, 15], head_indices=[0, 1, 2, 3, 4])
