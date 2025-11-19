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
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from artifacts.prompts.prompt_loader import load_prompt

from exploration.rules import (
  Rule,
  SimpleThresholdRule,
  MedianThresholdRule,
  MaxPoolThresholdRule,
  MagnitudeOrderRule,
  LaggedKLDivergenceRule
)
import matplotlib.colors as mcolors

MODEL = "Qwen/Qwen3-1.7B-Base"

TEXT = load_prompt("paragraphs_2")

SIMPLE_THRESHOLD: Rule = SimpleThresholdRule(tau=0.0005)
MEDIAN_THRESHOLD: Rule = MedianThresholdRule()
MAXPOOL_THRESHOLD: Rule = MaxPoolThresholdRule(tau=0.0005, kernel_size=30, stride=10)
MAGNITUDE_RULE: Rule = MagnitudeOrderRule()
DIVERGENCE_RULE: Rule = LaggedKLDivergenceRule(lag=6, threshold=10)


RULES = [
        LaggedKLDivergenceRule(lag=5),
        # LaggedKLDivergenceRule(lag=10),
        # LaggedKLDivergenceRule(lag=20)
        # CausalCrossMaxPool2DWindowRule(tau=0.005, up_radius=4, lr_radius=4, include_center=True),
        # CausalCrossMaxPool2DWindowRule(tau=0.01, up_radius=4, lr_radius=4, include_center=True)
    ]



MESSAGES = [
  {"role": "system",    "content": "You are a helpful assistant."},
  {"role": "user",      "content": f"{TEXT}"}
]

# Only recommended for a single rule
OVERWRITE_DIR = None

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
                          head_indices: List[int],
                          overwrite_dir: Optional[str] = None):
    if overwrite_dir is not None:
        directory = Path(f"profile__{model_name.replace("/","-")}") / overwrite_dir
    else:
        directory = Path(f"profile__{model_name.replace("/","-")}") / rule.name()
    os.makedirs(directory, exist_ok=True)

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
            mask_cmap = rule.color()
            mask_max = float(mask_h.max()) if mask_h.numel() > 0 else 0.0
            if mask_max == 0.0:
                mask_max = 1.0

            im1 = axes[i][1].imshow(
                mask_h.numpy(),
                aspect="auto",
                interpolation="nearest",
                # vmin=0.0,
                # vmax=mask_max,
                cmap=mask_cmap
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
        out_path = os.path.join(directory, f"layer{L}.png")
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"Saved {out_path}")


def plot_attention_layers_multiple_rules(out: CausalLMOutputWithPast,
                          rules: List[Rule],
                          model_name: str,
                          layer_indices: List[int],
                          head_indices: List[int],
                          overwrite_dir: Optional[str] = None):
    for rule in rules:
        plot_attention_layers(
            out,
            rule,
            model_name,
            layer_indices,
            head_indices,
            overwrite_dir=overwrite_dir
        )


if __name__ == "__main__":
    plot_attention_layers_multiple_rules(
        out,
        RULES,
        MODEL,
        layer_indices=[0, 13, 27],
        head_indices=[k for k in range(16)],
        overwrite_dir=OVERWRITE_DIR
    )
