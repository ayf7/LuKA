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

from exploration_scripts.rules import Rule, ThresholdRule

TEXT = "sample text"
MODEL = "meta-llama/Llama-3-8b-instruct"

RULE: Rule = ThresholdRule(tau=0.001)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- LOADING THINGS ---

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    enc = tokenizer(TEXT, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model(
            **enc,
            use_cache=False,
            output_attentions=True,
        )

    # Grab first layer & first head for the simplest view
    # out.attentions: list of length L; each tensor (B, H, Tq, Tk)
    attn = out.attentions[0][0]  # (H, Tq, Tk)
    attn_h0 = attn[0].detach().float().cpu()  # (Tq, Tk)

    # Build and apply rule
    mask_h0 = RULE.process(attn_h0).to(torch.float32).cpu()  # (Tq, Tk)

    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(attn_h0.numpy(), aspect="auto", interpolation="nearest", vmin=0.0, vmax=float(attn_h0.max()))
    axes[0].set_title("Attention (Layer 0, Head 0)")
    axes[0].set_xlabel("Key (k)")
    axes[0].set_ylabel("Query (q)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(mask_h0.numpy(), aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[1].set_title("Rule Mask (1 = LOW)")
    axes[1].set_xlabel("Key (k)")
    axes[1].set_ylabel("Query (q)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(MODEL, fontsize=11)
    fig.tight_layout()

    os.makedirs("attn_simple", exist_ok=True)
    out_path = os.path.join("attn_simple", "layer0_head0.png")
    fig.savefig(out_path, dpi=160)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()