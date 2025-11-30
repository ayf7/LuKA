"""
Test script for LuKA with HuggingFace Transformers.
Simple script to test boundary detection during generation.
"""

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoTokenizer
from modeling.qwen.luka_qwen3 import load_luka_model, set_luka_kv_params
from artifacts.prompts.prompt_loader import load_prompt

# Configuration
model_name = "Qwen/Qwen3-1.7B-Base"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configure LuKA cache parameters via registry (matches run_batch.py style)
set_luka_kv_params(
    compressor="mean",
    # segmenter="gaussian",
    # segmenter_kwargs={"mean": 16, "std": 8},
    segmenter="kl",
    segmenter_kwargs={"threshold": 999, "lag": 8}, # kl params
    refine_threshold=1,
)

# Load model and tokenizer
model = load_luka_model(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load prompts (single)
prompts = [load_prompt("paragraphs_2")]

# Tokenize with padding for batching
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(prompts, return_tensors="pt", padding=True)
if device == "cuda":
    inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate
outputs = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

# Decode and print each result
for i, (prompt, output) in enumerate(zip(prompts, outputs)):
    generated_text = tokenizer.decode(output, skip_special_tokens=True)
    new_text = generated_text[len(prompt):]

    print("\n" + "=" * 80)
    print(f"Prompt {i}:")
    print("=" * 80)
    print(prompt)
    print("=" * 80 + "\n")

    print("Generated output:")
    print(new_text)
    print("\n" + "=" * 80)

# Print LuKA debug summaries after generation
if hasattr(model, "model") and hasattr(model.model, "luka_kv_controller"):
    controller = model.model.luka_kv_controller
    print("\n=== LuKA Layer Summaries ===")
    for layer_idx in range(controller.num_layers):
        controller.print_layer_summary(layer_idx)

    def save_attention_heatmaps(ctrl, output_dir="artifacts/attention_plots", head_idx: int = 0, batch_idx: int = 0):
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for layer_idx in range(ctrl.num_layers):
            attn_weights, _, _ = ctrl.attn_buffer[layer_idx].get_data()
            if attn_weights is None:
                continue
            if batch_idx >= attn_weights.shape[0] or head_idx >= attn_weights.shape[1]:
                continue
            attn = attn_weights[batch_idx, head_idx].detach().cpu().float().numpy()  # [L, T]

            fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
            attn_norm = matplotlib.colors.LogNorm(vmin=max(attn.min(), 1e-8), vmax=max(attn.max(), 1e-8))
            im0 = ax.imshow(attn, aspect="auto", interpolation="nearest", origin="upper", norm=attn_norm)
            ax.set_title(f"Layer {layer_idx} Head {head_idx} Attention")
            ax.set_xlabel("Key positions")
            ax.set_ylabel("Query positions")
            fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

            fig.savefig(out_dir / f"attn_layer{layer_idx}_head{head_idx}.png")
            plt.close(fig)
            print(f"Saved attention heatmap: {out_dir / f'attn_layer{layer_idx}_head{head_idx}.png'}")

    save_attention_heatmaps(controller)
