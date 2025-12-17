"""
Test script for LuKA with HuggingFace Transformers (new controller path).
Simple script to test boundary detection during generation.

Usage:
    python scripts/run_batch.py --compressor attention_weighted --log-bias adaptive_k
    python scripts/run_batch.py --compressor mean --log-bias adaptive_k
    python scripts/run_batch.py --compressor mean --log-bias fixed_n
"""

import argparse
import torch
from transformers import AutoTokenizer
from modeling.qwen.luka_qwen3 import load_luka_model, set_luka_kv_params
from artifacts.prompts.prompt_loader import load_prompt

parser = argparse.ArgumentParser(description="Test LuKA with different compressors")
parser.add_argument("--compressor", type=str, default="attention_weighted",
                    choices=["attention_weighted", "mean"],
                    help="Compressor type (default: attention_weighted)")
parser.add_argument("--log-bias", type=str, default="adaptive_k",
                    choices=["none", "fixed_n", "adaptive_k"],
                    help="Log bias mode (default: adaptive_k)")
parser.add_argument("--max-tokens", type=int, default=256,
                    help="Max new tokens to generate (default: 256)")
args = parser.parse_args()

# Configuration
model_name = "Qwen/Qwen3-8B"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Compressor: {args.compressor}")
print(f"Log bias mode: {args.log_bias}")

# Active config (from CLI args)
compressor_kwargs = {"temperature": 7.0} if args.compressor == "attention_weighted" else {}
set_luka_kv_params(
    compressor=args.compressor,
    compressor_kwargs=compressor_kwargs,
    segmenter="dummy",
    refinement_rule="top_k",
    refinement_rule_kwargs={"k": 3},
    log_bias_mode=args.log_bias,
    segment_interval=16,
)

# Load model and tokenizer
model = load_luka_model(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

# Load prompts (batch)
prompts = [
    load_prompt("paragraphs_2"),
    load_prompt("paragraphs_1"),
]

# Tokenize with padding for batching
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(prompts, return_tensors="pt", padding=True)
if device == "cuda":
    inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate
outputs = model.generate(
    **inputs,
    max_new_tokens=args.max_tokens,
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

# Print LuKA refinement statistics
if hasattr(model, "model") and hasattr(model.model, "luka_kv_controller"):
    controller = model.model.luka_kv_controller
    stats = controller.get_refinement_stats()

    print("\n" + "=" * 60)
    print("LuKA Refinement Statistics")
    print("=" * 60)
    print(f"  Refinement rule:          {controller.refinement_rule}")
    print(f"  Layers:                   {stats['num_layers']}")
    print(f"  Pages (per layer avg):    {stats['avg_pages_per_layer']:.1f}")
    print(f"  Summary tokens (per layer): {stats['avg_summaries_per_layer']:.1f}")
    print(f"  Queries processed:        {stats['avg_queries_per_layer']:.0f}")
    print(f"  Total refinements:        {stats['total_refinements_made']}")
    print(f"  Refinement rate:          {stats['refinement_rate']:.4f} ({stats['refinement_rate']*100:.2f}%)")
    print(f"  Refinements per query:    {stats['refinements_per_query']:.2f}")
    print("=" * 60 + "\n")

    # Print log(k_eff) entropy statistics per layer
    print("=" * 60)
    print("Log Effective Support (Entropy) Statistics")
    print("=" * 60)
    print(f"  {'Layer':<8} {'Pages':<8} {'Mean':<10} {'Min':<10} {'Max':<10} {'Std':<10}")
    print("-" * 60)

    all_log_k = []
    for layer_idx, layer_cache in enumerate(controller.summary_cache):
        if layer_cache is not None and layer_cache.log_effective_support is not None:
            log_k = layer_cache.log_effective_support
            # Only consider non-zero entries (actual pages)
            num_pages = layer_cache.page_lens.sum().item()
            if num_pages > 0:
                # Flatten and get valid entries
                valid_log_k = []
                for b in range(log_k.shape[0]):
                    n_pages = layer_cache.page_lens[b].item()
                    if n_pages > 0:
                        valid_log_k.append(log_k[b, :n_pages])
                if valid_log_k:
                    valid_log_k = torch.cat(valid_log_k)
                    all_log_k.append(valid_log_k)
                    mean_val = valid_log_k.mean().item()
                    min_val = valid_log_k.min().item()
                    max_val = valid_log_k.max().item()
                    std_val = valid_log_k.std().item() if len(valid_log_k) > 1 else 0.0
                    print(f"  {layer_idx:<8} {int(num_pages):<8} {mean_val:<10.3f} {min_val:<10.3f} {max_val:<10.3f} {std_val:<10.3f}")

    if all_log_k:
        all_log_k = torch.cat(all_log_k)
        print("-" * 60)
        print(f"  {'TOTAL':<8} {len(all_log_k):<8} {all_log_k.mean().item():<10.3f} {all_log_k.min().item():<10.3f} {all_log_k.max().item():<10.3f} {all_log_k.std().item():<10.3f}")

        # Also show what this means in terms of k_eff = exp(log_k)
        k_eff = all_log_k.exp()
        print("\n  Effective support k_eff = exp(entropy):")
        print(f"    Mean: {k_eff.mean().item():.2f}, Min: {k_eff.min().item():.2f}, Max: {k_eff.max().item():.2f}")
        print(f"    (For reference: page_size=16 → uniform would give k_eff=16)")
    print("=" * 60 + "\n")
