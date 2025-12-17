"""
Test script for LuKA with mixed attention modes (per-layer configuration).
Compares topdown attention vs lined attention vs mixed configurations.

Usage:
    # Run all three modes sequentially
    python scripts/run_batch_mixed.py

    # Run specific mode
    python scripts/run_batch_mixed.py --mode topdown
    python scripts/run_batch_mixed.py --mode lined
    python scripts/run_batch_mixed.py --mode mixed

    # Custom mixed configuration (lined on layers 0-5 and 23-27)
    python scripts/run_batch_mixed.py --mode mixed --lined-layers 0,1,2,3,4,5,23,24,25,26,27
"""

import argparse
import torch
from transformers import AutoTokenizer
from modeling.qwen.luka_qwen3 import load_luka_model, set_luka_kv_params
from artifacts.prompts.prompt_loader import load_prompt

parser = argparse.ArgumentParser(description="Test LuKA with mixed attention modes")
parser.add_argument("--mode", type=str, default="all",
                    choices=["all", "topdown", "lined", "mixed"],
                    help="Attention mode (default: all - runs all three)")
parser.add_argument("--lined-layers", type=str, default=None,
                    help="Comma-separated layer indices for lined attention in mixed mode "
                         "(default: 0-5,23-27 for 28-layer model)")
parser.add_argument("--compressor", type=str, default="attention_weighted",
                    choices=["attention_weighted", "mean"],
                    help="Compressor type for topdown mode (default: attention_weighted)")
parser.add_argument("--log-bias", type=str, default="adaptive_k",
                    choices=["none", "fixed_n", "adaptive_k"],
                    help="Log bias mode for topdown (default: adaptive_k)")
parser.add_argument("--grid-top-k", type=int, default=16,
                    help="Number of grid tokens for lined attention (default: 16)")
parser.add_argument("--grid-update-interval", type=int, default=16,
                    help="Grid token update interval (default: 16)")
parser.add_argument("--grid-decay", type=float, default=0.99,
                    help="Grid score decay factor (default: 0.99)")
parser.add_argument("--max-tokens", type=int, default=256,
                    help="Max new tokens to generate (default: 256)")
parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                    help="Model name (default: Qwen/Qwen3-8B)")
args = parser.parse_args()

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load prompts
prompts = [
    load_prompt("paragraphs_2"),
    load_prompt("paragraphs_1"),
]

# Tokenizer (shared across runs)
tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token


def parse_lined_layers(layers_str: str, num_layers: int) -> set:
    """Parse comma-separated layer indices into a set."""
    if layers_str is None:
        # Default: early (0-5) and late (last 5) layers for lined attention
        return set(range(0, 6)) | set(range(num_layers - 5, num_layers))
    return set(int(x.strip()) for x in layers_str.split(","))


def run_generation(mode_name: str, use_lined: bool, lined_layers: set = None):
    """Run generation with specified attention mode."""
    print(f"\n{'='*80}")
    print(f"Running with {mode_name.upper()} ATTENTION")
    print(f"{'='*80}\n")

    # Set LuKA params
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

    # Load model
    model = load_luka_model(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    # Configure attention mode
    if hasattr(model, "model") and hasattr(model.model, "luka_kv_controller"):
        controller = model.model.luka_kv_controller
        controller.use_lined_attention = use_lined

        if lined_layers is not None:
            controller.lined_layers = lined_layers
        elif use_lined:
            # Default: all layers
            controller.lined_layers = set(range(controller.num_layers))
        else:
            controller.lined_layers = set()

        # Configure grid parameters
        controller.grid_top_k = args.grid_top_k
        controller.grid_update_interval = args.grid_update_interval
        controller.grid_decay = args.grid_decay

        print(f"Configuration:")
        print(f"  Mode: {mode_name}")
        print(f"  use_lined_attention: {controller.use_lined_attention}")
        print(f"  lined_layers: {sorted(controller.lined_layers) if controller.lined_layers else 'none'}")
        print(f"  topdown_layers: {sorted(set(range(controller.num_layers)) - controller.lined_layers)}")
        print(f"  grid_top_k: {controller.grid_top_k}")
        print(f"  grid_update_interval: {controller.grid_update_interval}")
        print(f"  grid_decay: {controller.grid_decay}")
        print(f"  compressor: {args.compressor}")
        print(f"  log_bias_mode: {args.log_bias}")
        print(f"  refinement_rule: {controller.refinement_rule}\n")

    # Prepare inputs
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

    # Decode and print results
    results = []
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        new_text = generated_text[len(prompt):]
        results.append((prompt, new_text))

        print(f"\n{'-'*80}")
        print(f"Prompt {i}: {prompt[:100]}...")
        print(f"{'-'*80}")
        print(f"Generated ({mode_name}):")
        print(new_text[:500] + ("..." if len(new_text) > 500 else ""))

    # Print statistics
    if hasattr(model, "model") and hasattr(model.model, "luka_kv_controller"):
        controller = model.model.luka_kv_controller
        stats = controller.get_refinement_stats()

        print(f"\n{'='*60}")
        print(f"{mode_name.upper()} Statistics")
        print(f"{'='*60}")
        print(f"  Layers:                   {stats['num_layers']}")
        print(f"  Pages (per layer avg):    {stats['avg_pages_per_layer']:.1f}")
        print(f"  Summary tokens (per layer): {stats['avg_summaries_per_layer']:.1f}")
        print(f"  Queries processed:        {stats['avg_queries_per_layer']:.0f}")
        print(f"  Total refinements:        {stats['total_refinements_made']}")
        print(f"  Refinement rate:          {stats['refinement_rate']:.4f} ({stats['refinement_rate']*100:.2f}%)")
        print(f"  Refinements per query:    {stats['refinements_per_query']:.2f}")
        print(f"{'='*60}\n")

    # Cleanup
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return results


def main():
    all_results = {}

    # Determine which modes to run
    modes_to_run = []
    if args.mode == "all":
        modes_to_run = ["topdown", "lined", "mixed"]
    else:
        modes_to_run = [args.mode]

    # Get num_layers for mixed mode default (load model briefly)
    num_layers = 28  # Default for Qwen3-8B

    for mode in modes_to_run:
        if mode == "topdown":
            results = run_generation("topdown", use_lined=False)
        elif mode == "lined":
            results = run_generation("lined", use_lined=True, lined_layers=None)
        elif mode == "mixed":
            lined_layers = parse_lined_layers(args.lined_layers, num_layers)
            results = run_generation("mixed", use_lined=True, lined_layers=lined_layers)

        all_results[mode] = results

    # Print comparison if multiple modes were run
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}\n")

        for i in range(len(prompts)):
            print(f"\nPrompt {i}:")
            print(f"{'-'*80}")
            print(f"Prompt: {prompts[i][:100]}...")

            for mode, results in all_results.items():
                output_preview = results[i][1][:200] + "..." if len(results[i][1]) > 200 else results[i][1]
                print(f"\n{mode.upper()} output: {output_preview}")

            print(f"{'-'*80}")


if __name__ == "__main__":
    main()
