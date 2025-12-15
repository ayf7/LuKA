"""
Comprehensive evaluation script for LuKA attention modes.
Evaluates perplexity, generation quality, and compression metrics with visualizations.

Usage:
    # With prompts from artifacts/prompts/
    python experiments/comprehensive_eval.py --model Qwen/Qwen3-1.7B-Base --prompt paragraphs_1
    
    # With WikiSalad dataset
    python experiments/comprehensive_eval.py --model Qwen/Qwen3-1.7B-Base \
        --prompt artifacts/hugging_face_wikipedia/wikisalad_datasets/easy_2topic_short.json \
        --use-wikisalad --wikisalad-example-id 0
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from artifacts.prompts.prompt_loader import load_prompt
from modeling.compressor import EncoderCompressor, MeanCompressor
from modeling.qwen.luka_qwen3 import (
    load_luka_model,
    set_luka_kv_params,
)
from modeling.segmenter import DummySegmenter


def get_compression_stats(model, layer_idx: int = 0) -> Dict:
    """Extract compression statistics from LuKA controller."""
    try:
        # Access the controller from the model
        controller = model.model.layers[layer_idx].self_attn.luka_kv
        
        # Get raw cache stats
        k_raw, v_raw, seq_start, raw_seq_start = controller.raw_cache.get_layer(
            layer_idx, with_offsets=True
        )
        raw_tokens = k_raw.shape[2] if k_raw is not None else 0
        
        # Get summary cache stats (for top-down attention)
        summary_cache = controller.summary_cache[layer_idx]
        num_pages = 0
        summary_tokens = 0
        if summary_cache.keys is not None:
            page_lens = summary_cache.page_lens
            if page_lens is not None and page_lens.numel() > 0:
                num_pages = int(page_lens[0].item())
                summary_tokens = num_pages  # Each page is 1 summary token
        
        # Get grid cache stats (for lined attention)
        grid_cache = controller.grid_cache[layer_idx]
        grid_tokens = 0
        if grid_cache.keys is not None and grid_cache.lens is not None:
            if grid_cache.lens.numel() > 0:
                grid_tokens = int(grid_cache.lens[0].item())
        
        # Get cover view stats
        cover_view = controller.cover_view[layer_idx]
        cover_tokens = 0
        if cover_view.cover_keys is not None:
            cover_tokens = cover_view.cover_keys.shape[2]
        
        # Calculate compression ratio
        if raw_tokens > 0:
            compression_ratio = raw_tokens / max(cover_tokens, 1)
        else:
            compression_ratio = 1.0
        
        # Get attention buffer stats
        attn_stats = controller.attn_buffer[layer_idx].get_stats()
        
        return {
            "raw_tokens": raw_tokens,
            "cover_tokens": cover_tokens,
            "summary_tokens": summary_tokens,
            "grid_tokens": grid_tokens,
            "num_pages": num_pages,
            "compression_ratio": compression_ratio,
            "refinement_rate": attn_stats.get("refinement_rate", 0.0) if attn_stats else 0.0,
        }
    except Exception as e:
        print(f"Warning: Could not extract compression stats: {e}")
        return {
            "raw_tokens": 0,
            "cover_tokens": 0,
            "summary_tokens": 0,
            "grid_tokens": 0,
            "num_pages": 0,
            "compression_ratio": 1.0,
            "refinement_rate": 0.0,
        }


def prefill_then_decode_perplexity(
    model,
    rollout_ids: torch.Tensor,
    prompt_len: int,
    layer_idx: int = 0,
) -> Tuple[float, List[float], float, Dict]:
    """
    Prefill with prompt, then teacher-force decode.
    Returns perplexity, per-token curve, tokens/sec, and compression stats.
    """
    device = rollout_ids.device
    B, T = rollout_ids.shape
    assert B == 1, "Only single-sequence evaluation supported."
    assert prompt_len < T, "Prompt length must be smaller than rollout length."
    
    # Prefill on prompt
    pre_ids = rollout_ids[:, :prompt_len]
    pre_mask = torch.ones_like(pre_ids)
    with torch.no_grad():
        pre_out = model(
            input_ids=pre_ids,
            attention_mask=pre_mask,
            use_cache=True,
            output_attentions=True,
        )
    past_key_values = pre_out.past_key_values
    
    # Decode token-by-token
    nll_list = []
    total_tokens = T - prompt_len
    start_time = time.perf_counter()
    compression_stats_history = []
    
    for t in range(prompt_len - 1, T - 1):
        cur_id = rollout_ids[:, t : t + 1]
        attn_mask = torch.ones(1, t + 1, device=device, dtype=rollout_ids.dtype)
        
        with torch.no_grad():
            out = model(
                input_ids=cur_id,
                attention_mask=attn_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        logits = out.logits[:, -1, :]
        target = rollout_ids[:, t + 1]
        
        log_probs = torch.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        nll_list.append(nll)
        
        # Collect compression stats periodically
        if t % 10 == 0 or t == T - 2:
            stats = get_compression_stats(model, layer_idx)
            compression_stats_history.append({
                "step": t - prompt_len + 1,
                **stats
            })
        
        past_key_values = out.past_key_values
    
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    toks_per_sec = total_tokens / max(elapsed, 1e-8)
    
    # Calculate perplexity
    nll_tensor = torch.stack(nll_list, dim=1)
    total_tokens_tensor = torch.tensor([[total_tokens]], device=device, dtype=nll_tensor.dtype)
    total_nll = nll_tensor.sum(dim=1, keepdim=True) / total_tokens_tensor
    perplexity = torch.exp(total_nll)[0, 0].item()
    
    # Cumulative curve
    cumsum = nll_tensor.cumsum(dim=1)
    counts = torch.arange(1, total_tokens + 1, device=device).unsqueeze(0)
    avg_nll = cumsum / counts
    curve = torch.exp(avg_nll)[0].tolist()
    
    # Final compression stats
    final_stats = get_compression_stats(model, layer_idx)
    
    return perplexity, curve, toks_per_sec, final_stats


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Tuple[str, Dict]:
    """Generate text and return generated text + compression stats."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs['input_ids'].shape[1]
    
    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            use_cache=True,
        )
    elapsed = time.perf_counter() - start_time
    
    # Decode only the newly generated tokens (after input_length)
    # This is more reliable than string slicing which can fail if prompt text changes
    generated_ids = outputs[0][input_length:]
    new_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Get compression stats
    stats = get_compression_stats(model)
    stats["generation_time"] = elapsed
    stats["tokens_per_sec"] = max_new_tokens / max(elapsed, 1e-8)
    
    return new_text, stats


def load_wikisalad_prompt(dataset_path: str, example_id: int = 0) -> str:
    """Load a prompt from WikiSalad dataset."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        examples = data
    else:
        examples = data.get('data', [])
    
    if example_id >= len(examples):
        raise ValueError(f"Example ID {example_id} out of range (max: {len(examples)-1})")
    
    example = examples[example_id]
    # WikiSalad format has 'prompt' field with full context
    if 'prompt' in example:
        return example['prompt']
    else:
        # Fallback: construct from segments if available
        return str(example)


def run_evaluation(
    model_name: str,
    prompt_name: str,
    device: str = "cuda",
    max_new_tokens: int = 256,
    prompt_len: Optional[int] = None,
    output_dir: str = "experiments/eval_results",
    use_wikisalad: bool = False,
    wikisalad_example_id: int = 0,
):
    """Run comprehensive evaluation for all attention modes.
    
    Args:
        use_wikisalad: If True, load prompt from WikiSalad dataset instead of prompts/
        wikisalad_example_id: Which example to use from WikiSalad dataset
    """
    device = device if torch.cuda.is_available() else "cpu"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load prompt
    if use_wikisalad:
        # Assume prompt_name is the path to WikiSalad dataset
        prompt = load_wikisalad_prompt(prompt_name, wikisalad_example_id)
        dataset_name = Path(prompt_name).stem
    else:
        prompt = load_prompt(prompt_name)
        dataset_name = prompt_name
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize full sequence for perplexity
    inputs = tokenizer(prompt, return_tensors="pt")
    if prompt_len is None:
        prompt_len = inputs["input_ids"].shape[1] // 2  # Use half as prompt
    
    # Generate baseline rollout for perplexity
    print("Generating baseline rollout...")
    set_luka_kv_params(
        default_tail_len=16,
        min_compress_chunk=16,
        max_pages=15,
        refine_threshold=-1.0,  # Force raw attention
        compressor=None,
        segmenter="dummy",
    )
    
    baseline_model = load_luka_model(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    ).to(device)
    baseline_model.eval()
    
    with torch.no_grad():
        baseline_gen = baseline_model.generate(
            **{k: v.to(device) for k, v in inputs.items()},
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            use_cache=True,
        )
    baseline_rollout = baseline_gen[0].unsqueeze(0)
    del baseline_model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # Define evaluation modes
    modes = {
        "baseline": {
            "use_lined_attention": False,
            "lined_layers": None,
            "refine_threshold": -1.0,  # Raw attention
            "description": "Baseline (raw attention)",
        },
        "topdown": {
            "use_lined_attention": False,
            "lined_layers": None,
            "refine_threshold": 0.05,
            "description": "Top-down (LuKA with pages)",
        },
        "lined": {
            "use_lined_attention": True,
            "lined_layers": None,  # All layers
            "refine_threshold": 0.05,
            "description": "Lined (H2O-style grid tokens)",
        },
        "mixed": {
            "use_lined_attention": True,
            "lined_layers": "auto",  # Will compute
            "refine_threshold": 0.05,
            "description": "Mixed (early/late lined, middle top-down)",
        },
    }
    
    results = {}
    
    for mode_name, mode_config in modes.items():
        print(f"\n{'='*80}")
        print(f"Evaluating: {mode_config['description']}")
        print(f"{'='*80}")
        
        # Configure LuKA
        compressor = EncoderCompressor(dim=128) if mode_name != "baseline" else None
        set_luka_kv_params(
            default_tail_len=16,
            min_compress_chunk=16,
            max_pages=15,
            refine_threshold=mode_config["refine_threshold"],
            compressor=compressor,
            segmenter="dummy",
        )
        
        # Load model
        model = load_luka_model(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        ).to(device)
        model.eval()
        
        # Configure lined attention if needed
        if mode_config["use_lined_attention"]:
            if not hasattr(model.model, 'layers') or len(model.model.layers) == 0:
                print(f"  Warning: Model has no layers. Skipping {mode_name} mode.")
                continue
            controller = model.model.layers[0].self_attn.luka_kv
            if controller is None:
                print(f"  Warning: Controller is None. Skipping {mode_name} mode.")
                continue
            # Set lined attention parameters (if they exist in controller)
            if hasattr(controller, 'use_lined_attention'):
                controller.use_lined_attention = True
                # Set lined attention thresholds
                if hasattr(controller, 'min_lined_seq_len'):
                    controller.min_lined_seq_len = 384
                if hasattr(controller, 'min_lined_tail_window'):
                    controller.min_lined_tail_window = 192
                if mode_config["lined_layers"] == "auto":
                    # Mixed: early 0-5 and late 23-27 lined, middle 6-22 top-down
                    num_layers = controller.num_layers
                    low_k, high_k = 6, 5
                    controller.lined_layers = set(range(0, low_k)) | set(range(num_layers - high_k, num_layers))
                elif mode_config["lined_layers"] is None:
                    # All layers
                    controller.lined_layers = set(range(controller.num_layers))
                else:
                    controller.lined_layers = mode_config["lined_layers"]
            else:
                print(f"  Warning: Controller does not have lined attention support. Skipping {mode_name} mode.")
                continue
        
        # 1. Perplexity evaluation
        print("  Computing perplexity...")
        rollout_ids = baseline_rollout.to(device)
        ppl, ppl_curve, tps, comp_stats = prefill_then_decode_perplexity(
            model, rollout_ids, prompt_len
        )
        
        # 2. Generation quality
        print("  Generating text...")
        generated_text, gen_stats = generate_text(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens
        )
        
        # Store results
        results[mode_name] = {
            "description": mode_config["description"],
            "perplexity": ppl,
            "perplexity_curve": ppl_curve,
            "tokens_per_sec": tps,
            "generated_text": generated_text,
            "compression_stats": comp_stats,
            "generation_stats": gen_stats,
        }
        
        print(f"  Perplexity: {ppl:.3f}")
        print(f"  Compression ratio: {comp_stats['compression_ratio']:.2f}x")
        print(f"  Tokens/sec: {tps:.2f}")
        
        # Cleanup
        del model
        torch.cuda.empty_cache() if device == "cuda" else None
    
    # Save results
    results_file = output_path / f"eval_results_{dataset_name}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")
    
    # Create visualizations
    create_plots(results, output_path, dataset_name)
    
    return results


def create_plots(results: Dict, output_path: Path, prompt_name: str):
    """Create visualization plots for all metrics."""
    fig_dir = output_path / "plots"
    fig_dir.mkdir(exist_ok=True)
    
    # Filter out any modes that failed (missing required keys)
    valid_modes = [m for m in results.keys() if "perplexity" in results[m] and "compression_stats" in results[m]]
    if not valid_modes:
        print("Warning: No valid results to plot. Skipping plot generation.")
        return
    
    # 1. Perplexity comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    modes = valid_modes
    perplexities = [results[m]["perplexity"] for m in modes]
    colors = ["black", "blue", "green", "orange"]
    
    bars = ax.bar(modes, perplexities, color=colors[:len(modes)], alpha=0.7)
    ax.set_ylabel("Perplexity", fontsize=12)
    ax.set_xlabel("Attention Mode", fontsize=12)
    ax.set_title("Perplexity Comparison", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels on bars
    for bar, ppl in zip(bars, perplexities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{ppl:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(fig_dir / f"perplexity_comparison_{prompt_name}.png", dpi=150)
    plt.close()
    
    # 2. Perplexity over time (curves)
    fig, ax = plt.subplots(figsize=(12, 6))
    for mode, color in zip(modes, colors[:len(modes)]):
        if "perplexity_curve" not in results[mode]:
            continue
        curve = results[mode]["perplexity_curve"]
        if not curve:
            continue
        steps = list(range(1, len(curve) + 1))
        ax.plot(steps, curve, label=results[mode]["description"], 
                color=color, linewidth=2, alpha=0.8)
    
    ax.set_xlabel("Generation Step", fontsize=12)
    ax.set_ylabel("Cumulative Perplexity", fontsize=12)
    ax.set_title("Perplexity Over Time", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / f"perplexity_curve_{prompt_name}.png", dpi=150)
    plt.close()
    
    # 3. Compression ratio comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    compression_ratios = [results[m]["compression_stats"].get("compression_ratio", 1.0) for m in modes]
    
    bars = ax.bar(modes, compression_ratios, color=colors[:len(modes)], alpha=0.7)
    ax.set_ylabel("Compression Ratio", fontsize=12)
    ax.set_xlabel("Attention Mode", fontsize=12)
    ax.set_title("Compression Ratio Comparison", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    
    for bar, ratio in zip(bars, compression_ratios):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.2f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(fig_dir / f"compression_ratio_{prompt_name}.png", dpi=150)
    plt.close()
    
    # 4. Memory usage (raw vs cover tokens)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(modes))
    width = 0.35
    
    raw_tokens = [results[m]["compression_stats"].get("raw_tokens", 0) for m in modes]
    cover_tokens = [results[m]["compression_stats"].get("cover_tokens", 0) for m in modes]
    
    bars1 = ax.bar(x - width/2, raw_tokens, width, label="Raw Tokens", 
                   color="red", alpha=0.7)
    bars2 = ax.bar(x + width/2, cover_tokens, width, label="Cover Tokens", 
                   color="blue", alpha=0.7)
    
    ax.set_ylabel("Number of Tokens", fontsize=12)
    ax.set_xlabel("Attention Mode", fontsize=12)
    ax.set_title("Memory Usage: Raw vs Cover Tokens", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / f"memory_usage_{prompt_name}.png", dpi=150)
    plt.close()
    
    # 5. Speed comparison (tokens/sec)
    fig, ax = plt.subplots(figsize=(10, 6))
    speeds = [results[m].get("tokens_per_sec", 0.0) for m in modes]
    
    bars = ax.bar(modes, speeds, color=colors[:len(modes)], alpha=0.7)
    ax.set_ylabel("Tokens per Second", fontsize=12)
    ax.set_xlabel("Attention Mode", fontsize=12)
    ax.set_title("Generation Speed Comparison", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    
    for bar, speed in zip(bars, speeds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{speed:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(fig_dir / f"speed_comparison_{prompt_name}.png", dpi=150)
    plt.close()
    
    # 6. Comprehensive comparison (all metrics normalized)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Perplexity (lower is better)
    axes[0, 0].bar(modes, perplexities, color=colors[:len(modes)], alpha=0.7)
    axes[0, 0].set_ylabel("Perplexity")
    axes[0, 0].set_title("Perplexity (lower is better)")
    axes[0, 0].grid(axis="y", alpha=0.3)
    
    # Compression ratio (higher is better)
    axes[0, 1].bar(modes, compression_ratios, color=colors[:len(modes)], alpha=0.7)
    axes[0, 1].set_ylabel("Compression Ratio")
    axes[0, 1].set_title("Compression Ratio (higher is better)")
    axes[0, 1].grid(axis="y", alpha=0.3)
    
    # Speed
    axes[1, 0].bar(modes, speeds, color=colors[:len(modes)], alpha=0.7)
    axes[1, 0].set_ylabel("Tokens/sec")
    axes[1, 0].set_title("Generation Speed")
    axes[1, 0].grid(axis="y", alpha=0.3)
    
    # Memory (cover tokens)
    axes[1, 1].bar(modes, cover_tokens, color=colors[:len(modes)], alpha=0.7)
    axes[1, 1].set_ylabel("Cover Tokens")
    axes[1, 1].set_title("Memory Usage (cover tokens)")
    axes[1, 1].grid(axis="y", alpha=0.3)
    
    plt.suptitle("Comprehensive Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / f"comprehensive_comparison_{prompt_name}.png", dpi=150)
    plt.close()
    
    print(f"\nPlots saved to: {fig_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive LuKA evaluation")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B-Base",
        help="Model name or path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="paragraphs_1",
        help="Prompt name from artifacts/prompts/",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=None,
        help="Prompt length for perplexity (default: half of prompt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/eval_results",
        help="Output directory for results and plots",
    )
    parser.add_argument(
        "--use-wikisalad",
        action="store_true",
        help="Use WikiSalad dataset instead of prompts/",
    )
    parser.add_argument(
        "--wikisalad-example-id",
        type=int,
        default=0,
        help="Which example to use from WikiSalad dataset (default: 0)",
    )
    
    args = parser.parse_args()
    
    results = run_evaluation(
        model_name=args.model,
        prompt_name=args.prompt,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        prompt_len=args.prompt_len,
        output_dir=args.output_dir,
        use_wikisalad=args.use_wikisalad,
        wikisalad_example_id=args.wikisalad_example_id,
    )
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    for mode, data in results.items():
        print(f"\n{mode.upper()} ({data['description']}):")
        print(f"  Perplexity: {data['perplexity']:.3f}")
        print(f"  Compression: {data['compression_stats']['compression_ratio']:.2f}x")
        print(f"  Speed: {data['tokens_per_sec']:.2f} tokens/sec")
        print(f"  Raw tokens: {data['compression_stats']['raw_tokens']}")
        print(f"  Cover tokens: {data['compression_stats']['cover_tokens']}")


if __name__ == "__main__":
    main()

