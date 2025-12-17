"""
Generate better plots from evaluation results.

This script runs all 4 evaluations (baseline, topdown, lined, mixed) and generates:
1. Bar chart: final average perplexity
2. Per-token perplexity over time (with smoothing)
3. Per-token perplexity difference vs baseline
4. Area-under-curve (AUC) bar chart
5. Throughput plot (tokens/sec)
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
from modeling.compressor import EncoderCompressor
from modeling.qwen.luka_qwen3 import (
    load_luka_model,
    set_luka_kv_params,
)

# Try to import scipy, fall back to numpy if not available
try:
    from scipy.ndimage import uniform_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


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
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_text = generated_text[len(prompt):]
    
    # Get compression stats
    stats = get_compression_stats(model)
    stats["generation_time"] = elapsed
    stats["tokens_per_sec"] = max_new_tokens / max(elapsed, 1e-8)
    
    return new_text, stats


def load_results(results_dir: Path, dataset_name: str) -> Dict:
    """Load results from all four evaluation scripts."""
    results = {}
    
    modes = ["baseline", "topdown", "lined", "mixed"]
    for mode in modes:
        result_file = results_dir / f"eval_{mode}_{dataset_name}.json"
        if result_file.exists():
            try:
                with open(result_file, "r") as f:
                    data = json.load(f)
                    if mode in data:
                        results[mode] = data[mode]
                    else:
                        # Try to get first (and only) key if structure is different
                        if len(data) == 1:
                            first_key = list(data.keys())[0]
                            results[mode] = data[first_key]
                        else:
                            print(f"Warning: {mode} not found in {result_file}")
            except (json.JSONDecodeError, KeyError, Exception) as e:
                print(f"Warning: Error loading {result_file}: {e}")
        else:
            print(f"Warning: {result_file} not found")
    
    return results


def cumulative_to_per_token(cumulative_curve: List[float]) -> List[float]:
    """
    Convert cumulative perplexity curve to per-token perplexity.
    
    Cumulative curve: PPL_cum[i] = exp(mean(NLL[0:i+1])) for i=0,1,2,...
    Per-token: PPL_token[i] = exp(NLL[i])
    
    We derive per-token from cumulative:
    - cumulative[i] = exp(sum(nll[0:i+1]) / (i+1))
    - cumulative[i-1] = exp(sum(nll[0:i]) / i)
    - From these, we can solve for nll[i]
    """
    if not cumulative_curve or len(cumulative_curve) == 0:
        return []
    
    # Filter out invalid values (None, NaN, inf)
    valid_curve = [x for x in cumulative_curve if x is not None and np.isfinite(x)]
    if not valid_curve:
        return []
    
    per_token = []
    
    for i, cum_ppl in enumerate(valid_curve):
        if not np.isfinite(cum_ppl) or cum_ppl <= 0:
            # Skip invalid values
            if per_token:
                per_token.append(per_token[-1])  # Use previous value
            else:
                per_token.append(1.0)  # Default
            continue
            
        if i == 0:
            # First token: per-token = cumulative
            token_ppl = cum_ppl
        else:
            # cumulative[i] = exp(sum(nll[0:i+1]) / (i+1))
            # cumulative[i-1] = exp(sum(nll[0:i]) / i)
            # sum(nll[0:i+1]) = (i+1) * log(cumulative[i])
            # sum(nll[0:i]) = i * log(cumulative[i-1])
            # nll[i] = (i+1) * log(cumulative[i]) - i * log(cumulative[i-1])
            prev_cum_ppl = valid_curve[i-1]
            if prev_cum_ppl <= 0 or not np.isfinite(prev_cum_ppl):
                # Fallback to previous per-token value
                token_ppl = per_token[-1] if per_token else cum_ppl
            else:
                try:
                    cum_nll_i = np.log(cum_ppl)
                    cum_nll_prev = np.log(prev_cum_ppl)
                    token_nll = (i + 1) * cum_nll_i - i * cum_nll_prev
                    token_ppl = np.exp(token_nll)
                    if not np.isfinite(token_ppl):
                        token_ppl = per_token[-1] if per_token else cum_ppl
                except (ValueError, OverflowError):
                    token_ppl = per_token[-1] if per_token else cum_ppl
        
        per_token.append(token_ppl)
    
    return per_token


def smooth_curve(curve: List[float], window_size: int = 5) -> np.ndarray:
    """Apply moving average smoothing to a curve."""
    if not curve or len(curve) == 0:
        return np.array([])
    
    arr = np.array(curve)
    if len(arr) < window_size:
        return arr
    
    if HAS_SCIPY:
        return uniform_filter1d(arr, size=window_size, mode='nearest')
    else:
        # Fallback: simple moving average using numpy
        kernel = np.ones(window_size) / window_size
        # Pad the array to handle boundaries
        padded = np.pad(arr, (window_size // 2, window_size // 2), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        return smoothed


def plot_final_perplexity(results: Dict, output_path: Path, dataset_name: str):
    """Plot 1: Bar chart of final average perplexity."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    modes = ["baseline", "topdown", "lined", "mixed"]
    mode_labels = {
        "baseline": "Baseline",
        "topdown": "Top-down",
        "lined": "Lined",
        "mixed": "Mixed"
    }
    colors = {"baseline": "black", "topdown": "blue", "lined": "green", "mixed": "orange"}
    
    perplexities = []
    labels = []
    bar_colors = []
    
    for mode in modes:
        if mode in results:
            ppl = results[mode].get("perplexity")
            if ppl is not None and np.isfinite(ppl):
                perplexities.append(ppl)
                labels.append(mode_labels[mode])
                bar_colors.append(colors[mode])
    
    if not perplexities:
        print("Warning: No valid perplexity data found. Skipping final perplexity plot.")
        plt.close()
        return
    
    bars = ax.bar(labels, perplexities, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel("Final Average Perplexity", fontsize=12, fontweight='bold')
    ax.set_xlabel("Attention Mode", fontsize=12, fontweight='bold')
    ax.set_title("Final Average Perplexity Comparison", fontsize=14, fontweight='bold')
    ax.grid(axis="y", alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, ppl in zip(bars, perplexities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{ppl:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / f"1_final_perplexity_{dataset_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: 1_final_perplexity_{dataset_name}.png")


def plot_per_token_perplexity(results: Dict, output_path: Path, dataset_name: str, smooth_window: int = 5):
    """Plot 2: Per-token perplexity over time (NOT cumulative)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    modes = ["baseline", "topdown", "lined", "mixed"]
    mode_labels = {
        "baseline": "Baseline",
        "topdown": "Top-down",
        "lined": "Lined",
        "mixed": "Mixed"
    }
    colors = {"baseline": "black", "topdown": "blue", "lined": "green", "mixed": "orange"}
    linestyles = {"baseline": "-", "topdown": "--", "lined": "-.", "mixed": "-"}
    
    for mode in modes:
        if mode not in results:
            continue
        
        cumulative_curve = results[mode].get("perplexity_curve")
        if cumulative_curve is None or not cumulative_curve:
            print(f"Warning: No perplexity curve for {mode}")
            continue
        
        # Convert cumulative to per-token
        per_token_curve = cumulative_to_per_token(cumulative_curve)
        if not per_token_curve or len(per_token_curve) == 0:
            print(f"Warning: Could not convert cumulative to per-token for {mode}")
            continue
        
        # Smooth the curve
        smoothed = smooth_curve(per_token_curve, window_size=smooth_window)
        if len(smoothed) == 0:
            continue
        
        # Plot
        token_indices = list(range(1, len(smoothed) + 1))
        ax.plot(token_indices, smoothed, 
                label=mode_labels[mode], 
                color=colors[mode], 
                linestyle=linestyles[mode],
                linewidth=2, 
                alpha=0.8)
    
    ax.set_xlabel("Token Index", fontsize=12, fontweight='bold')
    ax.set_ylabel("Per-Token Perplexity", fontsize=12, fontweight='bold')
    ax.set_title("Per-Token Perplexity Over Generation", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path / f"2_per_token_perplexity_{dataset_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: 2_per_token_perplexity_{dataset_name}.png")


def plot_perplexity_difference(results: Dict, output_path: Path, dataset_name: str, smooth_window: int = 5):
    """Plot 3: Per-token perplexity difference vs baseline."""
    if "baseline" not in results:
        print("Warning: Baseline results not found. Skipping difference plot.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get baseline per-token curve
    baseline_cumulative = results["baseline"].get("perplexity_curve")
    if baseline_cumulative is None or not baseline_cumulative:
        print("Warning: No baseline perplexity curve. Skipping difference plot.")
        plt.close()
        return
    
    baseline_per_token = cumulative_to_per_token(baseline_cumulative)
    if not baseline_per_token or len(baseline_per_token) == 0:
        print("Warning: Could not convert baseline cumulative to per-token. Skipping difference plot.")
        plt.close()
        return
    
    baseline_smoothed = smooth_curve(baseline_per_token, window_size=smooth_window)
    if len(baseline_smoothed) == 0:
        print("Warning: Baseline smoothed curve is empty. Skipping difference plot.")
        plt.close()
        return
    
    modes = ["topdown", "lined", "mixed"]
    mode_labels = {
        "topdown": "Top-down",
        "lined": "Lined",
        "mixed": "Mixed"
    }
    colors = {"topdown": "blue", "lined": "green", "mixed": "orange"}
    linestyles = {"topdown": "--", "lined": "-.", "mixed": "-"}
    
    # Plot baseline as zero line
    token_indices = list(range(1, len(baseline_smoothed) + 1))
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5, label='Baseline (reference)')
    
    for mode in modes:
        if mode not in results:
            continue
        
        cumulative_curve = results[mode].get("perplexity_curve")
        if cumulative_curve is None or not cumulative_curve:
            continue
        
        per_token_curve = cumulative_to_per_token(cumulative_curve)
        if not per_token_curve or len(per_token_curve) == 0:
            continue
        
        # Align lengths
        min_len = min(len(baseline_smoothed), len(per_token_curve))
        if min_len == 0:
            continue
        
        mode_smoothed = smooth_curve(per_token_curve[:min_len], window_size=smooth_window)
        if len(mode_smoothed) == 0:
            continue
        
        # Ensure both arrays have same length
        actual_min_len = min(len(mode_smoothed), len(baseline_smoothed))
        mode_smoothed = mode_smoothed[:actual_min_len]
        baseline_aligned = baseline_smoothed[:actual_min_len]
        
        # Compute difference
        diff = mode_smoothed - baseline_aligned
        if len(diff) == 0:
            continue
        
        # Plot
        token_indices_aligned = list(range(1, min_len + 1))
        ax.plot(token_indices_aligned, diff,
                label=mode_labels[mode],
                color=colors[mode],
                linestyle=linestyles[mode],
                linewidth=2,
                alpha=0.8)
    
    ax.set_xlabel("Token Index", fontsize=12, fontweight='bold')
    ax.set_ylabel("Δ Perplexity (vs Baseline)", fontsize=12, fontweight='bold')
    ax.set_title("Per-Token Perplexity Difference Relative to Baseline", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Add annotation about negative values
    ax.text(0.02, 0.98, "Negative values = improved prediction", 
            transform=ax.transAxes, fontsize=9, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path / f"3_perplexity_difference_{dataset_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: 3_perplexity_difference_{dataset_name}.png")


def plot_auc_perplexity(results: Dict, output_path: Path, dataset_name: str):
    """Plot 4: Area-under-curve (AUC) bar chart."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    modes = ["baseline", "topdown", "lined", "mixed"]
    mode_labels = {
        "baseline": "Baseline",
        "topdown": "Top-down",
        "lined": "Lined",
        "mixed": "Mixed"
    }
    colors = {"baseline": "black", "topdown": "blue", "lined": "green", "mixed": "orange"}
    
    aucs = []
    labels = []
    bar_colors = []
    
    for mode in modes:
        if mode not in results:
            continue
        
        cumulative_curve = results[mode].get("perplexity_curve")
        if cumulative_curve is None or not cumulative_curve:
            continue
        
        # Convert to per-token and compute mean (AUC approximation)
        per_token_curve = cumulative_to_per_token(cumulative_curve)
        if per_token_curve and len(per_token_curve) > 0:
            # Filter out invalid values
            valid_curve = [x for x in per_token_curve if np.isfinite(x) and x > 0]
            if valid_curve:
                auc = np.mean(valid_curve)  # Mean per-token perplexity
                if np.isfinite(auc):
                    aucs.append(auc)
                    labels.append(mode_labels[mode])
                    bar_colors.append(colors[mode])
    
    if not aucs:
        print("Warning: No valid AUC data found. Skipping AUC plot.")
        plt.close()
        return
    
    bars = ax.bar(labels, aucs, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel("Mean Per-Token Perplexity", fontsize=12, fontweight='bold')
    ax.set_xlabel("Attention Mode", fontsize=12, fontweight='bold')
    ax.set_title("Area-Under-Curve (Mean Per-Token Perplexity)", fontsize=14, fontweight='bold')
    ax.grid(axis="y", alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{auc:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / f"4_auc_perplexity_{dataset_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: 4_auc_perplexity_{dataset_name}.png")


def plot_throughput(results: Dict, output_path: Path, dataset_name: str):
    """Plot 5: Throughput (tokens/sec) bar chart."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    modes = ["baseline", "topdown", "lined", "mixed"]
    mode_labels = {
        "baseline": "Baseline",
        "topdown": "Top-down",
        "lined": "Lined",
        "mixed": "Mixed"
    }
    colors = {"baseline": "black", "topdown": "blue", "lined": "green", "mixed": "orange"}
    
    throughputs = []
    labels = []
    bar_colors = []
    
    for mode in modes:
        if mode not in results:
            continue
        
        tps = results[mode].get("tokens_per_sec")
        if tps is not None and np.isfinite(tps) and tps > 0:
            throughputs.append(tps)
            labels.append(mode_labels[mode])
            bar_colors.append(colors[mode])
    
    if not throughputs:
        print("Warning: No throughput data found. Skipping throughput plot.")
        plt.close()
        return
    
    bars = ax.bar(labels, throughputs, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel("Throughput (tokens/sec)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Attention Mode", fontsize=12, fontweight='bold')
    ax.set_title("Decoding Throughput Comparison", fontsize=14, fontweight='bold')
    ax.grid(axis="y", alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, tps in zip(bars, throughputs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{tps:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / f"5_throughput_{dataset_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: 5_throughput_{dataset_name}.png")


def run_all_evaluations(
    model_name: str,
    prompt_name: str,
    device: str = "cuda",
    max_new_tokens: int = 256,
    prompt_len: Optional[int] = None,
    use_wikisalad: bool = False,
    wikisalad_example_id: int = 0,
) -> Dict:
    """Run all 4 evaluations and return results."""
    device = device if torch.cuda.is_available() else "cpu"
    
    # Load prompt
    if use_wikisalad:
        prompt = load_wikisalad_prompt(prompt_name, wikisalad_example_id)
        dataset_name = Path(prompt_name).stem
    else:
        prompt = load_prompt(prompt_name)
        dataset_name = prompt_name
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize for perplexity
    inputs = tokenizer(prompt, return_tensors="pt")
    if prompt_len is None:
        prompt_len = inputs["input_ids"].shape[1] // 2
    
    # Generate baseline rollout
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
            "refine_threshold": -1.0,
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
            "lined_layers": None,
            "refine_threshold": 0.05,
            "description": "Lined (H2O-style grid tokens)",
        },
        "mixed": {
            "use_lined_attention": True,
            "lined_layers": "auto",
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
            if hasattr(controller, 'use_lined_attention'):
                controller.use_lined_attention = True
                if hasattr(controller, 'min_lined_seq_len'):
                    controller.min_lined_seq_len = 384
                if hasattr(controller, 'min_lined_tail_window'):
                    controller.min_lined_tail_window = 192
                if mode_config["lined_layers"] == "auto":
                    num_layers = controller.num_layers
                    low_k, high_k = 6, 5
                    controller.lined_layers = set(range(0, low_k)) | set(range(num_layers - high_k, num_layers))
                elif mode_config["lined_layers"] is None:
                    controller.lined_layers = set(range(controller.num_layers))
                else:
                    controller.lined_layers = mode_config["lined_layers"]
            else:
                print(f"  Warning: Controller does not have lined attention support. Skipping {mode_name} mode.")
                continue
        
        # Perplexity evaluation
        print("  Computing perplexity...")
        rollout_ids = baseline_rollout.to(device)
        ppl, ppl_curve, tps, comp_stats = prefill_then_decode_perplexity(model, rollout_ids, prompt_len)
        
        # Generation
        print("  Generating text...")
        generated_text, gen_stats = generate_text(model, tokenizer, prompt, max_new_tokens)
        
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
    
    return results, dataset_name


def main():
    parser = argparse.ArgumentParser(description="Run all evaluations and generate better plots")
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
        help="Prompt name from artifacts/prompts/ or path to WikiSalad JSON",
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
        default="experiments/better_plots",
        help="Output directory for plots"
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
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Window size for smoothing per-token perplexity curves"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running all evaluations and generating better plots...")
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print(f"Output directory: {output_dir}\n")
    
    # Run all evaluations
    results, dataset_name = run_all_evaluations(
        model_name=args.model,
        prompt_name=args.prompt,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        prompt_len=args.prompt_len,
        use_wikisalad=args.use_wikisalad,
        wikisalad_example_id=args.wikisalad_example_id,
    )
    
    if not results:
        print("Error: No results generated.")
        return
    
    # Validate results have required keys
    valid_results = {}
    for mode, data in results.items():
        if isinstance(data, dict) and "perplexity" in data and "perplexity_curve" in data:
            valid_results[mode] = data
        else:
            print(f"Warning: {mode} results missing required keys, skipping")
    
    if not valid_results:
        print("Error: No valid results to plot.")
        return
    
    print(f"\n{'='*80}")
    print("Generating plots...")
    print(f"{'='*80}\n")
    
    # Generate all plots (with error handling)
    try:
        plot_final_perplexity(valid_results, output_dir, dataset_name)
    except Exception as e:
        print(f"Error generating final perplexity plot: {e}")
    
    try:
        plot_per_token_perplexity(valid_results, output_dir, dataset_name, args.smooth_window)
    except Exception as e:
        print(f"Error generating per-token perplexity plot: {e}")
    
    try:
        plot_perplexity_difference(valid_results, output_dir, dataset_name, args.smooth_window)
    except Exception as e:
        print(f"Error generating perplexity difference plot: {e}")
    
    try:
        plot_auc_perplexity(valid_results, output_dir, dataset_name)
    except Exception as e:
        print(f"Error generating AUC plot: {e}")
    
    try:
        plot_throughput(valid_results, output_dir, dataset_name)
    except Exception as e:
        print(f"Error generating throughput plot: {e}")
    
    print(f"\n✓ All plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()

