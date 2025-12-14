"""
Generate better plots from evaluation results.

This script loads results from the four separate evaluation scripts and generates:
1. Bar chart: final average perplexity
2. Per-token perplexity over time (with smoothing)
3. Per-token perplexity difference vs baseline
4. Area-under-curve (AUC) bar chart
5. Throughput plot (tokens/sec)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# Try to import scipy, fall back to numpy if not available
try:
    from scipy.ndimage import uniform_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


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


def main():
    parser = argparse.ArgumentParser(description="Generate better plots from evaluation results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/eval_results",
        help="Directory containing evaluation JSON results"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Dataset name (e.g., 'easy_2topic_short' from WikiSalad)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/better_plots",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Window size for smoothing per-token perplexity curves"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {results_dir}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Output directory: {output_dir}\n")
    
    # Load results
    results = load_results(results_dir, args.dataset_name)
    
    if not results:
        print("Error: No results found. Make sure you've run all four evaluation scripts.")
        return
    
    print(f"Loaded results for: {', '.join(results.keys())}\n")
    
    # Generate all plots
    print("Generating plots...\n")
    
    plot_final_perplexity(results, output_dir, args.dataset_name)
    plot_per_token_perplexity(results, output_dir, args.dataset_name, args.smooth_window)
    plot_perplexity_difference(results, output_dir, args.dataset_name, args.smooth_window)
    plot_auc_perplexity(results, output_dir, args.dataset_name)
    plot_throughput(results, output_dir, args.dataset_name)
    
    print(f"\n✓ All plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()

