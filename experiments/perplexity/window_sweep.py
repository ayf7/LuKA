"""
Sweep over window (tail) length for lined attention.

Tests tail_len values: 8, 16, 32, 64, 128
Context: 2048 tokens

Run:
    python experiments/perplexity/window_sweep.py
    python experiments/perplexity/window_sweep.py --dataset pg19
    python experiments/perplexity/window_sweep.py --plot-only
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from experiments.perplexity.utils import (
    MODEL_NAME,
    get_device,
    get_tokenizer,
    get_prompt,
    get_baseline_perplexity,
    load_eval_text,
    get_output_dir,
)
from modeling.compressor import AttentionWeightedCompressor
from modeling.qwen.luka_qwen3 import load_luka_model, set_luka_kv_params


# Output paths
OUTPUT_DIR = get_output_dir("window_sweep")
CSV_PATH = OUTPUT_DIR / "results.csv"
PLOT_PATH = OUTPUT_DIR / "sweep.png"
PLOT_LOG_PATH = OUTPUT_DIR / "sweep_log.png"
CURVE_PLOT_PATH = OUTPUT_DIR / "curves.png"

# Sweep values
WINDOW_SIZES = [8, 16, 32, 64, 128]


def prefill_then_decode_perplexity(model, rollout_ids, prompt_len):
    """
    Prefill with the prompt, then teacher-force through the rest.
    Returns perplexity, running avg curve, tokens/sec.
    """
    import time

    device = rollout_ids.device
    B, T = rollout_ids.shape
    assert B == 1

    pre_ids = rollout_ids[:, :prompt_len]
    pre_mask = torch.ones_like(pre_ids)
    with torch.no_grad():
        pre_out = model(
            input_ids=pre_ids,
            attention_mask=pre_mask,
            use_cache=True,
        )
    past_key_values = pre_out.past_key_values

    nll_list = []
    total_tokens = T - prompt_len
    start_time = time.perf_counter()

    # First prediction: use prefill logits
    logits = pre_out.logits[:, -1, :]
    target = rollout_ids[:, prompt_len]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    nll_list.append(nll)

    # Subsequent predictions
    for t in range(prompt_len, T - 1):
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
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        nll_list.append(nll)
        past_key_values = out.past_key_values

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    tps = total_tokens / max(elapsed, 1e-8)

    nll_tensor = torch.stack(nll_list, dim=1)
    total_nll = nll_tensor.sum(dim=1) / total_tokens
    ppl = torch.exp(total_nll)[0].item()

    # Running average perplexity curve
    cumsum = nll_tensor.cumsum(dim=1)
    counts = torch.arange(1, total_tokens + 1, device=device).unsqueeze(0)
    avg_nll = cumsum / counts
    avg_curve = torch.exp(avg_nll)[0].tolist()

    return ppl, avg_curve, tps


def run_single_window(
    window_size: int,
    rollout_ids,
    prompt_len: int,
    device: str,
    num_layers: int,
    model_name: str,
):
    """Run perplexity evaluation for a single window size."""
    print(f"\nRunning window_size={window_size}...")

    # Set LuKA params with lined attention
    set_luka_kv_params(
        default_tail_len=window_size,
        min_compress_chunk=window_size,  # Match tail_len
        max_pages=15,
        refinement_rule="top_k",
        refinement_rule_kwargs={"k": 3},
        compressor=AttentionWeightedCompressor(temperature=1.0),
        log_bias_mode="none",
        segmenter="dummy",
        segment_interval=window_size,  # Match tail_len
        create_pages_in_generation=False,
        production_mode=True,
    )

    # Load model
    model = load_luka_model(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    ).to(device)
    model.eval()

    # Configure for lined attention (H2O-style)
    if hasattr(model, "model") and hasattr(model.model, "luka_kv_controller"):
        controller = model.model.luka_kv_controller
        controller.use_lined_attention = True
        controller.lined_layers = set(range(num_layers))  # All layers lined
        controller.grid_top_k = 96
        controller.grid_update_interval = window_size
        controller.grid_decay = 0.99

    # Run perplexity evaluation
    with torch.no_grad():
        ppl, avg_curve, tps = prefill_then_decode_perplexity(model, rollout_ids, prompt_len)

    print(f"  ppl={ppl:.3f}, tps={tps:.1f}")

    del model
    torch.cuda.empty_cache()

    return {
        "perplexity": ppl,
        "avg_curve": avg_curve,
        "tokens_per_sec": tps,
    }


def save_csv(results, base_ppl, base_tps, base_curve, num_layers, model_name):
    """Save results to CSV."""
    curve_len = len(base_curve) if base_curve else 0
    token_cols = [f"token_{i}_ppl" for i in range(1, curve_len + 1)]

    fieldnames = [
        "window_size", "model", "num_layers", "perplexity", "ppl_delta", "tokens_per_sec"
    ] + token_cols

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Baseline row
        row = {
            "window_size": "baseline",
            "model": model_name,
            "num_layers": num_layers,
            "perplexity": base_ppl,
            "ppl_delta": 0.0,
            "tokens_per_sec": base_tps,
        }
        for i, val in enumerate(base_curve or [], start=1):
            row[f"token_{i}_ppl"] = val
        writer.writerow(row)

        # Window size rows
        for ws in WINDOW_SIZES:
            r = results[ws]
            delta = r["perplexity"] - base_ppl
            row = {
                "window_size": ws,
                "model": model_name,
                "num_layers": num_layers,
                "perplexity": r["perplexity"],
                "ppl_delta": delta,
                "tokens_per_sec": r["tokens_per_sec"],
            }
            for i, val in enumerate(r.get("avg_curve") or [], start=1):
                row[f"token_{i}_ppl"] = val
            writer.writerow(row)

    print(f"\nSaved CSV to {CSV_PATH}")


def load_csv():
    """Load results from CSV file."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}. Run without --plot-only first.")

    results = {}
    base_ppl = None
    base_tps = None
    base_curve = []
    num_layers = 28
    model_name = MODEL_NAME

    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ws = row["window_size"]
            num_layers = int(row["num_layers"])
            model_name = row["model"]

            # Parse curve
            curve = []
            for key in row:
                if key.startswith("token_") and key.endswith("_ppl"):
                    try:
                        curve.append(float(row[key]))
                    except:
                        pass

            if ws == "baseline":
                base_ppl = float(row["perplexity"])
                base_tps = float(row["tokens_per_sec"])
                base_curve = curve
            else:
                results[int(ws)] = {
                    "perplexity": float(row["perplexity"]),
                    "avg_curve": curve,
                    "tokens_per_sec": float(row["tokens_per_sec"]),
                }

    return results, base_ppl, base_tps, base_curve, num_layers


def generate_plots(results, base_ppl, base_tps, base_curve):
    """Generate comparison plots."""
    import numpy as np

    # Colors for different window sizes
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(WINDOW_SIZES)))

    # --- Bar plot: Perplexity comparison ---
    plt.figure(figsize=(10, 6))

    modes = ["baseline"] + WINDOW_SIZES
    ppls = [base_ppl] + [results[ws]["perplexity"] for ws in WINDOW_SIZES]
    bar_colors = ["gray"] + list(colors)
    labels = ["Baseline"] + [f"window={ws}" for ws in WINDOW_SIZES]

    bars = plt.bar(range(len(modes)), ppls, color=bar_colors)
    # Baseline gets hatching
    bars[0].set_hatch("//")

    plt.xticks(range(len(modes)), labels, rotation=15, ha="right")
    plt.ylabel("Perplexity")
    plt.title("Perplexity by Window Size (Lined Attention)")

    # Add value labels
    for bar, ppl in zip(bars, ppls):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{ppl:.2f}", ha="center", va="bottom", fontsize=9)

    # Add delta labels
    for i, ws in enumerate(WINDOW_SIZES, start=1):
        delta = results[ws]["perplexity"] - base_ppl
        plt.text(i, ppls[i] * 0.5, f"{delta:+.2f}",
                 ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"Saved plot to {PLOT_PATH}")

    # --- Bar plot: Log scale ---
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(modes)), ppls, color=bar_colors)
    bars[0].set_hatch("//")
    plt.xticks(range(len(modes)), labels, rotation=15, ha="right")
    plt.ylabel("Perplexity (log scale)")
    plt.title("Perplexity by Window Size (log scale)")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(PLOT_LOG_PATH, dpi=150)
    print(f"Saved log plot to {PLOT_LOG_PATH}")

    # --- Line plot: TPS vs Window Size ---
    plt.figure(figsize=(10, 6))

    # Perplexity delta line
    fig, ax1 = plt.subplots(figsize=(10, 6))

    deltas = [results[ws]["perplexity"] - base_ppl for ws in WINDOW_SIZES]
    tps_vals = [results[ws]["tokens_per_sec"] for ws in WINDOW_SIZES]

    ax1.set_xlabel("Window Size")
    ax1.set_ylabel("PPL Delta (vs Baseline)", color="tab:blue")
    ax1.plot(WINDOW_SIZES, deltas, "o-", color="tab:blue", linewidth=2, markersize=8, label="PPL Delta")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(WINDOW_SIZES)
    ax1.set_xticklabels([str(ws) for ws in WINDOW_SIZES])

    ax2 = ax1.twinx()
    ax2.set_ylabel("Tokens/sec", color="tab:orange")
    ax2.plot(WINDOW_SIZES, tps_vals, "s-", color="tab:orange", linewidth=2, markersize=8, label="TPS")
    ax2.axhline(y=base_tps, color="tab:orange", linestyle="--", alpha=0.5, label=f"Baseline TPS ({base_tps:.1f})")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    plt.title("Window Size vs Performance (Lined Attention)")
    fig.tight_layout()
    tradeoff_path = OUTPUT_DIR / "tradeoff.png"
    plt.savefig(tradeoff_path, dpi=150)
    print(f"Saved tradeoff plot to {tradeoff_path}")

    # --- Running average curves ---
    skip_tokens = 100
    plt.figure(figsize=(12, 6))

    if base_curve and len(base_curve) > skip_tokens:
        tokens = list(range(skip_tokens + 1, len(base_curve) + 1))
        plt.plot(tokens, base_curve[skip_tokens:], label="Baseline (raw)",
                 color="gray", linestyle="--", linewidth=2)

    for ws, color in zip(WINDOW_SIZES, colors):
        curve = results[ws].get("avg_curve")
        if curve and len(curve) > skip_tokens:
            tokens = list(range(skip_tokens + 1, len(curve) + 1))
            plt.plot(tokens, curve[skip_tokens:], label=f"window={ws}",
                     color=color, linewidth=1.5)

    plt.xlabel("Token Position")
    plt.ylabel("Running Average Perplexity")
    plt.title("Running Average Perplexity by Window Size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CURVE_PLOT_PATH, dpi=150)
    print(f"Saved curve plot to {CURVE_PLOT_PATH}")

    plt.close("all")


def run(
    max_tokens: int = 2048,
    prompt_name: str = "paragraphs_1",
    eval_dataset: str = None,
    model_name: str = None,
    allow_skipping: bool = False,
):
    if model_name is None:
        model_name = MODEL_NAME

    # Check for existing results if allow_skipping is enabled
    if allow_skipping and CSV_PATH.exists():
        print(f"Found existing CSV at {CSV_PATH}, loading cached results...")
        try:
            results, base_ppl, base_tps, base_curve, _ = load_csv()
            print(f"  Loaded results for {len(results)} window sizes")
            print(f"  Baseline: ppl={base_ppl:.3f}, tps={base_tps:.1f}")
            generate_plots(results, base_ppl, base_tps, base_curve)
            return results, base_ppl, base_tps, base_curve
        except Exception as e:
            print(f"  Failed to load CSV: {e}, running experiments...")

    device = get_device()
    tokenizer = get_tokenizer(model_name)

    # Load evaluation data
    if eval_dataset:
        print(f"Loading eval text from {eval_dataset} (max {max_tokens} tokens)...")
        _, rollout_ids = load_eval_text(eval_dataset, max_tokens, tokenizer=tokenizer)
        rollout_ids = rollout_ids.to(device)
        prompt_len = 1
        print(f"Total tokens: {rollout_ids.shape[1]}, prompt_len: {prompt_len}")
    else:
        paragraph_text = get_prompt(prompt_name)
        print(f"Loading paragraph '{prompt_name}' for teacher-forcing evaluation...")
        rollout_ids = tokenizer(paragraph_text, return_tensors="pt")["input_ids"]
        if rollout_ids.shape[1] > max_tokens:
            rollout_ids = rollout_ids[:, :max_tokens]
        rollout_ids = rollout_ids.to(device)
        prompt_len = 1
        print(f"Total tokens: {rollout_ids.shape[1]}, prompt_len: {prompt_len}")

    # Get number of layers
    print(f"\nLoading model to get layer count: {model_name}")
    set_luka_kv_params(
        use_exact_attention=True,
        compressor="mean",
        segmenter="dummy",
        create_pages_in_generation=False,
        max_pages=128,
    )
    temp_model = load_luka_model(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if hasattr(temp_model, "model") and hasattr(temp_model.model, "luka_kv_controller"):
        num_layers = temp_model.model.luka_kv_controller.num_layers
    else:
        num_layers = 28
    del temp_model
    torch.cuda.empty_cache()
    print(f"  num_layers: {num_layers}")

    # Get baseline perplexity
    print("\nRunning baseline (raw attention)...")
    base_ppl, base_curve, _, base_tps = get_baseline_perplexity(rollout_ids, prompt_len, device, model_name)
    print(f"  ppl={base_ppl:.3f}, tps={base_tps:.1f}")

    # Run each window size
    results = {}
    for ws in WINDOW_SIZES:
        result = run_single_window(ws, rollout_ids, prompt_len, device, num_layers, model_name)
        results[ws] = result

    # Print summary table
    print("\n" + "=" * 70)
    print("Window Size Sweep Summary (Lined Attention)")
    print("=" * 70)
    print(f"\n{'Window Size':<15} {'PPL':>10} {'Delta':>10} {'TPS':>10}")
    print("-" * 50)
    print(f"{'Baseline':<15} {base_ppl:>10.3f} {0:>10.2f} {base_tps:>10.1f}")
    for ws in WINDOW_SIZES:
        r = results[ws]
        delta = r["perplexity"] - base_ppl
        print(f"{ws:<15} {r['perplexity']:>10.3f} {delta:>+10.2f} {r['tokens_per_sec']:>10.1f}")
    print("=" * 70)

    # Save CSV
    save_csv(results, base_ppl, base_tps, base_curve, num_layers, model_name)

    # Generate plots
    generate_plots(results, base_ppl, base_tps, base_curve)

    return results, base_ppl, base_tps, base_curve


def main():
    parser = argparse.ArgumentParser(description="Sweep window size for lined attention")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Model name (default: {MODEL_NAME})")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["wikitext", "pg19", "c4"],
                        help="Dataset for evaluation")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max tokens to evaluate (default: 2048)")
    parser.add_argument("--prompt", type=str, default="paragraphs_1",
                        help="Prompt name if not using dataset")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only generate plots from existing CSV")
    parser.add_argument("--allow-skipping", action="store_true",
                        help="Skip experiments if results already exist in CSV")
    args = parser.parse_args()

    if args.plot_only:
        print("Loading data from CSV...")
        results, base_ppl, base_tps, base_curve, _ = load_csv()
        print(f"Baseline: ppl={base_ppl:.3f}, tps={base_tps:.1f}")
        generate_plots(results, base_ppl, base_tps, base_curve)
    else:
        run(
            max_tokens=args.max_tokens,
            prompt_name=args.prompt,
            eval_dataset=args.dataset,
            model_name=args.model,
            allow_skipping=args.allow_skipping,
        )


if __name__ == "__main__":
    main()
