"""
Page fraction refinement sweep - analyzes PPL vs fraction of pages refined.

Sweeps the `frac` parameter of TopFracRule to understand the tradeoff between
refinement coverage and perplexity.

Run:
    python experiments/perplexity/page_fraction_sweep.py
    python experiments/perplexity/page_fraction_sweep.py --fracs 0 0.25 0.5 0.75 1.0
    python experiments/perplexity/page_fraction_sweep.py --max-tokens 1024 --dataset wikitext
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from experiments.perplexity.utils import (
    get_device,
    get_tokenizer,
    get_prompt,
    get_baseline_perplexity,
    run_single_config,
    load_eval_text,
    MODEL_NAME,
)
from modeling.compressor import AttentionWeightedCompressor, MeanCompressor, RandomCompressor


# Output paths
OUTPUT_DIR = Path("experiments/perplexity")
CSV_PATH = OUTPUT_DIR / "page_fraction_sweep.csv"
PLOT_PATH = OUTPUT_DIR / "page_fraction_sweep.png"
PLOT_LOG_PATH = OUTPUT_DIR / "page_fraction_sweep_log.png"
PLOT_LOGLOG_PATH = OUTPUT_DIR / "page_fraction_sweep_loglog.png"


def get_compressor_configs():
    """Compressor configurations to test."""
    return [
        {
            "name": "attn_weighted",
            "label": "Attn-Weighted",
            "compressor": AttentionWeightedCompressor(temperature=1.0),
            "log_bias_mode": "none",
            "color": "tab:blue",
            "linestyle": "-",
        },
        {
            "name": "mean",
            "label": "Mean",
            "compressor": MeanCompressor(),
            "log_bias_mode": "none",
            "color": "tab:orange",
            "linestyle": "-",
        },
        {
            "name": "random",
            "label": "Random",
            "compressor": RandomCompressor(),
            "log_bias_mode": "none",
            "color": "tab:red",
            "linestyle": "-",
        },
    ]


def run(
    fracs: list[float] = None,
    max_tokens: int = 512,
    prompt_name: str = "paragraphs_1",
    eval_dataset: str = None,
    model_name: str = None,
):
    if fracs is None:
        fracs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    if model_name is None:
        model_name = MODEL_NAME

    # Get compressor configs
    compressors = get_compressor_configs()

    # Load existing results if CSV exists
    existing_results = {}
    existing_base_ppl = None
    existing_base_tps = None
    if CSV_PATH.exists():
        print(f"Found existing CSV at {CSV_PATH}, loading...")
        existing_results, _, existing_base_ppl, existing_base_tps = load_csv()
        print(f"  Loaded {sum(len(v) for v in existing_results.values())} existing results")

    # Check what we need to run
    need_baseline = existing_base_ppl is None
    to_run = []  # List of (cfg, frac) pairs to run
    for cfg in compressors:
        existing_fracs = {r["frac"] for r in existing_results.get(cfg["name"], [])}
        for frac in fracs:
            if frac not in existing_fracs:
                to_run.append((cfg, frac))

    if not to_run and not need_baseline:
        print("All experiments already in CSV. Use --plot-only to regenerate plots.")
        results = existing_results
        base_ppl = existing_base_ppl
        base_tps = existing_base_tps
    else:
        device = get_device()
        tokenizer = get_tokenizer(model_name)

        # Load evaluation data
        if eval_dataset:
            print(f"Loading eval text from {eval_dataset} (max {max_tokens} tokens)...")
            _, rollout_ids = load_eval_text(eval_dataset, max_tokens, tokenizer=tokenizer)
            rollout_ids = rollout_ids.to(device)
            prompt_len = min(64, rollout_ids.shape[1] // 4)
            print(f"Total tokens: {rollout_ids.shape[1]}, prompt_len: {prompt_len}")
        else:
            from experiments.perplexity.utils import generate_baseline_rollout
            prompt = get_prompt(prompt_name)
            print("Generating baseline rollout...")
            rollout_ids, _ = generate_baseline_rollout(tokenizer, prompt, device, max_tokens)
            rollout_ids = rollout_ids.to(device)
            prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].size(1)
            print(f"Rollout: {rollout_ids.shape[1]} tokens, prompt_len={prompt_len}")

        # Get baseline perplexity
        if need_baseline:
            print("\nRunning baseline (raw attention)...")
            base_ppl, _, _, base_tps = get_baseline_perplexity(rollout_ids, prompt_len, device, model_name)
            print(f"Baseline: ppl={base_ppl:.3f}, tps={base_tps:.1f}")
        else:
            base_ppl = existing_base_ppl
            base_tps = existing_base_tps
            print(f"\nUsing cached baseline: ppl={base_ppl:.3f}, tps={base_tps:.1f}")

        # Start with existing results
        results = {cfg["name"]: list(existing_results.get(cfg["name"], [])) for cfg in compressors}

        # Run only missing experiments
        print(f"\nRunning {len(to_run)} missing experiments...")
        for cfg, frac in to_run:
            print(f"  {cfg['label']} frac={frac:.2f}...", end=" ", flush=True)

            result = run_single_config(
                rollout_ids,
                prompt_len,
                device,
                compressor=cfg["compressor"],
                log_bias_mode=cfg["log_bias_mode"],
                refinement_rule="top_frac",
                refinement_rule_kwargs={"frac": frac},
                model_name=model_name,
            )
            result["frac"] = frac
            results[cfg["name"]].append(result)

            frac_str = f"{result['summary_frac']:.3f}" if result["summary_frac"] is not None else "n/a"
            delta = result["perplexity"] - base_ppl
            delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
            print(f"ppl={result['perplexity']:.3f} ({delta_str}), summary_frac={frac_str}, tps={result['tokens_per_sec']:.1f}")

        # Sort results by frac
        for name in results:
            results[name].sort(key=lambda r: r["frac"])

    # Print summary table
    print("\n" + "=" * 80)
    print("Summary: Page Fraction Sweep")
    print("=" * 80)
    print(f"\nBaseline (raw attention): ppl={base_ppl:.3f}, tps={base_tps:.1f}")

    for cfg in compressors:
        print(f"\n{cfg['label']}:")
        print(f"  {'Frac':>6} | {'PPL':>8} | {'Delta':>8} | {'Summary%':>8} | {'TPS':>6}")
        print(f"  {'-'*6} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*6}")
        for r in results[cfg["name"]]:
            frac_str = f"{r['summary_frac']*100:.1f}%" if r["summary_frac"] else "n/a"
            delta = r["perplexity"] - base_ppl
            delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
            print(f"  {r['frac']:>6.2f} | {r['perplexity']:>8.3f} | {delta_str:>8} | {frac_str:>8} | {r['tokens_per_sec']:>6.1f}")

    # Save CSV
    save_csv(results, compressors, base_ppl, base_tps)

    # Generate plots
    generate_plots(results, compressors, base_ppl)

    return results, compressors, base_ppl, base_tps


def save_csv(results, compressors, base_ppl, base_tps):
    """Save results to CSV."""
    fieldnames = [
        "compressor", "frac", "perplexity", "ppl_delta",
        "summary_frac", "tokens_per_sec", "pages"
    ]

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Baseline row
        writer.writerow({
            "compressor": "baseline",
            "frac": "n/a",
            "perplexity": base_ppl,
            "ppl_delta": 0.0,
            "summary_frac": 0.0,
            "tokens_per_sec": base_tps,
            "pages": 0,
        })

        # Compressor rows
        for cfg in compressors:
            for r in results[cfg["name"]]:
                writer.writerow({
                    "compressor": cfg["name"],
                    "frac": r["frac"],
                    "perplexity": r["perplexity"],
                    "ppl_delta": r["perplexity"] - base_ppl,
                    "summary_frac": r["summary_frac"] if r["summary_frac"] else "n/a",
                    "tokens_per_sec": r["tokens_per_sec"],
                    "pages": r["pages"],
                })

    print(f"\nSaved CSV to {CSV_PATH}")


def generate_plots(results, compressors, base_ppl):
    """Generate plots from results."""
    # Plot 1: Perplexity vs Fraction (linear)
    plt.figure(figsize=(10, 6))
    plt.axhline(base_ppl, label="Baseline (raw)", linestyle="--", color="black", linewidth=2)

    for cfg in compressors:
        recs = results[cfg["name"]]
        xs = [r["frac"] for r in recs]
        ys = [r["perplexity"] for r in recs]
        plt.plot(xs, ys, marker="o", label=cfg["label"],
                 color=cfg["color"], linestyle=cfg["linestyle"], linewidth=2)

    plt.xlabel("Refinement Fraction (frac)")
    plt.ylabel("Perplexity")
    plt.title("Perplexity vs Page Refinement Fraction")
    plt.xlim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"Saved plot to {PLOT_PATH}")

    # Plot 2: Perplexity vs Fraction (log y)
    plt.figure(figsize=(10, 6))
    plt.axhline(base_ppl, label="Baseline (raw)", linestyle="--", color="black", linewidth=2)

    for cfg in compressors:
        recs = results[cfg["name"]]
        xs = [r["frac"] for r in recs]
        ys = [r["perplexity"] for r in recs]
        plt.plot(xs, ys, marker="o", label=cfg["label"],
                 color=cfg["color"], linestyle=cfg["linestyle"], linewidth=2)

    plt.xlabel("Refinement Fraction (frac)")
    plt.ylabel("Perplexity (log scale)")
    plt.title("Perplexity vs Page Refinement Fraction (log scale)")
    plt.xlim(-0.05, 1.05)
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_LOG_PATH, dpi=150)
    print(f"Saved log plot to {PLOT_LOG_PATH}")

    # Plot 3: Perplexity vs Fraction (log(log(y))) with raw PPL labels
    fig, ax = plt.subplots(figsize=(10, 6))
    if base_ppl > 1:
        baseline_loglog = np.log(np.log(base_ppl))
        ax.axhline(baseline_loglog, label="Baseline (raw)", linestyle="--", color="black", linewidth=2)

    all_ys = []
    for cfg in compressors:
        recs = results[cfg["name"]]
        xs = [r["frac"] for r in recs]
        ys_raw = [r["perplexity"] for r in recs]
        # Transform to log(log(y)), filtering values <= 1
        ys = [np.log(np.log(y)) if y > 1 else np.nan for y in ys_raw]
        all_ys.extend([y for y in ys if not np.isnan(y)])
        ax.plot(xs, ys, marker="o", label=cfg["label"],
                color=cfg["color"], linestyle=cfg["linestyle"], linewidth=2)

    # Custom formatter to show raw perplexity values on y-axis
    def loglog_to_ppl(y, pos):
        ppl = np.exp(np.exp(y))
        if ppl >= 1000:
            return f"{ppl:.0f}"
        elif ppl >= 100:
            return f"{ppl:.1f}"
        else:
            return f"{ppl:.2f}"

    ax.yaxis.set_major_formatter(FuncFormatter(loglog_to_ppl))

    ax.set_xlabel("Refinement Fraction (frac)")
    ax.set_ylabel("Perplexity (log-log scale)")
    ax.set_title("Perplexity vs Page Refinement Fraction")
    ax.set_xlim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_LOGLOG_PATH, dpi=150)
    print(f"Saved log-log plot to {PLOT_LOGLOG_PATH}")

    plt.close("all")


def load_csv():
    """Load results from CSV file."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}. Run without --plot-only first.")

    compressors = get_compressor_configs()
    results = {cfg["name"]: [] for cfg in compressors}
    base_ppl = None
    base_tps = None

    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["compressor"]
            if name == "baseline":
                base_ppl = float(row["perplexity"])
                base_tps = float(row["tokens_per_sec"])
                continue

            if name not in results:
                continue

            summary_frac = None if row["summary_frac"] == "n/a" else float(row["summary_frac"])
            results[name].append({
                "frac": float(row["frac"]),
                "perplexity": float(row["perplexity"]),
                "summary_frac": summary_frac,
                "tokens_per_sec": float(row["tokens_per_sec"]),
                "pages": float(row["pages"]),
            })

    return results, compressors, base_ppl, base_tps


def main():
    parser = argparse.ArgumentParser(description="Page fraction refinement sweep")
    parser.add_argument("--fracs", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        help="Fraction values to sweep")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens for evaluation")
    parser.add_argument("--prompt", type=str, default="paragraphs_1",
                        help="Prompt name (if not using dataset)")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["wikitext", "pg19", "c4"],
                        help="Use dataset instead of generation")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default: Qwen/Qwen3-1.7B-Base)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only generate plots from existing CSV")
    args = parser.parse_args()

    if args.plot_only:
        print("Loading data from CSV...")
        results, compressors, base_ppl, base_tps = load_csv()
        print(f"Baseline: ppl={base_ppl:.3f}, tps={base_tps:.1f}")
        generate_plots(results, compressors, base_ppl)
    else:
        run(
            fracs=args.fracs,
            max_tokens=args.max_tokens,
            prompt_name=args.prompt,
            eval_dataset=args.dataset,
            model_name=args.model,
        )


if __name__ == "__main__":
    main()
