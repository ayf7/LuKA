"""
Threshold sweep analyzing retention vs perplexity and speed.

Sweeps refine thresholds for each compressor and analyzes:
- Perplexity vs threshold
- Summary retention vs perplexity
- Speed vs summary retention

Run: python experiments/perplexity/retention_analysis.py
"""

import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt

from experiments.perplexity.utils import (
    get_device,
    get_tokenizer,
    get_prompt,
    get_compressor_configs,
    generate_baseline_rollout,
    get_baseline_perplexity,
    run_single_config,
    get_output_dir,
)


# Output paths
OUTPUT_DIR = get_output_dir("retention_analysis")
CSV_PATH = OUTPUT_DIR / "results.csv"
PPL_VS_THR_PLOT = OUTPUT_DIR / "ppl_vs_threshold.png"
RETENTION_VS_PPL_PLOT = OUTPUT_DIR / "retention_vs_ppl.png"
RETENTION_VS_SPEED_PLOT = OUTPUT_DIR / "retention_vs_speed.png"
PARETO_PLOT = OUTPUT_DIR / "pareto.png"


def load_csv():
    """Load results from CSV file."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}. Run without --allow-skipping first.")

    results = {}
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
                results[name] = []

            summary_frac = None if row["summary_frac"] == "n/a" else float(row["summary_frac"])
            results[name].append({
                "threshold": float(row["threshold"]),
                "perplexity": float(row["perplexity"]),
                "summary_frac": summary_frac,
                "tokens_per_sec": float(row["tokens_per_sec"]),
                "pages": float(row["pages"]),
            })

    return results, base_ppl, base_tps


def run(
    thresholds: list[float] = None,
    max_new_tokens: int = 1024,
    prompt_name: str = "paragraphs_1",
    allow_skipping: bool = False,
):
    if thresholds is None:
        thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    # Check for existing results if allow_skipping is enabled
    if allow_skipping and CSV_PATH.exists():
        print(f"Found existing CSV at {CSV_PATH}, loading cached results...")
        try:
            results, base_ppl, base_tps = load_csv()
            compressors = get_compressor_configs(include_trained_encoder=True)
            print(f"  Loaded results for {len(results)} compressors")
            print(f"  Baseline: ppl={base_ppl:.3f}, tps={base_tps:.1f}")
            # Generate plots from cached data
            generate_plots(results, compressors, base_ppl, base_tps)
            return results, compressors, base_ppl, base_tps
        except Exception as e:
            print(f"  Failed to load CSV: {e}, running experiments...")

    device = get_device()
    tokenizer = get_tokenizer()
    prompt = get_prompt(prompt_name)

    # Generate baseline rollout
    print("Generating baseline rollout...")
    rollout_ids, _ = generate_baseline_rollout(tokenizer, prompt, device, max_new_tokens)
    rollout_ids = rollout_ids.to(device)
    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].size(1)
    print(f"Rollout: {rollout_ids.shape[1]} tokens, prompt_len={prompt_len}")

    # Get baseline perplexity
    print("Running baseline (raw attention)...")
    base_ppl, base_curve, _, base_tps = get_baseline_perplexity(rollout_ids, prompt_len, device)
    print(f"Baseline: ppl={base_ppl:.3f}, tps={base_tps:.1f}")

    # Get compressor configs
    compressors = get_compressor_configs(include_trained_encoder=True)

    # Run sweep for each compressor
    results = {cfg["name"]: [] for cfg in compressors}

    for cfg in compressors:
        print(f"\nRunning {cfg['label']}...")

        for thr in thresholds:
            print(f"  threshold={thr}...", end=" ", flush=True)

            result = run_single_config(
                rollout_ids, prompt_len, device,
                compressor=cfg["compressor"],
                use_log_bias=cfg["use_log_bias"],
                threshold=thr,
            )
            result["threshold"] = thr
            results[cfg["name"]].append(result)

            frac_str = f"{result['summary_frac']:.3f}" if result["summary_frac"] is not None else "n/a"
            print(f"ppl={result['perplexity']:.3f}, summary_frac={frac_str}")

    # Print summary table
    print("\n" + "=" * 80)
    print("Summary vs Baseline")
    print("=" * 80)
    print(f"\nBaseline (raw attention): ppl={base_ppl:.3f}, tps={base_tps:.1f}")

    for cfg in compressors:
        print(f"\n{cfg['label']}:")
        for r in results[cfg["name"]]:
            frac_str = f"{r['summary_frac']:.3f}" if r["summary_frac"] else "n/a"
            delta = r["perplexity"] - base_ppl
            delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
            print(f"  thr={r['threshold']:>4}: ppl={r['perplexity']:.3f} ({delta_str}), "
                  f"summary_frac={frac_str}, tps={r['tokens_per_sec']:.1f}")

    # Generate plots
    generate_plots(results, compressors, base_ppl, base_tps)

    # CSV output
    fieldnames = [
        "compressor", "threshold", "perplexity", "ppl_delta",
        "summary_frac", "tokens_per_sec", "pages"
    ]

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Baseline row
        writer.writerow({
            "compressor": "baseline",
            "threshold": "n/a",
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
                    "threshold": r["threshold"],
                    "perplexity": r["perplexity"],
                    "ppl_delta": r["perplexity"] - base_ppl,
                    "summary_frac": r["summary_frac"] if r["summary_frac"] else "n/a",
                    "tokens_per_sec": r["tokens_per_sec"],
                    "pages": r["pages"],
                })

    print(f"Saved CSV to {CSV_PATH}")


def generate_plots(results, compressors, base_ppl, base_tps):
    """Generate all plots from results."""
    # Plot 1: Perplexity vs Threshold
    plt.figure(figsize=(10, 6))
    plt.axhline(base_ppl, label="Baseline (raw)", linestyle="--", color="black", linewidth=2)

    for cfg in compressors:
        if cfg["name"] not in results:
            continue
        recs = results[cfg["name"]]
        xs = [r["threshold"] for r in recs]
        ys = [r["perplexity"] for r in recs]
        plt.plot(xs, ys, marker="o", label=cfg["label"], color=cfg["color"])

    plt.xlabel("Refine Threshold")
    plt.ylabel("Perplexity")
    plt.title("Perplexity vs Threshold")
    plt.xscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PPL_VS_THR_PLOT, dpi=150)
    print(f"\nSaved perplexity plot to {PPL_VS_THR_PLOT}")

    # Plot 2: Summary Retention vs Perplexity
    plt.figure(figsize=(10, 6))
    plt.scatter([0.0], [base_ppl], color="black", s=150, marker="*", label="Baseline", zorder=5)

    for cfg in compressors:
        if cfg["name"] not in results:
            continue
        recs = results[cfg["name"]]
        xs = [r["summary_frac"] if r["summary_frac"] else 0.0 for r in recs]
        ys = [r["perplexity"] for r in recs]
        plt.scatter(xs, ys, label=cfg["label"], color=cfg["color"], s=60)
        plt.plot(xs, ys, color=cfg["color"], alpha=0.5, linewidth=1)

        for r in recs:
            x = r["summary_frac"] if r["summary_frac"] else 0.0
            y = r["perplexity"]
            plt.annotate(f"{r['threshold']}", (x, y), textcoords="offset points",
                        xytext=(3, 3), fontsize=7, alpha=0.7)

    plt.xlabel("Summary Retention (1 - refinement_rate)")
    plt.ylabel("Perplexity")
    plt.title("Perplexity vs Summary Retention")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RETENTION_VS_PPL_PLOT, dpi=150)
    print(f"Saved retention plot to {RETENTION_VS_PPL_PLOT}")

    # Plot 3: Speed vs Summary Retention
    plt.figure(figsize=(10, 6))
    plt.axhline(base_tps, label="Baseline (raw)", linestyle="--", color="black", linewidth=2)

    for cfg in compressors:
        if cfg["name"] not in results:
            continue
        recs = results[cfg["name"]]
        xs = [r["summary_frac"] if r["summary_frac"] else 0.0 for r in recs]
        ys = [r["tokens_per_sec"] for r in recs]
        plt.scatter(xs, ys, label=cfg["label"], color=cfg["color"], s=60)
        plt.plot(xs, ys, color=cfg["color"], alpha=0.5, linewidth=1)

    plt.xlabel("Summary Retention (1 - refinement_rate)")
    plt.ylabel("Tokens per Second")
    plt.title("Speed vs Summary Retention")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RETENTION_VS_SPEED_PLOT, dpi=150)
    print(f"Saved speed plot to {RETENTION_VS_SPEED_PLOT}")

    # Plot 4: Pareto frontier - Perplexity vs Speed (Attn-Weighted only)
    plt.figure(figsize=(10, 6))
    plt.scatter([base_tps], [base_ppl], color="black", s=150, marker="*", label="Baseline", zorder=5)

    attn_cfg = next((cfg for cfg in compressors if cfg["name"] == "attn_weighted"), None)
    if attn_cfg and attn_cfg["name"] in results:
        recs = results[attn_cfg["name"]]
        xs = [r["tokens_per_sec"] for r in recs]
        ys = [r["perplexity"] for r in recs]
        plt.scatter(xs, ys, label=attn_cfg["label"], color=attn_cfg["color"], s=60)

        for r in recs:
            plt.annotate(f"{r['threshold']}", (r["tokens_per_sec"], r["perplexity"]),
                        textcoords="offset points", xytext=(3, 3), fontsize=7, alpha=0.7)

    plt.xlabel("Tokens per Second")
    plt.ylabel("Perplexity")
    plt.title("Perplexity vs Speed (Attn-Weighted)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PARETO_PLOT, dpi=150)
    print(f"Saved Pareto plot to {PARETO_PLOT}")

    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retention analysis sweep")
    parser.add_argument("--thresholds", type=float, nargs="+",
                       default=[0.0001, 0.001, 0.01, 0.1, 1],
                       help="Thresholds to sweep")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--prompt", type=str, default="paragraphs_1", help="Prompt name")
    parser.add_argument("--allow-skipping", action="store_true",
                       help="Skip experiments if results already exist in CSV")
    parser.add_argument("--plot-only", action="store_true",
                       help="Only generate plots from existing CSV")
    args = parser.parse_args()

    if args.plot_only:
        print("Loading data from CSV...")
        results, base_ppl, base_tps = load_csv()
        compressors = get_compressor_configs(include_trained_encoder=True)
        print(f"Baseline: ppl={base_ppl:.3f}, tps={base_tps:.1f}")
        generate_plots(results, compressors, base_ppl, base_tps)
    else:
        run(
            thresholds=args.thresholds,
            max_new_tokens=args.max_tokens,
            prompt_name=args.prompt,
            allow_skipping=args.allow_skipping,
        )
