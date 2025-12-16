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
import matplotlib.pyplot as plt

from experiments.perplexity.utils import (
    get_device,
    get_tokenizer,
    get_prompt,
    get_compressor_configs,
    generate_baseline_rollout,
    get_baseline_perplexity,
    run_single_config,
)


def run(
    thresholds: list[float] = None,
    max_new_tokens: int = 1024,
    prompt_name: str = "paragraphs_1"
):
    if thresholds is None:
        thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

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
    base_ppl, base_curve, base_tps = get_baseline_perplexity(rollout_ids, prompt_len, device)
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

    # Plot 1: Perplexity vs Threshold
    plt.figure(figsize=(10, 6))
    plt.axhline(base_ppl, label="Baseline (raw)", linestyle="--", color="black", linewidth=2)

    for cfg in compressors:
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

    out_path = "experiments/perplexity/retention_ppl_vs_threshold.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved perplexity plot to {out_path}")

    # Plot 2: Summary Retention vs Perplexity
    plt.figure(figsize=(10, 6))
    plt.scatter([0.0], [base_ppl], color="black", s=150, marker="*", label="Baseline", zorder=5)

    for cfg in compressors:
        recs = results[cfg["name"]]
        xs = [r["summary_frac"] if r["summary_frac"] else 0.0 for r in recs]
        ys = [r["perplexity"] for r in recs]
        plt.scatter(xs, ys, label=cfg["label"], color=cfg["color"], s=60)
        plt.plot(xs, ys, color=cfg["color"], alpha=0.5, linewidth=1)

        # Annotate with threshold values
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

    out_path = "experiments/perplexity/retention_vs_ppl.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved retention plot to {out_path}")

    # Plot 3: Speed vs Summary Retention
    plt.figure(figsize=(10, 6))
    plt.axhline(base_tps, label="Baseline (raw)", linestyle="--", color="black", linewidth=2)

    for cfg in compressors:
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

    out_path = "experiments/perplexity/retention_vs_speed.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved speed plot to {out_path}")

    # Plot 4: Pareto frontier - Perplexity vs Speed
    plt.figure(figsize=(10, 6))
    plt.scatter([base_tps], [base_ppl], color="black", s=150, marker="*", label="Baseline", zorder=5)

    for cfg in compressors:
        recs = results[cfg["name"]]
        xs = [r["tokens_per_sec"] for r in recs]
        ys = [r["perplexity"] for r in recs]
        plt.scatter(xs, ys, label=cfg["label"], color=cfg["color"], s=60)

        # Annotate with threshold values
        for r in recs:
            plt.annotate(f"{r['threshold']}", (r["tokens_per_sec"], r["perplexity"]),
                        textcoords="offset points", xytext=(3, 3), fontsize=7, alpha=0.7)

    plt.xlabel("Tokens per Second")
    plt.ylabel("Perplexity")
    plt.title("Perplexity vs Speed (Pareto frontier)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = "experiments/perplexity/retention_pareto.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved Pareto plot to {out_path}")

    # CSV output
    csv_path = "experiments/perplexity/retention_analysis.csv"
    fieldnames = [
        "compressor", "threshold", "perplexity", "ppl_delta",
        "summary_frac", "tokens_per_sec", "pages"
    ]

    with open(csv_path, "w", newline="") as f:
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

    print(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retention analysis sweep")
    parser.add_argument("--thresholds", type=float, nargs="+",
                       default=[0.0001, 0.001, 0.01, 0.1, 1],
                       help="Thresholds to sweep")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--prompt", type=str, default="paragraphs_1", help="Prompt name")
    args = parser.parse_args()

    run(thresholds=args.thresholds, max_new_tokens=args.max_tokens, prompt_name=args.prompt)
