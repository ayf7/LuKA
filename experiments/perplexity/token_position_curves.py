"""
Per-token perplexity curves comparing compressors vs baseline.

Evaluates each compressor at a fixed threshold and plots cumulative
perplexity over token positions.

Run: python experiments/perplexity/token_position_curves.py
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


def run(threshold: float = 0.05, max_new_tokens: int = 1024, prompt_name: str = "paragraphs_1", include_log_bias: bool = True):
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
    compressors = get_compressor_configs(include_trained_encoder=True, include_log_bias=include_log_bias)

    # Run each compressor at the fixed threshold
    results = {}
    for cfg in compressors:
        print(f"\nRunning {cfg['label']} at threshold={threshold}...")
        result = run_single_config(
            rollout_ids, prompt_len, device,
            compressor=cfg["compressor"],
            use_log_bias=cfg["use_log_bias"],
            threshold=threshold,
        )
        results[cfg["name"]] = result
        frac_str = f"{result['summary_frac']:.3f}" if result["summary_frac"] is not None else "n/a"
        print(f"  ppl={result['perplexity']:.3f}, summary_frac={frac_str}, tps={result['tokens_per_sec']:.1f}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"Summary (threshold={threshold})")
    print("=" * 80)
    print(f"{'Compressor':<20} {'PPL':>10} {'Delta':>10} {'Summary%':>10} {'TPS':>10}")
    print("-" * 60)
    print(f"{'Baseline':<20} {base_ppl:>10.1f} {0:>10.1f} {'N/A':>10} {base_tps:>10.1f}")
    for cfg in compressors:
        r = results[cfg["name"]]
        delta = r["perplexity"] - base_ppl
        frac_str = f"{r['summary_frac']*100:.1f}%" if r["summary_frac"] is not None else "N/A"
        print(f"{cfg['label']:<20} {r['perplexity']:>10.1f} {delta:>+10.1f} {frac_str:>10} {r['tokens_per_sec']:>10.1f}")
    print("=" * 80)

    # Plot: Per-token perplexity curves
    plt.figure(figsize=(12, 6))
    token_axis = list(range(1, len(base_curve) + 1))

    plt.plot(token_axis, base_curve, label="Baseline (raw)", linestyle="--", color="black", linewidth=2)

    for cfg in compressors:
        curve = results[cfg["name"]]["curve"]
        plt.plot(token_axis, curve, label=cfg["label"], color=cfg["color"], linewidth=1.5)

    plt.xlabel("Token Position")
    plt.ylabel("Cumulative Perplexity")
    plt.title(f"Per-token Perplexity vs Baseline (threshold={threshold})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = f"experiments/perplexity/token_curves_thr{threshold}.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")

    # Plot: Log scale version
    plt.figure(figsize=(12, 6))
    plt.plot(token_axis, base_curve, label="Baseline (raw)", linestyle="--", color="black", linewidth=2)

    for cfg in compressors:
        curve = results[cfg["name"]]["curve"]
        plt.plot(token_axis, curve, label=cfg["label"], color=cfg["color"], linewidth=1.5)

    plt.xlabel("Token Position")
    plt.ylabel("Cumulative Perplexity (log scale)")
    plt.title(f"Per-token Perplexity vs Baseline (threshold={threshold}, log scale)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_log_path = f"experiments/perplexity/token_curves_thr{threshold}_log.png"
    plt.savefig(out_log_path, dpi=150)
    print(f"Saved log plot to {out_log_path}")

    # CSV output
    csv_path = f"experiments/perplexity/token_curves_thr{threshold}.csv"
    fieldnames = ["token_position", "baseline"] + [cfg["name"] for cfg in compressors]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, pos in enumerate(token_axis):
            row = {"token_position": pos, "baseline": base_curve[i]}
            for cfg in compressors:
                row[cfg["name"]] = results[cfg["name"]]["curve"][i]
            writer.writerow(row)

    print(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-token perplexity curves")
    parser.add_argument("--threshold", type=float, default=0.05, help="Refine threshold")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens to generate")
    parser.add_argument("--prompt", type=str, default="paragraphs_1", help="Prompt name")
    parser.add_argument("--log-bias", action="store_true", default=True, help="Include log(N) bias variants")
    parser.add_argument("--no-log-bias", dest="log_bias", action="store_false", help="Exclude log(N) bias variants")
    args = parser.parse_args()

    run(threshold=args.threshold, max_new_tokens=args.max_tokens, prompt_name=args.prompt, include_log_bias=args.log_bias)
