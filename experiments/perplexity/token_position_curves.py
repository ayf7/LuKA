"""
Per-token perplexity curves comparing compressors vs baseline.

Evaluates each compressor with a configurable refinement rule and plots
cumulative perplexity over token positions.

Run:
    python experiments/perplexity/token_position_curves.py                      # Default: top_frac(frac=0.1)
    python experiments/perplexity/token_position_curves.py --rule top_frac --frac 0.2
    python experiments/perplexity/token_position_curves.py --rule top_k --k 5
    python experiments/perplexity/token_position_curves.py --rule top_p --p 0.95
    python experiments/perplexity/token_position_curves.py --plot-only          # Regenerate plots from CSV
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
    load_eval_text,
)


DEFAULT_MODEL = "Qwen/Qwen3-1.7B-Base"


def load_csv(csv_path, compressors):
    """Load results from CSV file."""
    base_curve = []
    results = {cfg["name"]: {"curve": []} for cfg in compressors}
    token_axis = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            token_axis.append(int(row["token_position"]))
            base_curve.append(float(row["baseline"]))
            for cfg in compressors:
                if cfg["name"] in row:
                    results[cfg["name"]]["curve"].append(float(row[cfg["name"]]))

    return base_curve, results, token_axis


def generate_plots(base_curve, results, compressors, token_axis, rule_label, out_path, out_log_path):
    """Generate plots from results."""
    # Plot: Per-token perplexity curves
    plt.figure(figsize=(12, 6))

    plt.plot(token_axis, base_curve, label="Baseline (raw)", linestyle="--", color="black", linewidth=2)

    for cfg in compressors:
        if cfg["name"] in results and results[cfg["name"]]["curve"]:
            curve = results[cfg["name"]]["curve"]
            plt.plot(
                token_axis, curve,
                label=cfg["label"],
                color=cfg["color"],
                linestyle=cfg.get("linestyle", "-"),
                linewidth=cfg.get("linewidth", 1.5),
            )

    plt.xlabel("Token Position")
    plt.ylabel("Cumulative Perplexity")
    plt.title(f"Per-token Perplexity vs Baseline ({rule_label})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")

    # Plot: Log scale version
    plt.figure(figsize=(12, 6))
    plt.plot(token_axis, base_curve, label="Baseline (raw)", linestyle="--", color="black", linewidth=2)

    for cfg in compressors:
        if cfg["name"] in results and results[cfg["name"]]["curve"]:
            curve = results[cfg["name"]]["curve"]
            plt.plot(
                token_axis, curve,
                label=cfg["label"],
                color=cfg["color"],
                linestyle=cfg.get("linestyle", "-"),
                linewidth=cfg.get("linewidth", 1.5),
            )

    plt.xlabel("Token Position")
    plt.ylabel("Cumulative Perplexity (log scale)")
    plt.title(f"Per-token Perplexity vs Baseline ({rule_label}, log scale)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_log_path, dpi=150)
    print(f"Saved log plot to {out_log_path}")

    plt.close("all")


def run(
    refinement_rule: str = "top_frac",
    refinement_frac: float = 0.1,
    refinement_k: int = 3,
    refinement_p: float = 0.9,
    refinement_threshold: float = 0.05,
    always_refine_first_n: int = 0,
    max_tokens: int = 1024,
    prompt_name: str = "paragraphs_1",
    eval_dataset: str = None,  # "wikitext", "pg19", "c4", or None for generation
    no_generate: bool = False,  # Use prompt directly without generation
    model_name: str = None,
    include_log_bias: bool = True,
    include_adaptive_k: bool = True,
    include_encoder: bool = True,
    bias_comparison_mode: bool = False,
    plot_only: bool = False,
):
    if model_name is None:
        model_name = DEFAULT_MODEL

    # Build refinement kwargs based on rule
    if refinement_rule == "top_frac":
        refinement_kwargs = {"frac": refinement_frac}
        rule_label = f"top_frac(frac={refinement_frac})"
    elif refinement_rule == "top_k":
        refinement_kwargs = {"k": refinement_k}
        rule_label = f"top_k(k={refinement_k})"
    elif refinement_rule == "top_p":
        refinement_kwargs = {"p": refinement_p}
        rule_label = f"top_p(p={refinement_p})"
    elif refinement_rule == "top_k_top_p":
        refinement_kwargs = {"k": refinement_k, "p": refinement_p}
        rule_label = f"top_k_top_p(k={refinement_k}, p={refinement_p})"
    elif refinement_rule == "threshold":
        refinement_kwargs = {"threshold": refinement_threshold}
        rule_label = f"threshold={refinement_threshold}"
    elif refinement_rule == "none":
        refinement_kwargs = {}
        rule_label = "none"
    else:
        raise ValueError(f"Unknown refinement rule: {refinement_rule}")

    # Add attention sinks to kwargs if enabled
    if always_refine_first_n > 0:
        refinement_kwargs["always_refine_first_n"] = always_refine_first_n
        rule_label += f"+sink{always_refine_first_n}"

    # Build rule_suffix for filenames
    if refinement_rule == "top_frac":
        rule_suffix = f"topfrac{refinement_frac}"
    elif refinement_rule == "top_k":
        rule_suffix = f"topk{refinement_k}"
    elif refinement_rule == "top_p":
        rule_suffix = f"topp{refinement_p}"
    elif refinement_rule == "top_k_top_p":
        rule_suffix = f"topk{refinement_k}_topp{refinement_p}"
    elif refinement_rule == "threshold":
        rule_suffix = f"thr{refinement_threshold}"
    else:
        rule_suffix = refinement_rule

    if eval_dataset:
        rule_suffix += f"_{eval_dataset}"

    if bias_comparison_mode:
        rule_suffix += "_bias_comparison"

    csv_path = Path(f"experiments/perplexity/token_curves_{rule_suffix}.csv")
    out_path = Path(f"experiments/perplexity/token_curves_{rule_suffix}.png")
    out_log_path = Path(f"experiments/perplexity/token_curves_{rule_suffix}_log.png")

    # Get compressor configs
    compressors = get_compressor_configs(
        include_trained_encoder=include_encoder,
        include_log_bias=include_log_bias,
        include_adaptive_k=include_adaptive_k,
        bias_comparison_mode=bias_comparison_mode,
    )

    # Try to load from CSV if plot_only or CSV exists
    if plot_only or csv_path.exists():
        if csv_path.exists():
            print(f"Loading data from {csv_path}...")
            base_curve, results, token_axis = load_csv(csv_path, compressors)
            base_ppl = base_curve[-1] if base_curve else None
            print(f"Loaded {len(token_axis)} token positions")
            if plot_only:
                generate_plots(base_curve, results, compressors, token_axis, rule_label, out_path, out_log_path)
                return
        elif plot_only:
            raise FileNotFoundError(f"CSV not found: {csv_path}. Run without --plot-only first.")

    print(f"Refinement rule: {rule_label}")

    device = get_device()
    tokenizer = get_tokenizer(model_name)
    prompt = get_prompt(prompt_name)

    print(f"Model: {model_name}")

    if eval_dataset:
        # Load natural text from dataset
        print(f"\nLoading eval text from {eval_dataset} (max {max_tokens} tokens)...")
        eval_text, rollout_ids = load_eval_text(eval_dataset, max_tokens, tokenizer=tokenizer)
        rollout_ids = rollout_ids.to(device)
        # Use first 64 tokens as "prompt" for prefill, rest for eval
        prompt_len = min(64, rollout_ids.shape[1] // 4)
    elif no_generate:
        # Use prompt directly without generation
        print("\n" + "=" * 80)
        print("PROMPT (no generation):")
        print("=" * 80)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        print("=" * 80 + "\n")

        rollout_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        if rollout_ids.shape[1] > max_tokens:
            rollout_ids = rollout_ids[:, :max_tokens]
        # Use first 64 tokens as "prompt" for prefill, rest for eval
        prompt_len = min(64, rollout_ids.shape[1] // 4)
        print(f"Total tokens: {rollout_ids.shape[1]}, prompt_len (prefill): {prompt_len}")
    else:
        # Original behavior: generate from prompt
        print("\n" + "=" * 80)
        print("PROMPT:")
        print("=" * 80)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        print("=" * 80 + "\n")

        # Generate baseline rollout
        print("Generating baseline rollout...")
        rollout_ids, rollout_text = generate_baseline_rollout(tokenizer, prompt, device, max_tokens, model_name=model_name)
        rollout_ids = rollout_ids.to(device)
        prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].size(1)
        print(f"Rollout: {rollout_ids.shape[1]} tokens, prompt_len={prompt_len}")

        # Print generated continuation
        generated_text = rollout_text[len(prompt):]
        print("\nGenerated continuation:")
        print("-" * 40)
        print(generated_text)
        print("-" * 40 + "\n")

    # Get baseline perplexity
    print("Running baseline (raw attention)...")
    base_ppl, base_curve, base_tps = get_baseline_perplexity(rollout_ids, prompt_len, device, model_name=model_name)
    print(f"Baseline: ppl={base_ppl:.3f}, tps={base_tps:.1f}")

    # Run each compressor with the refinement rule
    results = {}
    for cfg in compressors:
        print(f"\nRunning {cfg['label']} with {rule_label}...")
        result = run_single_config(
            rollout_ids, prompt_len, device,
            compressor=cfg["compressor"],
            log_bias_mode=cfg["log_bias_mode"],
            refinement_rule=refinement_rule,
            refinement_rule_kwargs=refinement_kwargs,
            model_name=model_name,
        )
        results[cfg["name"]] = result
        frac_str = f"{result['summary_frac']:.3f}" if result["summary_frac"] is not None else "n/a"
        print(f"  ppl={result['perplexity']:.3f}, summary_frac={frac_str}, tps={result['tokens_per_sec']:.1f}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"Summary ({rule_label})")
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

    # Save CSV
    token_axis = list(range(1, len(base_curve) + 1))
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

    # Generate plots
    generate_plots(base_curve, results, compressors, token_axis, rule_label, out_path, out_log_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-token perplexity curves")
    parser.add_argument("--rule", type=str, default="top_frac",
                        choices=["top_frac", "top_k", "top_p", "top_k_top_p", "threshold", "none"],
                        help="Refinement rule (default: top_frac)")
    parser.add_argument("--frac", type=float, default=0.1, help="Top-Frac: fraction of pages to refine (default: 0.1)")
    parser.add_argument("--k", type=int, default=3, help="Top-K: number of summaries to refine (default: 3)")
    parser.add_argument("--p", type=float, default=0.9, help="Top-P: cumulative attention threshold (default: 0.9)")
    parser.add_argument("--threshold", type=float, default=0.05, help="Threshold rule: attention threshold (default: 0.05)")
    parser.add_argument("--first-n-pages", type=int, default=0, help="Always refine first N pages as attention sinks (default: 0)")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens to evaluate (default: 2048)")
    parser.add_argument("--prompt", type=str, default="paragraphs_1", help="Prompt name (only used if --dataset not set)")
    parser.add_argument("--dataset", type=str, default=None, choices=["wikitext", "pg19", "c4"],
                        help="Eval dataset to use instead of generation (recommended: wikitext)")
    parser.add_argument("--no-generate", action="store_true",
                        help="Use prompt text directly without generating (faster)")
    parser.add_argument("--plot-only", action="store_true", help="Only generate plots from existing CSV")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Model name/path (default: {DEFAULT_MODEL})")
    parser.add_argument("--log-bias", action="store_true", default=True, help="Include log(N) bias variants")
    parser.add_argument("--no-log-bias", dest="log_bias", action="store_false", help="Exclude log(N) bias variants")
    parser.add_argument("--adaptive-k", action="store_true", default=True, help="Include adaptive log(k) bias variants")
    parser.add_argument("--no-adaptive-k", dest="adaptive_k", action="store_false", help="Exclude adaptive log(k) bias variants")
    parser.add_argument("--encoder", action="store_true", default=False, help="Include trained encoder")
    parser.add_argument("--no-encoder", dest="encoder", action="store_false", help="Exclude trained encoder")
    parser.add_argument("--bias-comparison", action="store_true", default=False,
                        help="Bias comparison mode: Mean & Attn-Weighted only, same color per compressor, "
                             "different linestyle per bias mode (adaptive=bold, fixed=dotted, none=solid)")
    args = parser.parse_args()

    run(
        refinement_rule=args.rule,
        refinement_frac=args.frac,
        refinement_k=args.k,
        refinement_p=args.p,
        refinement_threshold=args.threshold,
        always_refine_first_n=args.first_n_pages,
        max_tokens=args.max_tokens,
        prompt_name=args.prompt,
        eval_dataset=args.dataset,
        no_generate=args.no_generate,
        model_name=args.model,
        include_log_bias=args.log_bias,
        include_adaptive_k=args.adaptive_k,
        include_encoder=args.encoder,
        bias_comparison_mode=args.bias_comparison,
        plot_only=args.plot_only,
    )
