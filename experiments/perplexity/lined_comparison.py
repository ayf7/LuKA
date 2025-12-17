"""
Compare perplexity across attention modes:
  1. Full top-down (LuKA paging on all layers)
  2. Full lined (H2O-style grid tokens on all layers)
  3. Hybrid (first 12 + last 12 layers lined, middle layers top-down)

Run:
    python experiments/perplexity/lined_comparison.py
    python experiments/perplexity/lined_comparison.py --dataset wikitext --max-tokens 2048
    python experiments/perplexity/lined_comparison.py --model Qwen/Qwen3-8B
    python experiments/perplexity/lined_comparison.py --plot-only
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.perplexity.utils import (
    MODEL_NAME,
    get_device,
    get_tokenizer,
    get_prompt,
    get_baseline_perplexity,
    load_eval_text,
    generate_baseline_rollout,
    get_output_dir,
)
from modeling.compressor import AttentionWeightedCompressor
from modeling.qwen.luka_qwen3 import load_luka_model, set_luka_kv_params


# Output paths
OUTPUT_DIR = get_output_dir("lined_comparison")
CSV_PATH = OUTPUT_DIR / "results.csv"
PLOT_PATH = OUTPUT_DIR / "comparison.png"
PLOT_LOG_PATH = OUTPUT_DIR / "comparison_log.png"
CURVE_PLOT_PATH = OUTPUT_DIR / "curves.png"
TOKEN_PLOT_PATH = OUTPUT_DIR / "token.png"
TOKEN_LOG_PLOT_PATH = OUTPUT_DIR / "token_log.png"


def get_attention_configs(num_layers: int):
    """
    Define attention mode configurations.

    Args:
        num_layers: Total number of transformer layers in the model.

    Returns:
        List of config dicts with: name, label, use_lined, lined_layers, color, linestyle
    """
    # Hybrid: first 12 + last 12 are lined, middle uses top-down
    first_n = min(12, num_layers // 3)
    last_n = min(12, num_layers // 3)
    hybrid_lined = set(range(first_n)) | set(range(num_layers - last_n, num_layers))

    return [
        {
            "name": "topdown",
            "label": "Top-Down (LuKA)",
            "use_lined": False,
            "lined_layers": set(),
            "color": "tab:blue",
            "linestyle": "-",
            "linewidth": 2.0,
        },
        {
            "name": "lined",
            "label": "Lined (H2O)",
            "use_lined": True,
            "lined_layers": set(range(num_layers)),
            "color": "tab:orange",
            "linestyle": "-",
            "linewidth": 2.0,
        },
        {
            "name": "hybrid",
            "label": f"Hybrid (0-{first_n-1}, {num_layers-last_n}-{num_layers-1} lined)",
            "use_lined": True,
            "lined_layers": hybrid_lined,
            "color": "tab:green",
            "linestyle": "-",
            "linewidth": 2.0,
        },
    ]


def prefill_then_decode_perplexity(model, rollout_ids, prompt_len):
    """
    Prefill with the prompt to build LuKA pages, then teacher-force through the generated tail.
    Returns perplexity, running avg curve, per-token curve, tokens/sec.
    """
    import time
    import torch

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

    # Subsequent predictions: feed each generated token
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

    # Per-token (instantaneous) perplexity curve
    token_curve = torch.exp(nll_tensor)[0].tolist()

    return ppl, avg_curve, token_curve, tps


def run_single_mode(
    cfg: dict,
    rollout_ids,
    prompt_len: int,
    device: str,
    num_layers: int,
    model_name: str,
):
    """Run perplexity evaluation for a single attention mode."""
    import torch

    print(f"\nRunning {cfg['label']}...")

    # Set LuKA params
    set_luka_kv_params(
        default_tail_len=32,
        min_compress_chunk=16,
        max_pages=256,
        refinement_rule="top_k",
        refinement_rule_kwargs={"k": 5, 'always_refine_first_n': 1},
        compressor=AttentionWeightedCompressor(temperature=1.0),
        log_bias_mode="none",
        segmenter="dummy",
        segment_interval=16,
        create_pages_in_generation=(len(cfg["lined_layers"]) < num_layers),  # Create pages unless all layers are lined
        production_mode=True,
    )

    # Load model
    model = load_luka_model(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    ).to(device)
    model.eval()

    # Configure attention mode
    if hasattr(model, "model") and hasattr(model.model, "luka_kv_controller"):
        controller = model.model.luka_kv_controller
        controller.use_lined_attention = cfg["use_lined"]
        controller.lined_layers = cfg["lined_layers"]
        controller.grid_top_k = 96
        controller.grid_update_interval = 16
        controller.grid_decay = 0.99

    # Run perplexity evaluation
    with torch.no_grad():
        ppl, avg_curve, token_curve, tps = prefill_then_decode_perplexity(model, rollout_ids, prompt_len)

    # Get stats
    summary_frac = None
    pages = 0
    if hasattr(model, "model") and hasattr(model.model, "luka_kv_controller"):
        stats = model.model.luka_kv_controller.get_refinement_stats()
        denom = stats.get("total_summaries_seen", 0)
        refinements = stats.get("total_refinements_made", 0)
        summary_frac = 1.0 - (refinements / denom) if denom > 0 else None
        pages = stats.get("avg_pages_per_layer", 0)

    print(f"  ppl={ppl:.3f}, tps={tps:.1f}")

    del model
    torch.cuda.empty_cache()

    return {
        "perplexity": ppl,
        "avg_curve": avg_curve,      # Running average perplexity
        "token_curve": token_curve,  # Per-token instantaneous perplexity
        "tokens_per_sec": tps,
        "summary_frac": summary_frac,
        "pages": pages,
    }


def save_csv(results, configs, base_ppl, base_tps, base_curve, num_layers, model_name):
    """Save results to CSV."""
    curve_len = len(base_curve) if base_curve else 0
    token_cols = [f"token_{i}_ppl" for i in range(1, curve_len + 1)]

    fieldnames = [
        "mode", "model", "num_layers", "perplexity", "ppl_delta",
        "tokens_per_sec", "summary_frac", "pages"
    ] + token_cols

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Baseline row
        row = {
            "mode": "baseline",
            "model": model_name,
            "num_layers": num_layers,
            "perplexity": base_ppl,
            "ppl_delta": 0.0,
            "tokens_per_sec": base_tps,
            "summary_frac": "n/a",
            "pages": 0,
        }
        for i, val in enumerate(base_curve or [], start=1):
            row[f"token_{i}_ppl"] = val
        writer.writerow(row)

        # Mode rows
        for cfg in configs:
            r = results[cfg["name"]]
            delta = r["perplexity"] - base_ppl
            row = {
                "mode": cfg["name"],
                "model": model_name,
                "num_layers": num_layers,
                "perplexity": r["perplexity"],
                "ppl_delta": delta,
                "tokens_per_sec": r["tokens_per_sec"],
                "summary_frac": r["summary_frac"] if r["summary_frac"] else "n/a",
                "pages": r["pages"],
            }
            for i, val in enumerate(r.get("avg_curve") or r.get("curve") or [], start=1):
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
            mode = row["mode"]
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

            if mode == "baseline":
                base_ppl = float(row["perplexity"])
                base_tps = float(row["tokens_per_sec"])
                base_curve = curve
            else:
                summary_frac = None if row["summary_frac"] == "n/a" else float(row["summary_frac"])
                results[mode] = {
                    "perplexity": float(row["perplexity"]),
                    "curve": curve,
                    "tokens_per_sec": float(row["tokens_per_sec"]),
                    "summary_frac": summary_frac,
                    "pages": float(row["pages"]),
                }

    configs = get_attention_configs(num_layers)
    return results, configs, base_ppl, base_tps, base_curve, num_layers


def generate_plots(results, configs, base_ppl, base_avg_curve, base_token_curve=None):
    """Generate comparison plots."""

    # Bar plot: Perplexity comparison
    plt.figure(figsize=(10, 6))

    modes = ["baseline"] + [cfg["name"] for cfg in configs]
    ppls = [base_ppl] + [results[cfg["name"]]["perplexity"] for cfg in configs]
    colors = ["gray"] + [cfg["color"] for cfg in configs]
    labels = ["Baseline (raw)"] + [cfg["label"] for cfg in configs]

    # Hatching: distinguish pure modes (baseline, pure top-down, pure lined)
    # Index 0 = baseline, 1 = top-down, 2 = lined, 3 = hybrid
    hatches = ["//", "//", "//", ""]  # Baseline, top-down, lined get hatching; hybrid doesn't

    bars = plt.bar(range(len(modes)), ppls, color=colors)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    plt.xticks(range(len(modes)), labels, rotation=15, ha="right")
    plt.ylabel("Perplexity")
    plt.title("Perplexity by Attention Mode")

    # Add value labels
    for bar, ppl in zip(bars, ppls):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{ppl:.2f}", ha="center", va="bottom", fontsize=10)

    # Add delta labels
    for i, cfg in enumerate(configs, start=1):
        delta = results[cfg["name"]]["perplexity"] - base_ppl
        plt.text(i, ppls[i] * 0.5, f"{delta:+.2f}",
                 ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"Saved plot to {PLOT_PATH}")

    # Bar plot: Log scale
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(modes)), ppls, color=colors)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    plt.xticks(range(len(modes)), labels, rotation=15, ha="right")
    plt.ylabel("Perplexity (log scale)")
    plt.title("Perplexity by Attention Mode (log scale)")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(PLOT_LOG_PATH, dpi=150)
    print(f"Saved log plot to {PLOT_LOG_PATH}")

    # Skip first N tokens where perplexity is artificially high
    skip_tokens = 100

    # --- Running average perplexity curves ---
    plt.figure(figsize=(12, 6))

    if base_avg_curve and len(base_avg_curve) > skip_tokens:
        tokens = list(range(skip_tokens + 1, len(base_avg_curve) + 1))
        plt.plot(tokens, base_avg_curve[skip_tokens:], label="Baseline (raw)",
                 color="gray", linestyle="--", linewidth=1.5)

    for cfg in configs:
        curve = results[cfg["name"]].get("avg_curve") or results[cfg["name"]].get("curve")
        if curve and len(curve) > skip_tokens:
            tokens = list(range(skip_tokens + 1, len(curve) + 1))
            plt.plot(tokens, curve[skip_tokens:], label=cfg["label"],
                     color=cfg["color"], linestyle=cfg["linestyle"], linewidth=cfg["linewidth"])

    plt.xlabel("Token Position")
    plt.ylabel("Running Average Perplexity")
    plt.title("Running Average Perplexity by Attention Mode")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CURVE_PLOT_PATH, dpi=150)
    print(f"Saved running avg curve plot to {CURVE_PLOT_PATH}")

    # --- Per-token (instantaneous) perplexity curves ---
    has_token_curves = base_token_curve is not None or any(
        results[cfg["name"]].get("token_curve") for cfg in configs
    )

    if has_token_curves:
        plt.figure(figsize=(12, 6))

        if base_token_curve and len(base_token_curve) > skip_tokens:
            tokens = list(range(skip_tokens + 1, len(base_token_curve) + 1))
            plt.plot(tokens, base_token_curve[skip_tokens:], label="Baseline (raw)",
                     color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

        for cfg in configs:
            curve = results[cfg["name"]].get("token_curve")
            if curve and len(curve) > skip_tokens:
                tokens = list(range(skip_tokens + 1, len(curve) + 1))
                plt.plot(tokens, curve[skip_tokens:], label=cfg["label"],
                         color=cfg["color"], linestyle=cfg["linestyle"],
                         linewidth=0.8, alpha=0.7)

        plt.xlabel("Token Position")
        plt.ylabel("Per-Token Perplexity")
        plt.title("Per-Token (Instantaneous) Perplexity by Attention Mode")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(TOKEN_PLOT_PATH, dpi=150)
        print(f"Saved per-token curve plot to {TOKEN_PLOT_PATH}")

        # Log scale version
        plt.figure(figsize=(12, 6))

        if base_token_curve and len(base_token_curve) > skip_tokens:
            tokens = list(range(skip_tokens + 1, len(base_token_curve) + 1))
            plt.plot(tokens, base_token_curve[skip_tokens:], label="Baseline (raw)",
                     color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

        for cfg in configs:
            curve = results[cfg["name"]].get("token_curve")
            if curve and len(curve) > skip_tokens:
                tokens = list(range(skip_tokens + 1, len(curve) + 1))
                plt.plot(tokens, curve[skip_tokens:], label=cfg["label"],
                         color=cfg["color"], linestyle=cfg["linestyle"],
                         linewidth=0.8, alpha=0.7)

        plt.xlabel("Token Position")
        plt.ylabel("Per-Token Perplexity (log scale)")
        plt.title("Per-Token Perplexity by Attention Mode (log scale)")
        plt.yscale("log")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(TOKEN_LOG_PLOT_PATH, dpi=150)
        print(f"Saved per-token log curve plot to {TOKEN_LOG_PLOT_PATH}")

    plt.close("all")


def run(
    max_tokens: int = 2048,
    prompt_name: str = "paragraphs_1",
    eval_dataset: str = None,
    model_name: str = None,
    prompt_len_override: int = None,
    allow_skipping: bool = False,
):
    import torch

    if model_name is None:
        model_name = MODEL_NAME

    # Check for existing results if allow_skipping is enabled
    if allow_skipping and CSV_PATH.exists():
        print(f"Found existing CSV at {CSV_PATH}, loading cached results...")
        try:
            results, configs, base_ppl, base_tps, base_curve, num_layers = load_csv()
            print(f"  Loaded results for {len(results)} configurations")
            print(f"  Baseline: ppl={base_ppl:.3f}, tps={base_tps:.1f}")
            # Generate plots from cached data
            generate_plots(results, configs, base_ppl, base_curve, base_token_curve=None)
            return results, configs, base_ppl, base_tps, base_curve
        except Exception as e:
            print(f"  Failed to load CSV: {e}, running experiments...")

    device = get_device()
    tokenizer = get_tokenizer(model_name)

    # Load evaluation data
    if eval_dataset:
        print(f"Loading eval text from {eval_dataset} (max {max_tokens} tokens)...")
        _, rollout_ids = load_eval_text(eval_dataset, max_tokens, tokenizer=tokenizer)
        rollout_ids = rollout_ids.to(device)
        # Pure teacher-forcing: use minimal prompt (just first token for context)
        # We evaluate perplexity on tokens 1 to N
        if prompt_len_override is not None:
            prompt_len = prompt_len_override
        else:
            prompt_len = 1  # Just seed with first token, evaluate on the rest
        print(f"Total tokens: {rollout_ids.shape[1]}, prompt_len: {prompt_len} (teacher-forcing on {rollout_ids.shape[1] - prompt_len} tokens)")
    else:
        # Load paragraph as evaluation text (teacher-forcing), not as generation prompt
        paragraph_text = get_prompt(prompt_name)
        print(f"Loading paragraph '{prompt_name}' for teacher-forcing evaluation...")
        rollout_ids = tokenizer(paragraph_text, return_tensors="pt")["input_ids"]
        if rollout_ids.shape[1] > max_tokens:
            rollout_ids = rollout_ids[:, :max_tokens]
        rollout_ids = rollout_ids.to(device)
        # Teacher-forcing: seed with first token, evaluate on the rest
        if prompt_len_override is not None:
            prompt_len = prompt_len_override
        else:
            prompt_len = 1
        print(f"Total tokens: {rollout_ids.shape[1]}, prompt_len: {prompt_len} (teacher-forcing on {rollout_ids.shape[1] - prompt_len} tokens)")

    # Get number of layers from a quick model load
    print(f"\nLoading model to get layer count: {model_name}")
    set_luka_kv_params(
        use_exact_attention=True,
        compressor="attention_weighted",
        segmenter="dummy",
        create_pages_in_generation=False,
        max_pages=256,
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

    # Get attention configs
    configs = get_attention_configs(num_layers)

    # Get baseline perplexity
    print("\nRunning baseline (raw attention)...")
    base_ppl, base_avg_curve, base_token_curve, base_tps = get_baseline_perplexity(rollout_ids, prompt_len, device, model_name)
    print(f"  ppl={base_ppl:.3f}, tps={base_tps:.1f}")

    # Run each attention mode
    results = {}
    for cfg in configs:
        result = run_single_mode(cfg, rollout_ids, prompt_len, device, num_layers, model_name)
        results[cfg["name"]] = result

    # Print summary table
    print("\n" + "=" * 80)
    print("Lined Attention Comparison Summary")
    print("=" * 80)
    print(f"\n{'Mode':<35} {'PPL':>10} {'Delta':>10} {'TPS':>10}")
    print("-" * 70)
    print(f"{'Baseline (raw)':<35} {base_ppl:>10.3f} {0:>10.2f} {base_tps:>10.1f}")
    for cfg in configs:
        r = results[cfg["name"]]
        delta = r["perplexity"] - base_ppl
        print(f"{cfg['label']:<35} {r['perplexity']:>10.3f} {delta:>+10.2f} {r['tokens_per_sec']:>10.1f}")
    print("=" * 80)

    # Save CSV (use avg_curve for backwards compat)
    save_csv(results, configs, base_ppl, base_tps, base_avg_curve, num_layers, model_name)

    # Generate plots
    generate_plots(results, configs, base_ppl, base_avg_curve, base_token_curve)

    return results, configs, base_ppl, base_tps, base_avg_curve


def main():
    parser = argparse.ArgumentParser(description="Compare attention modes on perplexity")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Model name (default: {MODEL_NAME})")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["wikitext", "pg19", "c4"],
                        help="Dataset for evaluation (default: generate from prompt)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max tokens to evaluate (default: 2048)")
    parser.add_argument("--prompt-len", type=int, default=None,
                        help="Prompt length for prefill (default: 1/4 of max-tokens for dataset, auto for generation)")
    parser.add_argument("--prompt", type=str, default="paragraphs_1",
                        help="Prompt name if not using dataset")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only generate plots from existing CSV")
    parser.add_argument("--allow-skipping", action="store_true",
                        help="Skip experiments if results already exist in CSV")
    args = parser.parse_args()

    if args.plot_only:
        print("Loading data from CSV...")
        results, configs, base_ppl, base_tps, base_curve, _ = load_csv()
        print(f"Baseline: ppl={base_ppl:.3f}, tps={base_tps:.1f}")
        # CSV only has running avg curves, not per-token; pass None for token curves
        generate_plots(results, configs, base_ppl, base_curve, base_token_curve=None)
    else:
        run(
            max_tokens=args.max_tokens,
            prompt_name=args.prompt,
            eval_dataset=args.dataset,
            model_name=args.model,
            prompt_len_override=args.prompt_len,
            allow_skipping=args.allow_skipping,
        )


if __name__ == "__main__":
    main()
