"""
Compare perplexity across different compressor strategies:
  1. AttentionWeightedCompressor (recommended)
  2. EncoderCompressor (trained, if checkpoint available)
  3. MeanCompressor with log(N) bias
  4. MeanCompressor without log(N) bias

Run: python experiments/perplexity/compressor_comparison.py
     python experiments/perplexity/compressor_comparison.py --plot-only
"""

import argparse
import csv
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoTokenizer

from modeling.compressor import (
    AttentionWeightedCompressor,
    EncoderCompressor,
    EvictionCompressor,
    MeanCompressor,
)
from modeling.qwen.luka_qwen3 import load_luka_model, set_luka_kv_params
from artifacts.prompts.prompt_loader import load_prompt


# Output paths
OUTPUT_DIR = Path("experiments/perplexity")
CSV_PATH = OUTPUT_DIR / "compressor_comparison.csv"
RETENTION_CSV_PATH = OUTPUT_DIR / "compressor_retention.csv"
PPL_PLOT_PATH = OUTPUT_DIR / "compressor_ppl_vs_threshold.png"
PPL_LOG_PLOT_PATH = OUTPUT_DIR / "compressor_ppl_vs_threshold_log.png"
RETENTION_PLOT_PATH = OUTPUT_DIR / "compressor_retention_vs_ppl.png"
RETENTION_LOG_PLOT_PATH = OUTPUT_DIR / "compressor_retention_vs_ppl_log.png"
RETENTION_LOGLOG_PLOT_PATH = OUTPUT_DIR / "compressor_retention_vs_ppl_loglog.png"
CURVE_PLOT_PATH = OUTPUT_DIR / "compressor_pertoken_curve.png"


# Compressor configurations
def get_compressor_configs(include_log_bias=True):
    """Define compressor configurations for testing."""
    compressors = [
        {
            "name": "attention_weighted",
            "label": "Attn-Weighted",
            "compressor": AttentionWeightedCompressor(temperature=1.0),
            "use_log_bias": False,
            "color": "tab:blue",
            "linestyle": "-",
        },
        {
            "name": "mean",
            "label": "Mean",
            "compressor": MeanCompressor(),
            "use_log_bias": False,
            "color": "tab:orange",
            "linestyle": "-",
        },
        {
            "name": "eviction",
            "label": "Eviction",
            "compressor": EvictionCompressor(temperature=1.0),
            "use_log_bias": False,
            "color": "tab:purple",
            "linestyle": "-",
        },
    ]

    if include_log_bias:
        compressors.extend([
            {
                "name": "attention_weighted_logbias",
                "label": "Attn-Weighted + log(N)",
                "compressor": AttentionWeightedCompressor(temperature=1.0),
                "use_log_bias": True,
                "color": "tab:blue",
                "linestyle": ":",
            },
            {
                "name": "mean_logbias",
                "label": "Mean + log(N)",
                "compressor": MeanCompressor(),
                "use_log_bias": True,
                "color": "tab:orange",
                "linestyle": ":",
            },
            {
                "name": "eviction_logbias",
                "label": "Eviction + log(N)",
                "compressor": EvictionCompressor(temperature=1.0),
                "use_log_bias": True,
                "color": "tab:purple",
                "linestyle": ":",
            },
        ])

    return compressors


def generate_baseline_rollout(tokenizer, prompt: str, device: str, max_new_tokens: int = 128):
    """
    Generate a greedy rollout using LuKA with refine_threshold=-1 (raw attention path).
    Returns token ids (including prompt) and decoded text.
    """
    set_luka_kv_params(
        default_tail_len=16,
        min_compress_chunk=16,
        max_pages=15,
        refine_threshold=-1.0,  # forces full raw attention
        compressor="mean",
        segmenter="dummy",
        create_pages_in_generation=False,  # No pages for baseline
    )

    model = load_luka_model(
        "Qwen/Qwen3-1.7B-Base",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    ).to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            use_cache=True,
        )
    text = tokenizer.decode(gen[0], skip_special_tokens=True)

    del model
    torch.cuda.empty_cache()

    return gen, text


def prefill_then_decode_perplexity(model, rollout_ids: torch.Tensor, prompt_len: int):
    """
    Prefill with the prompt to build LuKA pages, then teacher-force through the generated tail.
    Returns perplexity, per-token curve, tokens/sec.
    """
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
            output_attentions=True,
        )
    past_key_values = pre_out.past_key_values

    nll_list = []
    total_tokens = T - prompt_len
    start_time = time.perf_counter()

    # First prediction: use prefill logits (no extra forward pass needed)
    logits = pre_out.logits[:, -1, :]
    target = rollout_ids[:, prompt_len]
    log_probs = torch.log_softmax(logits, dim=-1)
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
        log_probs = torch.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        nll_list.append(nll)
        past_key_values = out.past_key_values

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    tps = total_tokens / max(elapsed, 1e-8)

    nll_tensor = torch.stack(nll_list, dim=1)
    total_tokens_tensor = torch.tensor([[total_tokens]], device=device, dtype=nll_tensor.dtype)
    total_nll = nll_tensor.sum(dim=1, keepdim=True) / total_tokens_tensor
    ppl = torch.exp(total_nll)[0, 0].item()

    cumsum = nll_tensor.cumsum(dim=1)
    counts = torch.arange(1, total_tokens + 1, device=device).unsqueeze(0)
    avg_nll = cumsum / counts
    curve = torch.exp(avg_nll)[0].tolist()

    return ppl, curve, tps


def get_stats_from_model(model):
    """Extract refinement statistics from the model."""
    if hasattr(model, "model") and hasattr(model.model, "luka_kv_controller"):
        stats = model.model.luka_kv_controller.get_refinement_stats()
        denom = stats.get("total_summaries_seen", 0)
        refinements = stats.get("total_refinements_made", 0)
        summary_frac = 1.0 - (refinements / denom) if denom > 0 else None
        return summary_frac, stats
    return None, {}


def run_experiments(thresholds, max_new_tokens, include_log_bias=True):
    """Run all compressor experiments and return records."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    paragraph = load_prompt("paragraphs_1")
    if isinstance(paragraph, list):
        paragraph = paragraph[0]

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    tokenizer.pad_token = tokenizer.eos_token

    # Generate baseline rollout
    print("Generating baseline rollout...")
    rollout_ids, _ = generate_baseline_rollout(tokenizer, paragraph, device, max_new_tokens)
    rollout_ids = rollout_ids.to(device)
    prompt_len = tokenizer(paragraph, return_tensors="pt")["input_ids"].size(1)
    print(f"Baseline rollout: {rollout_ids.shape[1]} tokens, prompt_len={prompt_len}")

    # Evaluate baseline perplexity (raw attention, no compression)
    print("\nEvaluating baseline (raw attention)...")
    set_luka_kv_params(
        default_tail_len=16,
        min_compress_chunk=16,
        max_pages=15,
        refine_threshold=-1,  # Raw attention path
        compressor="mean",
        segmenter="dummy",
        create_pages_in_generation=False,
        production_mode=True,
    )
    baseline_model = load_luka_model(
        "Qwen/Qwen3-1.7B-Base",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    ).to(device)
    baseline_model.eval()
    with torch.no_grad():
        baseline_ppl, baseline_curve, baseline_tps = prefill_then_decode_perplexity(
            baseline_model, rollout_ids, prompt_len
        )
    del baseline_model
    torch.cuda.empty_cache()
    print(f"Baseline perplexity: {baseline_ppl:.3f}, tps={baseline_tps:.1f}")

    # Get compressor configurations
    compressors = get_compressor_configs(include_log_bias=include_log_bias)
    records = {cfg["name"]: [] for cfg in compressors}

    for cfg in compressors:
        print(f"\nRunning {cfg['label']}...")

        for thr in thresholds:
            print(f"  threshold={thr}...", end=" ", flush=True)

            set_luka_kv_params(
                default_tail_len=16,
                min_compress_chunk=16,
                max_pages=15,
                refine_threshold=thr,
                compressor=cfg["compressor"],
                use_log_bias=cfg["use_log_bias"],
                segmenter="dummy",
                segment_interval=16,
                create_pages_in_generation=True,
            )

            model = load_luka_model(
                "Qwen/Qwen3-1.7B-Base",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
            ).to(device)
            model.eval()

            with torch.no_grad():
                ppl, curve, tps = prefill_then_decode_perplexity(
                    model, rollout_ids, prompt_len
                )

            summary_frac, stats = get_stats_from_model(model)

            records[cfg["name"]].append({
                "threshold": thr,
                "perplexity": ppl,
                "summary_frac": summary_frac,
                "tokens_per_sec": tps,
                "curve": curve,
                "pages": stats.get("avg_pages_per_layer", 0),
                "refinement_rate": stats.get("refinement_rate", 0),
            })

            frac_str = f"{summary_frac:.3f}" if summary_frac is not None else "n/a"
            print(f"ppl={ppl:.3f}, summary_frac={frac_str}")

            del model
            torch.cuda.empty_cache()

    # Add baseline info to records
    baseline_info = {
        "ppl": baseline_ppl,
        "curve": baseline_curve,
        "tps": baseline_tps,
    }

    return records, compressors, baseline_info


def save_csvs(records, compressors, baseline_info):
    """Save experiment data to CSV files."""
    # Full CSV with per-token curves
    first_curve_len = len(records[compressors[0]["name"]][0]["curve"]) if records else 0
    token_cols = [f"token_{i}_ppl" for i in range(1, first_curve_len + 1)]
    fieldnames = [
        "compressor", "threshold", "perplexity", "summary_frac",
        "tokens_per_sec", "pages", "refinement_rate"
    ] + token_cols

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cfg in compressors:
            for r in records[cfg["name"]]:
                row = {
                    "compressor": cfg["name"],
                    "threshold": r["threshold"],
                    "perplexity": r["perplexity"],
                    "summary_frac": r["summary_frac"] if r["summary_frac"] else "n/a",
                    "tokens_per_sec": r["tokens_per_sec"],
                    "pages": r["pages"],
                    "refinement_rate": r["refinement_rate"],
                }
                for i, val in enumerate(r["curve"], start=1):
                    row[f"token_{i}_ppl"] = val
                writer.writerow(row)
    print(f"Saved CSV to {CSV_PATH}")

    # Retention CSV (simpler, just key metrics)
    retention_fieldnames = ["compressor", "threshold", "perplexity", "summary_frac", "refinement_rate"]
    with open(RETENTION_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=retention_fieldnames)
        writer.writeheader()
        # Add baseline row
        writer.writerow({
            "compressor": "baseline",
            "threshold": "n/a",
            "perplexity": baseline_info["ppl"],
            "summary_frac": "n/a",
            "refinement_rate": "n/a",
        })
        for cfg in compressors:
            for r in records[cfg["name"]]:
                writer.writerow({
                    "compressor": cfg["name"],
                    "threshold": r["threshold"],
                    "perplexity": r["perplexity"],
                    "summary_frac": r["summary_frac"] if r["summary_frac"] else "n/a",
                    "refinement_rate": r["refinement_rate"],
                })
    print(f"Saved retention CSV to {RETENTION_CSV_PATH}")


def load_csvs(include_log_bias=True):
    """Load experiment data from CSV files."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}. Run without --plot-only first.")

    records = {}
    compressor_meta = {}  # Store metadata for each compressor

    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["compressor"]

            # Skip log bias variants if not included
            if not include_log_bias and "logbias" in name:
                continue

            if name not in records:
                records[name] = []
                # Infer config from name
                compressor_meta[name] = {
                    "name": name,
                    "label": name.replace("_", " ").title().replace("Logbias", "+ log(N)"),
                    "color": "tab:blue" if "attention" in name else ("tab:orange" if "mean" in name else "tab:purple"),
                    "linestyle": ":" if "logbias" in name else "-",
                }

            # Parse curve columns
            curve = []
            for key in row:
                if key.startswith("token_") and key.endswith("_ppl"):
                    curve.append(float(row[key]))

            summary_frac = None if row["summary_frac"] == "n/a" else float(row["summary_frac"])

            records[name].append({
                "threshold": float(row["threshold"]),
                "perplexity": float(row["perplexity"]),
                "summary_frac": summary_frac,
                "tokens_per_sec": float(row["tokens_per_sec"]),
                "curve": curve,
                "pages": float(row["pages"]),
                "refinement_rate": float(row["refinement_rate"]),
            })

    # Build compressors list from metadata
    compressors = list(compressor_meta.values())

    # Load baseline from retention CSV if available
    baseline_info = {"ppl": None, "curve": None, "tps": None}
    if RETENTION_CSV_PATH.exists():
        with open(RETENTION_CSV_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["compressor"] == "baseline":
                    baseline_info["ppl"] = float(row["perplexity"])
                    break

    return records, compressors, baseline_info


def generate_plots(records, compressors, baseline_info):
    """Generate all plots from records."""
    # Plot: Perplexity vs Threshold
    plt.figure(figsize=(10, 6))
    if baseline_info["ppl"]:
        plt.axhline(y=baseline_info["ppl"], color="black", linestyle="--", linewidth=1.5, label="Baseline (raw)")
    for cfg in compressors:
        recs = records[cfg["name"]]
        xs = [r["threshold"] for r in recs]
        ys = [r["perplexity"] for r in recs]
        plt.plot(xs, ys, marker="o", label=cfg["label"],
                 color=cfg["color"], linestyle=cfg["linestyle"], linewidth=2)
    plt.xlabel("Refine Threshold")
    plt.ylabel("Perplexity (tail)")
    plt.title("Perplexity vs Refine Threshold by Compressor")
    plt.xscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PPL_PLOT_PATH, dpi=150)
    print(f"Saved plot to {PPL_PLOT_PATH}")

    # Plot: Perplexity vs Threshold (log y)
    plt.figure(figsize=(10, 6))
    if baseline_info["ppl"]:
        plt.axhline(y=baseline_info["ppl"], color="black", linestyle="--", linewidth=1.5, label="Baseline (raw)")
    for cfg in compressors:
        recs = records[cfg["name"]]
        xs = [r["threshold"] for r in recs]
        ys = [r["perplexity"] for r in recs]
        plt.plot(xs, ys, marker="o", label=cfg["label"],
                 color=cfg["color"], linestyle=cfg["linestyle"], linewidth=2)
    plt.xlabel("Refine Threshold")
    plt.ylabel("Perplexity (log scale)")
    plt.title("Perplexity vs Refine Threshold by Compressor (log y)")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PPL_LOG_PLOT_PATH, dpi=150)
    print(f"Saved log plot to {PPL_LOG_PLOT_PATH}")

    # Plot: Summary Retention vs Perplexity
    plt.figure(figsize=(10, 6))
    if baseline_info["ppl"]:
        plt.axhline(y=baseline_info["ppl"], color="black", linestyle="--", linewidth=1.5, label="Baseline (raw)")
    for cfg in compressors:
        recs = records[cfg["name"]]
        xs = [r["summary_frac"] if r["summary_frac"] else 0.0 for r in recs]
        ys = [r["perplexity"] for r in recs]
        plt.plot(xs, ys, marker="o", label=cfg["label"],
                 color=cfg["color"], linestyle=cfg["linestyle"], linewidth=2)
        for x, y, r in zip(xs, ys, recs):
            plt.annotate(f"{r['threshold']}", (x, y), textcoords="offset points",
                        xytext=(3, 3), fontsize=6)
    plt.xlabel("Summary Retention (1 - refinement_rate)")
    plt.ylabel("Perplexity (tail)")
    plt.title("Perplexity vs Summary Retention by Compressor")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RETENTION_PLOT_PATH, dpi=150)
    print(f"Saved retention plot to {RETENTION_PLOT_PATH}")

    # Plot: Summary Retention vs Perplexity (log scale)
    plt.figure(figsize=(10, 6))
    if baseline_info["ppl"]:
        plt.axhline(y=baseline_info["ppl"], color="black", linestyle="--", linewidth=1.5, label="Baseline (raw)")
    for cfg in compressors:
        recs = records[cfg["name"]]
        xs = [r["summary_frac"] if r["summary_frac"] else 0.0 for r in recs]
        ys = [r["perplexity"] for r in recs]
        plt.plot(xs, ys, marker="o", label=cfg["label"],
                 color=cfg["color"], linestyle=cfg["linestyle"], linewidth=2)
        for x, y, r in zip(xs, ys, recs):
            plt.annotate(f"{r['threshold']}", (x, y), textcoords="offset points",
                        xytext=(3, 3), fontsize=6)
    plt.xlabel("Summary Retention (1 - refinement_rate)")
    plt.ylabel("Perplexity (log scale)")
    plt.title("Perplexity vs Summary Retention by Compressor (log y)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RETENTION_LOG_PLOT_PATH, dpi=150)
    print(f"Saved retention log plot to {RETENTION_LOG_PLOT_PATH}")

    # Plot: Summary Retention vs Perplexity (log(log(y)) scale)
    plt.figure(figsize=(10, 6))
    if baseline_info["ppl"] and baseline_info["ppl"] > 1:
        baseline_loglog = np.log(np.log(baseline_info["ppl"]))
        plt.axhline(y=baseline_loglog, color="black", linestyle="--", linewidth=1.5, label="Baseline (raw)")
    for cfg in compressors:
        recs = records[cfg["name"]]
        xs = [r["summary_frac"] if r["summary_frac"] else 0.0 for r in recs]
        # Transform y to log(log(y)), filtering out values <= 1
        ys_raw = [r["perplexity"] for r in recs]
        ys = [np.log(np.log(y)) if y > 1 else np.nan for y in ys_raw]
        plt.plot(xs, ys, marker="o", label=cfg["label"],
                 color=cfg["color"], linestyle=cfg["linestyle"], linewidth=2)
        for x, y, r in zip(xs, ys, recs):
            if not np.isnan(y):
                plt.annotate(f"{r['threshold']}", (x, y), textcoords="offset points",
                            xytext=(3, 3), fontsize=6)
    plt.xlabel("Summary Retention")
    plt.ylabel("log(log(Perplexity))")
    plt.title("log(log(Perplexity)) vs Summary Retention by Compressor")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RETENTION_LOGLOG_PLOT_PATH, dpi=150)
    print(f"Saved retention log-log plot to {RETENTION_LOGLOG_PLOT_PATH}")

    # Plot: Per-token perplexity curves (for threshold~0.05 or middle threshold)
    plt.figure(figsize=(12, 6))
    for cfg in compressors:
        recs = records[cfg["name"]]
        # Find threshold=0.05 or closest
        target_rec = None
        for r in recs:
            if abs(r["threshold"] - 0.05) < 0.01:
                target_rec = r
                break
        if target_rec is None:
            target_rec = recs[len(recs) // 2]

        curve = target_rec["curve"]
        if curve:
            token_axis = list(range(1, len(curve) + 1))
            plt.plot(token_axis, curve, label=f"{cfg['label']} (thr={target_rec['threshold']})",
                     color=cfg["color"], linestyle=cfg["linestyle"], linewidth=1.5)

    plt.xlabel("Token Position")
    plt.ylabel("Cumulative Perplexity")
    plt.title("Per-token Perplexity Curve by Compressor (threshold~0.05)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CURVE_PLOT_PATH, dpi=150)
    print(f"Saved per-token curve to {CURVE_PLOT_PATH}")


def print_summary(records, compressors, baseline_info):
    """Print summary table."""
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    if baseline_info["ppl"]:
        print(f"\nBaseline (raw attention): ppl={baseline_info['ppl']:.3f}")
    for cfg in compressors:
        print(f"\n{cfg['label']}:")
        for r in records[cfg["name"]]:
            frac_str = f"{r['summary_frac']:.3f}" if r["summary_frac"] else "n/a"
            print(f"  thr={r['threshold']:>6}: ppl={r['perplexity']:.3f}, "
                  f"summary_frac={frac_str}, tps={r['tokens_per_sec']:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Compare compressor perplexity")
    parser.add_argument("--thresholds", type=str, default="0,1e-6,1e-4,1e-3,1e-2,1e-1,1",
                        help="Comma-separated thresholds to test")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Max new tokens to generate for rollout")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only generate plots from existing CSVs")
    parser.add_argument("--no-log-bias", action="store_true",
                        help="Exclude log(N) bias variants")
    args = parser.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]
    include_log_bias = not args.no_log_bias

    if args.plot_only:
        print("Loading data from CSVs...")
        records, compressors, baseline_info = load_csvs(include_log_bias=include_log_bias)
        print_summary(records, compressors, baseline_info)
        generate_plots(records, compressors, baseline_info)
    else:
        records, compressors, baseline_info = run_experiments(
            thresholds, args.max_tokens, include_log_bias
        )
        print_summary(records, compressors, baseline_info)
        save_csvs(records, compressors, baseline_info)
        generate_plots(records, compressors, baseline_info)


if __name__ == "__main__":
    main()
