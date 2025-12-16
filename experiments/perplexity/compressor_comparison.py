"""
Compare perplexity across different compressor strategies:
  1. AttentionWeightedCompressor (recommended)
  2. EncoderCompressor (trained, if checkpoint available)
  3. MeanCompressor with log(N) bias
  4. MeanCompressor without log(N) bias

Run: python experiments/perplexity/compressor_comparison.py
"""

import csv
import time
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoTokenizer

from modeling.compressor import (
    AttentionWeightedCompressor,
    EncoderCompressor,
    MeanCompressor,
)
from modeling.qwen.luka_qwen3 import load_luka_model, set_luka_kv_params
from artifacts.prompts.prompt_loader import load_prompt


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


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    paragraph = load_prompt("paragraphs_1")
    if isinstance(paragraph, list):
        paragraph = paragraph[0]

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    tokenizer.pad_token = tokenizer.eos_token

    # Generate baseline rollout
    print("Generating baseline rollout...")
    rollout_ids, _ = generate_baseline_rollout(tokenizer, paragraph, device)
    rollout_ids = rollout_ids.to(device)
    prompt_len = tokenizer(paragraph, return_tensors="pt")["input_ids"].size(1)
    print(f"Baseline rollout: {rollout_ids.shape[1]} tokens, prompt_len={prompt_len}")

    # Check for trained encoder checkpoint
    # Try multiple possible paths
    checkpoint_paths = [
        Path("train_1/step_1000.pt"),
        Path("artifacts/compressor_checkpoints/layer_0_step_1000.pt"),
    ]
    checkpoint_path = None
    for cp in checkpoint_paths:
        if cp.exists():
            checkpoint_path = cp
            break
    has_trained_encoder = checkpoint_path is not None

    # Define compressor configurations
    compressors = [
        {
            "name": "attention_weighted",
            "label": "Attn-Weighted (no bias)",
            "compressor": AttentionWeightedCompressor(temperature=1.0),
            "use_log_bias": False,
        },
        {
            "name": "attention_weighted_bias",
            "label": "Attn-Weighted (log N bias)",
            "compressor": AttentionWeightedCompressor(temperature=1.0),
            "use_log_bias": True,
        },
        {
            "name": "mean_no_bias",
            "label": "Mean (no bias)",
            "compressor": MeanCompressor(),
            "use_log_bias": False,
        },
        {
            "name": "mean_with_bias",
            "label": "Mean (log N bias)",
            "compressor": MeanCompressor(),
            "use_log_bias": True,
        },
    ]

    if has_trained_encoder and checkpoint_path is not None:
        compressors.append({
            "name": "trained_encoder",
            "label": "Trained Encoder",
            "compressor": EncoderCompressor(checkpoint_path=str(checkpoint_path)),
            "use_log_bias": False,
        })
        print(f"Found trained encoder checkpoint at {checkpoint_path}")
    else:
        print(f"No trained encoder checkpoint found in {checkpoint_paths}, skipping.")

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
                create_pages_in_generation=True,  # Explicitly enable page creation during decode
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

    # Print summary table
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    for cfg in compressors:
        print(f"\n{cfg['label']}:")
        for r in records[cfg["name"]]:
            frac_str = f"{r['summary_frac']:.3f}" if r["summary_frac"] else "n/a"
            print(f"  thr={r['threshold']:>4}: ppl={r['perplexity']:.3f}, "
                  f"summary_frac={frac_str}, tps={r['tokens_per_sec']:.1f}")

    # Plot: Perplexity vs Threshold
    plt.figure(figsize=(8, 5))
    for cfg in compressors:
        recs = records[cfg["name"]]
        xs = [r["threshold"] for r in recs]
        ys = [r["perplexity"] for r in recs]
        plt.plot(xs, ys, marker="o", label=cfg["label"])
    plt.xlabel("Refine Threshold")
    plt.ylabel("Perplexity (tail)")
    plt.title("Perplexity vs Refine Threshold by Compressor")
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    out_path = "experiments/perplexity/compressor_ppl_vs_threshold.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")

    # Plot: Perplexity vs Threshold (log y)
    plt.figure(figsize=(8, 5))
    for cfg in compressors:
        recs = records[cfg["name"]]
        xs = [r["threshold"] for r in recs]
        ys = [r["perplexity"] for r in recs]
        plt.plot(xs, ys, marker="o", label=cfg["label"])
    plt.xlabel("Refine Threshold")
    plt.ylabel("Perplexity (log scale)")
    plt.title("Perplexity vs Refine Threshold by Compressor (log y)")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    out_log_path = "experiments/perplexity/compressor_ppl_vs_threshold_log.png"
    plt.savefig(out_log_path, dpi=150)
    print(f"Saved log plot to {out_log_path}")

    # Plot: Summary Retention vs Perplexity
    plt.figure(figsize=(8, 5))
    for cfg in compressors:
        recs = records[cfg["name"]]
        xs = [r["summary_frac"] if r["summary_frac"] else 0.0 for r in recs]
        ys = [r["perplexity"] for r in recs]
        plt.plot(xs, ys, marker="o", label=cfg["label"])
        for x, y, r in zip(xs, ys, recs):
            plt.annotate(f"{r['threshold']}", (x, y), textcoords="offset points",
                        xytext=(3, 3), fontsize=6)
    plt.xlabel("Summary Retention (1 - refinement_rate)")
    plt.ylabel("Perplexity (tail)")
    plt.title("Perplexity vs Summary Retention by Compressor")
    plt.legend()
    plt.tight_layout()
    retention_path = "experiments/perplexity/compressor_retention_vs_ppl.png"
    plt.savefig(retention_path, dpi=150)
    print(f"Saved retention plot to {retention_path}")

    # Plot: Per-token perplexity curves (for threshold=0.05)
    plt.figure(figsize=(10, 6))
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
        token_axis = list(range(1, len(curve) + 1))
        plt.plot(token_axis, curve, label=f"{cfg['label']} (thr={target_rec['threshold']})")

    plt.xlabel("Token Position")
    plt.ylabel("Cumulative Perplexity")
    plt.title("Per-token Perplexity Curve by Compressor (threshold~0.05)")
    plt.legend()
    plt.tight_layout()
    curve_path = "experiments/perplexity/compressor_pertoken_curve.png"
    plt.savefig(curve_path, dpi=150)
    print(f"Saved per-token curve to {curve_path}")

    # CSV output
    csv_path = "experiments/perplexity/compressor_comparison.csv"
    first_curve_len = len(records[compressors[0]["name"]][0]["curve"]) if records else 0
    token_cols = [f"token_{i}_ppl" for i in range(1, first_curve_len + 1)]
    fieldnames = [
        "compressor", "threshold", "perplexity", "summary_frac",
        "tokens_per_sec", "pages", "refinement_rate"
    ] + token_cols

    with open(csv_path, "w", newline="") as f:
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
    print(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    run()
