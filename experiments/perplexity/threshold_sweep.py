"""
Sweep LuKA refine thresholds and report perplexity on a sample paragraph.
Uses the DummySegmenter (no boundary detection) and the encoder compressor
from scripts/test.py. Run with `python experiments/perplexity/threshold_sweep.py`.
"""

import csv
import time
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from modeling.segmenter import DummySegmenter
from modeling.compressor import EncoderCompressor
from modeling.qwen.luka_qwen3 import (
    load_luka_model,
    set_luka_kv_params,
    set_luka_segmenter,
)
from artifacts.prompts.prompt_loader import load_prompt


def compute_token_perplexities(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    """Return per-token negative log-likelihood and cumulative perplexity."""
    log_probs = torch.log_softmax(logits, dim=-1)  # [B, T, V]
    nll = -log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B, T]
    nll = nll * mask
    token_counts = mask.sum(dim=1, keepdim=True)  # [B,1]

    # cumulative perplexity over time for the first example
    nll_cumsum = nll.cumsum(dim=1)
    counts = torch.arange(1, labels.size(1) + 1, device=labels.device).unsqueeze(0)
    counts = torch.minimum(counts, token_counts.expand_as(counts))
    avg_nll = nll_cumsum / counts.clamp_min(1)
    perplexity_over_time = torch.exp(avg_nll)[0].tolist()

    # overall perplexity (mask aware)
    total_tokens = token_counts.clamp_min(1)
    total_nll = nll.sum(dim=1, keepdim=True) / total_tokens
    perplexity = torch.exp(total_nll)[0, 0].item()
    return perplexity, perplexity_over_time


def prefill_then_decode_perplexity(model, rollout_ids: torch.Tensor, prompt_len: int):
    """
    Prefill with the prompt to build LuKA pages, then teacher-force through the generated tail
    using the cached state. Returns perplexity over the generated tail and per-token curve.
    """
    device = rollout_ids.device
    B, T = rollout_ids.shape
    assert B == 1, "Only single-sequence evaluation is supported."
    assert prompt_len < T, "Prompt length must be smaller than rollout length."

    # Prefill on the prompt (build cache/pages)
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

    # Decode the generated tail token-by-token
    nll_list = []
    total_tokens = T - prompt_len
    start_time = time.perf_counter()
    for t in range(prompt_len - 1, T - 1):
        cur_id = rollout_ids[:, t : t + 1]  # current token
        attn_mask = torch.ones(1, t + 1, device=device, dtype=rollout_ids.dtype)

        with torch.no_grad():
            out = model(
                input_ids=cur_id,
                attention_mask=attn_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        logits = out.logits[:, -1, :]  # predict next token
        target = rollout_ids[:, t + 1]

        log_probs = torch.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [B]
        nll_list.append(nll)

        past_key_values = out.past_key_values

    device_str = str(device)
    if device_str.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    toks_per_sec = total_tokens / max(elapsed, 1e-8)

    nll_tensor = torch.stack(nll_list, dim=1)  # [B, total_tokens]
    # Perplexity over generated tail only
    total_tokens_tensor = torch.tensor([[total_tokens]], device=device, dtype=nll_tensor.dtype)
    total_nll = nll_tensor.sum(dim=1, keepdim=True) / total_tokens_tensor
    perplexity = torch.exp(total_nll)[0, 0].item()

    # cumulative curve over generated tail
    cumsum = nll_tensor.cumsum(dim=1)
    counts = torch.arange(1, total_tokens + 1, device=device).unsqueeze(0)
    avg_nll = cumsum / counts
    curve = torch.exp(avg_nll)[0].tolist()

    return perplexity, curve, toks_per_sec


def generate_baseline_rollout(tokenizer, prompt: str, device: str, max_new_tokens: int = 128):
    """
    Generate a greedy rollout using LuKA with refine_threshold=-1 (raw attention path).
    Returns token ids (including prompt) and decoded text.
    """
    set_luka_kv_params(
        default_tail_len=16,
        min_compress_chunk=16,
        max_pages=15,
        refine_threshold=-1.0,  # forces full raw attention in top-down
        compressor=None,
    )
    set_luka_segmenter(DummySegmenter())

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
    return gen, text


def run_sweep():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    # Sample paragraph
    paragraph = load_prompt("paragraphs_1")
    if isinstance(paragraph, list):
        paragraph = paragraph[0]

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    tokenizer.pad_token = tokenizer.eos_token

    # 1) Greedy rollout baseline (raw attention path)
    rollout_ids, rollout_text = generate_baseline_rollout(tokenizer, paragraph, device)
    rollout_ids = rollout_ids.to(device)
    prompt_len = tokenizer(paragraph, return_tensors="pt")["input_ids"].size(1)
    # Baseline perplexity/throughput on rollout tail with raw attention
    baseline_model = load_luka_model(
        "Qwen/Qwen3-1.7B-Base",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    ).to(device)
    baseline_model.eval()
    with torch.no_grad():
        base_ppl, base_curve, base_tps = prefill_then_decode_perplexity(
            baseline_model, rollout_ids, prompt_len
        )

    results = []
    for thr in thresholds:
        # Configure LuKA globals before loading the model
        encoder_compressor = EncoderCompressor(dim=128)
        set_luka_kv_params(
            default_tail_len=16,
            min_compress_chunk=16,
            max_pages=15,
            refine_threshold=thr,
            compressor=encoder_compressor,
        )
        set_luka_segmenter(DummySegmenter())

        model = load_luka_model(
            "Qwen/Qwen3-1.7B-Base",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        ).to(device)
        model.eval()

        with torch.no_grad():
            # Prefill on the prompt, then teacher-force decode the generated tail
            ppl, ppl_over_time, tps = prefill_then_decode_perplexity(
                model, rollout_ids, prompt_len
            )

        # Fraction of summary tokens that remained summarized during decode (layer 0)
        # skip = summary tokens that stayed summarized; refine = those expanded
        stats = model.model.layers[0].self_attn.luka_kv_caches.refine_stats[0]
        denom = stats["refine"] + stats["skip"]
        summary_frac = (stats["skip"] / denom) if denom > 0 else None

        results.append((thr, ppl, ppl_over_time, summary_frac, tps))

        # Free GPU memory between runs
        del model
        torch.cuda.empty_cache()

    # Print summary table
    print("\nThreshold sweep (DummySegmenter, EncoderCompressor):")
    for thr, ppl, _, frac, tps in results:
        frac_str = f"{frac:.3f}" if frac is not None else "n/a"
        print(f"  threshold={thr:>4}: perplexity={ppl:.3f}, summary_frac={frac_str}, tps={tps:.2f}")
    print(f"\nBaseline (raw attention) perplexity: {base_ppl:.3f}, tps={base_tps:.2f}")

    # Print perplexity curve for the last run
    last_thr, _, curve, _, _ = results[-1]
    print(f"\nPer-token perplexity curve for threshold={last_thr} (first example):")
    for i, val in enumerate(curve, start=1):
        print(f"  token {i:3d}: {val:.3f}")

    # Plot all curves
    plt.figure(figsize=(8, 5))
    token_axis = list(range(1, len(results[0][2]) + 1))
    for thr, _, curve, _, _ in results:
        plt.plot(token_axis, curve, label=f"thr={thr}")
    plt.plot(token_axis, base_curve, label="baseline (raw)", linestyle="--", color="black")
    plt.xlabel("Token position")
    plt.ylabel("Per-token perplexity")
    plt.title("Per-token perplexity over time (threshold sweep)")
    plt.legend()
    plt.tight_layout()
    out_path = "experiments/perplexity/threshold_sweep.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")

    # Log-scale version of per-token plot (y-axis)
    plt.figure(figsize=(8, 5))
    for thr, _, curve, _, _ in results:
        plt.plot(token_axis, curve, label=f"thr={thr}")
    plt.plot(token_axis, base_curve, label="baseline (raw)", linestyle="--", color="black")
    plt.xlabel("Token position")
    plt.ylabel("Per-token perplexity (log scale)")
    plt.title("Per-token perplexity over time (log y)")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    out_log_path = "experiments/perplexity/threshold_sweep_log.png"
    plt.savefig(out_log_path, dpi=150)
    print(f"Saved log plot to {out_log_path}")

    # Write results to CSV (wide format: one row per threshold with per-token curves)
    csv_path = "experiments/perplexity/threshold_sweep.csv"
    token_axis = list(range(1, len(results[0][2]) + 1))
    fieldnames = ["threshold", "perplexity", "summary_frac", "tokens_per_sec"] + [f"token_{i}_ppl" for i in token_axis]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for thr, ppl, curve, frac, tps in results:
            row = {
                "threshold": thr,
                "perplexity": ppl,
                "summary_frac": frac if frac is not None else "n/a",
                "tokens_per_sec": tps,
            }
            for i, val in zip(token_axis, curve):
                row[f"token_{i}_ppl"] = val
            writer.writerow(row)
        base_row = {
            "threshold": "baseline_raw",
            "perplexity": base_ppl,
            "summary_frac": 0.0,
            "tokens_per_sec": base_tps,
        }
        for i, val in zip(token_axis, base_curve):
            base_row[f"token_{i}_ppl"] = val
        writer.writerow(base_row)
    print(f"Saved CSV to {csv_path}")

    # Plot summary_frac vs perplexity
    plt.figure(figsize=(6, 4))
    xs = []
    ys = []
    labels_ = []
    for thr, ppl, _, frac, _ in results:
        if frac is None:
            continue
        xs.append(frac)
        ys.append(ppl)
        labels_.append(thr)
    xs.append(0.0)
    ys.append(base_ppl)
    labels_.append("baseline")
    plt.scatter(xs, ys, color="tab:blue")
    for x, y, thr in zip(xs, ys, labels_):
        plt.annotate(f"{thr}", (x, y), textcoords="offset points", xytext=(4, 2), fontsize=8)
    plt.xlabel("Summary retention fraction (skip / (skip + refine))")
    plt.ylabel("Perplexity (tail)")
    plt.title("Perplexity vs summary retention")
    plt.tight_layout()
    corr_path = "experiments/perplexity/summary_vs_perplexity.png"
    plt.savefig(corr_path, dpi=150)
    print(f"Saved scatter to {corr_path}")

    # Log-scale scatter (y-axis), only for positive perplexities
    if xs and all(y > 0 for y in ys):
        plt.figure(figsize=(6, 4))
        plt.scatter(xs, ys, color="tab:blue")
        for x, y, thr in zip(xs, ys, labels_):
            plt.annotate(f"{thr}", (x, y), textcoords="offset points", xytext=(4, 2), fontsize=8)
        plt.xlabel("Summary retention fraction (skip / (skip + refine))")
        plt.ylabel("Perplexity (tail, log scale)")
        plt.title("Perplexity vs summary retention (log y)")
        plt.yscale("log")
        plt.tight_layout()
        corr_log_path = "experiments/perplexity/summary_vs_perplexity_log.png"
        plt.savefig(corr_log_path, dpi=150)
        print(f"Saved log scatter to {corr_log_path}")

    # Tokens/sec vs summary retention
    plt.figure(figsize=(6, 4))
    xs_speed = []
    ys_speed = []
    labels_speed = []
    for thr, _, _, frac, tps in results:
        if frac is None or tps is None:
            continue
        xs_speed.append(frac)
        ys_speed.append(tps)
        labels_speed.append(thr)
    xs_speed.append(0.0)
    ys_speed.append(base_tps)
    labels_speed.append("baseline")
    plt.scatter(xs_speed, ys_speed, color="tab:green")
    for x, y, thr in zip(xs_speed, ys_speed, labels_speed):
        plt.annotate(f"{thr}", (x, y), textcoords="offset points", xytext=(4, 2), fontsize=8)
    plt.axhline(base_tps, linestyle="--", color="gray", linewidth=1)
    plt.xlabel("Summary retention fraction (skip / (skip + refine))")
    plt.ylabel("Tokens per second (decode tail)")
    plt.title("Speed vs summary retention")
    plt.tight_layout()
    speed_path = "experiments/perplexity/speed_vs_summary.png"
    plt.savefig(speed_path, dpi=150)
    print(f"Saved speed scatter to {speed_path}")

    if ys_speed and all(y > 0 for y in ys_speed):
        plt.figure(figsize=(6, 4))
        plt.scatter(xs_speed, ys_speed, color="tab:green")
        for x, y, thr in zip(xs_speed, ys_speed, labels_speed):
            plt.annotate(f"{thr}", (x, y), textcoords="offset points", xytext=(4, 2), fontsize=8)
        plt.axhline(base_tps, linestyle="--", color="gray", linewidth=1)
        plt.xlabel("Summary retention fraction (skip / (skip + refine))")
        plt.ylabel("Tokens per second (log scale)")
        plt.title("Speed vs summary retention (log y)")
        plt.yscale("log")
        plt.tight_layout()
        speed_log_path = "experiments/perplexity/speed_vs_summary_log.png"
        plt.savefig(speed_log_path, dpi=150)
        print(f"Saved speed log scatter to {speed_log_path}")


if __name__ == "__main__":
    run_sweep()
