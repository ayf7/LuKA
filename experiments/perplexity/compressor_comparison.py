"""
Compare summary retention vs perplexity across compressors (Random vs Mean)
over a sweep of refine thresholds. Produces a line+scatter plot.
Run: python experiments/perplexity/compressor_comparison.py
"""

import csv
import time
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from modeling.segmenter import DummySegmenter
from modeling.compressor import Compressor, MeanCompressor
from modeling.qwen.luka_qwen3 import (
    load_luka_model,
    set_luka_kv_params,
    set_luka_segmenter,
)
from artifacts.prompts.prompt_loader import load_prompt


class RandomCompressor(Compressor):
    """
    Simple compressor that returns random summary vectors (matches shape/dtype/device).
    """

    def forward(self, k: torch.Tensor, v: torch.Tensor):
        return (
            torch.randn_like(k[:, :, :1, :]).squeeze(2),
            torch.randn_like(v[:, :, :1, :]).squeeze(2),
        )


class CentroidCompressor(Compressor):
    """
    Simple centroid compressor: mean over sequence length for both k and v.
    Kept distinct from MeanCompressor for comparison clarity.
    """

    def forward(self, k: torch.Tensor, v: torch.Tensor):
        return k.mean(dim=2), v.mean(dim=2)


def generate_baseline_rollout(tokenizer, prompt: str, device: str, max_new_tokens: int = 128):
    from experiments.perplexity.threshold_sweep import generate_baseline_rollout as base_rollout
    return base_rollout(tokenizer, prompt, device, max_new_tokens)


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


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    paragraph = load_prompt("paragraphs_1")
    if isinstance(paragraph, list):
        paragraph = paragraph[0]

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    tokenizer.pad_token = tokenizer.eos_token

    # Baseline rollout
    rollout_ids, _ = generate_baseline_rollout(tokenizer, paragraph, device)
    rollout_ids = rollout_ids.to(device)
    prompt_len = tokenizer(paragraph, return_tensors="pt")["input_ids"].size(1)

    compressors = [
        ("random", RandomCompressor()),
        ("mean", MeanCompressor()),
        ("centroid", CentroidCompressor()),
    ]

    records = {name: [] for name, _ in compressors}

    for name, compressor in compressors:
        for thr in thresholds:
            set_luka_kv_params(
                default_tail_len=16,
                min_compress_chunk=16,
                max_pages=15,
                refine_threshold=thr,
                compressor=compressor,
            )
            set_luka_segmenter(DummySegmenter())

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

            stats = model.model.layers[0].self_attn.luka_kv_caches.refine_stats[0]
            denom = stats["refine"] + stats["skip"]
            summary_frac = (stats["skip"] / denom) if denom > 0 else None

            records[name].append(
                {
                    "threshold": thr,
                    "perplexity": ppl,
                    "summary_frac": summary_frac,
                    "tokens_per_sec": tps,
                    "curve": curve,
                }
            )

            del model
            torch.cuda.empty_cache()

    # Plot summary retention vs perplexity (lines + points)
    plt.figure(figsize=(7, 5))
    for name, recs in records.items():
        xs = [r["summary_frac"] if r["summary_frac"] is not None else 0.0 for r in recs]
        ys = [r["perplexity"] for r in recs]
        plt.plot(xs, ys, label=name)
        plt.scatter(xs, ys)
    plt.xlabel("Summary retention fraction (skip / (skip + refine))")
    plt.ylabel("Perplexity (tail)")
    plt.title("Perplexity vs summary retention by compressor")
    plt.legend()
    plt.tight_layout()
    out_path = "experiments/perplexity/compressor_summary_vs_ppl.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    # Log-scale version (y-axis)
    plt.figure(figsize=(7, 5))
    for name, recs in records.items():
        xs = [r["summary_frac"] if r["summary_frac"] is not None else 0.0 for r in recs]
        ys = [r["perplexity"] for r in recs]
        plt.plot(xs, ys, label=name)
        plt.scatter(xs, ys)
    plt.xlabel("Summary retention fraction (skip / (skip + refine))")
    plt.ylabel("Perplexity (tail, log scale)")
    plt.title("Perplexity vs summary retention by compressor (log y)")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    out_log_path = "experiments/perplexity/compressor_summary_vs_ppl_log.png"
    plt.savefig(out_log_path, dpi=150)
    print(f"Saved log plot to {out_log_path}")

    # CSV output
    csv_path = "experiments/perplexity/compressor_comparison.csv"
    token_axis = list(range(1, len(records["random"][0]["curve"]) + 1))
    fieldnames = ["compressor", "threshold", "perplexity", "summary_frac", "tokens_per_sec"] + [
        f"token_{i}_ppl" for i in token_axis
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, recs in records.items():
            for r in recs:
                row = {
                    "compressor": name,
                    "threshold": r["threshold"],
                    "perplexity": r["perplexity"],
                    "summary_frac": r["summary_frac"] if r["summary_frac"] is not None else "n/a",
                    "tokens_per_sec": r["tokens_per_sec"],
                }
                for i, val in zip(token_axis, r["curve"]):
                    row[f"token_{i}_ppl"] = val
                writer.writerow(row)
    print(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    run()
