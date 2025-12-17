"""
Compare per-token perplexity curves across compressors at a fixed threshold.
Plots baseline (raw) vs random, mean, and trained compressors.
Run: python experiments/perplexity/fixed_threshold_comparison.py
"""

import time
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from modeling.compressor import Compressor, MeanCompressor, EncoderCompressor
from modeling.qwen.luka_qwen3 import (
    load_luka_model,
    set_luka_kv_params,
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


def generate_baseline_rollout(tokenizer, prompt: str, device: str, max_new_tokens: int = 128):
    """
    Generate a greedy rollout using LuKA with refine_threshold=0.0 (raw attention path).
    Returns token ids (including prompt) and decoded text.
    """
    set_luka_kv_params(
        default_tail_len=16,
        min_compress_chunk=16,
        max_pages=15,
        refine_threshold=0.0,  # forces full raw attention in top-down
        segment_interval=1,  # Allow page creation on every forward pass
        create_pages_in_generation=False,  # only create pages during prefill, not decode
        compressor=None,
        segmenter="dummy",
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


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    thresholds = [1e-4]  # List of thresholds to compare

    # Load prompt
    paragraph = load_prompt("paragraphs_1")
    if isinstance(paragraph, list):
        paragraph = paragraph[0]

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    tokenizer.pad_token = tokenizer.eos_token

    # 1) Baseline rollout (generate once, reuse for all thresholds)
    print("Generating baseline rollout...")
    rollout_ids, rollout_text = generate_baseline_rollout(tokenizer, paragraph, device)
    rollout_ids = rollout_ids.to(device)
    prompt_len = tokenizer(paragraph, return_tensors="pt")["input_ids"].size(1)

    print(f"Rollout length: {rollout_ids.shape[1]} tokens (prompt: {prompt_len}, generated: {rollout_ids.shape[1] - prompt_len})")

    # 2) Baseline perplexity (raw attention, no compression)
    print("\nEvaluating baseline (raw attention)...")
    set_luka_kv_params(
        default_tail_len=16,
        min_compress_chunk=16,
        max_pages=15,
        refine_threshold=0.0,  # forces full raw attention
        segment_interval=1,
        compressor=None,
        segmenter="dummy",
    )
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
    print(f"  Perplexity: {base_ppl:.3f}, Tokens/sec: {base_tps:.2f}")
    del baseline_model
    torch.cuda.empty_cache()

    # 3) Loop over thresholds
    for threshold in thresholds:
        print(f"\n{'=' * 80}")
        print(f"Processing threshold={threshold}")
        print(f"{'=' * 80}")

        # Load compressors (need fresh instances for each threshold)
        trained_compressor = EncoderCompressor(checkpoint_path="train_1/best.pt")
        compressors = [
            ("random", RandomCompressor()),
            ("mean", MeanCompressor()),
            ("trained", trained_compressor),
        ]

        results = {}
        for name, compressor in compressors:
            print(f"\nEvaluating {name} compressor at threshold={threshold}...")
            set_luka_kv_params(
                default_tail_len=16,
                min_compress_chunk=16,
                max_pages=15,
                refine_threshold=threshold,
                segment_interval=1,
                compressor=compressor,
                segmenter="dummy",
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

            # Get refinement stats
            stats = model.model.luka_kv_controller.attn_buffer[0].get_stats()
            summary_frac = 1.0 - stats["refinement_rate"] if stats["total_summaries_seen"] > 0 else None

            results[name] = {
                "perplexity": ppl,
                "curve": curve,
                "tokens_per_sec": tps,
                "summary_frac": summary_frac,
            }

            frac_str = f"{summary_frac:.3f}" if summary_frac is not None else "n/a"
            print(f"  Perplexity: {ppl:.3f}, Summary retention: {frac_str}, Tokens/sec: {tps:.2f}")

            del model
            torch.cuda.empty_cache()

        # 4) Plot per-token perplexity curves for this threshold
        plt.figure(figsize=(10, 6))
        token_axis = list(range(1, len(base_curve) + 1))

        # Plot baseline
        plt.plot(token_axis, base_curve, label="baseline (raw)", linestyle="--", color="black", linewidth=2)

        # Plot compressors
        colors = {"random": "tab:red", "mean": "tab:blue", "trained": "tab:green"}
        for name in ["random", "mean", "trained"]:
            r = results[name]
            plt.plot(token_axis, r["curve"], label=f"{name} (thr={threshold})", color=colors[name])

        plt.xlabel("Token position (in generated tail)")
        plt.ylabel("Cumulative perplexity")
        plt.title(f"Per-token perplexity curves (threshold={threshold})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = f"experiments/perplexity/fixed_threshold_curves_{threshold}.png"
        plt.savefig(out_path, dpi=150)
        print(f"\nSaved plot to {out_path}")
        plt.close()

        # 5) Log-scale version
        plt.figure(figsize=(10, 6))
        plt.plot(token_axis, base_curve, label="baseline (raw)", linestyle="--", color="black", linewidth=2)
        for name in ["random", "mean", "trained"]:
            r = results[name]
            plt.plot(token_axis, r["curve"], label=f"{name} (thr={threshold})", color=colors[name])
        plt.xlabel("Token position (in generated tail)")
        plt.ylabel("Cumulative perplexity (log scale)")
        plt.title(f"Per-token perplexity curves (threshold={threshold}, log y)")
        plt.yscale("log")
        plt.legend()
        plt.grid(True, alpha=0.3, which="both")
        plt.tight_layout()
        out_log_path = f"experiments/perplexity/fixed_threshold_curves_{threshold}_log.png"
        plt.savefig(out_log_path, dpi=150)
        print(f"Saved log plot to {out_log_path}")
        plt.close()

        # 6) Print summary table for this threshold
        print("\n" + "=" * 80)
        print(f"Summary (threshold={threshold}):")
        print("=" * 80)
        print(f"{'Compressor':<15} {'Perplexity':>12} {'Summary Frac':>15} {'Tokens/sec':>12}")
        print("-" * 80)
        print(f"{'baseline':<15} {base_ppl:>12.3f} {'0.000':>15} {base_tps:>12.2f}")
        for name in ["random", "mean", "trained"]:
            r = results[name]
            frac_str = f"{r['summary_frac']:.3f}" if r['summary_frac'] is not None else "n/a"
            print(f"{name:<15} {r['perplexity']:>12.3f} {frac_str:>15} {r['tokens_per_sec']:>12.2f}")
        print("=" * 80)


if __name__ == "__main__":
    run()
