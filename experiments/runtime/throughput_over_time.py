"""
Track throughput over time during generation.

Uses manual token-by-token generation to measure per-token latency,
then outputs throughput curves showing how performance evolves.
"""

import sys
import time
import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modeling.qwen.luka_qwen3 import load_luka_model, set_luka_kv_params
from modeling.compressor import AttentionWeightedCompressor
from artifacts.prompts.prompt_loader import load_prompt


MODEL_NAME = "Qwen/Qwen3-1.7B-Base"


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_prompt(name: str = "paragraphs_1") -> str:
    paragraph = load_prompt(name)
    if isinstance(paragraph, list):
        paragraph = paragraph[0]
    return paragraph


def generate_with_timing(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    device: str = "cuda",
):
    """
    Generate tokens one-by-one and record per-token latency.

    Returns:
        generated_text: str - the full generated text
        token_times: list[float] - time in seconds for each token
        cumulative_times: list[float] - cumulative time at each token
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Prefill
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]

    generated_ids = input_ids.clone()
    token_times = []
    cumulative_times = []
    total_time = 0.0

    for _ in range(max_new_tokens):
        # Sample next token (greedy)
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        # Don't stop at EOS - always generate full max_new_tokens for fair comparison

        # Time the forward pass for this token
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                attention_mask=torch.ones(1, generated_ids.shape[1], device=device),
                past_key_values=past_key_values,
                use_cache=True,
            )

        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        token_times.append(elapsed)
        total_time += elapsed
        cumulative_times.append(total_time)

        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

    # Flush any pending async page work
    if hasattr(model, "model") and hasattr(model.model, "luka_kv_controller"):
        controller = model.model.luka_kv_controller
        if controller.async_pages:
            controller.async_page_creator.flush_all(controller)

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text, token_times, cumulative_times


def apply_config(config_name: str):
    """Apply LuKA configuration settings."""
    if config_name == "raw_attention":
        set_luka_kv_params(
            default_tail_len=16,
            min_compress_chunk=16,
            max_pages=15,
            refine_threshold=-1,
            compressor="mean",
            segmenter="dummy",
            create_pages_in_generation=False,
            production_mode=True,
        )
    elif config_name == "cover_view_only":
        set_luka_kv_params(
            default_tail_len=16,
            min_compress_chunk=16,
            max_pages=15,
            refine_threshold=1.0,  # No refinement
            compressor=AttentionWeightedCompressor(temperature=1.0),
            segmenter="dummy",
            segment_interval=16,
            create_pages_in_generation=False,  # Pages only from prefill
            production_mode=True,
        )
    elif config_name == "pages_interval_1":
        set_luka_kv_params(
            default_tail_len=16,
            min_compress_chunk=16,
            max_pages=15,
            refine_threshold=1.0,  # No refinement
            compressor=AttentionWeightedCompressor(temperature=1.0),
            segmenter="dummy",
            segment_interval=1,
            create_pages_in_generation=True,
            production_mode=True,
        )
    elif config_name == "pages_with_refinement":
        set_luka_kv_params(
            default_tail_len=16,
            min_compress_chunk=16,
            max_pages=15,
            refine_threshold=0.05,
            compressor=AttentionWeightedCompressor(temperature=1.0),
            segmenter="dummy",
            segment_interval=1,
            create_pages_in_generation=True,
            production_mode=True,
        )
    elif config_name == "pages_interval_1_async":
        set_luka_kv_params(
            default_tail_len=16,
            min_compress_chunk=16,
            max_pages=15,
            refine_threshold=1.0,  # No refinement
            compressor=AttentionWeightedCompressor(temperature=1.0),
            segmenter="dummy",
            segment_interval=1,
            create_pages_in_generation=True,
            production_mode=True,
            async_pages=True,
        )
    elif config_name == "pages_with_refinement_async":
        set_luka_kv_params(
            default_tail_len=16,
            min_compress_chunk=16,
            max_pages=15,
            refine_threshold=0.05,
            compressor=AttentionWeightedCompressor(temperature=1.0),
            segmenter="dummy",
            segment_interval=1,
            create_pages_in_generation=True,
            production_mode=True,
            async_pages=True,
        )
    else:
        raise ValueError(f"Unknown config: {config_name}")


def load_fresh_model(device: str):
    """Load a fresh model instance."""
    model = load_luka_model(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    ).to(device)
    model.eval()
    return model


def run_config(config_name: str, prompt: str, tokenizer, device: str, max_new_tokens: int = 128):
    """Run a single configuration and return timing data."""
    apply_config(config_name)

    # Warmup run with a fresh model (then discard)
    warmup_model = load_fresh_model(device)
    _ = generate_with_timing(warmup_model, tokenizer, prompt, max_new_tokens=10, device=device)
    del warmup_model
    torch.cuda.empty_cache()

    # Actual timed run with a fresh model
    apply_config(config_name)  # Re-apply to ensure clean state
    model = load_fresh_model(device)

    text, token_times, cumulative_times = generate_with_timing(
        model, tokenizer, prompt, max_new_tokens=max_new_tokens, device=device
    )

    del model
    torch.cuda.empty_cache()

    return {
        "text": text,
        "token_times": token_times,
        "cumulative_times": cumulative_times,
    }


def compute_throughput_curves(token_times: list, window_size: int = 10):
    """
    Compute various throughput metrics from per-token times.

    Returns:
        instantaneous: list[float] - 1/token_time for each token (tok/s)
        rolling_avg: list[float] - rolling average throughput
        cumulative_avg: list[float] - cumulative average throughput
    """
    n = len(token_times)

    # Instantaneous throughput
    instantaneous = [1.0 / t if t > 0 else 0 for t in token_times]

    # Rolling average (window_size tokens)
    rolling_avg = []
    for i in range(n):
        start_idx = max(0, i - window_size + 1)
        window = token_times[start_idx:i + 1]
        avg_time = sum(window) / len(window)
        rolling_avg.append(1.0 / avg_time if avg_time > 0 else 0)

    # Cumulative average
    cumulative_avg = []
    total = 0.0
    for i, t in enumerate(token_times):
        total += t
        avg_time = total / (i + 1)
        cumulative_avg.append(1.0 / avg_time if avg_time > 0 else 0)

    return instantaneous, rolling_avg, cumulative_avg


def main():
    parser = argparse.ArgumentParser(description="Track throughput over time during generation")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--prompt", type=str, default="paragraphs_1", help="Prompt name")
    parser.add_argument("--window", type=int, default=10, help="Rolling average window size")
    parser.add_argument("--output", type=str, default=None, help="Output plot path")
    args = parser.parse_args()

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    prompt = get_prompt(args.prompt)

    configs = [
        ("raw_attention", "Raw Attention", "black"),
        ("cover_view_only", "Cover View Only", "tab:blue"),
        ("pages_interval_1", "Pages (sync)", "tab:orange"),
        ("pages_interval_1_async", "Pages (async)", "tab:red"),
        ("pages_with_refinement", "Refinement (sync)", "tab:green"),
        ("pages_with_refinement_async", "Refinement (async)", "tab:purple"),
    ]

    results = {}

    print(f"Running throughput over time analysis...")
    print(f"Prompt length: {len(tokenizer.encode(prompt))} tokens")
    print(f"Max new tokens: {args.max_tokens}")
    print()

    for config_name, label, color in configs:
        print(f"Running {label}...")
        data = run_config(config_name, prompt, tokenizer, device, args.max_tokens)

        instantaneous, rolling, cumulative = compute_throughput_curves(
            data["token_times"], window_size=args.window
        )

        results[config_name] = {
            "label": label,
            "color": color,
            "token_times": data["token_times"],
            "instantaneous": instantaneous,
            "rolling": rolling,
            "cumulative": cumulative,
        }

        # Print summary
        avg_tps = len(data["token_times"]) / sum(data["token_times"]) if data["token_times"] else 0
        print(f"  Generated {len(data['token_times'])} tokens")
        print(f"  Avg throughput: {avg_tps:.1f} tok/s")
        print(f"  Min latency: {min(data['token_times'])*1000:.2f} ms")
        print(f"  Max latency: {max(data['token_times'])*1000:.2f} ms")
        print()

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot 1: Rolling average throughput
    ax = axes[0, 0]
    for config_name, _, _ in configs:
        r = results[config_name]
        ax.plot(r["rolling"], label=r["label"], color=r["color"], linewidth=1.5)
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title(f"Rolling Average Throughput (window={args.window})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Cumulative average throughput
    ax = axes[0, 1]
    for config_name, _, _ in configs:
        r = results[config_name]
        ax.plot(r["cumulative"], label=r["label"], color=r["color"], linewidth=1.5)
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("Cumulative Average Throughput")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Per-token latency
    ax = axes[1, 0]
    for config_name, _, _ in configs:
        r = results[config_name]
        latency_ms = [t * 1000 for t in r["token_times"]]
        ax.plot(latency_ms, label=r["label"], color=r["color"], linewidth=1, alpha=0.7)
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-Token Latency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Latency distribution (box plot)
    ax = axes[1, 1]
    data_for_box = []
    labels_for_box = []
    colors_for_box = []
    for config_name, _, _ in configs:
        r = results[config_name]
        data_for_box.append([t * 1000 for t in r["token_times"]])
        labels_for_box.append(r["label"])
        colors_for_box.append(r["color"])

    bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors_for_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Distribution")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    output_path = args.output or str(Path(__file__).parent / "throughput_over_time.png")
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")

    # Also save CSV data
    csv_path = Path(output_path).with_suffix(".csv")
    with open(csv_path, "w") as f:
        f.write("config,token_idx,latency_ms,rolling_tps,cumulative_tps\n")
        for config_name, _, _ in configs:
            r = results[config_name]
            for i, (lat, roll, cum) in enumerate(zip(r["token_times"], r["rolling"], r["cumulative"])):
                f.write(f"{config_name},{i},{lat*1000:.4f},{roll:.2f},{cum:.2f}\n")
    print(f"Saved data to {csv_path}")


if __name__ == "__main__":
    main()
