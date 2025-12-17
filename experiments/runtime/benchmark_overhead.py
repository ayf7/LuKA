"""
Benchmark to isolate LuKA overhead sources.

Tests different configurations to determine where time is spent:
  1. Raw attention (baseline) - threshold < 0
  2. Cover view only (no page creation) - create_pages_in_generation=False
  3. With page creation, varying segment_interval
  4. With/without refinement

Run: python experiments/runtime/benchmark_overhead.py
"""

import csv
import time
import torch
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from transformers import AutoTokenizer
from modeling.qwen.luka_qwen3 import load_luka_model, set_luka_kv_params


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    name: str
    refine_threshold: float
    create_pages_in_generation: bool
    segment_interval: int
    compressor: str = "mean"
    description: str = ""


# Configurations to test (ordered to isolate overhead sources)
BENCHMARK_CONFIGS = [
    BenchmarkConfig(
        name="raw_attention",
        refine_threshold=-1.0,
        create_pages_in_generation=False,
        segment_interval=1,
        description="Baseline: raw attention, no LuKA overhead",
    ),
    BenchmarkConfig(
        name="cover_view_only",
        refine_threshold=1.0,  # High threshold = no refinement triggers
        create_pages_in_generation=False,
        segment_interval=1,
        description="Cover view maintained but no pages created, no refinement",
    ),
    BenchmarkConfig(
        name="pages_interval_1",
        refine_threshold=1.0,
        create_pages_in_generation=True,
        segment_interval=1,
        description="Pages created every step, no refinement",
    ),
    BenchmarkConfig(
        name="pages_interval_4",
        refine_threshold=1.0,
        create_pages_in_generation=True,
        segment_interval=4,
        description="Pages created every 4 steps, no refinement",
    ),
    BenchmarkConfig(
        name="pages_interval_8",
        refine_threshold=1.0,
        create_pages_in_generation=True,
        segment_interval=8,
        description="Pages created every 8 steps, no refinement",
    ),
    BenchmarkConfig(
        name="pages_with_refinement_0.2",
        refine_threshold=0.2,
        create_pages_in_generation=True,
        segment_interval=1,
        description="Full LuKA with refinement threshold 0.2",
    ),
    BenchmarkConfig(
        name="pages_with_refinement_0.05",
        refine_threshold=0.05,
        create_pages_in_generation=True,
        segment_interval=1,
        description="Full LuKA with refinement threshold 0.05 (more refinement)",
    ),
    BenchmarkConfig(
        name="attn_weighted_no_refine",
        refine_threshold=1.0,
        create_pages_in_generation=True,
        segment_interval=1,
        compressor="attention_weighted",
        description="AttentionWeightedCompressor, no refinement",
    ),
    BenchmarkConfig(
        name="attn_weighted_with_refine",
        refine_threshold=0.05,
        create_pages_in_generation=True,
        segment_interval=1,
        compressor="attention_weighted",
        description="AttentionWeightedCompressor with refinement",
    ),
]


def clear_memory():
    """Clear GPU memory between runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def reset_luka_state(model):
    """Reset LuKA controller state between runs."""
    if hasattr(model.model, 'luka_kv_controller'):
        controller = model.model.luka_kv_controller
        controller.raw_cache.cache = None
        for i in range(controller.num_layers):
            controller.raw_cache.seq_start[i] = None
            controller.raw_cache.raw_seq_start[i] = None
            controller.summary_cache[i] = type(controller.summary_cache[i])(controller.summary_cache[i].config)
            controller.cover_view[i] = type(controller.cover_view[i])()
            controller.attn_buffer[i] = type(controller.attn_buffer[i])()
            controller.seg_step_counters[i] = 0


def benchmark_config(
    config: BenchmarkConfig,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    device: str,
    num_warmup: int = 2,
    num_runs: int = 5,
) -> dict:
    """Benchmark a single configuration."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {config.name}")
    print(f"  {config.description}")
    print(f"{'='*60}")

    # Configure LuKA
    set_luka_kv_params(
        default_tail_len=16,
        min_compress_chunk=16,
        max_pages=15,
        refine_threshold=config.refine_threshold,
        create_pages_in_generation=config.create_pages_in_generation,
        segment_interval=config.segment_interval,
        compressor=config.compressor,
        segmenter="dummy",
        production_mode=True,
        print_stats_after_generate=False,
    )

    # Load model
    model = load_luka_model(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    # Warmup runs
    print(f"  Warmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        reset_luka_state(model)

    # Timed runs
    print(f"  Benchmarking ({num_runs} runs)...")
    times = []
    tokens_generated = []

    for run_idx in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        num_tokens = output.shape[1] - prompt_len

        times.append(elapsed)
        tokens_generated.append(num_tokens)
        reset_luka_state(model)

        print(f"    Run {run_idx + 1}: {num_tokens} tokens in {elapsed:.3f}s = {num_tokens/elapsed:.1f} tok/s")

    # Calculate statistics
    avg_time = sum(times) / len(times)
    avg_tokens = sum(tokens_generated) / len(tokens_generated)
    avg_throughput = avg_tokens / avg_time
    best_throughput = avg_tokens / min(times)

    del model
    clear_memory()

    result = {
        "name": config.name,
        "description": config.description,
        "refine_threshold": config.refine_threshold,
        "create_pages": config.create_pages_in_generation,
        "segment_interval": config.segment_interval,
        "compressor": config.compressor,
        "avg_time_s": avg_time,
        "min_time_s": min(times),
        "max_time_s": max(times),
        "avg_tokens": avg_tokens,
        "avg_throughput_tok_s": avg_throughput,
        "best_throughput_tok_s": best_throughput,
    }

    print(f"\n  Results: {avg_throughput:.1f} tok/s (best: {best_throughput:.1f})")
    return result


def run_benchmark(
    model_name: str = "Qwen/Qwen3-1.7B-Base",
    prompt: Optional[str] = None,
    max_new_tokens: int = 128,
    device: str = "cuda",
    output_csv: Optional[str] = None,
    num_warmup: int = 2,
    num_runs: int = 5,
    configs: Optional[list] = None,
):
    """Run the full benchmark suite."""
    if prompt is None:
        prompt = (
            "The history of artificial intelligence began in antiquity, with myths, stories and rumors of "
            "artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of "
            "modern AI were planted by philosophers who attempted to describe the process of human thinking "
            "as the mechanical manipulation of symbols. This work culminated in the invention of the "
            "programmable digital computer in the 1940s, a machine based on the abstract essence of "
            "mathematical reasoning. This device and the ideas behind it inspired a handful of scientists "
            "to begin seriously discussing the possibility of building an electronic brain."
        )

    if configs is None:
        configs = BENCHMARK_CONFIGS

    print("=" * 70)
    print("LuKA Runtime Overhead Benchmark")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Device: {device}")
    print(f"Configs to test: {len(configs)}")
    print("=" * 70)

    results = []
    for config in configs:
        try:
            result = benchmark_config(
                config=config,
                model_name=model_name,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                device=device,
                num_warmup=num_warmup,
                num_runs=num_runs,
            )
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        clear_memory()

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<30} {'Throughput':<15} {'vs Raw':<10}")
    print("-" * 70)

    baseline = None
    for r in results:
        tp = r["avg_throughput_tok_s"]
        if r["name"] == "raw_attention":
            baseline = tp
            print(f"{r['name']:<30} {tp:<15.1f} {'baseline':<10}")
        elif baseline:
            pct = (tp / baseline) * 100
            print(f"{r['name']:<30} {tp:<15.1f} {pct:.1f}%")
        else:
            print(f"{r['name']:<30} {tp:<15.1f}")

    # Save CSV
    if output_csv and results:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved to: {output_csv}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="experiments/runtime/benchmark_results.csv")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Test fewer configs")
    args = parser.parse_args()

    configs = BENCHMARK_CONFIGS
    if args.quick:
        configs = [c for c in BENCHMARK_CONFIGS if c.name in [
            "raw_attention", "cover_view_only", "pages_interval_1", "pages_with_refinement_0.05"
        ]]
        args.runs = 3

    run_benchmark(
        model_name=args.model,
        max_new_tokens=args.max_tokens,
        device=args.device,
        output_csv=args.output,
        num_warmup=args.warmup,
        num_runs=args.runs,
        configs=configs,
    )
