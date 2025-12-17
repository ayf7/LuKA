"""
Profile LuKA to identify exact bottlenecks.

Run: python experiments/runtime/profile_overhead.py
"""

import time
import torch
from contextlib import contextmanager
from collections import defaultdict

from transformers import AutoTokenizer
from modeling.qwen.luka_qwen3 import load_luka_model, set_luka_kv_params


class Timer:
    """Simple timer for profiling."""

    def __init__(self):
        self.times = defaultdict(list)
        self.enabled = True

    @contextmanager
    def track(self, name: str):
        if not self.enabled:
            yield
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.times[name].append(time.perf_counter() - start)

    def report(self):
        print("\n" + "=" * 70)
        print("TIMING BREAKDOWN")
        print("=" * 70)
        total = sum(sum(v) for v in self.times.values())
        for name, times in sorted(self.times.items(), key=lambda x: -sum(x[1])):
            total_time = sum(times)
            avg_time = total_time / len(times) * 1000  # ms
            pct = total_time / total * 100 if total > 0 else 0
            print(f"{name:<40} {avg_time:>8.3f}ms  ({pct:>5.1f}%)  n={len(times)}")
        print("=" * 70)


# Global timer
TIMER = Timer()


def patch_for_profiling():
    """Monkey-patch LuKA methods to add timing."""
    from modeling import kv_cache

    # Patch top_down_attention
    original_top_down = kv_cache.LukaKVController.top_down_attention
    def timed_top_down(self, *args, **kwargs):
        with TIMER.track("top_down_attention"):
            return original_top_down(self, *args, **kwargs)
    kv_cache.LukaKVController.top_down_attention = timed_top_down

    # Patch try_new_pages
    original_try_pages = kv_cache.LukaKVController.try_new_pages
    def timed_try_pages(self, *args, **kwargs):
        with TIMER.track("try_new_pages"):
            return original_try_pages(self, *args, **kwargs)
    kv_cache.LukaKVController.try_new_pages = timed_try_pages

    # Patch update (cover view update during decode)
    original_update = kv_cache.LukaKVController.update
    def timed_update(self, *args, **kwargs):
        with TIMER.track("controller.update"):
            return original_update(self, *args, **kwargs)
    kv_cache.LukaKVController.update = timed_update

    # Patch CoverView.update
    original_cv_update = kv_cache.CoverView.update
    def timed_cv_update(self, *args, **kwargs):
        with TIMER.track("cover_view.update"):
            return original_cv_update(self, *args, **kwargs)
    kv_cache.CoverView.update = timed_cv_update

    # Patch CoverView.update_cover_view (rebuild after pages)
    original_cv_rebuild = kv_cache.CoverView.update_cover_view
    def timed_cv_rebuild(self, *args, **kwargs):
        with TIMER.track("cover_view.rebuild"):
            return original_cv_rebuild(self, *args, **kwargs)
    kv_cache.CoverView.update_cover_view = timed_cv_rebuild

    # Patch AttentionScoreBuffer.push
    original_push = kv_cache.AttentionScoreBuffer.push
    def timed_push(self, *args, **kwargs):
        with TIMER.track("attn_buffer.push"):
            return original_push(self, *args, **kwargs)
    kv_cache.AttentionScoreBuffer.push = timed_push

    # Patch AttentionScoreBuffer.compress_and_trim
    original_compress = kv_cache.AttentionScoreBuffer.compress_and_trim
    def timed_compress(self, *args, **kwargs):
        with TIMER.track("attn_buffer.compress_trim"):
            return original_compress(self, *args, **kwargs)
    kv_cache.AttentionScoreBuffer.compress_and_trim = timed_compress

    # Patch segmenter
    from modeling import segmenter
    original_seg_process = segmenter.DummySegmenter.process
    def timed_seg_process(self, *args, **kwargs):
        with TIMER.track("segmenter.process"):
            return original_seg_process(self, *args, **kwargs)
    segmenter.DummySegmenter.process = timed_seg_process


def run_profile(
    model_name: str = "Qwen/Qwen3-1.7B-Base",
    max_new_tokens: int = 64,
    config_name: str = "pages_interval_1",
):
    """Run profiling for a specific config."""

    # Configure based on config name
    configs = {
        "raw_attention": dict(refine_threshold=-1.0, create_pages_in_generation=False),
        "cover_view_only": dict(refine_threshold=1.0, create_pages_in_generation=False),
        "pages_interval_1": dict(refine_threshold=1.0, create_pages_in_generation=True, segment_interval=1),
        "with_refinement": dict(refine_threshold=0.05, create_pages_in_generation=True, segment_interval=1),
    }

    if config_name not in configs:
        print(f"Unknown config: {config_name}")
        print(f"Available: {list(configs.keys())}")
        return

    cfg = configs[config_name]

    print(f"Profiling config: {config_name}")
    print(f"Settings: {cfg}")

    # Apply patches BEFORE loading model
    patch_for_profiling()

    set_luka_kv_params(
        production_mode=True,
        compressor="mean",
        segmenter="dummy",
        **cfg
    )

    model = load_luka_model(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = (
        "The history of artificial intelligence began in antiquity, with myths, stories and rumors of "
        "artificial beings endowed with intelligence or consciousness by master craftsmen."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Warmup
    print("Warmup...")
    TIMER.enabled = False
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=16, do_sample=False, use_cache=True)

    # Reset state
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

    # Timed run
    print(f"Generating {max_new_tokens} tokens...")
    TIMER.enabled = True
    TIMER.times.clear()

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    num_tokens = output.shape[1] - inputs["input_ids"].shape[1]
    print(f"\nGenerated {num_tokens} tokens in {elapsed:.3f}s = {num_tokens/elapsed:.1f} tok/s")

    TIMER.report()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--config", default="cover_view_only",
                        choices=["raw_attention", "cover_view_only", "pages_interval_1", "with_refinement"])
    args = parser.parse_args()

    run_profile(model_name=args.model, max_new_tokens=args.tokens, config_name=args.config)
