"""
Profile page creation breakdown to understand where time is spent.

This helps determine if async page creation can actually help.
"""

import sys
import time
from pathlib import Path
from functools import wraps

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modeling.qwen.luka_qwen3 import load_luka_model, set_luka_kv_params
from modeling.compressor import AttentionWeightedCompressor
from artifacts.prompts.prompt_loader import load_prompt


MODEL_NAME = "Qwen/Qwen3-1.7B-Base"


class PageCreationProfiler:
    """Instruments page creation to measure time spent in each phase."""

    def __init__(self):
        self.timings = {
            "segmenter_process": [],
            "kv_slicing": [],
            "importance_extraction": [],
            "padding_stacking": [],
            "compression": [],
            "state_updates": [],
            "total": [],
        }
        self.call_count = 0

    def record(self, phase: str, elapsed_ms: float):
        self.timings[phase].append(elapsed_ms)

    def summary(self):
        print("\n" + "="*60)
        print("PAGE CREATION PROFILING RESULTS")
        print("="*60)

        for phase, times in self.timings.items():
            if times:
                avg = sum(times) / len(times)
                total = sum(times)
                print(f"{phase:25s}: {avg:8.3f} ms avg, {total:8.1f} ms total ({len(times)} calls)")

        print("-"*60)
        if self.timings["total"]:
            avg_total = sum(self.timings["total"]) / len(self.timings["total"])
            print(f"Average page creation: {avg_total:.3f} ms")

            # Breakdown percentages
            print("\nBreakdown:")
            for phase in ["segmenter_process", "kv_slicing", "importance_extraction",
                         "padding_stacking", "compression", "state_updates"]:
                if self.timings[phase]:
                    phase_total = sum(self.timings[phase])
                    all_total = sum(self.timings["total"])
                    pct = 100 * phase_total / all_total if all_total > 0 else 0
                    print(f"  {phase:25s}: {pct:5.1f}%")


# Global profiler instance
profiler = PageCreationProfiler()


def patch_try_new_pages(controller):
    """Monkey-patch try_new_pages to add detailed timing."""
    original_try_new_pages = controller.try_new_pages.__func__

    def profiled_try_new_pages(self, layer_idx: int) -> bool:
        # Only profile layer 0 to avoid noise
        if layer_idx != 0:
            return original_try_new_pages(self, layer_idx)

        # Skip async path for profiling
        if self.async_pages:
            self.async_pages = False  # Temporarily disable
            result = profiled_try_new_pages(self, layer_idx)
            self.async_pages = True
            return result

        torch.cuda.synchronize()
        total_start = time.perf_counter()

        # Throttle check
        self.seg_step_counters[layer_idx] += 1
        if self.segment_interval > 1 and (self.seg_step_counters[layer_idx] % self.segment_interval) != 0:
            return False

        cover_view = self.cover_view[layer_idx]
        if cover_view.length < getattr(self.segmenter, '_min_tokens_needed', 32):
            return False

        attn_weights, _, _ = self.attn_buffer[layer_idx].get_data()
        _, _, cover_indices, cover_is_summary = cover_view.get_valid_kv()
        if attn_weights is None:
            return False

        # Phase 1: Segmentation
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        page_ends = self.segmenter.process(attn_weights, cover_indices, cover_is_summary, layer_idx=layer_idx)
        torch.cuda.synchronize()
        profiler.record("segmenter_process", (time.perf_counter() - t0) * 1000)

        if page_ends is None:
            return False

        summary_cache = self.summary_cache[layer_idx]
        raw_cache = self.raw_cache
        _, _, _, raw_seq_start = raw_cache.get_layer(layer_idx, with_offsets=True)
        if raw_seq_start is None:
            return False

        B = raw_seq_start.shape[0]
        device = raw_seq_start.device
        k_raw, v_raw, _, _ = raw_cache.get_layer(layer_idx, with_offsets=False)
        if k_raw is None:
            return False

        H_kv = k_raw.shape[1]
        D = k_raw.shape[3]
        new_frontiers = raw_seq_start.clone()
        has_updates = False
        all_new_pages = [[] for _ in range(B)]

        # Phase 2: K/V Slicing
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        all_k_slices = []
        all_v_slices = []
        page_metadata = []
        max_page_len = 0

        for b in range(B):
            frontier = raw_seq_start[b].item()
            p_ends = page_ends[b]
            valid_mask = (p_ends >= frontier) & (p_ends != -1)
            if not valid_mask.any():
                continue
            valid_ends = p_ends[valid_mask].sort().values
            current_start = frontier

            for end_idx in valid_ends.tolist():
                end_idx = int(end_idx)
                if end_idx < current_start:
                    continue
                page_len = end_idx - current_start + 1
                max_page_len = max(max_page_len, page_len)
                k_slice = k_raw[b, :, current_start:end_idx + 1, :]
                v_slice = v_raw[b, :, current_start:end_idx + 1, :]
                all_k_slices.append(k_slice)
                all_v_slices.append(v_slice)
                page_metadata.append((b, current_start, end_idx))
                all_new_pages[b].append((current_start, end_idx))
                current_start = end_idx + 1

            if current_start > frontier:
                new_frontiers[b] = current_start
                has_updates = True

        torch.cuda.synchronize()
        profiler.record("kv_slicing", (time.perf_counter() - t0) * 1000)

        if not page_metadata:
            return False

        # Phase 3: Importance extraction
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        all_importance = []
        for i, (b, start, end) in enumerate(page_metadata):
            cover_idx_b = cover_indices[b]
            is_sum_b = cover_is_summary[b]
            importance = None
            if attn_weights is not None:
                page_mask = (cover_idx_b >= start) & (cover_idx_b <= end) & (is_sum_b == 0)
                if page_mask.any():
                    page_attn = attn_weights[b, :, :, page_mask]
                    importance = page_attn.sum(dim=1)
                    H_q = importance.shape[0]
                    if H_q != H_kv:
                        num_groups = H_q // H_kv
                        importance = importance.view(H_kv, num_groups, -1).mean(dim=1)
            all_importance.append(importance)

        torch.cuda.synchronize()
        profiler.record("importance_extraction", (time.perf_counter() - t0) * 1000)

        # Phase 4: Padding and stacking
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        N_pages = len(all_k_slices)
        padded_k = torch.zeros(N_pages, H_kv, max_page_len, D, device=device, dtype=k_raw.dtype)
        padded_v = torch.zeros(N_pages, H_kv, max_page_len, D, device=device, dtype=v_raw.dtype)
        padded_importance = None

        has_any_importance = any(imp is not None for imp in all_importance)
        if has_any_importance:
            padded_importance = torch.zeros(N_pages, H_kv, max_page_len, device=device, dtype=k_raw.dtype)

        for i, (k_s, v_s, imp) in enumerate(zip(all_k_slices, all_v_slices, all_importance)):
            plen = k_s.shape[1]
            padded_k[i, :, :plen, :] = k_s
            padded_v[i, :, :plen, :] = v_s
            if imp is not None and padded_importance is not None:
                padded_importance[i, :, :plen] = imp

        torch.cuda.synchronize()
        profiler.record("padding_stacking", (time.perf_counter() - t0) * 1000)

        # Phase 5: Compression
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        k_compressed, v_compressed = self.compressor(padded_k, padded_v, padded_importance)

        torch.cuda.synchronize()
        profiler.record("compression", (time.perf_counter() - t0) * 1000)

        # Phase 6: State updates
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        batch_pages = {}
        for i, (b, start, end) in enumerate(page_metadata):
            if b not in batch_pages:
                batch_pages[b] = []
            batch_pages[b].append((k_compressed[i:i+1], v_compressed[i:i+1], start, end))

        for b, pages in batch_pages.items():
            k_list = [p[0] for p in pages]
            v_list = [p[1] for p in pages]
            starts = [p[2] for p in pages]
            ends = [p[3] for p in pages]
            k_stack = torch.cat(k_list, dim=0).unsqueeze(0).transpose(1, 2)
            v_stack = torch.cat(v_list, dim=0).unsqueeze(0).transpose(1, 2)
            summary_cache.add_pages(
                keys=k_stack, values=v_stack,
                batch_nums=torch.tensor([b], device=device),
                page_start=torch.tensor([starts], device=device),
                page_end=torch.tensor([ends], device=device),
                page_frontier=torch.tensor([new_frontiers[b].item()], device=device)
            )

        if has_updates:
            raw_cache.raw_seq_start[layer_idx] = new_frontiers
            self.cover_view[layer_idx].update_cover_view(layer_idx, raw_cache, summary_cache)
            self.attn_buffer[layer_idx].compress_and_trim(all_new_pages, new_frontiers)

        torch.cuda.synchronize()
        profiler.record("state_updates", (time.perf_counter() - t0) * 1000)

        # Total
        torch.cuda.synchronize()
        profiler.record("total", (time.perf_counter() - total_start) * 1000)
        profiler.call_count += 1

        return True

    import types
    controller.try_new_pages = types.MethodType(profiled_try_new_pages, controller)


def generate_tokens(model, tokenizer, prompt: str, max_new_tokens: int, device: str):
    """Generate tokens and return count."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    return outputs.shape[1] - inputs["input_ids"].shape[1]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    paragraph = load_prompt("paragraphs_1")
    if isinstance(paragraph, list):
        paragraph = paragraph[0]

    print(f"Prompt length: {len(tokenizer.encode(paragraph))} tokens")
    print(f"Generating 128 tokens with pages_interval_1 config...")
    print()

    # Set up pages config
    set_luka_kv_params(
        default_tail_len=16,
        min_compress_chunk=16,
        max_pages=15,
        refine_threshold=1.0,
        compressor=AttentionWeightedCompressor(temperature=1.0),
        segmenter="dummy",
        segment_interval=1,
        create_pages_in_generation=True,
        production_mode=True,
        async_pages=False,
    )

    model = load_luka_model(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    ).to(device)
    model.eval()

    # Patch the controller
    controller = model.model.luka_kv_controller
    patch_try_new_pages(controller)

    # Generate
    num_generated = generate_tokens(model, tokenizer, paragraph, 128, device)
    print(f"Generated {num_generated} tokens")
    print(f"Page creation called {profiler.call_count} times (layer 0 only)")

    # Print results
    profiler.summary()

    # Also print comparison with attention time
    print("\n" + "="*60)
    print("CONTEXT: How does this compare to attention?")
    print("="*60)
    if profiler.timings["total"]:
        avg_page_creation = sum(profiler.timings["total"]) / len(profiler.timings["total"])
        print(f"Average page creation time: {avg_page_creation:.2f} ms")
        print(f"Typical attention time:     ~25-30 ms (from previous profiling)")
        print()
        if avg_page_creation > 30:
            print("Page creation is SLOWER than attention.")
            print("Async can only hide up to ~30ms, so benefit is limited.")
            overlap_pct = 30 / avg_page_creation * 100
            print(f"Maximum theoretical overlap: {overlap_pct:.0f}%")
        else:
            print("Page creation is FASTER than attention.")
            print("Async could potentially hide most of page creation overhead.")


if __name__ == "__main__":
    main()
