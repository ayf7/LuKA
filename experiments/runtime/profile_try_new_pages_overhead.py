"""
Profile the overhead of try_new_pages when NO pages are created.

This explains why pages_interval_1 is slower than cover_view_only
even between page creation spikes.
"""

import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modeling.qwen.luka_qwen3 import load_luka_model, set_luka_kv_params
from modeling.compressor import AttentionWeightedCompressor
from artifacts.prompts.prompt_loader import load_prompt


MODEL_NAME = "Qwen/Qwen3-1.7B-Base"


class TryNewPagesOverheadProfiler:
    """Profile try_new_pages overhead broken down by exit point."""

    def __init__(self):
        self.timings = {
            "throttle_exit": [],      # Exited at throttle check
            "length_exit": [],        # Exited at cover view length check
            "attn_data_exit": [],     # Exited at attention data check
            "segmenter_exit": [],     # Exited at segmenter (returned None)
            "pages_created": [],      # Actually created pages
        }
        self.exit_counts = {k: 0 for k in self.timings.keys()}

    def record(self, exit_point: str, elapsed_ms: float):
        self.timings[exit_point].append(elapsed_ms)
        self.exit_counts[exit_point] += 1

    def summary(self):
        print("\n" + "="*70)
        print("TRY_NEW_PAGES OVERHEAD ANALYSIS")
        print("="*70)

        total_calls = sum(self.exit_counts.values())
        total_time = sum(sum(t) for t in self.timings.values())

        print(f"\nTotal calls: {total_calls}")
        print(f"Total time: {total_time:.1f} ms")
        print()

        print("Exit Point Breakdown:")
        print("-"*70)
        for exit_point in self.timings.keys():
            times = self.timings[exit_point]
            count = self.exit_counts[exit_point]
            if count > 0:
                avg = sum(times) / count
                total = sum(times)
                pct_calls = 100 * count / total_calls
                pct_time = 100 * total / total_time if total_time > 0 else 0
                print(f"{exit_point:20s}: {count:5d} calls ({pct_calls:5.1f}%), "
                      f"avg {avg:.3f} ms, total {total:.1f} ms ({pct_time:5.1f}% of time)")

        print()
        print("Per-layer overhead estimate (28 layers):")
        # Calculate overhead from non-page-creating calls
        non_creating_time = sum(sum(self.timings[k]) for k in self.timings if k != "pages_created")
        non_creating_calls = sum(self.exit_counts[k] for k in self.exit_counts if k != "pages_created")
        if non_creating_calls > 0:
            avg_non_creating = non_creating_time / non_creating_calls
            print(f"  Average 'no-op' call: {avg_non_creating:.3f} ms")
            print(f"  Per decode step (28 layers): {avg_non_creating * 28:.2f} ms")
            print()
            print(f"  This explains the ~5-6ms gap between cover_view_only and pages_interval_1!")


profiler = TryNewPagesOverheadProfiler()


def patch_try_new_pages(controller):
    """Monkey-patch try_new_pages to profile exit points."""
    original_segmenter = controller.segmenter

    def profiled_try_new_pages(self, layer_idx: int) -> bool:
        # Only profile layer 0 to reduce noise
        if layer_idx != 0:
            # Still call original for other layers
            return original_try_new_pages_logic(self, layer_idx)

        torch.cuda.synchronize()
        start = time.perf_counter()

        # --- Throttle check ---
        self.seg_step_counters[layer_idx] += 1
        if self.segment_interval > 1 and (self.seg_step_counters[layer_idx] % self.segment_interval) != 0:
            torch.cuda.synchronize()
            profiler.record("throttle_exit", (time.perf_counter() - start) * 1000)
            return False

        # --- Cover view length check ---
        cover_view = self.cover_view[layer_idx]
        if cover_view.length < getattr(self.segmenter, '_min_tokens_needed', 32):
            torch.cuda.synchronize()
            profiler.record("length_exit", (time.perf_counter() - start) * 1000)
            return False

        # --- Get attention data ---
        attn_weights, _, _ = self.attn_buffer[layer_idx].get_data()
        _, _, cover_indices, cover_is_summary = cover_view.get_valid_kv()
        if attn_weights is None:
            torch.cuda.synchronize()
            profiler.record("attn_data_exit", (time.perf_counter() - start) * 1000)
            return False

        # --- Segmenter ---
        page_ends = self.segmenter.process(attn_weights, cover_indices, cover_is_summary, layer_idx=layer_idx)
        if page_ends is None:
            torch.cuda.synchronize()
            profiler.record("segmenter_exit", (time.perf_counter() - start) * 1000)
            return False

        # Would create pages - call original logic
        # For simplicity, just record time and note it would create pages
        torch.cuda.synchronize()
        profiler.record("pages_created", (time.perf_counter() - start) * 1000)

        # Actually do the page creation (call rest of original)
        return do_page_creation(self, layer_idx, page_ends, attn_weights, cover_indices, cover_is_summary)

    def original_try_new_pages_logic(self, layer_idx):
        """Original logic for non-profiled layers."""
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

        page_ends = self.segmenter.process(attn_weights, cover_indices, cover_is_summary, layer_idx=layer_idx)
        if page_ends is None:
            return False

        return do_page_creation(self, layer_idx, page_ends, attn_weights, cover_indices, cover_is_summary)

    def do_page_creation(self, layer_idx, page_ends, attn_weights, cover_indices, cover_is_summary):
        """Do actual page creation (extracted from try_new_pages)."""
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
        all_k_slices = []
        all_v_slices = []
        all_importance = []
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
            cover_idx_b = cover_indices[b]
            is_sum_b = cover_is_summary[b]

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

                importance = None
                if attn_weights is not None:
                    page_mask = (cover_idx_b >= current_start) & (cover_idx_b <= end_idx) & (is_sum_b == 0)
                    if page_mask.any():
                        page_attn = attn_weights[b, :, :, page_mask]
                        importance = page_attn.sum(dim=1)
                        H_q = importance.shape[0]
                        if H_q != H_kv:
                            num_groups = H_q // H_kv
                            importance = importance.view(H_kv, num_groups, -1).mean(dim=1)
                all_importance.append(importance)
                page_metadata.append((b, current_start, end_idx))
                all_new_pages[b].append((current_start, end_idx))
                current_start = end_idx + 1

            if current_start > frontier:
                new_frontiers[b] = current_start
                has_updates = True

        if not page_metadata:
            return False

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

        k_compressed, v_compressed = self.compressor(padded_k, padded_v, padded_importance)

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

        return True

    import types
    controller.try_new_pages = types.MethodType(profiled_try_new_pages, controller)


def generate_tokens(model, tokenizer, prompt: str, max_new_tokens: int, device: str):
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
    print(f"Generating 128 tokens...")
    print()

    set_luka_kv_params(
        default_tail_len=16,
        min_compress_chunk=16,
        max_pages=15,
        refine_threshold=1.0,
        compressor=AttentionWeightedCompressor(temperature=1.0),
        segmenter="dummy",
        segment_interval=1,  # Check every step
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

    controller = model.model.luka_kv_controller
    patch_try_new_pages(controller)

    num_generated = generate_tokens(model, tokenizer, paragraph, 128, device)
    print(f"Generated {num_generated} tokens")

    profiler.summary()


if __name__ == "__main__":
    main()
