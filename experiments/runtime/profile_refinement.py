"""
Profile refinement overhead to understand why it's so slow.

Refinement (threshold=0.05) is ~20 tok/s vs ~40 tok/s for raw attention.
This script profiles where the time goes.
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


class RefinementProfiler:
    """Profile refinement overhead in top_down_attention."""

    def __init__(self):
        self.timings = {
            "cover_attention": [],      # Base attention over cover view
            "refinement_check": [],     # Checking which summaries need refinement
            "refinement_compute": [],   # Actually computing refined attention
            "total_attention": [],      # Total top_down_attention time
        }
        self.refinement_stats = {
            "total_calls": 0,
            "calls_with_refinement": 0,
            "total_summaries_checked": 0,
            "total_refined": 0,
        }

    def record(self, phase: str, elapsed_ms: float):
        self.timings[phase].append(elapsed_ms)

    def record_refinement(self, num_summaries: int, num_refined: int):
        self.refinement_stats["total_calls"] += 1
        self.refinement_stats["total_summaries_checked"] += num_summaries
        self.refinement_stats["total_refined"] += num_refined
        if num_refined > 0:
            self.refinement_stats["calls_with_refinement"] += 1

    def summary(self):
        print("\n" + "="*70)
        print("REFINEMENT PROFILING RESULTS")
        print("="*70)

        print("\nTiming Breakdown (layer 0 only):")
        print("-"*70)
        for phase, times in self.timings.items():
            if times:
                avg = sum(times) / len(times)
                total = sum(times)
                print(f"{phase:25s}: {avg:8.3f} ms avg, {total:8.1f} ms total ({len(times)} calls)")

        print("\nRefinement Statistics:")
        print("-"*70)
        stats = self.refinement_stats
        print(f"Total attention calls: {stats['total_calls']}")
        print(f"Calls with refinement: {stats['calls_with_refinement']} "
              f"({100*stats['calls_with_refinement']/max(1,stats['total_calls']):.1f}%)")
        print(f"Total summaries checked: {stats['total_summaries_checked']}")
        print(f"Total summaries refined: {stats['total_refined']} "
              f"({100*stats['total_refined']/max(1,stats['total_summaries_checked']):.1f}%)")

        if stats['calls_with_refinement'] > 0:
            avg_refined_per_call = stats['total_refined'] / stats['calls_with_refinement']
            print(f"Avg summaries refined per refinement call: {avg_refined_per_call:.1f}")

        # Calculate overhead
        if self.timings["total_attention"] and self.timings["cover_attention"]:
            avg_total = sum(self.timings["total_attention"]) / len(self.timings["total_attention"])
            avg_cover = sum(self.timings["cover_attention"]) / len(self.timings["cover_attention"])
            overhead = avg_total - avg_cover
            print(f"\nRefinement overhead per attention call: {overhead:.3f} ms")
            print(f"28 layers × {overhead:.3f} ms = {28 * overhead:.1f} ms per decode step")


profiler = RefinementProfiler()


def patch_top_down_attention(controller):
    """Monkey-patch top_down_attention to profile refinement."""
    original_top_down = controller.top_down_attention

    def profiled_top_down_attention(
        self,
        layer_idx: int,
        query_states,
        scaling: float,
        num_kv_groups: int,
        attention_mask=None,
        sliding_window=None,
        threshold: float = 0.2,
    ):
        # Only profile layer 0
        if layer_idx != 0:
            return original_top_down(
                layer_idx, query_states, scaling, num_kv_groups,
                attention_mask, sliding_window, threshold
            )

        torch.cuda.synchronize()
        total_start = time.perf_counter()

        # Get cover view data
        cover_k, cover_v, cover_indices, cover_is_summary = self.cover_view[layer_idx].get_valid_kv()

        if cover_k is None:
            return original_top_down(
                layer_idx, query_states, scaling, num_kv_groups,
                attention_mask, sliding_window, threshold
            )

        B, H_q, L_q, D = query_states.shape
        T_cover = cover_k.shape[2]

        # Expand for GQA
        cover_k_full = cover_k.repeat_interleave(num_kv_groups, dim=1)
        cover_v_full = cover_v.repeat_interleave(num_kv_groups, dim=1)

        # Phase 1: Base attention over cover view
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        attn_logits = torch.matmul(query_states, cover_k_full.transpose(-1, -2)) * scaling

        if attention_mask is not None:
            attn_logits = attn_logits + attention_mask[:, :, :, :cover_k_full.shape[-2]]

        attn_probs = torch.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_probs, cover_v_full)

        torch.cuda.synchronize()
        profiler.record("cover_attention", (time.perf_counter() - t0) * 1000)

        # Phase 2: Check for refinement
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        summary_mask = (cover_is_summary == 1)
        has_summaries = summary_mask.any()
        needs_refinement = (threshold is not None and 0 <= threshold < 1.0 and has_summaries)

        num_summaries = summary_mask.sum().item() if has_summaries else 0
        num_refined = 0

        if needs_refinement:
            refine_mask = (attn_probs > threshold) & summary_mask.view(B, 1, 1, T_cover)
            refine_positions = refine_mask.any(dim=(1, 2))  # [B, T_cover]
            num_refined = refine_positions.sum().item()

        torch.cuda.synchronize()
        profiler.record("refinement_check", (time.perf_counter() - t0) * 1000)

        # Phase 3: Refinement computation (if needed)
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        if needs_refinement and num_refined > 0:
            # This is where the expensive refinement happens
            # We need to fetch raw K/V and compute attention for refined positions
            summary_cache = self.summary_cache[layer_idx]
            page_start = summary_cache.page_start
            page_end = summary_cache.page_end

            if page_start.shape[0] < B:
                pad_b = B - page_start.shape[0]
                zeros = torch.zeros(pad_b, page_start.shape[1], device=page_start.device, dtype=page_start.dtype)
                page_start = torch.cat([page_start, zeros], dim=0)
                page_end = torch.cat([page_end, zeros], dim=0)

            k_raw, v_raw, _, _ = self.raw_cache.get_layer(layer_idx, with_offsets=False)
            b_idx, pos_idx = refine_positions.nonzero(as_tuple=True)
            page_ids = cover_indices[b_idx, pos_idx]
            starts = page_start[b_idx, page_ids]
            ends = page_end[b_idx, page_ids]
            lengths = ends - starts + 1

            valid = (page_ids >= 0) & (ends >= starts)
            if valid.any():
                b_idx, pos_idx = b_idx[valid], pos_idx[valid]
                starts, lengths = starts[valid], lengths[valid]
                page_ids = page_ids[valid]

                max_len = lengths.max().item()
                N = b_idx.shape[0]

                # Gather raw K/V for refined positions
                expanded_k = torch.zeros(N, k_raw.shape[1], max_len, D, device=k_raw.device, dtype=k_raw.dtype)
                expanded_v = torch.zeros(N, v_raw.shape[1], max_len, D, device=v_raw.device, dtype=v_raw.dtype)

                for i in range(N):
                    b, s, l = b_idx[i].item(), starts[i].item(), lengths[i].item()
                    expanded_k[i, :, :l, :] = k_raw[b, :, s:s+l, :]
                    expanded_v[i, :, :l, :] = v_raw[b, :, s:s+l, :]

                # Compute refined attention
                expanded_k = expanded_k.repeat_interleave(num_kv_groups, dim=1)
                expanded_v = expanded_v.repeat_interleave(num_kv_groups, dim=1)

                # Get query for each refined position
                q_for_refine = query_states[b_idx, :, :, :]  # [N, H_q, L_q, D]

                # Compute attention
                refined_logits = torch.matmul(q_for_refine, expanded_k.transpose(-1, -2)) * scaling
                refined_probs = torch.softmax(refined_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
                refined_output = torch.matmul(refined_probs, expanded_v)

                # Blend refined output back (simplified - actual code is more complex)
                # For profiling, we just measure the computation time

        torch.cuda.synchronize()
        profiler.record("refinement_compute", (time.perf_counter() - t0) * 1000)

        # Record stats
        profiler.record_refinement(num_summaries, num_refined)

        # Total time
        torch.cuda.synchronize()
        profiler.record("total_attention", (time.perf_counter() - total_start) * 1000)

        # Call original for correct output (our profiling may have changed state)
        return original_top_down(
            layer_idx, query_states, scaling, num_kv_groups,
            attention_mask, sliding_window, threshold
        )

    import types
    controller.top_down_attention = types.MethodType(profiled_top_down_attention, controller)


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
    print(f"Generating 64 tokens with refinement (threshold=0.05)...")
    print()

    set_luka_kv_params(
        default_tail_len=16,
        min_compress_chunk=16,
        max_pages=15,
        refine_threshold=0.05,  # Refinement enabled
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

    controller = model.model.luka_kv_controller
    patch_top_down_attention(controller)

    num_generated = generate_tokens(model, tokenizer, paragraph, 64, device)
    print(f"Generated {num_generated} tokens")

    profiler.summary()

    # Also print comparison
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    if profiler.timings["total_attention"]:
        avg_attn = sum(profiler.timings["total_attention"]) / len(profiler.timings["total_attention"])
        print(f"Average attention time (layer 0): {avg_attn:.2f} ms")
        print(f"Estimated per-token (28 layers): {avg_attn * 28:.1f} ms")
        print(f"Estimated throughput: {1000 / (avg_attn * 28):.1f} tok/s")

        if profiler.timings["cover_attention"]:
            avg_cover = sum(profiler.timings["cover_attention"]) / len(profiler.timings["cover_attention"])
            print(f"\nWithout refinement would be: {avg_cover:.2f} ms per layer")
            print(f"That's {1000 / (avg_cover * 28):.1f} tok/s")


if __name__ == "__main__":
    main()
