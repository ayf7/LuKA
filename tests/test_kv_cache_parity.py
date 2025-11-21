import types

import torch
import pytest

from modeling.kv_cache import LukaKVCaches
from modeling.segmenter import DummySegmenter
from transformers.models.qwen3.modeling_qwen3 import repeat_kv


def _make_setup(
    *,
    batch_size: int = 1,
    num_kv_heads: int = 1,
    num_groups: int = 2,
    seq_len: int = 6,
    head_dim: int = 8,
    num_queries: int = 2,
    page_size: int = 2,
):
    """
    Build a minimal LukaKVCaches state with:
    - raw keys/values (random)
    - two summary pages (size `page_size`) plus raw tail
    - no attention mask (to keep parity checks simple)
    """
    torch.manual_seed(42)

    H_k = num_kv_heads
    H_q = H_k * num_groups
    B = batch_size
    D = head_dim
    T = seq_len

    # Raw KV
    k_raw = torch.randn(B, H_k, T, D)
    v_raw = torch.randn(B, H_k, T, D)

    # Minimal cache object exposing .layers with .keys/.values
    layer = types.SimpleNamespace(keys=k_raw, values=v_raw)
    raw_cache = types.SimpleNamespace(layers=[layer])

    # Page boundaries (two pages of length page_size, then tail)
    # Example: for T=6, page_size=2 -> pages end at 1 and 3; tail = indices 4,5
    max_pages = 4
    page_ends = torch.full((B, max_pages), -1, dtype=torch.long)
    page_ends[:, 0] = page_size - 1          # first page end (inclusive)
    page_ends[:, 1] = (2 * page_size) - 1    # second page end (inclusive)

    luka = LukaKVCaches(
        raw_cache=raw_cache,
        segmenter=DummySegmenter(),
        num_layers=1,
        default_tail_len=0,      # include all remaining raw tokens in the cover
        min_compress_chunk=page_size,
        batch_size=B,
    )

    # Build summaries from raw using the helper (sets page_start/end/summary_len, etc.)
    luka.finalize_pages_and_build_summaries(
        layer_idx=0,
        k_raw=k_raw,
        v_raw=v_raw,
        page_ends=page_ends,
    )

    # Random queries
    q = torch.randn(B, H_q, num_queries, D)
    scaling = D**-0.5

    return luka, q, k_raw, v_raw, num_groups, scaling


def _raw_attention(q, k_raw, v_raw, num_groups, scaling, attention_mask=None):
    k_full = repeat_kv(k_raw, num_groups)
    v_full = repeat_kv(v_raw, num_groups)
    logits = torch.matmul(q, k_full.transpose(2, 3)) * scaling
    if attention_mask is not None:
        logits = logits + attention_mask
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(probs, v_full)
    return out, probs


@pytest.mark.parametrize("threshold", [-1.0, 0.0])
def test_top_down_matches_raw_when_forced(threshold):
    """
    - threshold < 0  -> fast-path raw attention
    - threshold == 0 -> refined path should still equal raw
    """
    luka, q, k_raw, v_raw, num_groups, scaling = _make_setup()

    expected_out, expected_probs = _raw_attention(q, k_raw, v_raw, num_groups, scaling, attention_mask=None)

    luka_out, luka_probs = luka.top_down_attention(
        layer_idx=0,
        query_states=q,
        scaling=scaling,
        num_kv_groups=num_groups,
        attention_mask=None,
        sliding_window=None,
        threshold=threshold,
    )

    # Output parity must always hold
    torch.testing.assert_close(luka_out, expected_out, atol=1e-5, rtol=1e-5)

    if threshold < 0:
        # Fast path returns raw-sized probabilities
        torch.testing.assert_close(luka_probs, expected_probs, atol=1e-5, rtol=1e-5)


def test_cover_represents_all_tokens_and_caches():
    """
    get_covering_kv should return a cover that represents every raw token
    either via a summary page or the raw tail, and should reuse the cached
    tensors when nothing has changed.
    """
    torch.manual_seed(0)
    B, H_k, D = 1, 2, 4
    T_raw = 10
    min_chunk = 2
    max_pages = 4

    k_raw = torch.randn(B, H_k, T_raw, D)
    v_raw = torch.randn(B, H_k, T_raw, D)

    layer = types.SimpleNamespace(keys=k_raw, values=v_raw)
    raw_cache = types.SimpleNamespace(layers=[layer])

    # Pages: [0-1], [2-3], [4-5] summarized; raw tail = [6,7,8,9]
    page_ends = torch.tensor([[1, 3, 5, -1]], dtype=torch.long)

    luka = LukaKVCaches(
        raw_cache=raw_cache,
        segmenter=DummySegmenter(),
        num_layers=1,
        default_tail_len=0,
        min_compress_chunk=min_chunk,
        batch_size=B,
    )
    luka.finalize_pages_and_build_summaries(0, k_raw, v_raw, page_ends)

    # First call builds and caches
    cover_k1, cover_v1, cover_is_summary1, cover_indices1 = luka.get_covering_kv(0)
    # Second call should hit cache and return identical tensors
    cover_k2, cover_v2, cover_is_summary2, cover_indices2 = luka.get_covering_kv(0)

    torch.testing.assert_close(cover_k1, cover_k2)
    torch.testing.assert_close(cover_v1, cover_v2)
    torch.testing.assert_close(cover_is_summary1, cover_is_summary2)
    torch.testing.assert_close(cover_indices1, cover_indices2)

    # Coverage check: every raw position 0..T_raw-1 is represented
    summary_mask = cover_is_summary1[0].bool()
    summary_indices = cover_indices1[0, summary_mask]
    raw_tail_indices = cover_indices1[0, (~summary_mask) & (cover_indices1[0] >= 0)]

    page_starts = luka.page_start[0][0, summary_indices]
    page_ends_used = luka.page_end[0][0, summary_indices]

    covered = set()
    for s, e in zip(page_starts.tolist(), page_ends_used.tolist()):
        covered.update(range(s, e + 1))
    covered.update(raw_tail_indices.tolist())

    assert covered == set(range(T_raw))
