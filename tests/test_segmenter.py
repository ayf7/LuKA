import types

import torch

from modeling.segmenter import DummySegmenter
from modeling.kv_cache import LukaKVCaches


def test_dummy_segmenter_matches_populate_pages():
    B = 2
    T = 10
    min_chunk = 3
    tail_len = 2
    max_pages = 4

    segmenter = DummySegmenter()
    luka = LukaKVCaches(
        raw_cache=None,
        segmenter=segmenter,
        num_layers=1,
        default_tail_len=tail_len,
        min_compress_chunk=min_chunk,
        batch_size=B,
        max_pages=max_pages,
    )

    # Manually set starts and attention buffer (content unused by DummySegmenter)
    seq_starts = torch.tensor([0, 2])
    luka.starts = seq_starts
    luka.attn_weight_buffer = torch.zeros(B, 1, 1, T)

    expected = segmenter.process(
        attention_scores=luka.attn_weight_buffer,
        seq_starts=seq_starts,
        min_chunk=min_chunk,
        tail_len=tail_len,
        max_pages=max_pages,
    )
    result = luka.populate_pages()

    torch.testing.assert_close(result, expected)


def test_kl_segmenter_spacing_and_threshold():
    from modeling.segmenter import KLDivergenceSegmenter

    B, H, L, T = 1, 2, 12, 12
    attn = torch.full((B, H, L, T), 1e-4)

    # Create two spikes in attention around positions 3 and 8
    for h in range(H):
        attn[0, h, 3, :4] = torch.tensor([0.6, 0.2, 0.15, 0.05])
        attn[0, h, 8, :9] = torch.tensor([0.05] * 8 + [0.6])

    seg = KLDivergenceSegmenter(lag=2, threshold=0.05)
    seq_starts = torch.tensor([0])
    page_ends = seg.process(attn, seq_starts, min_chunk=3, tail_len=2, max_pages=4)

    # Expect two boundaries that are at least min_chunk apart and within valid range
    ends = page_ends[0].tolist()
    ends = [e for e in ends if e >= 0]
    assert len(ends) == 2
    assert ends[0] >= 0
    assert ends[1] - ends[0] >= 3  # min_chunk spacing
    assert ends[1] <= T - 2  # respect tail_len


def test_topk_selection_overrides_threshold():
    from modeling.segmenter import KLDivergenceSegmenter

    B, H, L, T = 1, 1, 6, 6
    attn = torch.zeros((B, H, L, T))
    seg = KLDivergenceSegmenter(lag=1, threshold=None, top_k=2)

    # Monkeypatch _kl_scores to return known scores
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.4])
    seg._kl_scores = lambda x: scores

    page_ends = seg.process(attn, torch.tensor([0]), min_chunk=1, tail_len=0, max_pages=4)
    ends = page_ends[0].tolist()
    ends = [e for e in ends if e >= 0]

    # Top2 scores are at rows 1 and 3; expect sorted positions [1,3]
    assert ends[:2] == [1, 3]


def test_topk_with_threshold_filters():
    from modeling.segmenter import KLDivergenceSegmenter

    B, H, L, T = 1, 1, 6, 6
    attn = torch.zeros((B, H, L, T))
    seg = KLDivergenceSegmenter(lag=1, threshold=0.85, top_k=3)

    scores = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.95, 0.4])
    seg._kl_scores = lambda x: scores

    page_ends = seg.process(attn, torch.tensor([0]), min_chunk=1, tail_len=0, max_pages=6)
    ends = [e for e in page_ends[0].tolist() if e >= 0]

    # Only scores > 0.85 among top3 should remain: rows 1 and 4
    assert ends[:2] == [1, 4]
