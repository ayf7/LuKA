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
