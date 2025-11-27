import torch

from modeling.kv_cache_new import RawCache, LukaKVController


class _DummyLayer:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        self.keys = keys
        self.values = values


class _DummyCache:
    """Minimal DynamicCache stand-in for testing RawCache plumbing."""

    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        self.layers = [_DummyLayer(keys, values)]

    def update(self, keys: torch.Tensor, values: torch.Tensor, layer_idx: int, cache_kwargs=None):
        layer = self.layers[layer_idx]
        layer.keys = torch.cat([layer.keys, keys], dim=2)
        layer.values = torch.cat([layer.values, values], dim=2)
        return layer.keys, layer.values


class _DummyConfig:
    def __init__(self, num_hidden_layers: int = 1):
        self.num_hidden_layers = num_hidden_layers


def test_raw_cache_update_infers_padding_per_batch():
    """
    RawCache.update should append to the underlying DynamicCache and infer
    left padding (seq_start/raw_seq_start) from the attention mask on a per-batch basis.
    """
    torch.manual_seed(0)
    B, H, T0, D = 2, 1, 3, 4
    L_new = 1

    k0 = torch.randn(B, H, T0, D)
    v0 = torch.randn(B, H, T0, D)
    k_new = torch.randn(B, H, L_new, D)
    v_new = torch.randn(B, H, L_new, D)

    dummy_cache = _DummyCache(k0.clone(), v0.clone())
    config = _DummyConfig(num_hidden_layers=1)
    controller = LukaKVController(config)
    controller.raw_cache.initialize_with_cache(dummy_cache)

    # Padding: batch0 has 1 pad token, batch1 has 2 pad tokens on the left
    pad_counts = torch.tensor([1, 2])
    T_total = T0 + L_new
    attention_mask = torch.zeros(B, 1, 1, T_total)
    for b, p in enumerate(pad_counts.tolist()):
        attention_mask[b, 0, 0, :p] = -1e9

    k_all, v_all, seq_start, raw_seq_start = controller.update(
        layer_idx=0,
        keys=k_new,
        values=v_new,
        cache_kwargs={},
        attention_mask=attention_mask,
    )

    expected_k = torch.cat([k0, k_new], dim=2)
    expected_v = torch.cat([v0, v_new], dim=2)

    torch.testing.assert_close(k_all, expected_k)
    torch.testing.assert_close(v_all, expected_v)
    torch.testing.assert_close(seq_start, pad_counts)
    torch.testing.assert_close(raw_seq_start, pad_counts)

    # get_layer(with_offsets=True) should reflect the same state
    k_layer, v_layer, seq_off, raw_off = controller.raw_cache.get_layer(0, with_offsets=True)
    torch.testing.assert_close(k_layer, expected_k)
    torch.testing.assert_close(v_layer, expected_v)
    torch.testing.assert_close(seq_off, pad_counts)
    torch.testing.assert_close(raw_off, pad_counts)
