import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class Compressor(ABC, nn.Module):
    """
    Abstract interface for compressing a sequence of raw KV tokens into a single
    summary key/value pair.

    Inputs:
        k: [B, H, L, D]
        v: [B, H, L, D]
        importance_weights: Optional[torch.Tensor] [B, H, L] or [B, L]
            Per-token importance scores (e.g., accumulated attention).
            If provided, can be used for weighted compression.

    Returns:
        k_summary: [B, H, D]
        v_summary: [B, H, D]
    """

    @abstractmethod
    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        importance_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _validate_input(self, k: torch.Tensor, v: torch.Tensor):
        """
        Validate input tensor shapes for compression.

        Checks:
        - k is a 4D tensor [B, H, L, D]
        - v is a 4D tensor [B, H, L, D]
        - k and v have identical shapes

        Args:
            k: [B, H, L, D] Key tensor to compress.
            v: [B, H, L, D] Value tensor to compress.
        """
        assert k.ndim == 4, f"Invariant Violation: Expected 4D keys [B, H, L, D], got {k.shape}"
        assert v.ndim == 4, f"Invariant Violation: Expected 4D values [B, H, L, D], got {v.shape}"
        assert k.shape == v.shape, f"Invariant Violation: Key/Value shape mismatch: {k.shape} vs {v.shape}"

    def _validate_output(self, k_sum: torch.Tensor, v_sum: torch.Tensor, B: int, H: int, D: int):
        """
        Validate output tensor shapes after compression.

        Checks:
        - k_sum has shape [B, H, D]
        - v_sum has shape [B, H, D]
        - Dimensions match the expected batch (B), head (H), and hidden (D) sizes

        Args:
            k_sum: [B, H, D] Compressed key summary.
            v_sum: [B, H, D] Compressed value summary.
            B, H, D: Expected batch, head, and hidden dimensions.
        """
        assert k_sum.shape == (B, H, D), f"Invariant Violation: Output keys shape {k_sum.shape} != expected {(B, H, D)}"
        assert v_sum.shape == (B, H, D), f"Invariant Violation: Output values shape {v_sum.shape} != expected {(B, H, D)}"

class MeanCompressor(Compressor):
    """
    Baseline compressor that averages keys and values across the sequence length.
    Ignores importance_weights (uniform weighting).
    """

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        importance_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate_input(k, v)
        # k, v: [B, H, L, D]
        k_summary = k.mean(dim=2)
        v_summary = v.mean(dim=2)
        self._validate_output(k_summary, v_summary, k.shape[0], k.shape[1], k.shape[3])
        return k_summary, v_summary


class AttentionWeightedCompressor(Compressor):
    """
    Compressor that weights tokens by their accumulated attention (importance).

    This produces a convex combination biased toward tokens that historically
    received more attention, which better preserves attention patterns.

    Args:
        temperature: Softmax temperature for weight normalization.
            Lower = sharper (more weight on highest-attention token).
            Default 1.0 uses raw attention scores.
        fallback_to_mean: If importance_weights is None, fall back to uniform mean.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        fallback_to_mean: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.fallback_to_mean = fallback_to_mean

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        importance_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            k: [B, H, L, D] keys to compress
            v: [B, H, L, D] values to compress
            importance_weights: [B, H, L] per-token importance scores
                Higher values = more important tokens.

        Returns:
            k_summary: [B, H, D]
            v_summary: [B, H, D]
        """
        self._validate_input(k, v)
        B, H, L, D = k.shape

        if importance_weights is None:
            if self.fallback_to_mean:
                k_summary = k.mean(dim=2)
                v_summary = v.mean(dim=2)
                self._validate_output(k_summary, v_summary, B, H, D)
                return k_summary, v_summary
            else:
                raise ValueError("importance_weights required when fallback_to_mean=False")

        # importance_weights: [B, H, L]
        assert importance_weights.shape == (B, H, L), \
            f"importance_weights shape {importance_weights.shape} != expected {(B, H, L)}"

        # Normalize to get convex combination weights
        # softmax ensures weights sum to 1 and are non-negative
        weights = torch.softmax(importance_weights / self.temperature, dim=-1)  # [B, H, L]
        weights = weights.unsqueeze(-1)  # [B, H, L, 1]

        # Weighted sum (convex combination)
        k_summary = (k * weights).sum(dim=2)  # [B, H, D]
        v_summary = (v * weights).sum(dim=2)  # [B, H, D]

        self._validate_output(k_summary, v_summary, B, H, D)
        return k_summary, v_summary


class EvictionCompressor(Compressor):
    """
    Eviction-style compressor: attention-weighted keys, zero values.

    The summary key exists for routing/refinement decisions, but contributes
    no value information. Attention landing on the summary is effectively
    "wasted" unless refinement kicks in to fetch raw values.

    This tests whether summary keys alone (for attention routing) are sufficient
    when paired with refinement, without attempting to compress values.

    Args:
        temperature: Softmax temperature for key weight normalization.
        fallback_to_mean: If importance_weights is None, fall back to uniform mean for keys.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        fallback_to_mean: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.fallback_to_mean = fallback_to_mean

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        importance_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            k: [B, H, L, D] keys to compress
            v: [B, H, L, D] values to compress (ignored, output is zeros)
            importance_weights: [B, H, L] per-token importance scores

        Returns:
            k_summary: [B, H, D] - attention-weighted
            v_summary: [B, H, D] - zeros
        """
        self._validate_input(k, v)
        B, H, L, D = k.shape

        # Values: zeros (eviction - no value contribution)
        v_summary = torch.zeros(B, H, D, device=v.device, dtype=v.dtype)

        # Keys: use attention weighting if available
        if importance_weights is None:
            if self.fallback_to_mean:
                k_summary = k.mean(dim=2)
            else:
                raise ValueError("importance_weights required when fallback_to_mean=False")
        else:
            assert importance_weights.shape == (B, H, L), \
                f"importance_weights shape {importance_weights.shape} != expected {(B, H, L)}"

            weights = torch.softmax(importance_weights / self.temperature, dim=-1)
            weights = weights.unsqueeze(-1)  # [B, H, L, 1]
            k_summary = (k * weights).sum(dim=2)

        self._validate_output(k_summary, v_summary, B, H, D)
        return k_summary, v_summary


class EncoderCompressor(Compressor):
    """
    Simple encoder-based compressor (see PROJECT.md):

    - Concatenate key/value per token -> y = [k || v]
    - Run a single Transformer encoder layer over the sequence
    - Mean-pool encoder outputs -> split back into summary k/v
    """

    def __init__(
        self,
        dim: int | None = None,
        nhead: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.0,
        checkpoint_path: str | None = None,
    ):
        super().__init__()
        self.init_dim = dim
        self.nhead = nhead
        self.ff_mult = ff_mult
        self.dropout = dropout
        self.dim = None
        self.proj_in: nn.Linear | None = None
        self.encoder: nn.TransformerEncoder | None = None
        self.proj_out: nn.Linear | None = None

        if checkpoint_path is not None:
            # Load from checkpoint
            self._load_from_checkpoint(checkpoint_path)
        elif dim is not None:
            self._build(dim)

    def _load_from_checkpoint(self, checkpoint_path: str):
        """Load compressor weights from a checkpoint file."""
        import torch
        from pathlib import Path

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract architecture params from checkpoint
        head_dim = checkpoint["head_dim"]
        config = checkpoint.get("config", {})
        model_cfg = config.get("model_cfg", {})

        # Override init params with checkpoint values
        self.nhead = model_cfg.get("nhead", self.nhead)
        self.ff_mult = model_cfg.get("ff_mult", self.ff_mult)
        self.dropout = model_cfg.get("dropout", self.dropout)

        # Build architecture
        self._build(head_dim)

        # Load weights
        self.load_state_dict(checkpoint["compressor_state_dict"])
        self.eval()

        print(f"Loaded EncoderCompressor from {checkpoint_path}")
        print(f"  Layer: {checkpoint['layer_idx']}, Head dim: {head_dim}")
        print(f"  Step: {checkpoint.get('step', 'N/A')}, Loss: {checkpoint.get('loss', 'N/A'):.4f}")

    def _build(self, dim: int, device=None, dtype=None):
        d_model = 2 * dim
        self.proj_in = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=self.nhead,
            dim_feedforward=self.ff_mult * d_model,
            dropout=self.dropout,
            batch_first=True,
            device=device,
            dtype=dtype,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.proj_out = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.dim = dim

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        importance_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Note: importance_weights is accepted for API compatibility but currently
        ignored. The encoder learns its own weighting via attention.
        """
        self._validate_input(k, v)
        # k, v: [B, H, L, D]
        B, H, L, D = k.shape
        if self.encoder is None:
            self._build(D, device=k.device, dtype=k.dtype)
        elif D != self.dim:
            raise ValueError(f"EncoderCompressor expected dim={self.dim}, got {D}")

        # Ensure compressor is on same device/dtype as inputs
        if self.proj_in.weight.device != k.device or self.proj_in.weight.dtype != k.dtype:
            self.to(device=k.device, dtype=k.dtype)

        x = torch.cat([k, v], dim=-1)  # [B, H, L, 2D]
        x = x.view(B * H, L, 2 * D)
        x = self.proj_in(x)
        enc = self.encoder(x)  # [B*H, L, 2D]
        pooled = enc.mean(dim=1)  # [B*H, 2D]
        pooled = self.proj_out(pooled)
        pooled = pooled.view(B, H, 2 * D)
        k_summary, v_summary = torch.split(pooled, self.dim, dim=-1)
        self._validate_output(k_summary, v_summary, B, H, D)
        return k_summary, v_summary
