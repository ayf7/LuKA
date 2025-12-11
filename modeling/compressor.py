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

    Returns:
        k_summary: [B, H, D]
        v_summary: [B, H, D]
    """

    @abstractmethod
    def forward(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
    """

    def forward(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate_input(k, v)
        # k, v: [B, H, L, D]
        k_summary = k.mean(dim=2)
        v_summary = v.mean(dim=2)
        self._validate_output(k_summary, v_summary, k.shape[0], k.shape[1], k.shape[3])
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

    def forward(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
