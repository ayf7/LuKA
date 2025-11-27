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


class MeanCompressor(Compressor):
    """
    Baseline compressor that averages keys and values across the sequence length.
    """

    def forward(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # k, v: [B, H, L, D]
        k_summary = k.mean(dim=2)
        v_summary = v.mean(dim=2)
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
        if dim is not None:
            self._build(dim)

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
        # k, v: [B, H, L, D]
        B, H, L, D = k.shape
        if self.encoder is None:
            self._build(D, device=k.device, dtype=k.dtype)
        elif D != self.dim:
            raise ValueError(f"EncoderCompressor expected dim={self.dim}, got {D}")

        x = torch.cat([k, v], dim=-1)  # [B, H, L, 2D]
        x = x.view(B * H, L, 2 * D)
        x = self.proj_in(x)
        enc = self.encoder(x)  # [B*H, L, 2D]
        pooled = enc.mean(dim=1)  # [B*H, 2D]
        pooled = self.proj_out(pooled)
        pooled = pooled.view(B, H, 2 * D)
        k_summary, v_summary = torch.split(pooled, self.dim, dim=-1)
        return k_summary, v_summary
