"""
Training package scaffolding for compressor distillation.

This module exports the key configuration, data, capture, sampler, loss, and
training entrypoints. Implementations live in the sibling modules; this file
only defines import convenience and the public surface.
"""

from .config import DataConfig, LossConfig, SamplerConfig, SamplerParams, TrainConfig
from .data import build_dataloader, build_tokenizer
from .capture import LayerCapture, QKVCapture, patch_qwen3_for_training, restore_qwen3_attention
from .sampler import sample_segments
from .losses import alignment_losses
from .trainer import train

__all__ = [
    "DataConfig",
    "LossConfig",
    "SamplerConfig",
    "SamplerParams",
    "TrainConfig",
    "build_dataloader",
    "build_tokenizer",
    "LayerCapture",
    "QKVCapture",
    "patch_qwen3_for_training",
    "restore_qwen3_attention",
    "sample_segments",
    "alignment_losses",
    "train",
]
