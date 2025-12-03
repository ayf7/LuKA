"""
Training package scaffolding for compressor distillation.

This module exports the key configuration, data, capture, sampler, loss, and
training entrypoints. Implementations live in the sibling modules; this file
only defines import convenience and the public surface.
"""

from training.config import DataConfig, LossConfig, SamplerConfig, SamplerParams, TrainConfig
from training.data import build_dataloader, build_tokenizer
from training.capture import LayerCapture, QKVCapture, patch_qwen3_for_training, restore_qwen3_attention
from training.sampler import sample_segments
from training.losses import compute_losses
from training.trainer import train

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
    "compute_losses",
    "train",
]
