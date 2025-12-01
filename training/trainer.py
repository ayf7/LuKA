"""
Training loop scaffolding for compressor distillation.

Coordinates data loading, model capture, segment sampling, loss computation,
and optimizer updates. Implementations are deferred; this file defines the
interface and documents tensor shapes for each stage.
"""

from typing import Optional

import torch

from training.config import DataConfig, LossConfig, SamplerConfig, TrainConfig
from training.capture import QKVCapture


def train(
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    sampler_cfg: SamplerConfig,
    loss_cfg: LossConfig,
    recorder: Optional[QKVCapture] = None,
) -> None:
    """
    Run the compressor training loop.

    High-level steps (to be implemented):
        1) Build dataloader from `data_cfg`.
        2) Load base model `train_cfg.model_name`, freeze parameters, move to device/dtype.
        3) Patch Qwen3 attention to capture q/k/v for layer(s) specified by train_cfg.layer_idx.
        4) For each batch (prefill only, use_cache=False):
            a) Run model forward under no grad to populate captures.
            b) Sample segments via sampler_cfg on batch["attention_mask"].
            c) Compute alignment losses with captured q/k/v and compressor outputs.
            d) Backpropagate through compressor parameters only; optimizer step.
        5) Log metrics every `train_cfg.log_every` steps and save weights to train_cfg.save_path.

    Args:
        train_cfg: TrainConfig containing model, optimizer, device, and run-length params.
        data_cfg: DataConfig for dataloader construction.
        sampler_cfg: SamplerConfig controlling page sampling.
        loss_cfg: LossConfig with lambda weights.
        recorder: Optional existing QKVCapture; if None, should be created internally.

    Returns:
        None.

    Side effects:
        Should read/write model weights, perform GPU computation, and write the
        trained compressor weights to disk. May start dataloader workers and
        download model/tokenizer assets.
    """
    raise NotImplementedError("Training loop is not implemented in scaffolding.")
