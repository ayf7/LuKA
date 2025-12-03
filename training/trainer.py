"""
Training loop for compressor distillation.

Coordinates data loading, model capture, segment sampling, loss computation,
and optimizer updates.
"""

import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM

from modeling.compressor import EncoderCompressor, MeanCompressor
from training.capture import QKVCapture, patch_qwen3_for_training, restore_qwen3_attention
from training.config import DataConfig, LossConfig, ModelConfig, SamplerConfig, TrainConfig
from training.data import build_dataloader
from training.losses import compute_losses
from training.sampler import sample_segments


def _save_checkpoint(
    save_path: str,
    compressor: torch.nn.Module,
    train_cfg: TrainConfig,
    loss_cfg: LossConfig,
    model_cfg: ModelConfig,
    sampler_params: any,
    head_dim: int,
    step: int,
    loss: float,
) -> None:
    """Save a checkpoint to disk."""
    torch.save({
        "compressor_state_dict": compressor.state_dict(),
        "compressor_type": train_cfg.compressor,
        "layer_idx": train_cfg.layer_idx,
        "head_dim": head_dim,
        "step": step,
        "loss": loss,
        "config": {
            "train_cfg": vars(train_cfg),
            "loss_cfg": vars(loss_cfg),
            "model_cfg": vars(model_cfg),
            "sampler_cfg": vars(sampler_params),
        },
    }, save_path)


def train(
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    sampler_cfg: SamplerConfig,
    loss_cfg: LossConfig,
    model_cfg: ModelConfig,
    recorder: Optional[QKVCapture] = None,
) -> None:
    """
    Run the compressor training loop.

    Steps:
        1) Build dataloader from data_cfg.
        2) Load base model, freeze parameters, move to device/dtype.
        3) Patch Qwen3 attention to capture q/k/v for train_cfg.layer_idx.
        4) For each batch (prefill only, use_cache=False):
            a) Run model forward under no_grad to populate captures.
            b) Sample segments via sampler_cfg on batch["attention_mask"].
            c) Compute alignment losses with captured q/k/v and compressor outputs.
            d) Backpropagate through compressor parameters only; optimizer step.
        5) Log metrics every train_cfg.log_every steps and save checkpoints to train_cfg.save_dir.

    Args:
        train_cfg: TrainConfig containing model, optimizer, device, and run-length params.
        data_cfg: DataConfig for dataloader construction.
        sampler_cfg: SamplerConfig controlling page sampling.
        loss_cfg: LossConfig with lambda weights.
        model_cfg: ModelConfig with compressor architecture hyperparameters.
        recorder: Optional existing QKVCapture; if None, created internally.

    Returns:
        None.

    Side effects:
        Reads/writes model weights, performs GPU computation, writes trained
        compressor weights to disk, starts dataloader workers, downloads assets.
    """
    device = torch.device(train_cfg.device)
    dtype = getattr(torch, train_cfg.dtype) if train_cfg.dtype else None

    print(f"=== Compressor Training ===")
    print(f"Model: {train_cfg.model_name}")
    print(f"Layer: {train_cfg.layer_idx}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Steps: {train_cfg.num_steps}")
    print(f"Learning rate: {train_cfg.lr}")
    print(f"Save directory: {train_cfg.save_dir}")

    # Create save directory
    os.makedirs(train_cfg.save_dir, exist_ok=True)

    # 1. Build dataloader
    print("\n[1/6] Building dataloader...")
    dataloader = build_dataloader(data_cfg)
    data_iter = iter(dataloader)

    # 2. Load and freeze base model
    print("[2/6] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(train_cfg.model_name)
    model = model.to(device)
    if dtype:
        model = model.to(dtype)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"Model loaded and frozen: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    # 3. Patch attention for q/k/v capture
    print("[3/6] Patching attention for capture...")
    if recorder is None:
        recorder = patch_qwen3_for_training(model, layers=[train_cfg.layer_idx])
    else:
        patch_qwen3_for_training(model, layers=[train_cfg.layer_idx])

    # Get model config for compressor initialization
    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    num_q_heads = config.num_attention_heads
    num_kv_groups = num_q_heads // num_kv_heads
    scaling = head_dim ** -0.5

    # 4. Initialize compressor
    print("[4/6] Initializing compressor...")
    if train_cfg.compressor == "mean":
        compressor = MeanCompressor()
        print("Using MeanCompressor (baseline, not trainable)")
    elif train_cfg.compressor == "encoder":
        compressor = EncoderCompressor(
            dim=head_dim,
            nhead=model_cfg.nhead,
            ff_mult=model_cfg.ff_mult,
            dropout=model_cfg.dropout,
        )
        print(f"Using EncoderCompressor: {sum(p.numel() for p in compressor.parameters()) / 1e3:.1f}K params")
    else:
        raise ValueError(f"Unknown compressor type: {train_cfg.compressor}")

    compressor = compressor.to(device)
    if dtype:
        compressor = compressor.to(dtype)

    # 5. Create optimizer (only for trainable compressors)
    trainable_params = [p for p in compressor.parameters() if p.requires_grad]
    if trainable_params:
        optimizer = torch.optim.Adam(trainable_params, lr=train_cfg.lr)
        print(f"Optimizer: Adam with {len(trainable_params)} parameter groups")
    else:
        optimizer = None
        print("No trainable parameters (using baseline compressor)")

    # Get sampler params for the target layer
    sampler_params = sampler_cfg.per_layer.get(train_cfg.layer_idx, sampler_cfg.default)

    # 6. Training loop
    print(f"\n[5/6] Starting training for {train_cfg.num_steps} steps...")
    print(f"Sampler: tail_len={sampler_params.tail_len}, mean={sampler_params.mean}, "
          f"std={sampler_params.std}, max_pages={sampler_params.max_pages}")
    print(f"Loss weights: lambda_key={loss_cfg.lambda_key}")

    step = 0
    total_loss = 0.0
    total_loss_key = 0.0
    total_loss_out = 0.0
    total_pages = 0
    best_loss = float('inf')
    best_step = 0
    last_loss = float('inf')

    compressor.train()

    try:
        while step < train_cfg.num_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward through frozen model to capture q/k/v
            recorder.clear()
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

            # Get captured tensors for target layer
            capture = recorder.records.get(train_cfg.layer_idx)
            if capture is None:
                print(f"Warning: No capture for layer {train_cfg.layer_idx}, skipping batch")
                continue

            # Sample segments
            segments = sample_segments(attention_mask, sampler_params)

            # Check if any segments were generated
            total_segments = sum(len(seg_list) for seg_list in segments)
            if total_segments == 0:
                continue  # Skip batch with no segments

            # Compute losses
            loss_dict = compute_losses(
                q=capture.q,
                k=capture.k,
                v=capture.v,
                segments=segments,
                compressor=compressor,
                token_mask=attention_mask,
                attention_mask=capture.attention_mask,
                num_kv_groups=num_kv_groups,
                scaling=scaling,
                lambda_key=loss_cfg.lambda_key,
            )

            loss = loss_dict["loss"]
            last_loss = loss.item()

            # Backprop and optimizer step
            if optimizer is not None:
                loss.backward()

                if (step + 1) % train_cfg.grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # Logging
            total_loss += loss.item()
            total_loss_key += loss_dict["loss_key"].item()
            total_loss_out += loss_dict["loss_out"].item()
            total_pages += loss_dict["num_pages"].item()

            step += 1

            if step % train_cfg.log_every == 0:
                avg_loss = total_loss / train_cfg.log_every
                avg_loss_key = total_loss_key / train_cfg.log_every
                avg_loss_out = total_loss_out / train_cfg.log_every
                avg_pages = total_pages / train_cfg.log_every

                print(f"Step {step}/{train_cfg.num_steps} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Key: {avg_loss_key:.4f} | "
                      f"Out: {avg_loss_out:.4f} | "
                      f"Pages: {avg_pages:.1f}")

                # Track best loss and save checkpoint
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_step = step
                    best_path = os.path.join(train_cfg.save_dir, "best.pt")
                    _save_checkpoint(
                        best_path, compressor, train_cfg, loss_cfg, model_cfg,
                        sampler_params, head_dim, step, avg_loss
                    )
                    print(f"  → New best! Saved to {best_path}")

                # Periodic checkpoint save
                if train_cfg.save_every > 0 and step % train_cfg.save_every == 0:
                    periodic_path = os.path.join(train_cfg.save_dir, f"step_{step}.pt")
                    _save_checkpoint(
                        periodic_path, compressor, train_cfg, loss_cfg, model_cfg,
                        sampler_params, head_dim, step, avg_loss
                    )
                    print(f"  → Saved checkpoint to {periodic_path}")

                total_loss = 0.0
                total_loss_key = 0.0
                total_loss_out = 0.0
                total_pages = 0

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    # 7. Save compressor weights
    last_path = os.path.join(train_cfg.save_dir, "last.pt")
    print(f"\n[6/6] Saving last checkpoint to {last_path}...")
    _save_checkpoint(
        last_path, compressor, train_cfg, loss_cfg, model_cfg,
        sampler_params, head_dim, step, last_loss
    )

    # Cleanup
    restore_qwen3_attention(model)
    print("Training complete!")
