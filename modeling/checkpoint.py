"""
Utilities for loading trained compressor checkpoints.
"""

import torch
from pathlib import Path
from typing import Optional

from modeling.compressor import EncoderCompressor, MeanCompressor


def load_compressor_checkpoint(
    checkpoint_path: str,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.nn.Module:
    """
    Load a trained compressor from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file (e.g., "train_1/best.pt").
        device: Device to load the compressor to (e.g., "cuda", "cpu").
            If None, uses the device from training.
        dtype: Data type for the compressor parameters.
            If None, uses the dtype from training.

    Returns:
        Loaded compressor module with trained weights.

    Example:
        >>> compressor = load_compressor_checkpoint("train_1/best.pt", device="cuda")
        >>> # Use with LuKA
        >>> set_luka_kv_params(compressor=compressor)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract config
    compressor_type = checkpoint["compressor_type"]
    head_dim = checkpoint["head_dim"]
    config = checkpoint.get("config", {})

    # Get model config if available
    model_cfg = config.get("model_cfg", {})
    nhead = model_cfg.get("nhead", 4)
    ff_mult = model_cfg.get("ff_mult", 4)
    dropout = model_cfg.get("dropout", 0.0)

    # Initialize compressor
    if compressor_type == "mean":
        compressor = MeanCompressor()
    elif compressor_type == "encoder":
        compressor = EncoderCompressor(
            dim=head_dim,
            nhead=nhead,
            ff_mult=ff_mult,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown compressor type: {compressor_type}")

    # Load weights
    compressor.load_state_dict(checkpoint["compressor_state_dict"])

    # Move to device/dtype if specified
    if device is not None:
        compressor = compressor.to(device)
    if dtype is not None:
        compressor = compressor.to(dtype)

    # Set to eval mode
    compressor.eval()

    print(f"Loaded {compressor_type} compressor from {checkpoint_path}")
    print(f"  Layer: {checkpoint['layer_idx']}")
    print(f"  Head dim: {head_dim}")
    print(f"  Step: {checkpoint.get('step', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    if compressor_type == "encoder":
        num_params = sum(p.numel() for p in compressor.parameters())
        print(f"  Parameters: {num_params / 1e3:.1f}K")

    return compressor
