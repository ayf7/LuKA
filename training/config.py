"""
Configuration dataclasses for compressor training.

Each config isolates a concern (data, sampling, losses, training) so the
trainer can receive strongly typed, self-documenting parameters.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    """
    Data loading hyperparameters.

    Fields:
        model_name: Hugging Face model id used for tokenizer and model weights.
        split: Dataset split name (e.g., "train").
        seq_len: Integer max sequence length for truncation/padding.
        batch_size: Per-step batch size.
        num_workers: DataLoader worker count.
        streaming: Whether to stream the dataset from HF Hub.
        docs_per_sequence: Number of documents to concatenate per training sequence.
            If > 1, documents are joined with EOS token separator.
    """

    model_name: str = "Qwen/Qwen3-1.7B-Base"
    split: str = "train"
    seq_len: int = 2048
    batch_size: int = 1
    num_workers: int = 0
    streaming: bool = True
    docs_per_sequence: int = 2


@dataclass
class SamplerParams:
    """
    Parameters for a single page-sampling policy.

    Fields:
        tail_len: Trailing tokens to exclude from compression.
        mean: Base page length/stride (tokens) for jittered tiling.
        std: Standard deviation of Gaussian jitter applied to page starts.
        max_pages: Upper bound on pages per batch row.
    """

    tail_len: int = 16
    mean: float = 64.0
    std: float = 8.0
    max_pages: int = 15


@dataclass
class SamplerConfig:
    """
    Sampler configuration with optional layer-specific overrides.

    Fields:
        default: SamplerParams applied when no layer-specific override exists.
        per_layer: Mapping from layer index to SamplerParams for that layer.
    """

    default: SamplerParams = field(default_factory=SamplerParams)
    per_layer: dict[int, SamplerParams] = field(default_factory=dict)


@dataclass
class LossConfig:
    """
    Loss weighting parameters.

    Fields:
        lambda_key: Weight on key alignment MSE.
        lambda_out: Weight on output alignment MSE.
    """

    lambda_key: float = 1.0
    lambda_out: float = 0.1


@dataclass
class ModelConfig:
    """
    Compressor model architecture hyperparameters.

    Fields:
        nhead: Number of attention heads in EncoderCompressor.
        ff_mult: Feed-forward dimension multiplier (ff_dim = ff_mult * d_model).
        dropout: Dropout probability for EncoderCompressor layers.
    """

    nhead: int = 4
    ff_mult: int = 4
    dropout: float = 0.0


@dataclass
class TrainConfig:
    """
    End-to-end training hyperparameters.

    Fields:
        model_name: Hugging Face model id for the base LM.
        layer_idx: Target layer index to capture/align; can extend to list later.
        lr: Learning rate for compressor parameters.
        device: Torch device string ("cuda", "cpu", etc.).
        dtype: Optional torch dtype override (e.g., torch.float16).
        num_steps: Number of optimization steps.
        grad_accum: Gradient accumulation steps to simulate larger batches.
        log_every: Logging frequency (steps).
        save_dir: Directory to save checkpoints (best.pt and last.pt).
        save_every: Save checkpoint every N steps (0 to disable periodic saving).
        compressor: Compressor type identifier (e.g., "mean", "encoder").
    """

    model_name: str = "Qwen/Qwen3-1.7B-Base"
    layer_idx: int = 0
    lr: float = 1e-4
    device: str = "cuda"
    dtype: Optional[str] = None
    num_steps: int = 1000
    grad_accum: int = 1
    log_every: int = 10
    save_dir: str = "checkpoints"
    save_every: int = 100
    compressor: str = "mean"
