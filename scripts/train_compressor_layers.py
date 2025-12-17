"""
Training script for compressor with explicit configuration.

Run with: python scripts/train_compressor.py
"""

from training.trainer import train
from training.config import TrainConfig, DataConfig, SamplerConfig, SamplerParams, LossConfig, ModelConfig


# Data configuration
data_cfg = DataConfig(
    model_name="Qwen/Qwen3-1.7B-Base",
    split="train",
    seq_len=4096,
    batch_size=1,
    num_workers=0,
    streaming=True,
    docs_per_sequence=2,
)

# Sampler configuration
sampler_cfg = SamplerConfig(
    default=SamplerParams(
        tail_len=16,
        mean=16.0,
        std=4.0,
        max_pages=128,
    ),
    per_layer={},
)

# Loss configuration
loss_cfg = LossConfig(
    lambda_key=0.25,
    lambda_out=1.0,
)

# Model configuration
model_cfg = ModelConfig(
    nhead=4,
    ff_mult=4,
    dropout=0.0,
)

# Training configuration

layers = [

    [4, 1],
    [4, 1],
    [4, 1],
    [4, 1],
    [4, 1],
    [4, 1],
    [4, 1],

    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [0.25, 1],
    [0.25, 1],
    [0.25, 1],

    [0.25, 1],
    [0.25, 1],
    [0.25, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],

    [4, 1],
    [4, 1],
    [4, 1],
    [4, 1],
    [4, 1],
    [4, 1],
    [4, 1],
]

# Run training
if __name__ == "__main__":
    for i in range(14, 28):
        train_cfg = TrainConfig(
            model_name="Qwen/Qwen3-1.7B-Base",
            layer_idx=i,
            lr=1e-4,
            device="cuda",
            dtype=None,
            num_steps=200,
            grad_accum=1,
            log_every=10,
            save_dir=f"train_layer_{i}",
            compressor="encoder",
        )

        loss_cfg = LossConfig(
            lambda_key=layers[i][0],
            lambda_out=layers[i][1],
        )
        print(f"Training for layer {i}")
        train(train_cfg, data_cfg, sampler_cfg, loss_cfg, model_cfg)
