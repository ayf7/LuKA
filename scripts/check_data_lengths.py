"""
Minimal script to check token length distribution in a dataset.
"""

import sys
import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    # Config
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "lucadiliello/bookcorpusopen"
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    model_name = "Qwen/Qwen3-1.7B-Base"

    print(f"Dataset: {dataset_name}")
    print(f"Samples: {num_samples}")
    print(f"Model: {model_name}\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Stream dataset
    print("Loading dataset...")
    if dataset_name == "allenai/c4":
        # C4 requires a config (e.g., "en", "realnewslike")
        config = sys.argv[3] if len(sys.argv) > 3 else "realnewslike"
        print(f"C4 config: {config}")
        dataset = load_dataset(dataset_name, config, split="train", streaming=True)
    else:
        dataset = load_dataset(dataset_name, split="train", streaming=True)

    # Collect lengths
    print("Tokenizing...")
    lengths = []
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        tokens = tokenizer(example["text"], truncation=False, return_attention_mask=False)
        lengths.append(len(tokens["input_ids"]))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{num_samples}")

    # Stats
    lengths_t = torch.tensor(lengths, dtype=torch.float32)
    print(f"\n{'='*60}")
    print(f"Token Length Statistics (n={len(lengths)})")
    print(f"{'='*60}")
    print(f"Mean:   {lengths_t.mean().item():.1f}")
    print(f"Median: {lengths_t.median().item():.1f}")
    print(f"Std:    {lengths_t.std().item():.1f}")
    print(f"Min:    {lengths_t.min().item():.0f}")
    print(f"Max:    {lengths_t.max().item():.0f}")

    print(f"\nPercentiles:")
    for p in [50, 75, 90, 95, 99]:
        val = torch.quantile(lengths_t, p / 100.0).item()
        print(f"  {p:2d}th: {val:.0f}")

    print(f"\nDistribution:")
    for bucket in [512, 1024, 2048, 4096, 8192]:
        count = (lengths_t < bucket).sum().item()
        pct = count / len(lengths) * 100
        print(f"  < {bucket:5d}: {count:4.0f} ({pct:5.1f}%)")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
