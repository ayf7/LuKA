"""
Data loading utilities for compressor training.

This module will stream C4-realnewslike, tokenize to fixed-length sequences,
and return batches shaped:
    input_ids:      [B, L] int64
    attention_mask: [B, L] int64 (1=token, 0=pad; left-padded)
"""

from typing import Callable, Dict

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from training.config import DataConfig


def build_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """
    Load the tokenizer for `model_name` and ensure a pad token exists.

    Args:
        model_name: Hugging Face model id.

    Returns:
        tokenizer: PreTrainedTokenizerBase with pad_token_id set (uses eos if missing).

    Side effects:
        May download tokenizer assets from HF Hub; no file writes beyond cache.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def build_dataloader(cfg: DataConfig) -> DataLoader:
    """
    Create a DataLoader over C4-realnewslike.

    Args:
        cfg: DataConfig with model/tokenization and batching parameters.

    Returns:
        torch.utils.data.DataLoader yielding dict batches:
            input_ids:      torch.LongTensor [cfg.batch_size, cfg.seq_len]
            attention_mask: torch.LongTensor [cfg.batch_size, cfg.seq_len], 1=token, 0=pad

    Side effects:
        Streams/loads dataset shards; may start background workers and read from disk/network.
    """
    tokenizer = build_tokenizer(cfg.model_name)
    dataset = load_dataset(
        "allenai/c4",
        "realnewslike",
        split=cfg.split,
        streaming=cfg.streaming,
    )
    encoded = dataset.map(
        _encode_batch,
        batched=True,
        remove_columns=dataset.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "seq_len": cfg.seq_len,
            "docs_per_sequence": cfg.docs_per_sequence,
        },
    )
    loader = DataLoader(
        encoded,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=_collate_fn(tokenizer.pad_token_id),
    )
    return loader


def _collate_fn(pad_token_id: int) -> Callable[[list[Dict]], Dict[str, torch.Tensor]]:
    """
    Build a collate function that converts tokenized samples into tensors.

    Args:
        pad_token_id: Integer pad id used where attention_mask == 0.

    Returns:
        Callable that maps a list of dicts with keys {"input_ids", "attention_mask"}
        into a batch dict:
            input_ids: torch.LongTensor [B, L]
            attention_mask: torch.LongTensor [B, L]

    Side effects:
        None. Intended to be passed into torch.utils.data.DataLoader.
    """
    def _collate(batch: list[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
        attention_mask = torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long)
        input_ids = torch.where(
            attention_mask == 0, torch.full_like(input_ids, pad_token_id), input_ids
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    return _collate


def _encode_batch(
    examples: Dict,
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
    docs_per_sequence: int = 1,
) -> Dict[str, list[list[int]]]:
    """
    Tokenize raw text examples to fixed-length sequences.

    Args:
        examples: Mapping with key "text"; a batch of strings from C4.
        tokenizer: Hugging Face tokenizer to apply.
        seq_len: Target sequence length for truncation/padding.
        docs_per_sequence: Number of documents to concatenate per sequence.

    Returns:
        dict with:
            input_ids: List[List[int]] shaped [batch, seq_len]
            attention_mask: List[List[int]] shaped [batch, seq_len], 1=token, 0=pad

    Side effects:
        None; pure transformation.
    """
    texts = examples["text"]

    # Concatenate documents if requested
    if docs_per_sequence > 1:
        # Group documents into chunks of docs_per_sequence
        concatenated_texts = []
        eos_token = tokenizer.eos_token or ""

        for i in range(0, len(texts), docs_per_sequence):
            chunk = texts[i : i + docs_per_sequence]
            # Join with EOS token separator
            concatenated = eos_token.join(chunk)
            concatenated_texts.append(concatenated)

        texts = concatenated_texts

    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=seq_len,
        padding="max_length",
        return_attention_mask=True,
    )
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}


if __name__ == "__main__":
    """
    Lightweight manual test for streaming dataloader construction.

    Prints the shapes of one batch, the first few token ids/mask entries, and
    the per-row non-padding token counts (sum of attention_mask). Intended for
    ad-hoc verification; not a unit test.
    """
    cfg = DataConfig(batch_size=16)
    loader = build_dataloader(cfg)
    first = next(iter(loader))
    ids = first["input_ids"]
    mask = first["attention_mask"]
    token_counts = mask.sum(dim=1)
    print(f"input_ids shape: {ids.shape}, attention_mask shape: {mask.shape}")
    print("input_ids[0][:20]:", ids[0, :20].tolist())
    print("attention_mask[0][:20]:", mask[0, :20].tolist())
    print("non-pad token counts per row:", token_counts.tolist())
