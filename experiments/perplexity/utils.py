"""
Shared utilities for perplexity experiments.
"""

import time
import torch
from pathlib import Path
from transformers import AutoTokenizer

from modeling.compressor import (
    AttentionWeightedCompressor,
    EncoderCompressor,
    EvictionCompressor,
    MeanCompressor,
)
from modeling.qwen.luka_qwen3 import load_luka_model, set_luka_kv_params
from artifacts.prompts.prompt_loader import load_prompt


MODEL_NAME = "Qwen/Qwen3-1.7B-Base"

CHECKPOINT_PATHS = [
    Path("train_1/step_1000.pt"),
    Path("artifacts/compressor_checkpoints/layer_0_step_1000.pt"),
]


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_prompt(name: str = "paragraphs_1") -> str:
    paragraph = load_prompt(name)
    if isinstance(paragraph, list):
        paragraph = paragraph[0]
    return paragraph


def find_encoder_checkpoint():
    """Find trained encoder checkpoint if available."""
    for cp in CHECKPOINT_PATHS:
        if cp.exists():
            return cp
    return None


def get_compressor_configs(include_trained_encoder: bool = True, include_log_bias: bool = True):
    """
    Get list of compressor configurations to test.

    Returns list of dicts with keys: name, label, compressor, use_log_bias, color
    """
    compressors = [
        {
            "name": "attn_weighted",
            "label": "Attn-Weighted",
            "compressor": AttentionWeightedCompressor(temperature=1.0),
            "use_log_bias": False,
            "color": "tab:blue",
        },
        {
            "name": "mean",
            "label": "Mean",
            "compressor": MeanCompressor(),
            "use_log_bias": False,
            "color": "tab:orange",
        },
        {
            "name": "eviction",
            "label": "Eviction (attn-K, zero-V)",
            "compressor": EvictionCompressor(temperature=1.0),
            "use_log_bias": False,
            "color": "tab:purple",
        },
    ]

    if include_log_bias:
        compressors.extend([
            {
                "name": "attn_weighted_logbias",
                "label": "Attn-Weighted + log(N)",
                "compressor": AttentionWeightedCompressor(temperature=1.0),
                "use_log_bias": True,
                "color": "tab:cyan",
            },
            {
                "name": "mean_logbias",
                "label": "Mean + log(N)",
                "compressor": MeanCompressor(),
                "use_log_bias": True,
                "color": "tab:red",
            },
            {
                "name": "eviction_logbias",
                "label": "Eviction + log(N)",
                "compressor": EvictionCompressor(temperature=1.0),
                "use_log_bias": True,
                "color": "tab:pink",
            },
        ])

    if include_trained_encoder:
        checkpoint_path = find_encoder_checkpoint()
        if checkpoint_path is not None:
            compressors.append({
                "name": "trained_encoder",
                "label": "Trained Encoder",
                "compressor": EncoderCompressor(checkpoint_path=str(checkpoint_path)),
                "use_log_bias": False,
                "color": "tab:green",
            })
            # Note: log(N) bias is catastrophic with trained encoder, not included
            print(f"Found trained encoder at {checkpoint_path}")
        else:
            print("No trained encoder checkpoint found, skipping.")

    return compressors


def generate_baseline_rollout(
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 128
):
    """
    Generate a greedy rollout using raw attention (no compression).
    Returns token ids (including prompt) and decoded text.
    """
    set_luka_kv_params(
        default_tail_len=16,
        min_compress_chunk=16,
        max_pages=15,
        refine_threshold=-1,  # Negative = exact raw attention path
        compressor="mean",
        segmenter="dummy",
        create_pages_in_generation=False,
    )

    model = load_luka_model(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    ).to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    text = tokenizer.decode(gen[0], skip_special_tokens=True)

    del model
    torch.cuda.empty_cache()

    return gen, text


def prefill_then_decode_perplexity(model, rollout_ids: torch.Tensor, prompt_len: int):
    """
    Prefill with the prompt to build LuKA pages, then teacher-force through the generated tail.

    Returns:
        perplexity: float - overall perplexity on generated tail
        curve: list[float] - cumulative perplexity at each token position
        tps: float - tokens per second
    """
    device = rollout_ids.device
    B, T = rollout_ids.shape
    assert B == 1

    pre_ids = rollout_ids[:, :prompt_len]
    pre_mask = torch.ones_like(pre_ids)
    with torch.no_grad():
        pre_out = model(
            input_ids=pre_ids,
            attention_mask=pre_mask,
            use_cache=True,
            output_attentions=True,
        )
    past_key_values = pre_out.past_key_values

    nll_list = []
    total_tokens = T - prompt_len
    start_time = time.perf_counter()

    # First prediction: use prefill logits (no extra forward pass needed)
    logits = pre_out.logits[:, -1, :]
    target = rollout_ids[:, prompt_len]
    log_probs = torch.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    nll_list.append(nll)

    # Subsequent predictions: feed each generated token
    for t in range(prompt_len, T - 1):
        cur_id = rollout_ids[:, t : t + 1]
        attn_mask = torch.ones(1, t + 1, device=device, dtype=rollout_ids.dtype)
        with torch.no_grad():
            out = model(
                input_ids=cur_id,
                attention_mask=attn_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        logits = out.logits[:, -1, :]
        target = rollout_ids[:, t + 1]
        log_probs = torch.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        nll_list.append(nll)
        past_key_values = out.past_key_values

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    tps = total_tokens / max(elapsed, 1e-8)

    nll_tensor = torch.stack(nll_list, dim=1)
    total_tokens_tensor = torch.tensor([[total_tokens]], device=device, dtype=nll_tensor.dtype)
    total_nll = nll_tensor.sum(dim=1, keepdim=True) / total_tokens_tensor
    ppl = torch.exp(total_nll)[0, 0].item()

    cumsum = nll_tensor.cumsum(dim=1)
    counts = torch.arange(1, total_tokens + 1, device=device).unsqueeze(0)
    avg_nll = cumsum / counts
    curve = torch.exp(avg_nll)[0].tolist()

    return ppl, curve, tps


def get_baseline_perplexity(rollout_ids: torch.Tensor, prompt_len: int, device: str):
    """Run baseline (raw attention, no compression) perplexity evaluation."""
    set_luka_kv_params(
        default_tail_len=16,
        min_compress_chunk=16,
        max_pages=15,
        refine_threshold=-1,  # Negative = exact raw attention path
        compressor="mean",
        segmenter="dummy",
        create_pages_in_generation=False,
        production_mode=True
    )

    model = load_luka_model(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    ).to(device)
    model.eval()

    with torch.no_grad():
        ppl, curve, tps = prefill_then_decode_perplexity(model, rollout_ids, prompt_len)

    del model
    torch.cuda.empty_cache()

    return ppl, curve, tps


def get_stats_from_model(model):
    """
    Extract refinement statistics from the model.

    Returns:
        summary_frac: float or None - fraction of summaries that were NOT refined
        stats: dict - full stats from controller
    """
    if hasattr(model, "model") and hasattr(model.model, "luka_kv_controller"):
        stats = model.model.luka_kv_controller.get_refinement_stats()
        denom = stats.get("total_summaries_seen", 0)
        refinements = stats.get("total_refinements_made", 0)
        summary_frac = 1.0 - (refinements / denom) if denom > 0 else None
        return summary_frac, stats
    return None, {}


def run_single_config(
    rollout_ids: torch.Tensor,
    prompt_len: int,
    device: str,
    compressor,
    use_log_bias: bool,
    threshold: float,
):
    """
    Run perplexity evaluation with a single compressor configuration.

    Returns dict with: perplexity, curve, tokens_per_sec, summary_frac, pages
    """
    set_luka_kv_params(
        default_tail_len=16,
        min_compress_chunk=16,
        max_pages=15,
        refine_threshold=threshold,
        compressor=compressor,
        use_log_bias=use_log_bias,
        segmenter="dummy",
        segment_interval=16,
        create_pages_in_generation=True,
        production_mode=True
    )

    model = load_luka_model(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    ).to(device)
    model.eval()

    with torch.no_grad():
        ppl, curve, tps = prefill_then_decode_perplexity(model, rollout_ids, prompt_len)

    summary_frac, stats = get_stats_from_model(model)

    del model
    torch.cuda.empty_cache()

    return {
        "perplexity": ppl,
        "curve": curve,
        "tokens_per_sec": tps,
        "summary_frac": summary_frac,
        "pages": stats.get("avg_pages_per_layer", 0),
    }
