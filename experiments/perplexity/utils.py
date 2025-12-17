"""
Shared utilities for perplexity experiments.
"""

import time
import torch
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset

from modeling.compressor import (
    AttentionWeightedCompressor,
    AttentionWeightedZeroVCompressor,
    EncoderCompressor,
    MeanCompressor,
    MeanZeroVCompressor,
)
from modeling.qwen.luka_qwen3 import load_luka_model, set_luka_kv_params
from artifacts.prompts.prompt_loader import load_prompt


MODEL_NAME = "Qwen/Qwen3-1.7B-Base"


def load_eval_text(dataset_name: str = "wikitext", max_tokens: int = 2048, tokenizer=None) -> tuple[str, torch.Tensor]:
    """
    Load evaluation text from a standard dataset.

    Args:
        dataset_name: Dataset to load ("wikitext", "pg19", "c4")
        max_tokens: Maximum number of tokens to include
        tokenizer: Tokenizer to use for counting tokens

    Returns:
        (text, token_ids) - the text and its tokenized form
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    if dataset_name == "wikitext":
        # WikiText-103 test set - long Wikipedia articles
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        # Concatenate articles until we have enough tokens
        text = ""
        for item in dataset:
            if item["text"].strip():
                text += item["text"] + "\n"
            # Tokenize to check length
            tokens = tokenizer(text, return_tensors="pt")["input_ids"]
            if tokens.shape[1] >= max_tokens:
                break

    elif dataset_name == "pg19":
        # PG-19: Long books from Project Gutenberg (use streaming to avoid downloading 7GB)
        dataset = load_dataset("emozilla/pg19", split="test", streaming=True)
        # Take first book, it's usually long enough
        text = next(iter(dataset))["text"]

    elif dataset_name == "c4":
        # C4 validation set
        dataset = load_dataset("c4", "en", split="validation", streaming=True)
        text = ""
        for item in dataset:
            text += item["text"] + "\n"
            tokens = tokenizer(text, return_tensors="pt")["input_ids"]
            if tokens.shape[1] >= max_tokens:
                break

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'wikitext', 'pg19', or 'c4'.")

    # Tokenize and truncate to max_tokens
    token_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    if token_ids.shape[1] > max_tokens:
        token_ids = token_ids[:, :max_tokens]
        # Decode back to get the truncated text
        text = tokenizer.decode(token_ids[0], skip_special_tokens=True)

    return text, token_ids

CHECKPOINT_PATHS = [
    Path("train_1/step_1000.pt"),
    Path("artifacts/compressor_checkpoints/layer_0_step_1000.pt"),
]


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_tokenizer(model_name: str = None):
    if model_name is None:
        model_name = MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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


def get_compressor_configs(
    include_trained_encoder: bool = True,
    include_log_bias: bool = True,
    include_adaptive_k: bool = True,
    bias_comparison_mode: bool = False,
):
    """
    Get list of compressor configurations to test.

    Returns list of dicts with keys: name, label, compressor, log_bias_mode, color, linestyle, linewidth

    Args:
        include_trained_encoder: Include trained encoder compressor if checkpoint found.
        include_log_bias: Include fixed log(N) bias variants.
        include_adaptive_k: Include adaptive log(k_eff) bias variants.
        bias_comparison_mode: If True, only include Mean and Attn-Weighted with all 3 bias modes.
            Same color per compressor, different linestyles per bias mode:
            - adaptive_k: bold solid
            - fixed_n: dotted
            - none: solid (normal weight)
    """
    if bias_comparison_mode:
        # Focused comparison: same compressor = same color, different bias = different linestyle
        compressors = [
            # Mean variants (orange)
            {
                "name": "mean_logk",
                "label": "Mean + log(k)",
                "compressor": MeanCompressor(),
                "log_bias_mode": "adaptive_k",
                "color": "tab:orange",
                "linestyle": "-",
                "linewidth": 3.0,  # bold
            },
            {
                "name": "mean_logn",
                "label": "Mean + log(N)",
                "compressor": MeanCompressor(),
                "log_bias_mode": "fixed_n",
                "color": "tab:orange",
                "linestyle": ":",
                "linewidth": 1.5,
            },
            {
                "name": "mean",
                "label": "Mean",
                "compressor": MeanCompressor(),
                "log_bias_mode": "none",
                "color": "tab:orange",
                "linestyle": "-",
                "linewidth": 1.5,
            },
            # Attention-Weighted variants (blue)
            {
                "name": "attn_weighted_logk",
                "label": "Attn-Weighted + log(k)",
                "compressor": AttentionWeightedCompressor(temperature=1.0),
                "log_bias_mode": "adaptive_k",
                "color": "tab:blue",
                "linestyle": "-",
                "linewidth": 3.0,  # bold
            },
            {
                "name": "attn_weighted_logn",
                "label": "Attn-Weighted + log(N)",
                "compressor": AttentionWeightedCompressor(temperature=1.0),
                "log_bias_mode": "fixed_n",
                "color": "tab:blue",
                "linestyle": ":",
                "linewidth": 1.5,
            },
            {
                "name": "attn_weighted",
                "label": "Attn-Weighted",
                "compressor": AttentionWeightedCompressor(temperature=1.0),
                "log_bias_mode": "none",
                "color": "tab:blue",
                "linestyle": "-",
                "linewidth": 1.5,
            },
        ]
        return compressors

    # Original behavior
    compressors = [
        {
            "name": "attn_weighted",
            "label": "Attn-Weighted",
            "compressor": AttentionWeightedCompressor(temperature=1.0),
            "log_bias_mode": "none",
            "color": "tab:blue",
            "linestyle": "-",
            "linewidth": 1.5,
        },
        {
            "name": "mean",
            "label": "Mean",
            "compressor": MeanCompressor(),
            "log_bias_mode": "none",
            "color": "tab:orange",
            "linestyle": "-",
            "linewidth": 1.5,
        },
        {
            "name": "attn_weighted_zero_v",
            "label": "Attn-K, Zero-V",
            "compressor": AttentionWeightedZeroVCompressor(temperature=1.0),
            "log_bias_mode": "none",
            "color": "tab:purple",
            "linestyle": "-",
            "linewidth": 1.5,
        },
        {
            "name": "mean_zero_v",
            "label": "Mean-K, Zero-V",
            "compressor": MeanZeroVCompressor(),
            "log_bias_mode": "none",
            "color": "tab:brown",
            "linestyle": "-",
            "linewidth": 1.5,
        },
    ]

    if include_log_bias:
        compressors.extend([
            {
                "name": "attn_weighted_logn",
                "label": "Attn-Weighted + log(N)",
                "compressor": AttentionWeightedCompressor(temperature=1.0),
                "log_bias_mode": "fixed_n",
                "color": "tab:cyan",
                "linestyle": "-",
                "linewidth": 1.5,
            },
            {
                "name": "mean_logn",
                "label": "Mean + log(N)",
                "compressor": MeanCompressor(),
                "log_bias_mode": "fixed_n",
                "color": "tab:red",
                "linestyle": "-",
                "linewidth": 1.5,
            },
        ])

    if include_adaptive_k:
        compressors.extend([
            {
                "name": "attn_weighted_logk",
                "label": "Attn-Weighted + log(k)",
                "compressor": AttentionWeightedCompressor(temperature=1.0),
                "log_bias_mode": "adaptive_k",
                "color": "tab:green",
                "linestyle": "-",
                "linewidth": 1.5,
            },
            # Note: Mean doesn't have importance weights, so adaptive_k falls back to log(N)
            # Still useful to test to verify fallback behavior
            {
                "name": "mean_logk",
                "label": "Mean + log(k)",
                "compressor": MeanCompressor(),
                "log_bias_mode": "adaptive_k",
                "color": "tab:pink",
                "linestyle": "-",
                "linewidth": 1.5,
            },
        ])

    if include_trained_encoder:
        checkpoint_path = find_encoder_checkpoint()
        if checkpoint_path is not None:
            compressors.append({
                "name": "trained_encoder",
                "label": "Trained Encoder",
                "compressor": EncoderCompressor(checkpoint_path=str(checkpoint_path)),
                "log_bias_mode": "none",
                "color": "tab:gray",
                "linestyle": "-",
                "linewidth": 1.5,
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
    max_new_tokens: int = 128,
    model_name: str = None,
):
    """
    Generate a greedy rollout using raw attention (no compression).
    Returns token ids (including prompt) and decoded text.
    """
    if model_name is None:
        model_name = MODEL_NAME

    set_luka_kv_params(
        default_tail_len=16,
        min_compress_chunk=16,
        max_pages=15,
        use_exact_attention=True,  # Bypass cover view, use raw attention
        compressor="mean",
        segmenter="dummy",
        create_pages_in_generation=False,
    )

    model = load_luka_model(
        model_name,
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
    has_nan = False

    # First prediction: use prefill logits (no extra forward pass needed)
    logits = pre_out.logits[:, -1, :]
    target = rollout_ids[:, prompt_len]
    # Check for NaN/Inf in logits
    if logits.isnan().any() or logits.isinf().any():
        has_nan = True
    log_probs = torch.log_softmax(logits.float(), dim=-1)  # float32 for stability
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
        # Check for NaN/Inf in logits
        if logits.isnan().any() or logits.isinf().any():
            has_nan = True
        log_probs = torch.log_softmax(logits.float(), dim=-1)  # float32 for stability
        nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        nll_list.append(nll)
        past_key_values = out.past_key_values

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    tps = total_tokens / max(elapsed, 1e-8)

    nll_tensor = torch.stack(nll_list, dim=1)

    # If we detected NaN/Inf during forward pass, return inf perplexity
    if has_nan or nll_tensor.isnan().any() or nll_tensor.isinf().any():
        print("  [WARNING: NaN/Inf detected in logits, returning inf perplexity]")
        return float('inf'), [float('inf')] * total_tokens, [float('inf')] * total_tokens, tps

    total_tokens_tensor = torch.tensor([[total_tokens]], device=device, dtype=nll_tensor.dtype)
    total_nll = nll_tensor.sum(dim=1, keepdim=True) / total_tokens_tensor
    ppl = torch.exp(total_nll)[0, 0].item()

    # Running average perplexity curve
    cumsum = nll_tensor.cumsum(dim=1)
    counts = torch.arange(1, total_tokens + 1, device=device).unsqueeze(0)
    avg_nll = cumsum / counts
    avg_curve = torch.exp(avg_nll)[0].tolist()

    # Per-token (instantaneous) perplexity curve
    token_curve = torch.exp(nll_tensor)[0].tolist()

    return ppl, avg_curve, token_curve, tps


def get_baseline_perplexity(rollout_ids: torch.Tensor, prompt_len: int, device: str, model_name: str = None):
    """Run baseline (raw attention, no compression) perplexity evaluation.

    Returns:
        ppl: float - overall perplexity
        avg_curve: list[float] - running average perplexity at each token
        token_curve: list[float] - per-token instantaneous perplexity
        tps: float - tokens per second
    """
    if model_name is None:
        model_name = MODEL_NAME

    set_luka_kv_params(
        default_tail_len=16,
        min_compress_chunk=16,
        max_pages=15,
        use_exact_attention=True,  # Bypass cover view, use raw attention
        compressor="mean",
        segmenter="dummy",
        create_pages_in_generation=False,
        production_mode=True,
    )

    model = load_luka_model(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    ).to(device)
    model.eval()

    with torch.no_grad():
        ppl, avg_curve, token_curve, tps = prefill_then_decode_perplexity(model, rollout_ids, prompt_len)

    del model
    torch.cuda.empty_cache()

    return ppl, avg_curve, token_curve, tps


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
    use_log_bias: bool = None,  # DEPRECATED: use log_bias_mode
    log_bias_mode: str = None,  # "none", "fixed_n", or "adaptive_k"
    threshold: float = None,
    refinement_rule: str = "threshold",
    refinement_rule_kwargs: dict = None,
    model_name: str = None,
):
    """
    Run perplexity evaluation with a single compressor configuration.

    Args:
        rollout_ids: Token IDs to evaluate
        prompt_len: Length of prompt (prefix)
        device: Device to run on
        compressor: Compressor instance or string
        use_log_bias: DEPRECATED. Use log_bias_mode instead.
        log_bias_mode: Log bias mode ("none", "fixed_n", or "adaptive_k")
        threshold: (Deprecated) Threshold for refinement, use refinement_rule instead
        refinement_rule: Refinement rule name ("threshold", "top_k", "top_p", etc.)
        refinement_rule_kwargs: Kwargs for refinement rule.
            e.g. {"k": 3, "always_refine_first_n": 1} for top_k with attention sinks

    Returns dict with: perplexity, curve, tokens_per_sec, summary_frac, pages
    """
    # Build refinement kwargs
    if refinement_rule_kwargs is None:
        refinement_rule_kwargs = {}

    # Backwards compat: if threshold provided and rule is threshold, use it
    if threshold is not None and refinement_rule == "threshold" and "threshold" not in refinement_rule_kwargs:
        refinement_rule_kwargs["threshold"] = threshold

    # Handle log_bias_mode vs use_log_bias (backwards compat)
    effective_log_bias_mode = log_bias_mode
    if effective_log_bias_mode is None:
        if use_log_bias is True:
            effective_log_bias_mode = "fixed_n"
        elif use_log_bias is False:
            effective_log_bias_mode = "none"
        else:
            effective_log_bias_mode = "none"  # default

    if model_name is None:
        model_name = MODEL_NAME

    set_luka_kv_params(
        default_tail_len=16,
        min_compress_chunk=16,
        max_pages=15,
        refinement_rule=refinement_rule,
        refinement_rule_kwargs=refinement_rule_kwargs,
        compressor=compressor,
        log_bias_mode=effective_log_bias_mode,
        segmenter="dummy",
        segment_interval=16,
        create_pages_in_generation=True,
        production_mode=True
    )

    model = load_luka_model(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    ).to(device)
    model.eval()

    with torch.no_grad():
        ppl, avg_curve, token_curve, tps = prefill_then_decode_perplexity(model, rollout_ids, prompt_len)

    summary_frac, stats = get_stats_from_model(model)

    del model
    torch.cuda.empty_cache()

    return {
        "perplexity": ppl,
        "curve": avg_curve,           # Backwards compat alias
        "avg_curve": avg_curve,
        "token_curve": token_curve,
        "tokens_per_sec": tps,
        "summary_frac": summary_frac,
        "pages": stats.get("avg_pages_per_layer", 0),
        "page_selection_counts": stats.get("page_selection_counts", {}),
    }
