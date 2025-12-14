"""
Evaluate baseline (raw attention) mode only.

Usage:
    python experiments/eval_baseline.py --model Qwen/Qwen3-1.7B-Base --prompt paragraphs_1
    python experiments/eval_baseline.py --model Qwen/Qwen3-1.7B-Base \
        --prompt artifacts/hugging_face_wikipedia/wikisalad_datasets/easy_2topic_short.json \
        --use-wikisalad --wikisalad-example-id 0
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from artifacts.prompts.prompt_loader import load_prompt
from modeling.compressor import EncoderCompressor, MeanCompressor
from modeling.qwen.luka_qwen3 import (
    load_luka_model,
    set_luka_kv_params,
)
from modeling.segmenter import DummySegmenter

# Import shared functions from comprehensive_eval
import sys
sys.path.insert(0, str(Path(__file__).parent))
from comprehensive_eval import (
    get_compression_stats,
    prefill_then_decode_perplexity,
    generate_text,
    load_wikisalad_prompt,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline (raw attention) mode")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B-Base", help="Model name")
    parser.add_argument("--prompt", type=str, default="paragraphs_1", help="Prompt name")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prompt-len", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="experiments/eval_results")
    parser.add_argument("--use-wikisalad", action="store_true")
    parser.add_argument("--wikisalad-example-id", type=int, default=0)
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load prompt
    if args.use_wikisalad:
        prompt = load_wikisalad_prompt(args.prompt, args.wikisalad_example_id)
        dataset_name = Path(args.prompt).stem
    else:
        prompt = load_prompt(args.prompt)
        dataset_name = args.prompt
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize for perplexity
    inputs = tokenizer(prompt, return_tensors="pt")
    if args.prompt_len is None:
        prompt_len = inputs["input_ids"].shape[1] // 2
    
    # Generate baseline rollout
    print("Generating baseline rollout...")
    set_luka_kv_params(
        default_tail_len=16,
        min_compress_chunk=16,
        max_pages=15,
        refine_threshold=-1.0,  # Force raw attention
        compressor=None,
        segmenter="dummy",
    )
    
    baseline_model = load_luka_model(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    ).to(device)
    baseline_model.eval()
    
    with torch.no_grad():
        baseline_gen = baseline_model.generate(
            **{k: v.to(device) for k, v in inputs.items()},
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            use_cache=True,
        )
    baseline_rollout = baseline_gen[0].unsqueeze(0)
    del baseline_model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # Evaluate baseline mode
    print("\n" + "="*80)
    print("Evaluating: Baseline (raw attention)")
    print("="*80)
    
    set_luka_kv_params(
        default_tail_len=16,
        min_compress_chunk=16,
        max_pages=15,
        refine_threshold=-1.0,  # Raw attention
        compressor=None,
        segmenter="dummy",
    )
    
    model = load_luka_model(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    ).to(device)
    model.eval()
    
    # Perplexity
    print("  Computing perplexity...")
    rollout_ids = baseline_rollout.to(device)
    ppl, ppl_curve, tps, comp_stats = prefill_then_decode_perplexity(model, rollout_ids, prompt_len)
    
    # Generation
    print("  Generating text...")
    generated_text, gen_stats = generate_text(model, tokenizer, prompt, args.max_new_tokens)
    
    results = {
        "baseline": {
            "description": "Baseline (raw attention)",
            "perplexity": ppl,
            "perplexity_curve": ppl_curve,
            "tokens_per_sec": tps,
            "generated_text": generated_text,
            "compression_stats": comp_stats,
            "generation_stats": gen_stats,
        }
    }
    
    print(f"  Perplexity: {ppl:.3f}")
    print(f"  Compression ratio: {comp_stats['compression_ratio']:.2f}x")
    print(f"  Tokens/sec: {tps:.2f}")
    
    # Save
    results_file = output_path / f"eval_baseline_{dataset_name}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
