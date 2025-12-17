#!/usr/bin/env python
"""
Unified LuKA Evaluation Runner

Single entrypoint for all evaluation scenarios: baseline, LuKA, and API models.

Usage Examples:
    # Baseline HuggingFace model
    python -m evaluation.run_eval --model-type baseline --model qwen-3b --dataset easy

    # LuKA with top-down attention
    python -m evaluation.run_eval --model-type luka --model qwen-1.7b --attention top_down --dataset hard

    # LuKA with lined attention + scoring
    python -m evaluation.run_eval --model-type luka --attention lined --dataset medium --score

    # LuKA + compare against baseline
    python -m evaluation.run_eval --model-type luka --attention mix --compare baseline.json --dataset hard

    # API model (OpenAI)
    python -m evaluation.run_eval --model-type api --api-type openai --model gpt-4 --dataset easy

Available Models:
    Baseline: qwen-0.5b, qwen-1.5b, qwen-3b, qwen-7b (or any HuggingFace model ID)
    LuKA: qwen-1.7b (or any Qwen3 model ID)

Available Datasets:
    easy, medium, hard, very_hard (or path to custom JSON)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from evaluation.evaluator import QAEvaluator
from evaluation.model_interface import GenerationConfig
from evaluation.model_implementations import HuggingFaceModel, APIModel, LuKAQwenModel
from evaluation.metrics import (
    compute_exact_match,
    compute_f1,
    compute_qa_accuracy,
    resolve_dataset_path,
    save_results,
)


# ============================================================================
# Predefined Models and Datasets
# ============================================================================

BASELINE_MODELS = {
    'qwen-0.5b': 'Qwen/Qwen2.5-0.5B-Instruct',
    'qwen-1.5b': 'Qwen/Qwen2.5-1.5B-Instruct',
    'qwen-3b': 'Qwen/Qwen2.5-3B-Instruct',
    'qwen-7b': 'Qwen/Qwen2.5-7B-Instruct',
    'flan-t5-small': 'google/flan-t5-small',
    'flan-t5-base': 'google/flan-t5-base',
}

LUKA_MODELS = {
    'qwen-1.7b': 'Qwen/Qwen3-1.7B-Base',
    'qwen-4b': 'Qwen/Qwen3-4B-Instruct-2507',
    'qwen-4b-instruct': 'Qwen/Qwen3-4B-Instruct-2507',
}

DATASETS = {
    'easy': 'easy_2topic_short.json',
    'medium': 'medium_2topic_medium.json',
    'hard': 'hard_3topic_short.json',
    'very_hard': 'very_hard_3topic_long.json',
}


# ============================================================================
# Model Loading
# ============================================================================

def load_model(args):
    """Load model based on command-line arguments."""
    print(f"\n{'='*60}")
    print(f"Loading Model")
    print(f"{'='*60}")
    print(f"Type: {args.model_type}")
    print(f"Model: {args.model}")

    if args.model_type == "baseline":
        model_name = BASELINE_MODELS.get(args.model, args.model)
        print(f"Resolved: {model_name}")
        print(f"Device: {args.device}")
        print(f"{'='*60}\n")

        return HuggingFaceModel(model_name=model_name, device=args.device)

    elif args.model_type == "luka":
        model_name = LUKA_MODELS.get(args.model, args.model)
        print(f"Resolved: {model_name}")
        print(f"Attention: {args.attention}")
        print(f"Device: {args.device}")

        luka_config = {
            "compressor": args.compressor,
            "segmenter": args.segmenter,
            "segment_interval": args.segment_interval,
            "refinement_rule": args.refinement_rule,
            "refinement_rule_kwargs": {"k": args.refinement_k},
            "log_bias_mode": args.log_bias,
            # H2O-style parameters
            "heavy_ratio": args.heavy_ratio,
            "recent_ratio": args.recent_ratio,
        }
        print(f"LuKA Config:")
        print(f"  Compressor: {args.compressor}")
        print(f"  Segmenter: {args.segmenter} (min_chunk=8, tail_len=128, max_pages=256)")
        print(f"  Segment Interval: {args.segment_interval}")
        print(f"  Refinement: {args.refinement_rule}" + (f" (k={args.refinement_k})" if args.refinement_rule != "none" else ""))
        print(f"  Log Bias: {args.log_bias}")
        print(f"  H2O (lined): heavy_ratio={args.heavy_ratio}, recent_ratio={args.recent_ratio}")
        print(f"{'='*60}\n")

        lined_layers = None
        if args.lined_layers:
            lined_layers = [int(x.strip()) for x in args.lined_layers.split(",")]

        return LuKAQwenModel(
            model_name=model_name,
            device=args.device,
            attention_mode=args.attention,
            lined_layers=lined_layers,
            luka_config=luka_config,
        )

    elif args.model_type == "api":
        if not args.api_type:
            raise ValueError("--api-type is required for API models")
        print(f"API Type: {args.api_type}")
        print(f"{'='*60}\n")

        return APIModel(
            api_type=args.api_type,
            model_name=args.model,
            api_key=args.api_key,
            base_url=args.api_base_url,
        )

    else:
        raise ValueError(f"Unknown model type: {args.model_type}")


# ============================================================================
# Scoring
# ============================================================================

def score_predictions(result, dataset) -> Dict[str, Any]:
    """Score predictions against ground truth."""
    gt_lookup = {ex['example_id']: ex for ex in dataset}

    scores = []
    per_example = []

    for pred in result.predictions:
        example_id = pred['example_id']
        example = gt_lookup.get(example_id)
        if example is None:
            continue

        questions = example['questions']
        example_scores = []

        for i, (pred_answer, q_data) in enumerate(zip(pred['answers'], questions)):
            answers_data = q_data['answers']
            if isinstance(answers_data, dict) and 'text' in answers_data:
                true_answers = answers_data['text']
            else:
                true_answers = answers_data
            if not isinstance(true_answers, list):
                true_answers = [true_answers]

            em = compute_exact_match(pred_answer, true_answers)
            f1 = compute_f1(pred_answer, true_answers)
            qa = compute_qa_accuracy(pred_answer, true_answers)

            example_scores.append({'question_idx': i, 'exact_match': em, 'f1': f1, 'qa_accuracy': qa})
            scores.append({'em': em, 'f1': f1, 'qa': qa})

        per_example.append({
            'example_id': example_id,
            'scores': example_scores,
            'avg_em': np.mean([s['exact_match'] for s in example_scores]),
            'avg_f1': np.mean([s['f1'] for s in example_scores]),
            'avg_qa': np.mean([s['qa_accuracy'] for s in example_scores]),
        })

    aggregate = {
        'exact_match': np.mean([s['em'] for s in scores]) if scores else 0.0,
        'f1_score': np.mean([s['f1'] for s in scores]) if scores else 0.0,
        'qa_accuracy': np.mean([s['qa'] for s in scores]) if scores else 0.0,
        'num_questions': len(scores),
        'num_examples': len(per_example),
    }

    return {'aggregate': aggregate, 'per_example': per_example}


# ============================================================================
# Comparison
# ============================================================================

def compare_results(current_results: Dict, baseline_path: str):
    """Compare current results against baseline."""
    try:
        from evaluation.compare_results import create_summary_table, analyze_per_example
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
        print("\n" + create_summary_table(baseline, current_results))
        print(analyze_per_example(baseline, current_results))
    except ImportError:
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
        baseline_agg = baseline.get('aggregate', {})
        current_agg = current_results.get('aggregate', {})
        print(f"\n{'Metric':<25} {'Baseline':<15} {'Current':<15} {'Delta':<15}")
        print("-" * 70)
        for metric in ['exact_match', 'f1_score', 'qa_accuracy']:
            b_val = baseline_agg.get(metric, 0)
            c_val = current_agg.get(metric, 0)
            delta = c_val - b_val
            print(f"{metric:<25} {b_val:.3f}{'':<9} {c_val:.3f}{'':<9} {delta:+.3f}")
        print("=" * 60)


# ============================================================================
# Main Evaluation Pipeline
# ============================================================================

def run_evaluation(args):
    """Run the evaluation pipeline."""
    # Resolve dataset path
    if args.dataset in DATASETS:
        dataset_path = resolve_dataset_path(DATASETS[args.dataset])
    else:
        dataset_path = resolve_dataset_path(args.dataset)

    print(f"\n{'='*60}")
    print("Evaluation Configuration")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_path.name}")
    print(f"Model Type: {args.model_type}")
    print(f"Eval Mode: {args.eval_mode}")
    print(f"Max Examples: {args.max_examples or 'all'}")
    print(f"{'='*60}")

    model = load_model(args)
    evaluator = QAEvaluator(str(dataset_path))

    with open(dataset_path, 'r') as f:
        raw_dataset = json.load(f)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
    )

    if args.output:
        output_path = Path(args.output)
    else:
        attn_tag = args.attention if args.model_type == "luka" else "baseline"
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        output_path = results_dir / f"eval_{attn_tag}_{args.dataset}.json"

    all_results = []

    if args.eval_mode in ["simple", "both"]:
        print("\n" + "=" * 60)
        print("RUNNING SIMPLE Q&A EVALUATION")
        print("=" * 60)
        result = evaluator.evaluate_simple_qa(model=model, config=gen_config, num_examples=args.max_examples)
        all_results.append(('simple', result))

    if args.eval_mode in ["sequential", "both"]:
        print("\n" + "=" * 60)
        print("RUNNING SEQUENTIAL MULTI-ANSWER EVALUATION")
        print("=" * 60)
        result = evaluator.evaluate_sequential_multi_answer(model=model, config=gen_config, num_examples=args.max_examples)
        all_results.append(('sequential', result))

    if args.eval_mode == "batched":
        print("\n" + "=" * 60)
        print("RUNNING BATCHED EVALUATION")
        print("=" * 60)
        result = evaluator.evaluate_batched(
            model=model,
            config=gen_config,
            num_examples=args.max_examples,
            batch_size=args.batch_size,
        )
        all_results.append(('batched', result))

    for mode, result in all_results:
        output_data = result.to_dict()
        output_data['model_info'] = model.get_model_info()

        if args.score:
            print(f"\n{'='*60}")
            print(f"SCORING ({mode.upper()} mode)")
            print(f"{'='*60}")
            scores = score_predictions(result, raw_dataset)
            output_data['scores'] = scores
            agg = scores['aggregate']
            print(f"\nResults:")
            print(f"  Exact Match: {agg['exact_match']:.1%}")
            print(f"  F1 Score:    {agg['f1_score']:.1%}")
            print(f"  QA Accuracy: {agg['qa_accuracy']:.1%}")

        suffix = f"_{mode}" if len(all_results) > 1 else ""
        final_output = str(output_path).replace('.json', f'{suffix}.json')
        save_results(output_data, final_output)

        if args.compare and args.score:
            compare_results(output_data['scores'], args.compare)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    for mode, result in all_results:
        print(f"\n{mode.upper()} MODE:")
        print(f"  Model: {result.model_name}")
        print(f"  Examples: {result.num_examples}")
        if result.compression_stats:
            ratios = [s.get('compression_ratio', 1.0) for s in result.compression_stats]
            print(f"  Avg Compression: {np.mean(ratios):.2f}x")
        if result.performance_stats:
            ps = result.performance_stats
            print(f"  Performance:")
            print(f"    Total tokens generated: {ps.total_tokens_generated}")
            print(f"    Decode throughput: {ps.decode_throughput_tps:.1f} tokens/sec")
            print(f"    Decode latency: {ps.decode_latency_ms_per_token:.2f} ms/token")
            print(f"    Peak memory (allocated): {ps.peak_memory_allocated_mb:.1f} MB")
    print(f"\nResults saved to: {output_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified LuKA Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model arguments
    parser.add_argument("--model-type", type=str, required=True, choices=["baseline", "luka", "api"])
    parser.add_argument("--model", type=str, required=True, help="Model name/ID")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"])

    # LuKA-specific
    parser.add_argument("--attention", type=str, default="top_down", choices=["baseline", "top_down", "lined", "mix"])
    parser.add_argument("--lined-layers", type=str, default=None, help="Comma-separated layer indices")
    parser.add_argument("--compressor", type=str, default="attention_weighted", choices=["mean", "attention_weighted"])
    parser.add_argument("--segmenter", type=str, default="dummy")
    parser.add_argument("--segment-interval", type=int, default=1)
    parser.add_argument("--refinement-rule", type=str, default="top_k", choices=["none", "threshold", "top_k", "top_p", "top_frac"])
    parser.add_argument("--refinement-k", type=int, default=3, help="K for top_k refinement rule")
    parser.add_argument("--log-bias", type=str, default="adaptive_k", choices=["none", "fixed_n", "adaptive_k"])
    # H2O-style parameters (for lined attention)
    parser.add_argument("--heavy-ratio", type=float, default=0.1, help="Fraction of tokens to keep as heavy hitters (H2O)")
    parser.add_argument("--recent-ratio", type=float, default=0.1, help="Fraction of tokens to keep as recent window (H2O)")

    # API-specific
    parser.add_argument("--api-type", type=str, choices=["openai", "dashscope"])
    parser.add_argument("--api-key", type=str)
    parser.add_argument("--api-base-url", type=str)

    # Dataset
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-examples", type=int, default=None)

    # Evaluation
    parser.add_argument("--eval-mode", type=str, default="simple", choices=["simple", "sequential", "batched", "both"])
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for batched eval mode")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)

    # Output
    parser.add_argument("--output", type=str)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--compare", type=str, help="Path to baseline results for comparison")

    args = parser.parse_args()

    if args.model_type == "api" and not args.api_type:
        parser.error("--api-type is required for API models")
    if args.compare and not args.score:
        parser.error("--compare requires --score")

    try:
        run_evaluation(args)
    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
