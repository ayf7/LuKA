"""
Main Evaluation Runner

Command-line interface for running LuKA evaluations.
Supports both baseline and LuKA-enabled models.

Example usage:
    # Run simple Q&A evaluation with local Qwen model
    python -m evaluation.run_evaluation \
        --dataset artifacts/hugging_face_wikipedia/wikisalad_eval_example.json \
        --model-type huggingface \
        --model-name Qwen/Qwen2.5-3B-Instruct \
        --eval-mode simple \
        --output results/baseline_simple.json

    # Run sequential evaluation with API model
    python -m evaluation.run_evaluation \
        --dataset artifacts/hugging_face_wikipedia/wikisalad_eval_example.json \
        --model-type api \
        --model-name qwen-turbo \
        --api-type dashscope \
        --eval-mode sequential \
        --output results/api_sequential.json

    # Run both modes
    python -m evaluation.run_evaluation \
        --dataset artifacts/hugging_face_wikipedia/wikisalad_eval_example.json \
        --model-type huggingface \
        --model-name Qwen/Qwen2.5-3B-Instruct \
        --eval-mode both \
        --output results/baseline.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from evaluation.evaluator import QAEvaluator, GenerationConfig
from evaluation.model_implementations import HuggingFaceModel, APIModel, LuKAQwenModel


def load_model(args):
    """Load model based on command-line arguments."""
    print(f"\n{'='*60}")
    print(f"Loading model: {args.model_name}")
    print(f"Type: {args.model_type}")
    print(f"{'='*60}\n")

    if args.model_type == "huggingface":
        return HuggingFaceModel(
            model_name=args.model_name,
            device=args.device
        )

    elif args.model_type == "api":
        return APIModel(
            api_type=args.api_type,
            model_name=args.model_name,
            api_key=args.api_key,
            base_url=args.api_base_url
        )

    elif args.model_type == "luka":
        # Load LuKA configuration if provided
        luka_config = None
        if args.luka_config:
            with open(args.luka_config, 'r') as f:
                luka_config = json.load(f)

        return LuKAQwenModel(
            model_name=args.model_name,
            device=args.device,
            luka_config=luka_config
        )

    else:
        raise ValueError(f"Unknown model type: {args.model_type}")


def run_evaluation(args):
    """Run the evaluation pipeline."""
    # Load model
    model = load_model(args)

    # Load evaluator
    evaluator = QAEvaluator(args.dataset)

    # Setup generation config
    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        do_sample=args.temperature > 0
    )

    # Run evaluation(s)
    results = []

    if args.eval_mode in ["simple", "both"]:
        print("\n" + "="*60)
        print("RUNNING SIMPLE Q&A EVALUATION")
        print("="*60)
        result = evaluator.evaluate_simple_qa(
            model=model,
            config=gen_config,
            num_examples=args.num_examples
        )
        results.append(result)

        # Save results
        output_path = args.output.replace('.json', '_simple.json') if args.eval_mode == "both" else args.output
        evaluator.save_results(result, output_path)

    if args.eval_mode in ["sequential", "both"]:
        print("\n" + "="*60)
        print("RUNNING SEQUENTIAL MULTI-ANSWER EVALUATION")
        print("="*60)
        result = evaluator.evaluate_sequential_multi_answer(
            model=model,
            config=gen_config,
            num_examples=args.num_examples
        )
        results.append(result)

        # Save results
        output_path = args.output.replace('.json', '_sequential.json') if args.eval_mode == "both" else args.output
        evaluator.save_results(result, output_path)

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {args.output}")

    # Print summary
    for result in results:
        print(f"\n{result.eval_mode.upper()} MODE:")
        print(f"  Model: {result.model_name}")
        print(f"  Examples evaluated: {result.num_examples}")
        print(f"  Total predictions: {len(result.predictions)}")
        if result.compression_stats:
            avg_ratio = sum(s['compression_ratio'] for s in result.compression_stats) / len(result.compression_stats)
            print(f"  Avg compression ratio: {avg_ratio:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Run LuKA evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Dataset args
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to WikiSalad evaluation dataset JSON"
    )

    # Model args
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["huggingface", "api", "luka"],
        help="Type of model to use"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name/ID (e.g., 'Qwen/Qwen2.5-3B-Instruct' or 'qwen-turbo')"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for local models (default: auto)"
    )

    # API-specific args
    parser.add_argument(
        "--api-type",
        type=str,
        choices=["openai", "dashscope"],
        help="API type (required if model-type is 'api')"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key (or set via environment variable)"
    )
    parser.add_argument(
        "--api-base-url",
        type=str,
        help="Base URL for API (optional)"
    )

    # LuKA-specific args
    parser.add_argument(
        "--luka-config",
        type=str,
        help="Path to LuKA configuration JSON (optional)"
    )

    # Evaluation args
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="both",
        choices=["simple", "sequential", "both"],
        help="Evaluation mode (default: both)"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        help="Number of examples to evaluate (default: all)"
    )

    # Generation args
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Max tokens to generate (default: 100)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for deterministic)"
    )

    # Output args
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for results JSON"
    )

    args = parser.parse_args()

    # Validate API args
    if args.model_type == "api" and not args.api_type:
        parser.error("--api-type is required when using API models")

    # Run evaluation
    try:
        run_evaluation(args)
    except Exception as e:
        print(f"\nError during evaluation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
