#!/usr/bin/env python
"""
Quick script to run evaluations on different WikiSalad datasets.

Usage:
    python run_eval.py --model qwen --dataset medium --max-examples 20
    python run_eval.py --model flan-t5 --dataset hard --max-examples 50
    python run_eval.py --model qwen --dataset very_hard  # Run all examples
"""

import argparse
from pathlib import Path
from simple_qa_evaluator import SimpleQAEvaluator, HuggingFaceModelAdapter

# Available datasets
DATASETS = {
    'easy': 'easy_2topic_short.json',
    'medium': 'medium_2topic_medium.json',
    'hard': 'hard_3topic_short.json',
    'very_hard': 'very_hard_3topic_long.json'
}

# Available models
MODELS = {
    'flan-t5-small': 'google/flan-t5-small',
    'flan-t5-base': 'google/flan-t5-base',
    'flan-t5-large': 'google/flan-t5-large',
    'qwen-0.5b': 'Qwen/Qwen2.5-0.5B-Instruct',
    'qwen-1.5b': 'Qwen/Qwen2.5-1.5B-Instruct',
    'qwen-3b': 'Qwen/Qwen2.5-3B-Instruct',
    'qwen-7b': 'Qwen/Qwen2.5-7B-Instruct',
}

def main():
    parser = argparse.ArgumentParser(description='Evaluate models on WikiSalad datasets')
    parser.add_argument('--model', type=str, required=True,
                       choices=list(MODELS.keys()),
                       help='Model to evaluate')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=list(DATASETS.keys()),
                       help='Dataset difficulty level')
    parser.add_argument('--max-examples', type=int, default=None,
                       help='Limit number of examples (default: all)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to run on (default: cpu)')

    args = parser.parse_args()

    # Get paths
    model_name = MODELS[args.model]
    dataset_file = DATASETS[args.dataset]
    dataset_path = Path(__file__).parent.parent / "artifacts/hugging_face_wikipedia/wikisalad_datasets" / dataset_file

    # Output file
    output_file = f"eval_results_{args.model}_{args.dataset}.json"

    print("="*60)
    print(f"Model: {args.model} ({model_name})")
    print(f"Dataset: {args.dataset} ({dataset_file})")
    print(f"Device: {args.device}")
    if args.max_examples:
        print(f"Max examples: {args.max_examples}")
    print(f"Output: {output_file}")
    print("="*60)
    print()

    # Load model
    print(f"Loading model: {model_name}")
    model = HuggingFaceModelAdapter(model_name, device=args.device)
    print()

    # Load dataset
    evaluator = SimpleQAEvaluator(str(dataset_path))
    print()

    # Run evaluation
    results = evaluator.evaluate(model, max_examples=args.max_examples)

    # Print and save results
    evaluator.print_summary(results)
    evaluator.save_results(results, output_file)

    print()
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()
