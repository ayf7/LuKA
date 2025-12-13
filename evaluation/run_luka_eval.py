#!/usr/bin/env python
"""
Quick script to run evaluations for LuKA on different WikiSalad datasets.

Usage:
    For Lined Attention Only:
        python run_luka_eval.py --model qwen --use_lined_attn --dataset easy --max-examples 20
    
    For Ours:
        python run_luka_eval.py --model qwen --use_lined_attn --dataset easy
"""

import argparse
from pathlib import Path
from simple_qa_evaluator import SimpleQAEvaluator, HuggingFaceModelAdapter
import torch
from modeling.qwen.luka_qwen3 import load_luka_model, set_luka_kv_params

# Configuration
model_name = "Qwen/Qwen3-1.7B-Base"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Available datasets
DATASETS = {
    'easy': 'easy_2topic_short.json',
    'medium': 'medium_2topic_medium.json',
    'hard': 'hard_3topic_short.json',
    'very_hard': 'very_hard_3topic_long.json'
}

# Available models
MODELS = {
    'qwen': 'Qwen/Qwen3-1.7B-Base'
}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LuKA models on WikiSalad datasets')
    parser.add_argument('--model', type=str, required=True,
                       choices=list(MODELS.keys()),
                       help='Model to evaluate')
    parser.add_argument('--use_lined_attn', type=bool, required=True,
                       help='Use lined attention')
    parser.add_argument('--num_lined_layers', type=bool, default=None, required=False,
                       help='number of lined layers to use')
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
    print(f"LuKA Model: {args.model} ({model_name})")
    print(f"Attention Type: {"Lined Attention Only" if args.use_lined_attn else "Top-Down Attention Only"} ")
    print(f"Dataset: {args.dataset} ({dataset_file})")
    print(f"Device: {args.device}")
    if args.max_examples:
        print(f"Max examples: {args.max_examples}")
    print(f"Output: {output_file}")
    print("="*60)
    print()

    set_luka_kv_params(
        compressor="mean",
        segmenter="dummy",
        refine_threshold=1,
        segment_interval=16,
    )

    # Load model
    print(f"Loading model with LuKA configuration: {model_name}")
    model = load_luka_model(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )    
    print()

    # Configure attention mode
    if hasattr(model, "model") and hasattr(model.model, "luka_kv_controller"):
        controller = model.model.luka_kv_controller
        controller.use_lined_attention = args.use_lined_attn
        if args.num_lined_layers is not None:
            # Explicitly provided layers
            controller.lined_layers = set(args.num_lined_layers)
        else:
            if args.use_lined_attention:
                # Default: use all layers for lined attention
                controller.lined_layers = set(range(controller.num_layers))
            else:
                controller.lined_layers = set()
        
        print(f"Configuration:")
        print(f"  use_lined_attention: {controller.use_lined_attention}")
        print(f"  lined_layers: {controller.lined_layers}")
        print(f"  grid_top_k: {controller.grid_top_k}")
        print(f"  grid_update_interval: {controller.grid_update_interval}")
        print(f"  grid_decay: {controller.grid_decay}\n")

    

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
