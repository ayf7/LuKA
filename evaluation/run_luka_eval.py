#!/usr/bin/env python
"""
Quick script to run evaluations for LuKA on different WikiSalad datasets.
"""

import argparse
from pathlib import Path
import torch

from simple_qa_evaluator import SimpleQAEvaluator, HuggingFaceModelAdapter
from modeling.qwen.luka_qwen3 import load_luka_model, set_luka_kv_params

# Available datasets
DATASETS = {
    'easy': 'easy_2topic_short.json',
    'medium': 'medium_2topic_medium.json',
    'hard': 'hard_3topic_short.json',
    'very_hard': 'very_hard_3topic_long.json'
}

MODELS = {
    'qwen': 'Qwen/Qwen3-1.7B-Base'
}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LuKA models on WikiSalad datasets')

    parser.add_argument('--model', type=str, required=True,
                        choices=list(MODELS.keys()))
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASETS.keys()))
    parser.add_argument('--max-examples', type=int, default=None)

    # Attention flags
    parser.add_argument('--use_lined_attn', action='store_true',
                        help='Enable lined attention')
    parser.add_argument('--no_lined_attn', dest='use_lined_attn', action='store_false')
    parser.set_defaults(use_lined_attn=False)

    parser.add_argument('--num_lined_layers', type=int, default=None,
                        help='Number of initial layers to use lined attention')

    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'])

    args = parser.parse_args()

    device = args.device
    model_name = MODELS[args.model]
    dataset_file = DATASETS[args.dataset]

    dataset_path = (
        Path(__file__).parent.parent /
        "artifacts/hugging_face_wikipedia/wikisalad_datasets" /
        dataset_file
    )

    output_file = Path(__file__).parent / f"eval_results_{args.model}_{args.dataset}.json"

    attn_type = "Lined Attention Only" if args.use_lined_attn else "Top-Down Attention Only"

    print("=" * 60)
    print(f"LuKA Model: {args.model} ({model_name})")
    print(f"Attention Type: {attn_type}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"Output: {output_file}")
    print("=" * 60)
    print()

    set_luka_kv_params(
        compressor="mean",
        segmenter="dummy",
        refine_threshold=1,
        segment_interval=16,
    )

    model = load_luka_model(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    if hasattr(model, "model") and hasattr(model.model, "luka_kv_controller"):
        controller = model.model.luka_kv_controller
        controller.use_lined_attention = args.use_lined_attn

        if args.num_lined_layers is not None:
            controller.lined_layers = set(range(args.num_lined_layers))
        else:
            controller.lined_layers = (
                set(range(controller.num_layers)) if args.use_lined_attn else set()
            )

        print("Configuration:")
        print(f"  use_lined_attention: {controller.use_lined_attention}")
        print(f"  lined_layers: {controller.lined_layers}")
        print()


    
    evaluator = SimpleQAEvaluator(str(dataset_path))
    
    results = evaluator.evaluate(model, max_examples=args.max_examples)

    evaluator.print_summary(results)
    evaluator.save_results(results, output_file)

    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
