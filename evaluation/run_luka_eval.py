#!/usr/bin/env python
"""
Quick script to run evaluations for LuKA on different WikiSalad datasets.
"""

import argparse
from pathlib import Path
import torch

from simple_qa_evaluator import SimpleQAEvaluator
from modeling.qwen.luka_qwen3 import load_luka_model, set_luka_kv_params
from transformers import AutoTokenizer

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
    parser.add_argument(
        '--attention_type',
        type=str,
        required=True,
        choices=['top_down', 'lined', 'mix'],
        help='Attention mode: top_down, lined, or mix'
    )

    parser.add_argument(
        '--num_lined_layers',
        type=int,
        default=None,
        help='Number of initial layers for lined attention (used in mix mode)'
    )

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

    attn_type = args.attention_type

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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Ensure model is on the requested device
    if device == "cpu":
        model.to("cpu")

    # Configure attention mode
    if hasattr(model, "model") and hasattr(model.model, "luka_kv_controller"):
        controller = model.model.luka_kv_controller
        if args.attention_type == "top_down":
            controller.use_lined_attention = False
            controller.lined_layers = set()

        elif args.attention_type == "lined":
            controller.use_lined_attention = True
            controller.lined_layers = set(range(controller.num_layers))

        elif args.attention_type == "mix":
            controller.use_lined_attention = True
            controller.lined_layers = set(range(0, 6)) | set(range(23, 28))
                    
        print(f"Configuration:")
        print(f"  use_lined_attention: {controller.use_lined_attention}")
        print(f"  lined_layers: {controller.lined_layers}")
        print(f"  grid_top_k: {controller.grid_top_k}")
        print(f"  grid_update_interval: {controller.grid_update_interval}")
        print(f"  grid_decay: {controller.grid_decay}\n")

    def model_fn(context: str, question: str) -> str:
        """Wrap the LuKA/Qwen model to the (context, question) -> answer interface."""
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based only on the provided context.",
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer the question concisely based only on the context.",
                },
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt", max_length=4096, truncation=True).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True)
            

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    evaluator = SimpleQAEvaluator(str(dataset_path))
    results = evaluator.evaluate(model_fn, max_examples=args.max_examples)

    evaluator.print_summary(results)
    evaluator.save_results(results, output_file)

    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
