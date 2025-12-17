#!/usr/bin/env python
"""
Evaluation script for LuKA with configurable attention modes.

Supports three attention modes:
- topdown: Traditional top-down attention (default)
- lined: H2O-style lined attention on all layers
- mixed: Lined attention on early layers, top-down on last layers

Usage:
    python3 luka_attention_eval.py --dataset easy --attention lined --max-examples 1
    python3 luka_attention_eval.py --dataset hard --attention lined --max-examples 5
    python3 luka_attention_eval.py --dataset very_hard --attention mixed --protect-last 4
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer
from datetime import datetime

from modeling.qwen.luka_qwen3 import load_luka_model, set_luka_kv_params
from simple_qa_evaluator import SimpleQAEvaluator

# Available datasets
DATASETS = {
    "easy": "easy_2topic_short.json",
    "medium": "medium_2topic_medium.json",
    "hard": "hard_3topic_short.json",
    "very_hard": "very_hard_3topic_long.json",
}

MODEL_NAME = "Qwen/Qwen3-1.7B-Base"


def configure_attention_mode(model, attention_mode, protect_last_n=4, custom_layers=None):
    """
    Configure the LuKA controller's attention mode.
    
    Args:
        model: The loaded LuKA model
        attention_mode: One of 'topdown', 'lined', or 'mixed'
        protect_last_n: Number of last layers to protect in mixed mode
        custom_layers: Optional list of specific layer indices to use lined attention
    """
    if not (hasattr(model, "model") and hasattr(model.model, "luka_kv_controller")):
        print("Warning: Model does not have luka_kv_controller")
        return
    
    controller = model.model.luka_kv_controller
    num_layers = controller.num_layers
    
    if attention_mode == "topdown":
        # Top-down attention on all layers
        controller.use_lined_attention = False
        controller.lined_layers = set()
        print(f"✓ Configured TOP-DOWN attention on all {num_layers} layers")
        
    elif attention_mode == "lined":
        # Lined attention on all layers
        controller.use_lined_attention = True
        if custom_layers is not None:
            controller.lined_layers = set(custom_layers)
            print(f"✓ Configured LINED attention on custom layers: {sorted(custom_layers)}")
        else:
            controller.lined_layers = set(range(num_layers))
            print(f"✓ Configured LINED attention on all {num_layers} layers")
    
    elif attention_mode == "mixed":
        # Mixed: lined on early layers, top-down on last layers
        controller.use_lined_attention = True
        if custom_layers is not None:
            controller.lined_layers = set(custom_layers)
            print(f"✓ Configured MIXED attention on custom layers: {sorted(custom_layers)}")
        else:
            lined_layer_indices = list(range(max(0, num_layers - protect_last_n)))
            controller.lined_layers = set(lined_layer_indices)
            topdown_layer_indices = list(range(max(0, num_layers - protect_last_n), num_layers))
            print(f"✓ Configured MIXED attention:")
            print(f"  - LINED on layers {lined_layer_indices[0]}-{lined_layer_indices[-1]} ({len(lined_layer_indices)} layers)")
            print(f"  - TOP-DOWN on layers {topdown_layer_indices[0]}-{topdown_layer_indices[-1]} ({len(topdown_layer_indices)} layers)")
    
    else:
        raise ValueError(f"Unknown attention mode: {attention_mode}")
    
    # Print controller configuration
    print(f"\nController Configuration:")
    print(f"  use_lined_attention: {controller.use_lined_attention}")
    print(f"  lined_layers: {sorted(controller.lined_layers) if controller.lined_layers else 'None'}")
    print(f"  grid_top_k: {controller.grid_top_k}")
    print(f"  grid_update_interval: {controller.grid_update_interval}")
    print(f"  grid_decay: {controller.grid_decay}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate LuKA with different attention modes")
    
    # Dataset options
    parser.add_argument(
        "--dataset",
        type=str,
        default="easy",
        choices=DATASETS.keys(),
        help="WikiSalad dataset difficulty"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (default: all)"
    )
    
    # Attention mode options
    parser.add_argument(
        "--attention",
        type=str,
        default="topdown",
        choices=["topdown", "lined", "mixed"],
        help="Attention mode: topdown, lined, or mixed"
    )
    parser.add_argument(
        "--protect-last",
        type=int,
        default=4,
        help="Number of last layers to protect in mixed mode (default: 4)"
    )
    parser.add_argument(
        "--custom-layers",
        type=str,
        default=None,
        help="Comma-separated layer indices for custom lined attention (e.g., '0,1,2,5,10')"
    )
    
    # LuKA KV parameters
    parser.add_argument(
        "--compressor",
        type=str,
        default="mean",
        help="LuKA compressor type (default: mean)"
    )
    parser.add_argument(
        "--segmenter",
        type=str,
        default="dummy",
        help="LuKA segmenter type (default: dummy)"
    )
    parser.add_argument(
        "--segment-interval",
        type=int,
        default=16,
        help="Segment interval for dummy segmenter (default: 16)"
    )
    parser.add_argument(
        "--refine-threshold",
        type=int,
        default=1,
        help="Refine threshold (default: 1)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling (default: 0.9)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate (default: 256)"
    )
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Dataset path
    dataset_file = DATASETS[args.dataset]
    dataset_path = (
        Path(__file__).parent.parent
        / "artifacts/hugging_face_wikipedia/wikisalad_datasets"
        / dataset_file
    )
    
    if not dataset_path.exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        return

    # Parse custom layers if provided
    custom_layers = None
    if args.custom_layers:
        custom_layers = [int(x.strip()) for x in args.custom_layers.split(",")]
        print(f"Using custom layers: {custom_layers}\n")

    # Set LuKA KV params
    print("Setting LuKA KV parameters...")
    set_luka_kv_params(
        compressor=args.compressor,
        segmenter=args.segmenter,
        refine_threshold=args.refine_threshold,
        segment_interval=args.segment_interval,
    )

    # Load tokenizer
    print(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # Load LuKA model
    print(f"Loading model: {MODEL_NAME}...")
    model = load_luka_model(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    print("Model loaded successfully!\n")

    # Configure attention mode
    print(f"{'='*80}")
    print(f"Configuring {args.attention.upper()} attention mode")
    print(f"{'='*80}")
    configure_attention_mode(
        model,
        args.attention,
        protect_last_n=args.protect_last,
        custom_layers=custom_layers
    )

    # Create custom model adapter for pre-loaded LuKA model
    class LuKAModelAdapter:
        """Adapter for pre-loaded LuKA model."""
        
        def __init__(self, model, tokenizer, device, max_new_tokens=256, 
                     temperature=0.7, top_p=0.9, do_sample=True):
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            self.max_new_tokens = max_new_tokens
            self.temperature = temperature
            self.top_p = top_p
            self.do_sample = do_sample
        
        def __call__(self, context: str, question: str) -> str:
            """Generate answer given context and question."""
            # Format prompt for causal models
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context. Provide concise, direct answers."},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nProvide a brief, direct answer to the question based only on the context."}
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback for models without chat template
                prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=4096,
                truncation=True
            )
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the answer part (remove prompt)
            if "assistant" in answer:
                parts = answer.split("assistant")
                answer = parts[-1].strip()
            
            if prompt in answer:
                answer = answer[len(prompt):].strip()
            
            # Try to extract after common markers
            for marker in ["Answer:", "answer:", "A:"]:
                if marker in answer:
                    answer = answer.split(marker)[-1].strip()
                    break
            
            return answer
    
    # Wrap model with custom adapter
    model_fn = LuKAModelAdapter(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
    )

    # Load evaluator
    print(f"Loading WikiSalad dataset: {args.dataset}")
    print(f"Dataset path: {dataset_path}")
    evaluator = SimpleQAEvaluator(str(dataset_path))
    print()

    # Run evaluation
    print(f"{'='*80}")
    print(f"Starting evaluation")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Attention mode: {args.attention}")
    print(f"Max examples: {args.max_examples if args.max_examples else 'all'}")
    print(f"{'='*80}\n")
    
    results = evaluator.evaluate(
        model_fn=model_fn,
        max_examples=args.max_examples,
        verbose=True,
    )

    # Generate output filename with timestamp and configuration
    output_file = f"eval_{args.dataset}_{args.attention}.json"
    evaluator.save_results(results, output_file)

    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Attention mode: {args.attention}")
    print(f"Examples evaluated: {len(results.results) if hasattr(results, 'results') else 'N/A'}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()