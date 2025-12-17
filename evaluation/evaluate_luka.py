"""
Generate text using lined (H2O-style grid tokens) attention on WikiSalad.

Datasets are:
- easy_2topic_short
- medium_2topic_short
- hard_3topic_short
- very_hard_3topic_long.json

- Set max examples with --max-examples, default is 100

- attention modes are {lined, top_down, mixed}

- Results will save in evaluation/results/eval_results_{attention_mode}_attention_{dataset}.json

How to run:
python3 evaluate_luka.py --attention_mode lined --dataset easy_2topic_short --max-examples 1
python3 evaluate_luka.py --attention_mode top_down --dataset easy_2topic_short --max-examples 1
python3 evaluate_luka.py --attention_mode mixed --dataset easy_2topic_short --max-examples 1

"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from experiments.comprehensive_eval import generate_text
from modeling.compressor import EncoderCompressor
from modeling.qwen.luka_qwen3 import load_luka_model, set_luka_kv_params
from utils import (
    EvalResult,
    compute_exact_match,
    compute_f1,
    format_qa_prompt,
    normalize_answer,
    parse_example,
    print_summary,
    resolve_dataset_path,
    save_results,
)

MAX_NEW_TOKENS = 256

def main():
    parser = argparse.ArgumentParser(description="Generate with lined attention on WikiSalad")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to WikiSalad JSON (list of examples)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B-Base",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=100,
        help="Maximum number of examples to evaluate (default: 10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu or cuda)",
    )
    parser.add_argument(
        "--attention_mode",
        type=str,
        choices=["lined", "top_down", "mixed"],
        default="lined",
        help="Attention mode to use (lined, top_down, or mixed)",
    )
    args = parser.parse_args()

    device = args.device

    # Load WikiSalad Dataset JSON (list of examples)
    dataset_path = resolve_dataset_path(args.dataset)
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # Configure LuKA
    use_controller = False

    # ********************************************************************** #
    # LINED ATTENTION SETUP
    # ********************************************************************** #
    if args.attention_mode == "lined":
        use_controller = True
        set_luka_kv_params(
            default_tail_len=16,
            min_compress_chunk=16,
            max_pages=15,
            refine_threshold=0.05,
            compressor=None,
            segmenter="dummy",
        )
        model = load_luka_model(
            args.model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        ).to(device)
        model.eval()

        # Enable lined attention globally
        controller = model.model.layers[0].self_attn.luka_kv
        controller.use_lined_attention = True
        controller.min_lined_seq_len = 384
        controller.min_lined_tail_window = 512 
        controller.grid_min_change_ratio = 0.0  
        controller.grid_update_interval = 4     
        controller.grid_decay = 0.95            
        controller.debug = False                 
        controller.lined_layers = set(range(controller.num_layers))
    
    # ********************************************************************** #
    # TOP-DOWN ATTENTION SETUP
    # ********************************************************************** #
    elif args.attention_mode == "top_down":
        use_controller = False
        compressor = EncoderCompressor(dim=128)
        set_luka_kv_params(
            default_tail_len=16,
            min_compress_chunk=16,
            max_pages=15,
            refine_threshold=0.05,
            compressor=compressor,
            segmenter="dummy",
        )
        model = load_luka_model(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,).to(device)
        model.eval()

    # ********************************************************************** #
    # MIXED ATTENTION SETUP
    # ********************************************************************** #
    elif args.attention_mode == "mixed":
        use_controller = True
        compressor = EncoderCompressor(dim=128)
        set_luka_kv_params(
            default_tail_len=16,
            min_compress_chunk=16,
            max_pages=15,
        
            refine_threshold=0.05,
            compressor=compressor,
            segmenter="dummy",
            )
        model = load_luka_model(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        ).to(device)
        model.eval()
        controller = model.model.layers[0].self_attn.luka_kv
        controller.use_lined_attention = True
        controller.min_lined_seq_len = 384
        controller.min_lined_tail_window = 192
        num_layers = controller.num_layers
        low_k, high_k = 6, 5
        controller.lined_layers = set(range(0, low_k)) | set(range(num_layers - high_k, num_layers))


    # ********************************************************************** #
    # Run Generations and Evaluate
    # ********************************************************************** #
    results = []
    print("Num examples: ", len(dataset))

    examples_to_eval = dataset[:args.max_examples]
    exact_matches = []
    f1_scores = []
    per_example = []
    verbose = True

    print(f"Evaluating {len(examples_to_eval)} examples (max: {args.max_examples})")
    iterator = tqdm(examples_to_eval, desc="Evaluating") if verbose else examples_to_eval

    for i, example in enumerate(iterator):
            # Extract context and questions
            context, questions = parse_example(example)

            for q_idx, question_data in enumerate(questions):
                question = question_data['question']
                # Extract answer text (handles both dict and list formats)
                answers_data = question_data['answers']
                if isinstance(answers_data, dict) and 'text' in answers_data:
                    true_answers = answers_data['text']
                else:
                    true_answers = answers_data
                
                try:
                    # CRITICAL FIX: Reset cache before each question to prevent contamination
                    if use_controller:
                        controller.reset()
                    
                    # Format prompt to encourage short answers
                    input_prompt = format_qa_prompt(context, question)
                    
                    # Debug: dump prompt tokens for Kerry question (question_id=3)
                    if q_idx == 3 and verbose:
                        prompt_tokens = tokenizer.encode(input_prompt, add_special_tokens=False)
                        prompt_text = tokenizer.decode(prompt_tokens[-80:], skip_special_tokens=False)
                        print(f"\n[DEBUG] Last 80 prompt tokens for question_id=3:")
                        print(f"  Text: {prompt_text}")
                        print(f"  Token count: {len(prompt_tokens)}")
                    
                    if verbose:
                        print(f"\n[Example {i}, Question {q_idx}]")
                        print(f"Question: {question}")
                        print(f"True answers: {true_answers}")

                    predicted_answer, _ = generate_text(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=input_prompt,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=0.3,  # Lower temperature for more focused answers
                        top_p=0.9,
                    )
                    
                    # Extract first sentence or phrase (stop at newline or period)
                    # This helps get just the answer, not the continuation
                    if '\n' in predicted_answer:
                        predicted_answer = predicted_answer.split('\n')[0].strip()
                    # Also try to stop after first sentence if it's clearly an answer
                    if predicted_answer and len(predicted_answer) > 100:
                        # If it's very long, try to extract just the first part
                        sentences = predicted_answer.split('.')
                        if len(sentences) > 1:
                            # Take first sentence if it seems like an answer
                            first_sent = sentences[0].strip()
                            if len(first_sent) > 10:  # If first sentence is substantial
                                predicted_answer = first_sent
                    
                    if verbose:
                        print(f"Predicted: {predicted_answer}")
                    
                    # Compute metrics
                    em = compute_exact_match(predicted_answer, true_answers)
                    f1 = compute_f1(predicted_answer, true_answers)
                    
                    if verbose:
                        print(f"Exact Match: {em:.2f}, F1: {f1:.2f}")
                    
                    exact_matches.append(em)
                    f1_scores.append(f1)
                    
                    per_example.append({
                        'example_id': i,
                        'question_id': q_idx,
                        'question': question,
                        'predicted': predicted_answer,
                        'true_answers': true_answers,
                        'exact_match': em,
                        'f1': f1
                    })
                    
                except Exception as e:
                    if verbose:
                        print(f"\nError on example {i}, question {q_idx}: {e}")
                        import traceback
                        traceback.print_exc()
                    per_example.append({
                        'example_id': i,
                        'question_id': q_idx,
                        'error': str(e)
                    })

    results = EvalResult(
        dataset_name=Path(dataset_path).stem,
        num_examples=len(per_example),
        exact_match=float(np.mean(exact_matches)) if exact_matches else 0.0,
        f1_score=float(np.mean(f1_scores)) if f1_scores else 0.0,
        per_example_results=per_example
    )
    
    print_summary(results)
    save_results(results, f"results/eval_results_{args.attention_mode}_attention_{Path(dataset_path).stem}.json")


if __name__ == "__main__":
    main()

