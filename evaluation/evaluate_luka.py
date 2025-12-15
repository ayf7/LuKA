"""
Generate text using lined (H2O-style grid tokens) attention on WikiSalad.
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import re
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable


import torch
from transformers import AutoTokenizer

from modeling.qwen.luka_qwen3 import (
    load_luka_model,
    set_luka_kv_params,
)

# Reuse eval helper for generation
from experiments.comprehensive_eval import generate_text


# ---------------------------------------------------------------------
# Fixed configuration
# ---------------------------------------------------------------------
OUTPUT_DIR = Path("experiments/wikisalad_generations")
MAX_NEW_TOKENS = 256

WIKISALAD_DIR = (
    Path(__file__).resolve().parent.parent
    / "artifacts/hugging_face_wikipedia/wikisalad_datasets"
)

def save_results(results, output_path: str):
    """Save evaluation results to JSON."""
    output = {
        'dataset': results.dataset_name,
        'num_examples': results.num_examples,
        'exact_match': results.exact_match,
        'f1_score': results.f1_score,
        'per_example': results.per_example_results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")

def print_summary(results):
    """Print evaluation summary."""
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS: {results.dataset_name}")
    print("="*60)
    print(f"Examples evaluated: {results.num_examples}")
    print(f"Exact Match:        {results.exact_match:.1%}")
    print(f"F1 Score:           {results.f1_score:.1%}")
    print("="*60)

@dataclass
class EvalResult:
    """Results from evaluating a model."""
    dataset_name: str
    num_examples: int
    exact_match: float
    f1_score: float
    per_example_results: List[Dict]


def resolve_dataset_path(dataset_arg: str) -> Path:
    """
    Resolve the dataset path, trying a few sensible defaults:
    - The exact path provided
    - The path relative to the repo root
    - The path inside the WikiSalad datasets directory
    """
    repo_root = Path(__file__).resolve().parent.parent
    dataset_path = Path(dataset_arg)

    candidates = [
        dataset_path,
        repo_root / dataset_path,
        WIKISALAD_DIR / dataset_path.name,
    ]

    # If the user passed a bare stem like "easy_2topic_short", try adding .json
    if dataset_path.suffix != ".json":
        candidates.append(WIKISALAD_DIR / f"{dataset_path.name}.json")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Dataset not found for '{dataset_arg}'. Tried:\n  {searched}")

def parse_example(example):
        """
        Parse example into context and questions.
        Handles WikiSalad format and standard SQuAD format.
        """
        # WikiSalad format
        if 'prompt' in example and 'questions' in example:
            return example['prompt'], example['questions']
        
        # SQuAD format
        elif 'context' in example and 'question' in example:
            return example['context'], [{
                'question': example['question'],
                'answers': example['answers']['text'] if isinstance(example['answers'], dict) else example['answers']
            }]
        
        # SQuAD nested format
        elif 'paragraphs' in example:
            all_questions = []
            context = ""
            for para in example['paragraphs']:
                context += para['context'] + "\n\n"
                for qa in para.get('qas', []):
                    all_questions.append({
                        'question': qa['question'],
                        'answers': [a['text'] for a in qa.get('answers', [])]
                    })
            return context, all_questions
        
        else:
            raise ValueError(f"Unknown dataset format: {example.keys()}")

def compute_exact_match(prediction: str, true_answers):
        """
        Check if prediction matches any true answer.
        Uses both exact match and substring containment (like SQuAD).
        """
        # Handle unanswerable questions (empty answer list)
        if not true_answers:
            return 0.0

        pred_norm = normalize_answer(prediction)

        for true in true_answers:
            true_norm = normalize_answer(true)
            # Exact match
            if pred_norm == true_norm:
                return 1.0
            # Substring match: true answer is contained in prediction
            if true_norm in pred_norm:
                return 1.0
            # Reverse substring: prediction is contained in true answer
            # (for cases where prediction is more concise)
            if pred_norm in true_norm:
                return 1.0

        return 0.0

def normalize_answer(text: str) -> str:
        """Normalize answer for comparison."""
        text = text.lower()
        # Remove articles
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()

def compute_f1(prediction: str, true_answers) -> float:
    """Compute F1 score against best matching answer."""
    # Handle unanswerable questions (empty answer list)
    if not true_answers:
        return 0.0

    pred_tokens = normalize_answer(prediction).split()

    if len(pred_tokens) == 0:
        return 1.0 if all(len(normalize_answer(t)) == 0 for t in true_answers) else 0.0

    f1_scores = []
    for true in true_answers:
        true_tokens = normalize_answer(true).split()

        if len(true_tokens) == 0:
            f1_scores.append(0.0)
            continue

        common = set(pred_tokens) & set(true_tokens)

        if len(common) == 0:
            f1_scores.append(0.0)
            continue

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(true_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)

    return max(f1_scores) if f1_scores else 0.0
    

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
    args = parser.parse_args()

    device = "cpu"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load WikiSalad JSON (list of examples)
    # ------------------------------------------------------------------
    dataset_path = resolve_dataset_path(args.dataset)
    print(f"Loading dataset from: {dataset_path}")

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Configure LuKA + lined attention
    # ------------------------------------------------------------------
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
    controller.min_lined_tail_window = 192
    controller.lined_layers = set(range(controller.num_layers))

    # ------------------------------------------------------------------
    # Generate for every WikiSalad example
    # ------------------------------------------------------------------
    results = []
    print("Num examples: ", len(dataset))

    examples_to_eval = dataset
    exact_matches = []
    f1_scores = []
    per_example = []
    verbose = True

    iterator = tqdm(examples_to_eval, desc="Evaluating") if verbose else examples_to_eval

    count = 1
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
                    # Get model prediction

                    input_prompt = context + " " + question
                    print("INPUT PROMPT IS: " , input_prompt)

                    predicted_answer, _ = generate_text(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=input_prompt,
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
                    
                    # Compute metrics
                    em = compute_exact_match(predicted_answer, true_answers)
                    f1 = compute_f1(predicted_answer, true_answers)
                    
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
                    print(per_example)
                    
                except Exception as e:
                    if verbose:
                        print(f"\nError on example {i}, question {q_idx}: {e}")
                    per_example.append({
                        'example_id': i,
                        'question_id': q_idx,
                        'error': str(e)
                    })
                break
            break

    results = EvalResult(
            dataset_name=Path(dataset_path).stem,
            num_examples=len(per_example),
            exact_match=float(np.mean(exact_matches)),
            f1_score=float(np.mean(f1_scores)),
            per_example_results=per_example
        )
    
    print_summary(results)
    save_results(results, "eval_results_lined_attention.json")


    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_file = OUTPUT_DIR / dataset_path.name.replace(".json", "_lined.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved generations to {out_file}")


if __name__ == "__main__":
    main()
