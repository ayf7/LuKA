import argparse
import json
from pathlib import Path
from tqdm import tqdm
import re
import numpy as np
from dataclasses import dataclass
from typing import List, Dict



@dataclass
class EvalResult:
    """Results from evaluating a model."""
    dataset_name: str
    num_examples: int
    exact_match: float
    f1_score: float
    per_example_results: List[Dict]


def print_summary(results):
    """Print evaluation summary."""
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS: {results.dataset_name}")
    print("="*60)
    print(f"Examples evaluated: {results.num_examples}")
    print(f"Exact Match:        {results.exact_match:.1%}")
    print(f"F1 Score:           {results.f1_score:.1%}")
    print("="*60)


def save_results(results, output_path: str):
    """Save evaluation results to JSON."""
    output = {
        'dataset': results.dataset_name,
        'num_examples': results.num_examples,
        'exact_match': results.exact_match,
        'f1_score': results.f1_score,
        'per_example': results.per_example_results
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")



def resolve_dataset_path(dataset_arg: str) -> Path:
    """
    Resolve the dataset path, trying a few sensible defaults:
    - The exact path provided
    - The path relative to the repo root
    - The path inside the WikiSalad datasets directory
    """
    WIKISALAD_DIR = (
    Path(__file__).resolve().parent.parent
    / "artifacts/hugging_face_wikipedia/wikisalad_datasets"
        )
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
        
        # Handle empty predictions
        if not prediction or len(prediction.strip()) == 0:
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

def format_qa_prompt(context: str, question: str) -> str:
    prompt = f"{context}\n\nAnswer the following question in one short phrase. Do not include any other text.\n\nQuestion: {question}\n\nAnswer:"
    return prompt
