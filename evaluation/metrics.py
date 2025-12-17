"""
Shared metrics for LuKA evaluation.

Consolidated scoring functions for QA accuracy, compression ratios, and boundary detection.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class EvalResult:
    """Results from evaluating a model."""
    dataset_name: str
    num_examples: int
    exact_match: float
    f1_score: float
    per_example_results: List[Dict]


def normalize_answer(text: str) -> str:
    """
    Normalize answer for comparison.

    Applies: lowercase, remove articles, remove punctuation, collapse whitespace.
    """
    text = text.lower()
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.strip()


def compute_exact_match(prediction: str, true_answers: List[str]) -> float:
    """
    Check if prediction matches any true answer.

    Uses exact match and bidirectional substring containment (lenient matching).

    Args:
        prediction: Model's predicted answer
        true_answers: List of acceptable ground truth answers

    Returns:
        1.0 if match found, 0.0 otherwise
    """
    if not true_answers:
        return 0.0
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
        if pred_norm in true_norm:
            return 1.0

    return 0.0


def compute_f1(prediction: str, true_answers: List[str]) -> float:
    """
    Compute token-level F1 score against best matching answer.

    Args:
        prediction: Model's predicted answer
        true_answers: List of acceptable ground truth answers

    Returns:
        Maximum F1 score across all true answers
    """
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


def compute_qa_accuracy(prediction: str, true_answers: List[str]) -> float:
    """
    Combined QA accuracy: average of exact match and F1.

    Args:
        prediction: Model's predicted answer
        true_answers: List of acceptable ground truth answers

    Returns:
        Average of EM and F1 scores
    """
    em = compute_exact_match(prediction, true_answers)
    f1 = compute_f1(prediction, true_answers)
    return (em + f1) / 2.0


def compute_compression_ratio(
    original_tokens: int,
    compressed_tokens: int,
    summary_tokens: int = 0
) -> float:
    """
    Calculate compression ratio (higher is better).

    Args:
        original_tokens: Number of tokens before compression
        compressed_tokens: Number of raw tokens after compression
        summary_tokens: Number of summary page tokens

    Returns:
        Compression ratio (original / total_after)
    """
    total_compressed = compressed_tokens + summary_tokens
    return original_tokens / total_compressed if total_compressed > 0 else 1.0


def compute_boundary_f1(
    predicted_boundaries: List[int],
    true_boundaries: List[int],
    tolerance: int = 5
) -> float:
    """
    Compute F1 for boundary detection with tolerance.

    Args:
        predicted_boundaries: List of predicted segment boundary positions
        true_boundaries: List of ground truth boundary positions
        tolerance: Number of tokens within which a prediction is considered correct

    Returns:
        F1 score for boundary detection
    """
    if not true_boundaries:
        return 1.0 if not predicted_boundaries else 0.0
    if not predicted_boundaries:
        return 0.0

    # Count true positives (predicted boundaries within tolerance of any true boundary)
    tp = 0
    matched_true = set()

    for pred in predicted_boundaries:
        for i, true in enumerate(true_boundaries):
            if i not in matched_true and abs(pred - true) <= tolerance:
                tp += 1
                matched_true.add(i)
                break

    precision = tp / len(predicted_boundaries) if predicted_boundaries else 0.0
    recall = tp / len(true_boundaries) if true_boundaries else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def resolve_dataset_path(dataset_arg: str) -> Path:
    """
    Resolve the dataset path, trying a few sensible defaults.

    Tries:
    - The exact path provided
    - The path relative to the repo root
    - The path inside the WikiSalad datasets directory

    Args:
        dataset_arg: Dataset path or name (e.g., "easy" or "easy_2topic_short.json")

    Returns:
        Resolved Path object

    Raises:
        FileNotFoundError: If dataset cannot be found
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


def parse_example(example: Dict) -> tuple:
    """
    Parse example into context and questions.

    Handles WikiSalad format and standard SQuAD format.

    Args:
        example: Dataset example dict

    Returns:
        Tuple of (context_str, list_of_question_dicts)
    """
    # WikiSalad format
    if 'prompt' in example and 'questions' in example:
        return example['prompt'], example['questions']

    # SQuAD format
    elif 'context' in example and 'question' in example:
        answers = example['answers']
        if isinstance(answers, dict) and 'text' in answers:
            answers = answers['text']
        return example['context'], [{
            'question': example['question'],
            'answers': answers
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


def format_qa_prompt(context: str, question: str) -> str:
    """
    Format a QA prompt for model input.

    Args:
        context: Document context
        question: Question to answer

    Returns:
        Formatted prompt string
    """
    return (
        f"{context}\n\n"
        f"Answer the following question in one short phrase. "
        f"Do not include any other text.\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def print_summary(results: EvalResult):
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS: {results.dataset_name}")
    print("=" * 60)
    print(f"Examples evaluated: {results.num_examples}")
    print(f"Exact Match:        {results.exact_match:.1%}")
    print(f"F1 Score:           {results.f1_score:.1%}")
    print("=" * 60)


def save_results(results: Union[EvalResult, Dict], output_path: str):
    """
    Save evaluation results to JSON.

    Args:
        results: EvalResult dataclass or dict
        output_path: Output file path
    """
    import json

    if isinstance(results, EvalResult):
        output = {
            'dataset': results.dataset_name,
            'num_examples': results.num_examples,
            'exact_match': results.exact_match,
            'f1_score': results.f1_score,
            'per_example': results.per_example_results
        }
    else:
        output = results

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")
