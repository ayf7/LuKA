"""
Evaluation Pipeline for LuKA

Runs two evaluation modes:
1. Simple Q&A: Each question gets full context, model generates answer
2. Sequential Multi-Answer: Build up context incrementally
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from tqdm import tqdm
from dataclasses import dataclass, asdict, field

import torch

from evaluation.model_interface import ModelInterface, GenerationConfig, LuKAModelInterface


@dataclass
class PerformanceStats:
    """Performance statistics from evaluation."""
    total_generations: int = 0
    total_tokens_generated: int = 0
    total_decode_time_sec: float = 0.0
    # Throughput: tokens per second
    decode_throughput_tps: float = 0.0
    # Latency: milliseconds per token
    decode_latency_ms_per_token: float = 0.0
    # Memory
    peak_memory_mb: float = 0.0
    peak_memory_allocated_mb: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class EvaluationResult:
    """Results from a single evaluation run."""
    model_name: str
    eval_mode: str  # 'simple' or 'sequential'
    dataset_name: str
    num_examples: int
    predictions: List[Dict]  # List of {example_id, questions, answers}
    compression_stats: Optional[List[Dict]] = None  # Only for LuKA models
    performance_stats: Optional[PerformanceStats] = None

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        if self.performance_stats:
            d['performance_stats'] = self.performance_stats.to_dict()
        return d


class QAEvaluator:
    """
    Evaluation pipeline for Q&A tasks.
    Model-agnostic: works with any ModelInterface implementation.
    """

    def __init__(self, dataset_path: str):
        """
        Load evaluation dataset.

        Args:
            dataset_path: Path to WikiSalad JSON dataset
        """
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)

        self.dataset_name = Path(dataset_path).stem
        print(f"Loaded {len(self.dataset)} examples from {self.dataset_name}")

    def evaluate_simple_qa(
        self,
        model: ModelInterface,
        config: Optional[GenerationConfig] = None,
        num_examples: Optional[int] = None
    ) -> EvaluationResult:
        """
        Simple Q&A evaluation: Each question gets full context.

        For each example:
        - Provide the full interlaced document
        - Ask all questions
        - Collect answers

        Args:
            model: Model implementing ModelInterface
            config: Generation configuration
            num_examples: Number of examples to evaluate (all if None)

        Returns:
            EvaluationResult with predictions
        """
        if config is None:
            config = GenerationConfig(max_new_tokens=100)

        examples_to_eval = self.dataset[:num_examples] if num_examples else self.dataset
        predictions = []
        compression_stats = [] if isinstance(model, LuKAModelInterface) else None

        # Performance tracking
        generation_times = []
        tokens_generated = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Get tokenizer for counting output tokens
        tokenizer = getattr(model, 'tokenizer', None)

        print(f"\n{'='*60}")
        print(f"Running Simple Q&A Evaluation")
        print(f"Model: {model.get_model_info()['name']}")
        print(f"Examples: {len(examples_to_eval)}")
        print(f"{'='*60}\n")

        for example in tqdm(examples_to_eval, desc="Evaluating examples"):
            example_id = example['example_id']
            prompt = example['prompt']
            questions = example['questions']

            # For simple mode, we ask each question independently with full context
            answers = []
            for q_data in questions:
                question = q_data['question']

                # Build prompt: full context + question
                full_prompt = f"{prompt}\n\nQuestion: {question}\nAnswer:"

                # Reset model state for each question
                model.reset()

                # Generate answer with timing
                start_time = time.perf_counter()
                answer = model.generate(full_prompt, config)
                end_time = time.perf_counter()
                generation_times.append(end_time - start_time)

                # Count tokens in generated output
                if tokenizer is not None:
                    num_tokens = len(tokenizer.encode(answer, add_special_tokens=False))
                else:
                    num_tokens = len(answer.split())  # Fallback: word count
                tokens_generated.append(num_tokens)

                answers.append(answer.strip())

            predictions.append({
                'example_id': example_id,
                'questions': [q['question'] for q in questions],
                'answers': answers
            })

            # Collect compression stats if LuKA model
            if isinstance(model, LuKAModelInterface):
                compression_stats.append({
                    'example_id': example_id,
                    **model.get_compression_stats()
                })

        # Compute performance stats
        total_generations = len(generation_times)
        total_tokens = sum(tokens_generated)
        total_time = sum(generation_times)

        # Throughput: tokens per second
        throughput = total_tokens / total_time if total_time > 0 else 0.0
        # Latency: ms per token
        latency_ms = (total_time * 1000) / total_tokens if total_tokens > 0 else 0.0

        peak_memory_mb = 0.0
        peak_allocated_mb = 0.0
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)
            peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        perf_stats = PerformanceStats(
            total_generations=total_generations,
            total_tokens_generated=total_tokens,
            total_decode_time_sec=total_time,
            decode_throughput_tps=throughput,
            decode_latency_ms_per_token=latency_ms,
            peak_memory_mb=peak_memory_mb,
            peak_memory_allocated_mb=peak_allocated_mb,
        )

        result = EvaluationResult(
            model_name=model.get_model_info()['name'],
            eval_mode='simple',
            dataset_name=self.dataset_name,
            num_examples=len(examples_to_eval),
            predictions=predictions,
            compression_stats=compression_stats,
            performance_stats=perf_stats,
        )

        return result

    def evaluate_sequential_multi_answer(
        self,
        model: ModelInterface,
        config: Optional[GenerationConfig] = None,
        num_examples: Optional[int] = None
    ) -> EvaluationResult:
        """
        Sequential Multi-Answer evaluation: Build up context incrementally.

        Pattern: <docs> <q1> <ans1> <docs> <q2> <ans2> ...

        For each question i:
        - Prompt includes all previous (docs, question, answer) pairs
        - Plus current docs and question
        - Model generates answer_i

        Args:
            model: Model implementing ModelInterface
            config: Generation configuration
            num_examples: Number of examples to evaluate (all if None)

        Returns:
            EvaluationResult with predictions
        """
        if config is None:
            config = GenerationConfig(max_new_tokens=100)

        examples_to_eval = self.dataset[:num_examples] if num_examples else self.dataset
        predictions = []
        compression_stats = [] if isinstance(model, LuKAModelInterface) else None

        # Performance tracking
        generation_times = []
        tokens_generated = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Get tokenizer for counting output tokens
        tokenizer = getattr(model, 'tokenizer', None)

        print(f"\n{'='*60}")
        print(f"Running Sequential Multi-Answer Evaluation")
        print(f"Model: {model.get_model_info()['name']}")
        print(f"Examples: {len(examples_to_eval)}")
        print(f"{'='*60}\n")

        for example in tqdm(examples_to_eval, desc="Evaluating examples"):
            example_id = example['example_id']
            questions = example['questions']
            metadata = example['metadata']

            # Build up context incrementally
            answers = []
            accumulated_context = ""

            # Reset model state at start of each example
            model.reset()

            for i, q_data in enumerate(questions):
                question = q_data['question']
                segment_id = q_data['segment_id']

                # Get the document segment for this question
                # From metadata, find the segment text
                segment_boundaries = metadata['segment_boundaries'][segment_id]
                start_char = segment_boundaries['start_char']
                end_char = segment_boundaries['end_char']

                # Extract segment from original prompt
                full_prompt_text = example['prompt']
                segment_text = full_prompt_text[start_char:end_char]

                # Build incremental prompt:
                # accumulated_context + current_segment + question
                if i == 0:
                    # First question: just the segment and question
                    prompt = f"{segment_text}\n\nQuestion: {question}\nAnswer:"
                else:
                    # Subsequent questions: accumulated context + new segment + question
                    prompt = f"{accumulated_context}\n\n{segment_text}\n\nQuestion: {question}\nAnswer:"

                # Generate answer with timing
                start_time = time.perf_counter()
                answer = model.generate(prompt, config)
                end_time = time.perf_counter()
                generation_times.append(end_time - start_time)

                # Count tokens in generated output
                if tokenizer is not None:
                    num_tokens = len(tokenizer.encode(answer, add_special_tokens=False))
                else:
                    num_tokens = len(answer.split())  # Fallback: word count
                tokens_generated.append(num_tokens)

                answer = answer.strip()
                answers.append(answer)

                # Update accumulated context for next iteration
                accumulated_context += f"\n\n{segment_text}\n\nQuestion: {question}\nAnswer: {answer}"

            predictions.append({
                'example_id': example_id,
                'questions': [q['question'] for q in questions],
                'answers': answers
            })

            # Collect compression stats if LuKA model
            if isinstance(model, LuKAModelInterface):
                compression_stats.append({
                    'example_id': example_id,
                    **model.get_compression_stats()
                })

        # Compute performance stats
        total_generations = len(generation_times)
        total_tokens = sum(tokens_generated)
        total_time = sum(generation_times)

        # Throughput: tokens per second
        throughput = total_tokens / total_time if total_time > 0 else 0.0
        # Latency: ms per token
        latency_ms = (total_time * 1000) / total_tokens if total_tokens > 0 else 0.0

        peak_memory_mb = 0.0
        peak_allocated_mb = 0.0
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)
            peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        perf_stats = PerformanceStats(
            total_generations=total_generations,
            total_tokens_generated=total_tokens,
            total_decode_time_sec=total_time,
            decode_throughput_tps=throughput,
            decode_latency_ms_per_token=latency_ms,
            peak_memory_mb=peak_memory_mb,
            peak_memory_allocated_mb=peak_allocated_mb,
        )

        result = EvaluationResult(
            model_name=model.get_model_info()['name'],
            eval_mode='sequential',
            dataset_name=self.dataset_name,
            num_examples=len(examples_to_eval),
            predictions=predictions,
            compression_stats=compression_stats,
            performance_stats=perf_stats
        )

        return result

    def evaluate_batched(
        self,
        model: ModelInterface,
        config: Optional[GenerationConfig] = None,
        num_examples: Optional[int] = None,
        batch_size: int = 8,
    ) -> EvaluationResult:
        """
        Batched evaluation: all questions across all examples, processed in batches.

        Args:
            model: Model implementing ModelInterface
            config: Generation configuration
            num_examples: Number of examples to evaluate (all if None)
            batch_size: Number of prompts per batch

        Returns:
            EvaluationResult with predictions
        """
        if config is None:
            config = GenerationConfig(max_new_tokens=100)

        examples_to_eval = self.dataset[:num_examples] if num_examples else self.dataset

        # Performance tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Get tokenizer for counting output tokens
        tokenizer = getattr(model, 'tokenizer', None)

        # Build all prompts upfront - ALL questions from ALL examples
        all_prompts = []
        prompt_metadata = []  # Track (example_idx, question_idx) for each prompt

        for ex_idx, example in enumerate(examples_to_eval):
            prompt = example['prompt']
            questions = example['questions']

            for q_idx, q_data in enumerate(questions):
                question = q_data['question']
                full_prompt = f"{prompt}\n\nQuestion: {question}\nAnswer:"
                all_prompts.append(full_prompt)
                prompt_metadata.append((ex_idx, q_idx, example['example_id'], question))

        total_questions = len(all_prompts)
        print(f"\n{'='*60}")
        print(f"Running Batched Evaluation")
        print(f"Model: {model.get_model_info()['name']}")
        print(f"Examples: {len(examples_to_eval)}")
        print(f"Total questions: {total_questions}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*60}\n")

        # Process in batches
        all_answers = []
        total_tokens = 0
        total_time = 0.0

        num_batches = (len(all_prompts) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_prompts))
            batch_prompts = all_prompts[start_idx:end_idx]

            # Reset model state before each batch
            model.reset()

            # Generate with timing
            start_time = time.perf_counter()
            batch_answers = model.batch_generate(batch_prompts, config)
            end_time = time.perf_counter()
            total_time += end_time - start_time

            # Count tokens
            for answer in batch_answers:
                if tokenizer is not None:
                    total_tokens += len(tokenizer.encode(answer, add_special_tokens=False))
                else:
                    total_tokens += len(answer.split())

            all_answers.extend(batch_answers)

        # Group answers back by example
        example_answers: Dict[str, Dict] = {}
        for (ex_idx, q_idx, example_id, question), answer in zip(prompt_metadata, all_answers):
            if example_id not in example_answers:
                example_answers[example_id] = {'questions': [], 'answers': []}
            example_answers[example_id]['questions'].append(question)
            example_answers[example_id]['answers'].append(answer.strip())

        # Build predictions in original order
        predictions = []
        for example in examples_to_eval:
            example_id = example['example_id']
            if example_id in example_answers:
                predictions.append({
                    'example_id': example_id,
                    'questions': example_answers[example_id]['questions'],
                    'answers': example_answers[example_id]['answers'],
                })

        # Compute performance stats
        throughput = total_tokens / total_time if total_time > 0 else 0.0
        latency_ms = (total_time * 1000) / total_tokens if total_tokens > 0 else 0.0

        peak_memory_mb = 0.0
        peak_allocated_mb = 0.0
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)
            peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        perf_stats = PerformanceStats(
            total_generations=len(all_prompts),
            total_tokens_generated=total_tokens,
            total_decode_time_sec=total_time,
            decode_throughput_tps=throughput,
            decode_latency_ms_per_token=latency_ms,
            peak_memory_mb=peak_memory_mb,
            peak_memory_allocated_mb=peak_allocated_mb,
        )

        # No compression stats for batched mode (would need per-example tracking)
        result = EvaluationResult(
            model_name=model.get_model_info()['name'],
            eval_mode='batched',
            dataset_name=self.dataset_name,
            num_examples=len(examples_to_eval),
            predictions=predictions,
            compression_stats=None,
            performance_stats=perf_stats,
        )

        return result

    def save_results(self, result: EvaluationResult, output_path: str):
        """
        Save evaluation results to JSON.

        Args:
            result: EvaluationResult object
            output_path: Where to save results
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        print(f"\nResults saved to {output_path}")

    def score_predictions(self, result: EvaluationResult) -> Dict:
        """
        Score predictions against ground truth.

        Args:
            result: EvaluationResult with predictions

        Returns:
            Dict with aggregate and per-example scores
        """
        from evaluation.metrics import compute_exact_match, compute_f1, compute_qa_accuracy
        import numpy as np

        gt_lookup = {ex['example_id']: ex for ex in self.dataset}

        scores = []
        per_example = []

        for pred in result.predictions:
            example_id = pred['example_id']
            example = gt_lookup.get(example_id)
            if example is None:
                continue

            questions = example['questions']
            example_scores = []

            for i, (pred_answer, q_data) in enumerate(zip(pred['answers'], questions)):
                answers_data = q_data['answers']
                if isinstance(answers_data, dict) and 'text' in answers_data:
                    true_answers = answers_data['text']
                else:
                    true_answers = answers_data
                if not isinstance(true_answers, list):
                    true_answers = [true_answers]

                em = compute_exact_match(pred_answer, true_answers)
                f1 = compute_f1(pred_answer, true_answers)
                qa = compute_qa_accuracy(pred_answer, true_answers)

                example_scores.append({
                    'question_idx': i,
                    'exact_match': em,
                    'f1': f1,
                    'qa_accuracy': qa,
                })
                scores.append({'em': em, 'f1': f1, 'qa': qa})

            per_example.append({
                'example_id': example_id,
                'scores': example_scores,
                'avg_em': float(np.mean([s['exact_match'] for s in example_scores])),
                'avg_f1': float(np.mean([s['f1'] for s in example_scores])),
                'avg_qa': float(np.mean([s['qa_accuracy'] for s in example_scores])),
            })

        aggregate = {
            'exact_match': float(np.mean([s['em'] for s in scores])) if scores else 0.0,
            'f1_score': float(np.mean([s['f1'] for s in scores])) if scores else 0.0,
            'qa_accuracy': float(np.mean([s['qa'] for s in scores])) if scores else 0.0,
            'num_questions': len(scores),
            'num_examples': len(per_example),
        }

        return {'aggregate': aggregate, 'per_example': per_example}
