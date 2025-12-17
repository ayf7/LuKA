"""
LuKA Evaluation Pipeline

Unified evaluation framework for baseline and LuKA-enabled models.

Usage:
    python -m evaluation.run_eval --model-type luka --model qwen-1.7b --dataset easy --score
"""

from evaluation.model_interface import (
    ModelInterface,
    BaselineModelInterface,
    LuKAModelInterface,
    GenerationConfig,
)
from evaluation.evaluator import QAEvaluator, EvaluationResult, PerformanceStats
from evaluation.model_implementations import (
    HuggingFaceModel,
    APIModel,
    LuKAQwenModel,
)
from evaluation.metrics import (
    normalize_answer,
    compute_exact_match,
    compute_f1,
    compute_qa_accuracy,
    compute_compression_ratio,
    compute_boundary_f1,
    resolve_dataset_path,
    parse_example,
    EvalResult,
)

__all__ = [
    # Interfaces
    'ModelInterface',
    'BaselineModelInterface',
    'LuKAModelInterface',
    'GenerationConfig',
    # Evaluator
    'QAEvaluator',
    'EvaluationResult',
    'PerformanceStats',
    # Model implementations
    'HuggingFaceModel',
    'APIModel',
    'LuKAQwenModel',
    # Metrics
    'normalize_answer',
    'compute_exact_match',
    'compute_f1',
    'compute_qa_accuracy',
    'compute_compression_ratio',
    'compute_boundary_f1',
    'resolve_dataset_path',
    'parse_example',
    'EvalResult',
]
