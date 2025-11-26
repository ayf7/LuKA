"""
Score Evaluation Results

Bridges the evaluation pipeline with LuKAEvaluator to compute metrics:
- QA Accuracy (Exact Match + F1)
- Compression metrics (for LuKA models)
- Boundary detection (for LuKA models)

Usage:
    python -m evaluation.score_results \
        --results results/qwen3b_baseline_sequential.json \
        --dataset artifacts/hugging_face_wikipedia/wikisalad_eval_example.json \
        --output results/qwen3b_baseline_sequential_scores.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from artifacts.hugging_face_wikipedia.luka_evaluator import LuKAEvaluator
import numpy as np


def convert_predictions_for_scoring(eval_result: dict) -> dict:
    """
    Convert evaluation result format to LuKAEvaluator format.

    Args:
        eval_result: Output from QAEvaluator

    Returns:
        Dict in format expected by LuKAEvaluator.run_full_evaluation()
    """
    predictions = eval_result['predictions']
    compression_stats = eval_result.get('compression_stats', None)

    # Build the format expected by LuKAEvaluator
    model_predictions = {
        'qa_predictions': [pred['answers'] for pred in predictions],
        'boundaries': [],
        'compression_stats': [],
        'decompressed_segments': []
    }

    # If we have compression stats (LuKA model), include them
    if compression_stats:
        for stats in compression_stats:
            model_predictions['boundaries'].append(stats.get('boundaries', []))
            model_predictions['compression_stats'].append({
                'original_tokens': stats.get('original_tokens', 0),
                'compressed_tokens': stats.get('compressed_tokens', 0),
                'summary_tokens': stats.get('summary_tokens', 0)
            })
            model_predictions['decompressed_segments'].append(
                stats.get('decompressed_segments', [])
            )
    else:
        # For baseline models, create dummy compression stats
        num_examples = len(predictions)
        model_predictions['boundaries'] = [[] for _ in range(num_examples)]
        model_predictions['compression_stats'] = [
            {
                'original_tokens': 0,
                'compressed_tokens': 0,
                'summary_tokens': 0
            } for _ in range(num_examples)
        ]
        model_predictions['decompressed_segments'] = [[] for _ in range(num_examples)]

    return model_predictions


def _create_bar(value: float, max_value: float = 1.0, width: int = 40) -> str:
    """Create an ASCII bar chart."""
    filled = int((value / max_value) * width)
    bar = '█' * filled + '░' * (width - filled)
    return bar


def _get_color_indicator(value: float, thresholds: dict) -> str:
    """Get color indicator based on thresholds."""
    if value >= thresholds.get('good', 0.8):
        return '✓'  # Good
    elif value >= thresholds.get('ok', 0.6):
        return '○'  # OK
    else:
        return '✗'  # Bad


def _print_enhanced_summary(metrics, eval_result):
    """Print enhanced summary with ASCII visualizations."""
    is_luka = eval_result.get('compression_stats') is not None

    print(f"\n{'='*70}")
    print(f"{'SCORING RESULTS':^70}")
    print(f"{'='*70}")
    print(f"\nModel: {eval_result['model_name']}")
    print(f"Mode: {eval_result['eval_mode']}")
    print(f"Type: {'LuKA-enabled' if is_luka else 'Baseline'}")
    print(f"Examples: {eval_result['num_examples']}")

    print(f"\n{'-'*70}")
    print("METRICS")
    print(f"{'-'*70}")

    # QA Accuracy
    qa_acc = metrics.qa_accuracy
    qa_indicator = _get_color_indicator(qa_acc, {'good': 0.7, 'ok': 0.5})
    qa_bar = _create_bar(qa_acc, 1.0, 40)
    print(f"\nQA Accuracy:                    {qa_acc:.3f} {qa_indicator}")
    print(f"  {qa_bar} {qa_acc*100:.1f}%")

    if is_luka:
        # Boundary Detection F1
        boundary_f1 = metrics.boundary_f1
        boundary_indicator = _get_color_indicator(boundary_f1, {'good': 0.7, 'ok': 0.5})
        boundary_bar = _create_bar(boundary_f1, 1.0, 40)
        print(f"\nBoundary Detection F1:          {boundary_f1:.3f} {boundary_indicator}")
        print(f"  {boundary_bar} {boundary_f1*100:.1f}%")

        # Compression Ratio
        comp_ratio = metrics.compression_ratio
        # For compression, higher is better (normalize to reasonable range)
        comp_normalized = min(comp_ratio / 10.0, 1.0)  # Assume 10x is max
        comp_indicator = _get_color_indicator(comp_normalized, {'good': 0.3, 'ok': 0.15})
        comp_bar = _create_bar(comp_normalized, 1.0, 40)
        print(f"\nCompression Ratio:              {comp_ratio:.2f}x {comp_indicator}")
        print(f"  {comp_bar} {comp_ratio:.1f}x")
        print(f"  (Memory saved: {(1 - 1/comp_ratio)*100:.1f}%)")

        # Selective Decompression
        decomp_acc = metrics.selective_decompression_accuracy
        decomp_indicator = _get_color_indicator(decomp_acc, {'good': 0.7, 'ok': 0.5})
        decomp_bar = _create_bar(decomp_acc, 1.0, 40)
        print(f"\nSelective Decompression Acc:    {decomp_acc:.3f} {decomp_indicator}")
        print(f"  {decomp_bar} {decomp_acc*100:.1f}%")

    print(f"\n{'='*70}")


def _print_per_example_breakdown(full_results: dict, eval_result: dict):
    """Print per-example breakdown table."""
    per_example = full_results.get('per_example', [])

    if not per_example or len(per_example) > 20:
        # Skip detailed breakdown if too many examples
        if len(per_example) > 20:
            print(f"\n(Skipping per-example breakdown - too many examples: {len(per_example)})")
        return

    is_luka = eval_result.get('compression_stats') is not None

    print(f"\n{'='*70}")
    print("PER-EXAMPLE BREAKDOWN")
    print(f"{'='*70}")

    if is_luka:
        # Header
        print(f"\n{'ID':<6} {'QA Acc':<10} {'Bound F1':<10} {'Comp Ratio':<12} {'Decomp Acc':<12}")
        print(f"{'-'*6} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")

        # Rows
        for ex in per_example:
            ex_id = ex['example_id']
            qa_acc = ex['qa_accuracy']
            bound_f1 = ex['boundary_f1']
            comp_ratio = ex['compression_ratio']
            decomp_acc = ex['decompression_accuracy']

            qa_ind = _get_color_indicator(qa_acc, {'good': 0.7, 'ok': 0.5})
            bound_ind = _get_color_indicator(bound_f1, {'good': 0.7, 'ok': 0.5})
            decomp_ind = _get_color_indicator(decomp_acc, {'good': 0.7, 'ok': 0.5})

            print(f"{ex_id:<6} {qa_acc:.3f} {qa_ind:<4} {bound_f1:.3f} {bound_ind:<4} "
                  f"{comp_ratio:.2f}x {' '*6} {decomp_acc:.3f} {decomp_ind:<4}")
    else:
        # Baseline: simpler table
        print(f"\n{'ID':<6} {'QA Accuracy':<15}")
        print(f"{'-'*6} {'-'*15}")

        for ex in per_example:
            ex_id = ex['example_id']
            qa_acc = ex['qa_accuracy']
            qa_ind = _get_color_indicator(qa_acc, {'good': 0.7, 'ok': 0.5})

            print(f"{ex_id:<6} {qa_acc:.3f} {qa_ind}")

    # Summary statistics
    qa_scores = [ex['qa_accuracy'] for ex in per_example]
    print(f"\n{'-'*70}")
    print(f"Summary: Mean={np.mean(qa_scores):.3f}, "
          f"Std={np.std(qa_scores):.3f}, "
          f"Min={np.min(qa_scores):.3f}, "
          f"Max={np.max(qa_scores):.3f}")

    if is_luka:
        comp_ratios = [ex['compression_ratio'] for ex in per_example]
        print(f"Compression: Mean={np.mean(comp_ratios):.2f}x, "
              f"Std={np.std(comp_ratios):.2f}x, "
              f"Min={np.min(comp_ratios):.2f}x, "
              f"Max={np.max(comp_ratios):.2f}x")

    print(f"{'='*70}")


def score_results(results_path: str, dataset_path: str, output_path: str):
    """
    Score evaluation results using LuKAEvaluator.

    Args:
        results_path: Path to evaluation results JSON
        dataset_path: Path to WikiSalad dataset (ground truth)
        output_path: Where to save scored results
    """
    print(f"\n{'='*60}")
    print("Scoring Evaluation Results")
    print(f"{'='*60}\n")

    # Load evaluation results
    print(f"Loading results from: {results_path}")
    with open(results_path, 'r') as f:
        eval_result = json.load(f)

    print(f"Model: {eval_result['model_name']}")
    print(f"Mode: {eval_result['eval_mode']}")
    print(f"Examples: {eval_result['num_examples']}")

    # Load full dataset
    print(f"\nLoading ground truth from: {dataset_path}")
    with open(dataset_path, 'r') as f:
        full_dataset = json.load(f)

    # Filter dataset to only include evaluated examples
    evaluated_example_ids = [pred['example_id'] for pred in eval_result['predictions']]
    filtered_dataset = [ex for ex in full_dataset if ex['example_id'] in evaluated_example_ids]

    print(f"Filtered to {len(filtered_dataset)} examples (matching predictions)")

    # Create evaluator with filtered dataset
    # We need to temporarily save the filtered dataset
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json.dump(filtered_dataset, tmp)
        tmp_path = tmp.name

    evaluator = LuKAEvaluator(tmp_path)

    # Clean up temp file
    import os
    os.unlink(tmp_path)

    # Convert predictions to scoring format
    print("\nConverting predictions to scoring format...")
    model_predictions = convert_predictions_for_scoring(eval_result)

    # Run scoring
    print("\nComputing metrics...")
    try:
        metrics = evaluator.run_full_evaluation(
            model_predictions,
            output_file=output_path
        )

        # Print enhanced summary
        _print_enhanced_summary(metrics, eval_result)

        print(f"\nDetailed results saved to: {output_path}")

        # Load and print per-example breakdown
        with open(output_path, 'r') as f:
            full_results = json.load(f)

        _print_per_example_breakdown(full_results, eval_result)

        return metrics

    except Exception as e:
        print(f"\nError during scoring: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Score evaluation results using LuKAEvaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to evaluation results JSON (from run_evaluation.py)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to WikiSalad dataset JSON (ground truth)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for scored results"
    )

    args = parser.parse_args()

    score_results(args.results, args.dataset, args.output)


if __name__ == "__main__":
    main()
