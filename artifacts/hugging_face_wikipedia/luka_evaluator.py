"""
LuKA Evaluation Script for WikiSalad Dataset

Evaluates the LuKA compression scheme on interlaced Wikipedia articles:
1. Boundary Detection: How well does LuKA identify topic transitions?
2. Compression Efficiency: Token savings while maintaining accuracy
3. Selective Attention: Does it decompress the right segments for questions?
4. QA Performance: Accuracy on questions after compression
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class CompressionMetrics:
    """Metrics for evaluating compression performance."""
    boundary_precision: float
    boundary_recall: float
    boundary_f1: float
    compression_ratio: float
    tokens_original: int
    tokens_compressed: int
    qa_accuracy: float
    selective_decompression_accuracy: float


class LuKAEvaluator:
    """Evaluates LuKA's performance on WikiSalad datasets."""
    
    def __init__(self, dataset_path: str):
        """Load the WikiSalad evaluation dataset."""
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)
        print(f"Loaded {len(self.dataset)} evaluation examples")
    
    def evaluate_boundary_detection(
        self,
        predicted_boundaries: List[int],
        true_boundaries: List[int],
        tolerance: int = 50
    ) -> Tuple[float, float, float]:
        """
        Evaluate boundary detection with position tolerance.
        
        Args:
            predicted_boundaries: List of character positions where boundaries were predicted
            true_boundaries: List of true boundary positions from metadata
            tolerance: How many characters off a prediction can be to count as correct
            
        Returns:
            (precision, recall, f1)
        """
        if len(predicted_boundaries) == 0:
            return 0.0, 0.0, 0.0
        
        # Count true positives
        matched_predictions = set()
        matched_ground_truth = set()
        
        for pred_idx, pred_pos in enumerate(predicted_boundaries):
            for true_idx, true_pos in enumerate(true_boundaries):
                if true_idx not in matched_ground_truth:
                    if abs(pred_pos - true_pos) <= tolerance:
                        matched_predictions.add(pred_idx)
                        matched_ground_truth.add(true_idx)
                        break
        
        tp = len(matched_predictions)
        fp = len(predicted_boundaries) - tp
        fn = len(true_boundaries) - len(matched_ground_truth)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def evaluate_compression_ratio(
        self,
        original_tokens: int,
        compressed_tokens: int,
        summary_tokens: int = 0
    ) -> float:
        """
        Calculate compression ratio.
        
        Args:
            original_tokens: Number of tokens in original context
            compressed_tokens: Number of raw tokens still in cache
            summary_tokens: Number of summary key-value tokens
            
        Returns:
            Compression ratio (higher is better)
        """
        total_compressed = compressed_tokens + summary_tokens
        return original_tokens / total_compressed if total_compressed > 0 else 1.0
    
    def evaluate_selective_decompression(
        self,
        question_segment_ids: List[int],
        decompressed_segment_ids: List[int],
        total_segments: int
    ) -> float:
        """
        Evaluate whether LuKA decompresses the correct segments.
        
        For each question, we know which segment(s) contain the answer.
        LuKA should decompress those segments and keep others compressed.
        
        Args:
            question_segment_ids: Which segments contain answers for each question
            decompressed_segment_ids: Which segments LuKA chose to decompress
            total_segments: Total number of segments
            
        Returns:
            Accuracy of selective decompression (0-1)
        """
        # Convert to sets
        should_decompress = set(question_segment_ids)
        did_decompress = set(decompressed_segment_ids)
        
        # Calculate metrics
        tp = len(should_decompress & did_decompress)  # Correctly decompressed
        fp = len(did_decompress - should_decompress)  # Unnecessarily decompressed
        fn = len(should_decompress - did_decompress)  # Missed necessary decompressions
        tn = total_segments - len(should_decompress | did_decompress)  # Correctly kept compressed
        
        accuracy = (tp + tn) / total_segments if total_segments > 0 else 0.0
        return accuracy
    
    def evaluate_qa_performance(
        self,
        predictions: List[str],
        ground_truths: List[Dict]
    ) -> float:
        """
        Evaluate QA accuracy using exact match and F1.
        
        Args:
            predictions: List of predicted answers
            ground_truths: List of answer dictionaries from dataset
            
        Returns:
            Combined accuracy score
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Number of predictions must match ground truths")
        
        exact_matches = []
        f1_scores = []
        
        for pred, gt_dict in zip(predictions, ground_truths):
            # Handle SQuAD format where answers might be empty (unanswerable)
            gt_answers = gt_dict.get('text', [])
            if not gt_answers:  # Unanswerable question
                exact_matches.append(1.0 if pred.strip() == "" else 0.0)
                f1_scores.append(1.0 if pred.strip() == "" else 0.0)
                continue
            
            # Check exact match against any answer
            exact_match = max(
                1.0 if self._normalize_answer(pred) == self._normalize_answer(gt) else 0.0
                for gt in gt_answers
            )
            exact_matches.append(exact_match)
            
            # Calculate F1 against best matching answer
            f1 = max(self._compute_f1(pred, gt) for gt in gt_answers)
            f1_scores.append(f1)
        
        return (np.mean(exact_matches) + np.mean(f1_scores)) / 2
    
    def _normalize_answer(self, text: str) -> str:
        """Normalize answer for comparison."""
        import re
        text = text.lower()
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        text = ' '.join(text.split())
        return text.strip()
    
    def _compute_f1(self, prediction: str, ground_truth: str) -> float:
        """Compute F1 score between prediction and ground truth."""
        pred_tokens = self._normalize_answer(prediction).split()
        gt_tokens = self._normalize_answer(ground_truth).split()
        
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return int(pred_tokens == gt_tokens)
        
        common_tokens = set(pred_tokens) & set(gt_tokens)
        
        if len(common_tokens) == 0:
            return 0.0
        
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(gt_tokens)
        
        return 2 * precision * recall / (precision + recall)
    
    def create_baseline_report(self, output_file: str = "baseline_analysis.json"):
        """
        Create a baseline analysis of the dataset:
        - Average segment lengths
        - Boundary positions
        - Question distribution across segments
        - Difficulty metrics
        """
        analysis = {
            'num_examples': len(self.dataset),
            'examples': []
        }
        
        for example in self.dataset:
            metadata = example['metadata']
            questions = example['questions']
            
            # Analyze segments
            segment_lengths = []
            boundary_positions = []
            
            for i, boundary in enumerate(metadata['segment_boundaries']):
                length = boundary['end_char'] - boundary['start_char']
                segment_lengths.append(length)
                boundary_positions.append(boundary['end_char'])
            
            # Analyze question distribution
            question_segments = defaultdict(int)
            for q in questions:
                question_segments[q['topic']] += 1
            
            analysis['examples'].append({
                'example_id': example['example_id'],
                'pattern': metadata['pattern'],
                'num_segments': len(metadata['segment_boundaries']),
                'num_topics': len(set(b['topic'] for b in metadata['segment_boundaries'])),
                'avg_segment_length': np.mean(segment_lengths),
                'total_length': sum(segment_lengths),
                'boundary_positions': boundary_positions,
                'question_distribution': dict(question_segments),
                'num_questions': len(questions)
            })
        
        # Compute aggregate statistics
        analysis['aggregate'] = {
            'avg_segments_per_example': np.mean([e['num_segments'] for e in analysis['examples']]),
            'avg_prompt_length': np.mean([e['total_length'] for e in analysis['examples']]),
            'avg_questions_per_example': np.mean([e['num_questions'] for e in analysis['examples']])
        }
        
        # Save analysis
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Baseline analysis saved to {output_file}")
        return analysis
    
    def run_full_evaluation(
        self,
        model_predictions: Dict,
        output_file: str = "luka_evaluation_results.json"
    ) -> CompressionMetrics:
        """
        Run complete evaluation on model predictions.
        
        Args:
            model_predictions: Dict containing:
                - 'boundaries': List of predicted boundary positions per example
                - 'compression_stats': Dict with token counts
                - 'decompressed_segments': Which segments were decompressed
                - 'qa_predictions': Predicted answers to questions
            output_file: Where to save results
            
        Returns:
            CompressionMetrics object
        """
        all_boundary_metrics = []
        all_compression_ratios = []
        all_decompression_accuracies = []
        all_qa_scores = []
        
        results = {'per_example': []}
        
        for i, example in enumerate(self.dataset):
            example_id = example['example_id']
            metadata = example['metadata']
            
            # Get true boundary positions
            true_boundaries = [b['end_char'] for b in metadata['segment_boundaries']]
            
            # Evaluate boundaries
            pred_boundaries = model_predictions['boundaries'][i]
            p, r, f1 = self.evaluate_boundary_detection(pred_boundaries, true_boundaries)
            all_boundary_metrics.append((p, r, f1))
            
            # Evaluate compression
            comp_stats = model_predictions['compression_stats'][i]
            comp_ratio = self.evaluate_compression_ratio(
                comp_stats['original_tokens'],
                comp_stats['compressed_tokens'],
                comp_stats.get('summary_tokens', 0)
            )
            all_compression_ratios.append(comp_ratio)
            
            # Evaluate selective decompression
            question_segments = [q['segment_id'] for q in example['questions']]
            decompressed = model_predictions['decompressed_segments'][i]
            decomp_acc = self.evaluate_selective_decompression(
                question_segments,
                decompressed,
                len(metadata['segment_boundaries'])
            )
            all_decompression_accuracies.append(decomp_acc)
            
            # Evaluate QA
            qa_preds = model_predictions['qa_predictions'][i]
            qa_gts = [q['answers'] for q in example['questions']]
            qa_acc = self.evaluate_qa_performance(qa_preds, qa_gts)
            all_qa_scores.append(qa_acc)
            
            results['per_example'].append({
                'example_id': example_id,
                'boundary_precision': p,
                'boundary_recall': r,
                'boundary_f1': f1,
                'compression_ratio': comp_ratio,
                'decompression_accuracy': decomp_acc,
                'qa_accuracy': qa_acc
            })
        
        # Aggregate results
        avg_p, avg_r, avg_f1 = np.mean(all_boundary_metrics, axis=0)
        
        metrics = CompressionMetrics(
            boundary_precision=float(avg_p),
            boundary_recall=float(avg_r),
            boundary_f1=float(avg_f1),
            compression_ratio=float(np.mean(all_compression_ratios)),
            tokens_original=int(np.mean([s['original_tokens'] 
                for s in model_predictions['compression_stats']])),
            tokens_compressed=int(np.mean([s['compressed_tokens'] + s.get('summary_tokens', 0)
                for s in model_predictions['compression_stats']])),
            qa_accuracy=float(np.mean(all_qa_scores)),
            selective_decompression_accuracy=float(np.mean(all_decompression_accuracies))
        )
        
        results['aggregate'] = metrics.__dict__
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Evaluation results saved to {output_file}")
        return metrics


# Example usage
def example_evaluation():
    """Example of how to use the evaluator with mock predictions."""
    
    # Load evaluator
    evaluator = LuKAEvaluator("wikisalad_datasets/easy_2topic_short.json")
    
    # Create baseline analysis
    evaluator.create_baseline_report()
    
    # Mock predictions (you'll replace this with actual LuKA output)
    mock_predictions = {
        'boundaries': [[500, 1000, 1500] for _ in range(len(evaluator.dataset))],
        'compression_stats': [
            {
                'original_tokens': 1000,
                'compressed_tokens': 200,
                'summary_tokens': 50
            } for _ in range(len(evaluator.dataset))
        ],
        'decompressed_segments': [[0, 2] for _ in range(len(evaluator.dataset))],
        'qa_predictions': [
            ["answer1", "answer2", "answer3", "answer4"] 
            for _ in range(len(evaluator.dataset))
        ]
    }
    
    # Run evaluation
    metrics = evaluator.run_full_evaluation(mock_predictions)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Boundary Detection F1: {metrics.boundary_f1:.3f}")
    print(f"Compression Ratio: {metrics.compression_ratio:.2f}x")
    print(f"QA Accuracy: {metrics.qa_accuracy:.3f}")
    print(f"Selective Decompression: {metrics.selective_decompression_accuracy:.3f}")
    print("="*60)


if __name__ == "__main__":
    example_evaluation()
