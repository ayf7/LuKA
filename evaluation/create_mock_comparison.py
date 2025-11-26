"""
Create mock baseline and LuKA results for testing comparison script.
"""

import json
import random
import numpy as np

# Create mock baseline results
baseline = {
    "model_name": "Qwen/Qwen2.5-3B-Instruct",
    "eval_mode": "simple",
    "dataset_name": "wikisalad_eval_example",
    "num_examples": 10,
    "aggregate": {
        "qa_accuracy": 0.75,
        "boundary_precision": 0.0,
        "boundary_recall": 0.0,
        "boundary_f1": 0.0,
        "compression_ratio": 1.0,
        "tokens_original": 0,
        "tokens_compressed": 0,
        "selective_decompression_accuracy": 0.0
    },
    "per_example": []
}

# Create mock LuKA results (slightly lower accuracy but with compression)
luka = {
    "model_name": "Qwen/Qwen2.5-3B-Instruct_luka",
    "eval_mode": "simple",
    "dataset_name": "wikisalad_eval_example",
    "num_examples": 10,
    "aggregate": {
        "qa_accuracy": 0.72,
        "boundary_precision": 0.78,
        "boundary_recall": 0.82,
        "boundary_f1": 0.80,
        "compression_ratio": 4.5,
        "tokens_original": 2000,
        "tokens_compressed": 445,
        "selective_decompression_accuracy": 0.85
    },
    "per_example": []
}

# Generate per-example data
random.seed(42)
np.random.seed(42)

for i in range(10):
    # Baseline example
    baseline_qa = max(0.0, min(1.0, np.random.normal(0.75, 0.15)))

    baseline['per_example'].append({
        'example_id': i,
        'boundary_precision': 0.0,
        'boundary_recall': 0.0,
        'boundary_f1': 0.0,
        'compression_ratio': 1.0,
        'decompression_accuracy': 0.0,
        'qa_accuracy': baseline_qa
    })

    # LuKA example - generally maintains accuracy with some variation
    # Some examples do better, some worse
    if i < 6:
        # Most examples: slight drop but acceptable
        luka_qa = max(0.0, baseline_qa - np.random.uniform(0.01, 0.08))
    elif i < 8:
        # A few examples: actually improve
        luka_qa = min(1.0, baseline_qa + np.random.uniform(0.0, 0.05))
    else:
        # Some examples: larger drop
        luka_qa = max(0.0, baseline_qa - np.random.uniform(0.10, 0.15))

    # Compression varies by example
    compression = np.random.uniform(3.5, 5.5)

    luka['per_example'].append({
        'example_id': i,
        'boundary_precision': np.random.uniform(0.7, 0.85),
        'boundary_recall': np.random.uniform(0.75, 0.90),
        'boundary_f1': np.random.uniform(0.75, 0.85),
        'compression_ratio': compression,
        'decompression_accuracy': np.random.uniform(0.80, 0.90),
        'qa_accuracy': luka_qa
    })

# Save results
with open('results/mock_baseline_scores.json', 'w') as f:
    json.dump(baseline, f, indent=2)

with open('results/mock_luka_scores.json', 'w') as f:
    json.dump(luka, f, indent=2)

print("Mock results created:")
print("  - results/mock_baseline_scores.json")
print("  - results/mock_luka_scores.json")
print("\nYou can now test the comparison script with:")
print("  python -m evaluation.compare_results \\")
print("    --baseline results/mock_baseline_scores.json \\")
print("    --luka results/mock_luka_scores.json \\")
print("    --output results/comparison_report.html")
