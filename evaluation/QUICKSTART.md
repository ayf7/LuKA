# LuKA Evaluation Pipeline - Quick Start Guide

## What This Pipeline Does

Evaluates **baseline vs LuKA-enabled models** to compare:
- **QA Accuracy**: Can LuKA answer questions as well as baseline?
- **Compression Ratio**: How much memory does LuKA save?
- **Quality vs Efficiency**: Finding the sweet spot

## Three-Step Workflow

### Step 1: Run Baseline Model

```bash
python -m evaluation.run_evaluation \
    --dataset artifacts/hugging_face_wikipedia/wikisalad_eval_example.json \
    --model-type huggingface \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --eval-mode both \
    --num-examples 10 \
    --output results/baseline_qwen3b.json
```

This runs both evaluation modes:
- **Simple Q&A**: Each question answered independently
- **Sequential**: Questions build on each other (Anthony's `<docs><q1><ans1><docs><q2>...` pattern)

### Step 2: Run LuKA Model

```bash
python -m evaluation.run_evaluation \
    --dataset artifacts/hugging_face_wikipedia/wikisalad_eval_example.json \
    --model-type luka \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --luka-config configs/luka_default.json \
    --eval-mode both \
    --num-examples 10 \
    --output results/luka_qwen3b.json
```

**Note**: The `LuKAQwenModel` class in `model_implementations.py` is currently a stub. You'll need to:
1. Integrate with actual LuKA compression from `modeling/qwen/luka_qwen3.py`
2. Or use a pre-configured model from Anthony

### Step 3: Score Both Results

```bash
# Score baseline
python -m evaluation.score_results \
    --results results/baseline_qwen3b_simple.json \
    --dataset artifacts/hugging_face_wikipedia/wikisalad_eval_example.json \
    --output results/baseline_qwen3b_simple_scores.json

# Score LuKA
python -m evaluation.score_results \
    --results results/luka_qwen3b_simple.json \
    --dataset artifacts/hugging_face_wikipedia/wikisalad_eval_example.json \
    --output results/luka_qwen3b_simple_scores.json
```

This computes:
- QA Accuracy (Exact Match + F1)
- Compression ratio (for LuKA)
- Boundary detection F1 (for LuKA)
- Selective decompression accuracy (for LuKA)

## Alternative: Using API Instead of Local Models

If you don't want to download model weights locally:

```bash
# Using Alibaba DashScope API
export DASHSCOPE_API_KEY="your-key-here"

python -m evaluation.run_evaluation \
    --dataset artifacts/hugging_face_wikipedia/wikisalad_eval_example.json \
    --model-type api \
    --model-name qwen-turbo \
    --api-type dashscope \
    --eval-mode sequential \
    --num-examples 10 \
    --output results/api_baseline.json
```

**Limitation**: APIs won't let you run LuKA compression (no KV cache access).
Use APIs for **baseline comparison only**.

## Adding Your Own Model

If Anthony gives you a pre-configured Qwen-LuKA model:

```python
# In evaluation/model_implementations.py or a new file

from evaluation import LuKAModelInterface, GenerationConfig

class AnthonysLuKAModel(LuKAModelInterface):
    def __init__(self, model_path: str):
        # Load Anthony's model
        self.model = load_model_somehow(model_path)

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        # Generate with LuKA compression
        return self.model.generate(prompt, ...)

    def get_compression_stats(self) -> dict:
        # Extract stats from LuKA cache
        return {
            'original_tokens': ...,
            'compressed_tokens': ...,
            'compression_ratio': ...,
        }

    # ... implement other required methods
```

Then use it:

```python
from evaluation import QAEvaluator
from your_file import AnthonysLuKAModel

model = AnthonysLuKAModel("/path/to/model")
evaluator = QAEvaluator("artifacts/hugging_face_wikipedia/wikisalad_eval_example.json")

result = evaluator.evaluate_sequential_multi_answer(model)
evaluator.save_results(result, "results/anthonys_model.json")
```

## Testing Without a Real Model

Run the example with a mock model:

```bash
python -m evaluation.example_usage
```

This creates `results/example_simple.json` and `results/example_sequential.json` using a dummy model that generates random answers.

## Output Files Explained

### Evaluation Results (`results/baseline_qwen3b_simple.json`)

```json
{
  "model_name": "Qwen/Qwen2.5-3B-Instruct",
  "eval_mode": "simple",
  "dataset_name": "wikisalad_eval_example",
  "num_examples": 10,
  "predictions": [
    {
      "example_id": 0,
      "questions": ["What is...", "When did..."],
      "answers": ["The answer is...", "It happened in..."]
    }
  ],
  "compression_stats": null  // Only present for LuKA models
}
```

### Scored Results (`results/baseline_qwen3b_simple_scores.json`)

```json
{
  "aggregate": {
    "qa_accuracy": 0.75,
    "boundary_f1": 0.82,  // LuKA only
    "compression_ratio": 4.5,  // LuKA only
    "selective_decompression_accuracy": 0.88  // LuKA only
  },
  "per_example": [
    {
      "example_id": 0,
      "qa_accuracy": 0.8,
      "compression_ratio": 4.2
    }
  ]
}
```

## What to Report to Anthony

After running evaluations, report:

1. **QA Accuracy Comparison**
   - Baseline: X.XX
   - LuKA: X.XX
   - Difference: X.XX%

2. **Compression Achieved**
   - Average ratio: X.Xx
   - Memory saved: XX%

3. **Quality Maintained?**
   - If accuracy drop < 5% and compression > 2x: **Success!**
   - If accuracy drop > 10%: **Need to tune LuKA parameters**

## Next Steps

1. **Complete LuKA Integration**: Fill in the `LuKAQwenModel` stub with actual compression
2. **Run Full Evaluation**: Remove `--num-examples` limit to evaluate all 50 examples
3. **Try Different Datasets**: WikiSalad has multiple difficulty levels in `artifacts/hugging_face_wikipedia/wikisalad_datasets/`
4. **Parameter Sweep**: Try different LuKA configurations to find optimal compression/accuracy tradeoff

## Troubleshooting

**Problem**: "Model not found" error
**Solution**: Check model name is correct on HuggingFace. For Qwen models, use `Qwen/Qwen2.5-3B-Instruct` (not `qwen2.5-3b`)

**Problem**: "CUDA out of memory"
**Solution**: Use a smaller model (3B instead of 7B) or add `--device cpu`

**Problem**: "API key not found"
**Solution**: Export API key: `export DASHSCOPE_API_KEY=your-key`

**Problem**: Scoring fails with "list index out of range"
**Solution**: This is now fixed! But if it happens, make sure the results file contains predictions.

## Files Overview

```
evaluation/
├── model_interface.py        # Abstract interface for models
├── evaluator.py              # Core evaluation logic
├── model_implementations.py  # HuggingFace, API, LuKA wrappers
├── run_evaluation.py         # CLI entry point
├── score_results.py          # Score predictions
├── example_usage.py          # Demo with mock model
├── README.md                 # Detailed documentation
└── QUICKSTART.md             # This file
```
