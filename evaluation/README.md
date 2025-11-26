# LuKA Evaluation Pipeline

Model-agnostic evaluation framework for comparing baseline and LuKA-enabled models on long-context Q&A tasks.

## Overview

This pipeline evaluates models on WikiSalad datasets using two modes:
1. **Simple Q&A**: Each question gets full context independently
2. **Sequential Multi-Answer**: Build context incrementally (`<docs><q1><ans1><docs><q2>...`)

The framework is designed to be **model-agnostic** - it works with:
- Local HuggingFace models
- API endpoints (OpenAI-compatible, DashScope, etc.)
- LuKA-enabled models
- Any custom implementation of `ModelInterface`

## Quick Start

### 1. Simple Evaluation with Local Model

```bash
python -m evaluation.run_evaluation \
    --dataset artifacts/hugging_face_wikipedia/wikisalad_eval_example.json \
    --model-type huggingface \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --eval-mode both \
    --output results/qwen3b_baseline.json
```

### 2. API-Based Evaluation

```bash
# Using Alibaba DashScope API for Qwen
export DASHSCOPE_API_KEY="your-api-key"

python -m evaluation.run_evaluation \
    --dataset artifacts/hugging_face_wikipedia/wikisalad_eval_example.json \
    --model-type api \
    --model-name qwen-turbo \
    --api-type dashscope \
    --eval-mode sequential \
    --output results/qwen_api.json
```

### 3. LuKA-Enabled Evaluation

```bash
python -m evaluation.run_evaluation \
    --dataset artifacts/hugging_face_wikipedia/wikisalad_eval_example.json \
    --model-type luka \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --luka-config configs/luka_default.json \
    --eval-mode both \
    --output results/qwen3b_luka.json
```

## Architecture

### Core Components

1. **`model_interface.py`** - Abstract interface for models
   - `ModelInterface`: Base interface all models implement
   - `BaselineModelInterface`: For standard models
   - `LuKAModelInterface`: For compression-enabled models
   - `GenerationConfig`: Generation parameters

2. **`evaluator.py`** - Evaluation pipeline
   - `QAEvaluator`: Runs evaluations on WikiSalad datasets
   - `evaluate_simple_qa()`: Simple Q&A mode
   - `evaluate_sequential_multi_answer()`: Sequential mode
   - `EvaluationResult`: Results container

3. **`model_implementations.py`** - Concrete model wrappers
   - `HuggingFaceModel`: Local HuggingFace models
   - `APIModel`: API-based models
   - `LuKAQwenModel`: LuKA-enabled models (stub)

4. **`run_evaluation.py`** - CLI runner

### Adding Custom Models

To add a custom model backend, implement `ModelInterface`:

```python
from evaluation import ModelInterface, GenerationConfig

class MyCustomModel(ModelInterface):
    def generate(self, prompt: str, config: GenerationConfig) -> str:
        # Your generation logic
        return generated_text

    def get_model_info(self) -> dict:
        return {'name': 'my-model', 'type': 'custom'}

    def reset(self):
        # Optional: reset state between calls
        pass
```

For LuKA models, implement `LuKAModelInterface` and add:
```python
def get_compression_stats(self) -> dict:
    return {
        'original_tokens': ...,
        'compressed_tokens': ...,
        'compression_ratio': ...,
    }

def get_decompressed_segments(self) -> list:
    return [...]  # List of segment IDs
```

## Evaluation Modes

### Simple Q&A
Each question is asked independently with full context:
```
For each question:
    Prompt = full_context + question
    Answer = model.generate(prompt)
```

### Sequential Multi-Answer
Questions build on each other with accumulated context:
```
For each question i:
    Prompt = previous_context + current_segment + question_i
    Answer_i = model.generate(prompt)
    Accumulate: context += segment + question_i + answer_i
```

This mimics realistic long-context scenarios where models must:
- Maintain context across multiple turns
- Compress irrelevant information
- Retrieve relevant details when needed

## Output Format

Results are saved as JSON:

```json
{
  "model_name": "Qwen/Qwen2.5-3B-Instruct",
  "eval_mode": "sequential",
  "dataset_name": "wikisalad_eval_example",
  "num_examples": 50,
  "predictions": [
    {
      "example_id": 0,
      "questions": ["What is...", "When did..."],
      "answers": ["The answer is...", "It happened in..."]
    }
  ],
  "compression_stats": [  // Only for LuKA models
    {
      "example_id": 0,
      "original_tokens": 2000,
      "compressed_tokens": 500,
      "compression_ratio": 4.0,
      "boundaries": [150, 300, 450]
    }
  ]
}
```

## Scoring Results

Use the existing `LuKAEvaluator` to score predictions:

```python
from artifacts.hugging_face_wikipedia.luka_evaluator import LuKAEvaluator

# Load evaluator with ground truth
evaluator = LuKAEvaluator("artifacts/hugging_face_wikipedia/wikisalad_eval_example.json")

# Format predictions for scoring
model_predictions = {
    'boundaries': [...],
    'compression_stats': [...],
    'decompressed_segments': [...],
    'qa_predictions': [...]
}

# Run evaluation
metrics = evaluator.run_full_evaluation(model_predictions)

print(f"QA Accuracy: {metrics.qa_accuracy:.3f}")
print(f"Compression Ratio: {metrics.compression_ratio:.2f}x")
```

## Configuration

### Generation Config
```python
GenerationConfig(
    max_new_tokens=100,      # Max tokens to generate
    temperature=0.0,          # 0 = deterministic
    top_p=1.0,               # Nucleus sampling
    do_sample=False,         # Greedy if False
    stop_strings=None        # Optional stop sequences
)
```

### LuKA Config (example)
```json
{
  "segmenter_type": "kl_divergence",
  "page_size": 128,
  "compression_threshold": 0.5,
  "top_k_boundaries": 10
}
```

## Next Steps

1. **Complete LuKA Integration**: Fill in `LuKAQwenModel` stub with actual compression
2. **Add Scoring Pipeline**: Integrate with `LuKAEvaluator` for automatic metric computation
3. **Comparison Script**: Create script to compare baseline vs LuKA results
4. **Visualization**: Add plots for compression vs accuracy tradeoffs

## Requirements

```
torch>=2.1
transformers>=4.40
tqdm
datasets  # For loading SQuAD/WikiSalad data
```