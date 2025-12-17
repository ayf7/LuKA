# Comprehensive Evaluation Script

This script evaluates LuKA attention modes (baseline, top-down, lined, mixed) across three dimensions:

1. **Perplexity Evaluation** - Measures language modeling quality
2. **Generation Quality** - Compares generated text outputs
3. **Compression Metrics** - Tracks memory usage and compression ratios

## Usage

```bash
# Basic usage
python experiments/comprehensive_eval.py --model Qwen/Qwen3-1.7B-Base --prompt paragraphs_1

# With custom settings
python experiments/comprehensive_eval.py \
    --model Qwen/Qwen3-1.7B-Base \
    --prompt paragraphs_1 \
    --device cuda \
    --max-new-tokens 256 \
    --output-dir experiments/eval_results
```

## Output

The script generates:

1. **JSON Results** (`eval_results_<prompt>.json`):
   - Perplexity scores for each mode
   - Generated text samples
   - Compression statistics
   - Speed metrics

2. **Visualization Plots** (in `plots/` directory):
   - `perplexity_comparison_*.png` - Bar chart comparing perplexities
   - `perplexity_curve_*.png` - Line chart showing perplexity over time
   - `compression_ratio_*.png` - Compression ratio comparison
   - `memory_usage_*.png` - Raw vs cover tokens comparison
   - `speed_comparison_*.png` - Generation speed (tokens/sec)
   - `comprehensive_comparison_*.png` - All metrics in one view

## Evaluation Modes

- **baseline**: Raw attention (no compression)
- **topdown**: LuKA with page summaries
- **lined**: H2O-style grid tokens
- **mixed**: Hybrid (early/late lined, middle top-down)

## Requirements

- matplotlib (for plots)
- torch
- transformers
- All LuKA dependencies

