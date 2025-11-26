# Comparison Script Guide

## Overview

The comparison script creates comprehensive reports comparing baseline and LuKA-enabled models, with both ASCII terminal output and rich HTML visualizations.

## Quick Usage

```bash
# 1. Run baseline evaluation
python -m evaluation.run_evaluation \
    --dataset artifacts/hugging_face_wikipedia/wikisalad_eval_example.json \
    --model-type huggingface \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --eval-mode simple \
    --output results/baseline_simple.json

# 2. Score baseline
python -m evaluation.score_results \
    --results results/baseline_simple.json \
    --dataset artifacts/hugging_face_wikipedia/wikisalad_eval_example.json \
    --output results/baseline_simple_scores.json

# 3. Run LuKA evaluation
python -m evaluation.run_evaluation \
    --dataset artifacts/hugging_face_wikipedia/wikisalad_eval_example.json \
    --model-type luka \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --eval-mode simple \
    --output results/luka_simple.json

# 4. Score LuKA
python -m evaluation.score_results \
    --results results/luka_simple.json \
    --dataset artifacts/hugging_face_wikipedia/wikisalad_eval_example.json \
    --output results/luka_simple_scores.json

# 5. Compare!
python -m evaluation.compare_results \
    --baseline results/baseline_simple_scores.json \
    --luka results/luka_simple_scores.json \
    --output results/comparison_report.html
```

## Output Features

### Terminal Output

**1. Summary Comparison Table**
```
==========================================================================================
                               BASELINE vs LuKA COMPARISON
==========================================================================================

Metric                              Baseline             LuKA                 Δ
------------------------------------------------------------------------------------------
QA Accuracy                         0.750 (75.0%)     0.720 (72.0%)     -0.030 (-4.0%) ✓
Boundary Detection F1               N/A                  0.800 (80.0%)     N/A
Compression Ratio                   1.00x (no compression) 4.50x                ✓
Memory Saved                        0%                   77.8%
Selective Decompression Acc         N/A                  0.850 (85.0%)     N/A
==========================================================================================

ASSESSMENT:
  ✓ SUCCESS: LuKA maintains accuracy while achieving good compression
==========================================================================================
```

**Indicators:**
- ✓ = Good (accuracy drop < 5%, compression > 2x)
- ○ = Acceptable (accuracy drop < 10%)
- ✗ = Needs attention (accuracy drop > 10% or low compression)

**2. Per-Example Analysis**
- Best cases: Examples where LuKA maintains or improves accuracy
- Worst cases: Examples where LuKA degrades accuracy most
- Statistics: Mean, std dev, min, max for differences and compression

**3. ASCII Scatter Plot**
- Visual representation of accuracy vs compression tradeoff
- Each ● represents one example
- Shows if there's a correlation between compression and accuracy drop

**4. Per-Example Heatmap**
- Bar charts showing baseline vs LuKA accuracy for each example
- Compression ratio and indicator for each example
- Quickly spot problematic examples

### HTML Report

Generated HTML includes:

**1. Summary Metrics Table**
- Side-by-side comparison with color coding
- Overall assessment (success/acceptable/needs tuning)

**2. Four Interactive Plots**:
1. **Bar chart**: Baseline vs LuKA accuracy by example
2. **Scatter plot**: Accuracy vs compression tradeoff
   - Green dots: < 5% accuracy drop
   - Orange dots: 5-10% accuracy drop
   - Red dots: > 10% accuracy drop
3. **Delta chart**: Shows accuracy difference per example
   - Reference lines at -5% and -10%
4. **Histogram**: Compression ratio distribution

**3. Statistics Summary**
- Detailed breakdown of performance across all examples
- Count of problematic examples

## Interpreting Results

### Success Criteria

**✓ Good Result:**
- QA Accuracy drop < 5%
- Compression ratio > 2.0x
- Most examples are green in scatter plot

**○ Acceptable:**
- QA Accuracy drop 5-10%
- Compression ratio > 2.0x
- Some orange dots acceptable

**✗ Needs Tuning:**
- QA Accuracy drop > 10%
- Many red dots in scatter plot
- High variance in per-example results

### What to Report to Anthony

When you have results, report:

1. **Aggregate Metrics**:
   - "LuKA achieves X.Xx compression with only Y.Y% accuracy drop"
   - "Z out of N examples maintain >95% accuracy"

2. **Tradeoff Analysis**:
   - "Best case: X.Xx compression with no accuracy loss (example ID)"
   - "Worst case: X.Xx compression with Y% accuracy drop (example ID)"

3. **Recommendations**:
   - If accuracy drop > 10%: "Consider tuning segmentation parameters"
   - If compression < 2x: "Consider more aggressive compression settings"
   - If success: "Ready for larger-scale evaluation"

## Advanced Usage

### Comparing Multiple Modes

```bash
# Compare both simple and sequential modes
python -m evaluation.compare_results \
    --baseline results/baseline_simple_scores.json results/baseline_sequential_scores.json \
    --luka results/luka_simple_scores.json results/luka_sequential_scores.json \
    --output results/full_comparison.html
```

(Note: Currently only compares first pair, multi-comparison support coming soon)

### Text-Only Output

Skip HTML generation for quick terminal-only analysis:

```bash
python -m evaluation.compare_results \
    --baseline results/baseline_simple_scores.json \
    --luka results/luka_simple_scores.json \
    --output results/comparison_report.html \
    --no-html
```

## Testing Without Real Models

Use the mock data generator to test the pipeline:

```bash
python evaluation/create_mock_comparison.py

python -m evaluation.compare_results \
    --baseline results/mock_baseline_scores.json \
    --luka results/mock_luka_scores.json \
    --output results/mock_comparison.html
```

This creates realistic mock results with:
- 10 examples
- ~4.5x average compression
- ~4% accuracy drop
- Some examples improve, some degrade

## Customization

### Threshold Tuning

Edit thresholds in `compare_results.py`:

```python
# Success criteria
if qa_diff >= -0.05 and comp_ratio > 2.0:  # Adjust -0.05 and 2.0
    assessment = "SUCCESS"
```

### Plot Styling

For HTML plots, modify matplotlib settings in `generate_html_report()`:

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # Adjust size
ax.set_style(...)  # Custom styling
```

## Dependencies

Required:
- numpy (for statistics)

Optional (for HTML reports):
- matplotlib (auto-imported if available)
- If not installed: `pip install matplotlib`

## Troubleshooting

**Issue**: "No common examples found"
**Solution**: Make sure baseline and LuKA were evaluated on same dataset subset

**Issue**: HTML plots not generated
**Solution**: Install matplotlib: `pip install matplotlib`

**Issue**: "list index out of range" in scoring
**Solution**: This should be fixed. Make sure you're using scored results (output of `score_results.py`), not raw evaluation results

**Issue**: Scatter plot looks empty
**Solution**: Check that per-example data exists. Some examples may have zero accuracy causing them to be at bottom.

## Files Generated

```
results/
├── baseline_simple.json              # Raw baseline predictions
├── baseline_simple_scores.json       # Scored baseline (input to compare)
├── luka_simple.json                  # Raw LuKA predictions
├── luka_simple_scores.json           # Scored LuKA (input to compare)
├── comparison_report.html            # Final HTML report
└── comparison_report_plots.png       # Embedded plots
```

## Next Steps

1. **Run on Full Dataset**: Remove `--num-examples` limit
2. **Try Different Modes**: Compare simple vs sequential
3. **Parameter Sweep**: Test different LuKA configurations
4. **Analyze Failures**: Inspect examples with high accuracy drop
