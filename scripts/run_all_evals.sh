#!/bin/bash
#
# Run all evaluation combinations: datasets x attention modes
#
# Usage:
#   ./scripts/run_all_evals.sh                    # Run all combinations
#   ./scripts/run_all_evals.sh --max-examples 5   # Quick test with 5 examples
#   ./scripts/run_all_evals.sh --dry-run          # Show commands without running
#   ./scripts/run_all_evals.sh --refinement none  # No refinement (for fair comparison)
#   ./scripts/run_all_evals.sh --modes top_down   # Run only top_down
#   ./scripts/run_all_evals.sh --modes "top_down lined"  # Run top_down and lined
#   ./scripts/run_all_evals.sh --resume           # Skip existing result files
#
# Output: results/eval_{attention}_{dataset}.json
#
# Current settings:
#   - Compressor: attention_weighted (temperature=7.0)
#   - Segmenter: dummy (min_chunk=8, tail_len=128, max_pages=256)
#   - H2O (lined): heavy_ratio=0.1, recent_ratio=0.1 (20% total retention)
#
# H2O-style lined attention parameters:
#   --heavy-ratio 0.1  : Keep 10% of tokens as heavy hitters
#   --recent-ratio 0.1 : Keep 10% of tokens as recent window

set -e

# Configuration
DATASETS=("easy" "medium" "hard" "very_hard")
ATTENTION_MODES=("baseline" "top_down" "lined" "mix")
MODEL="qwen-4b"
OUTPUT_DIR="evaluation/results"
REFINEMENT_RULE="none"
REFINEMENT_K="3"
LOG_BIAS="adaptive_k"
CUSTOM_MODES=""
HEAVY_RATIO="0.1"
RECENT_RATIO="0.1"

# Parse arguments
DRY_RUN=false
RESUME=false
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --max-examples)
            EXTRA_ARGS="$EXTRA_ARGS --max-examples $2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --refinement)
            REFINEMENT_RULE="$2"
            shift 2
            ;;
        --refinement-k)
            REFINEMENT_K="$2"
            shift 2
            ;;
        --log-bias)
            LOG_BIAS="$2"
            shift 2
            ;;
        --modes)
            CUSTOM_MODES="$2"
            shift 2
            ;;
        --heavy-ratio)
            HEAVY_RATIO="$2"
            shift 2
            ;;
        --recent-ratio)
            RECENT_RATIO="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Override attention modes if custom modes specified
if [[ -n "$CUSTOM_MODES" ]]; then
    read -ra ATTENTION_MODES <<< "$CUSTOM_MODES"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Count total runs
TOTAL=$((${#DATASETS[@]} * ${#ATTENTION_MODES[@]}))
CURRENT=0

echo "=============================================="
echo "LuKA Batch Evaluation"
echo "=============================================="
echo "Model:       $MODEL"
echo "Datasets:    ${DATASETS[*]}"
echo "Modes:       ${ATTENTION_MODES[*]}"
echo "Output:      $OUTPUT_DIR"
echo "Refinement:  $REFINEMENT_RULE (k=$REFINEMENT_K)"
echo "Log Bias:    $LOG_BIAS"
echo "H2O:         heavy_ratio=$HEAVY_RATIO, recent_ratio=$RECENT_RATIO"
echo "Total:       $TOTAL runs"
echo "Resume:      $RESUME"
echo "Extra:       $EXTRA_ARGS"
echo ""
echo "Compression settings:"
echo "  Segmenter: min_chunk=8, tail_len=128, max_pages=256"
echo "  H2O (lined): heavy=${HEAVY_RATIO}, recent=${RECENT_RATIO}"
echo "=============================================="
echo ""

# Run all combinations
for DATASET in "${DATASETS[@]}"; do
    for ATTENTION in "${ATTENTION_MODES[@]}"; do
        CURRENT=$((CURRENT + 1))
        OUTPUT_FILE="$OUTPUT_DIR/eval_${ATTENTION}_${DATASET}.json"

        echo "[$CURRENT/$TOTAL] $ATTENTION on $DATASET"
        echo "  -> $OUTPUT_FILE"

        # Check if should skip (resume mode and file exists)
        if [ "$RESUME" = true ] && [ -f "$OUTPUT_FILE" ]; then
            echo "  (skipping - file exists)"
            echo ""
            continue
        fi

        CMD="python -m evaluation.run_eval \
            --model-type luka \
            --model $MODEL \
            --attention $ATTENTION \
            --dataset $DATASET \
            --compressor attention_weighted \
            --refinement-rule $REFINEMENT_RULE \
            --refinement-k $REFINEMENT_K \
            --log-bias $LOG_BIAS \
            --heavy-ratio $HEAVY_RATIO \
            --recent-ratio $RECENT_RATIO \
            --score \
            --output $OUTPUT_FILE \
            $EXTRA_ARGS"

        if [ "$DRY_RUN" = true ]; then
            echo "  (dry-run) $CMD"
        else
            echo "  Running..."
            eval $CMD
            echo "  Done."
        fi
        echo ""
    done
done

echo "=============================================="
echo "Batch evaluation complete!"
echo "Results in: $OUTPUT_DIR"
echo "=============================================="

# Generate summary if not dry run
if [ "$DRY_RUN" = false ]; then
    echo ""
    echo "Summary:"
    echo "--------"
    for ATTENTION in "${ATTENTION_MODES[@]}"; do
        echo ""
        echo "$ATTENTION:"
        for DATASET in "${DATASETS[@]}"; do
            FILE="$OUTPUT_DIR/eval_${ATTENTION}_${DATASET}.json"
            if [ -f "$FILE" ]; then
                # Extract scores and performance using python
                SCORES=$(python3 -c "
import json
with open('$FILE') as f:
    d = json.load(f)
scores = d.get('scores', {}).get('aggregate', {})
perf = d.get('performance_stats', {})
em = scores.get('exact_match', 0)
f1 = scores.get('f1_score', 0)
throughput = perf.get('decode_throughput_tps', 0)
latency = perf.get('decode_latency_ms_per_token', 0)
peak_mem = perf.get('peak_memory_allocated_mb', 0)
print(f'  {\"$DATASET\":<12} EM={em:.1%} F1={f1:.1%} | {throughput:.1f} tok/s, {latency:.1f}ms/tok, {peak_mem:.0f}MB')
" 2>/dev/null || echo "  $DATASET: (no scores)")
                echo "$SCORES"
            fi
        done
    done
fi
