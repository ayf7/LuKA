#!/bin/bash
#
# Run all evaluation combinations: datasets x attention modes
#
# Usage:
#   ./scripts/run_all_evals.sh                    # Run all combinations
#   ./scripts/run_all_evals.sh --max-examples 5   # Quick test with 5 examples
#   ./scripts/run_all_evals.sh --dry-run          # Show commands without running
#
# Output: results/eval_{attention}_{dataset}.json

set -e

# Configuration
DATASETS=("easy" "medium" "hard" "very_hard")
ATTENTION_MODES=("baseline" "top_down" "lined" "mix")
MODEL="qwen-4b"
OUTPUT_DIR="evaluation/results"

# Parse arguments
DRY_RUN=false
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
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
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Count total runs
TOTAL=$((${#DATASETS[@]} * ${#ATTENTION_MODES[@]}))
CURRENT=0

echo "=============================================="
echo "LuKA Batch Evaluation"
echo "=============================================="
echo "Model:     $MODEL"
echo "Datasets:  ${DATASETS[*]}"
echo "Modes:     ${ATTENTION_MODES[*]}"
echo "Output:    $OUTPUT_DIR"
echo "Total:     $TOTAL runs"
echo "Extra:     $EXTRA_ARGS"
echo "=============================================="
echo ""

# Run all combinations
for DATASET in "${DATASETS[@]}"; do
    for ATTENTION in "${ATTENTION_MODES[@]}"; do
        CURRENT=$((CURRENT + 1))
        OUTPUT_FILE="$OUTPUT_DIR/eval_${ATTENTION}_${DATASET}.json"

        echo "[$CURRENT/$TOTAL] $ATTENTION on $DATASET"
        echo "  -> $OUTPUT_FILE"

        CMD="python -m evaluation.run_eval \
            --model-type luka \
            --model $MODEL \
            --attention $ATTENTION \
            --dataset $DATASET \
            --compressor attention_weighted \
            --refinement-rule top_k \
            --refinement-k 3 \
            --log-bias adaptive_k \
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
                # Extract scores using python
                SCORES=$(python3 -c "
import json
with open('$FILE') as f:
    d = json.load(f)
scores = d.get('scores', {}).get('aggregate', {})
em = scores.get('exact_match', 0)
f1 = scores.get('f1_score', 0)
print(f'  {\"$DATASET\":<12} EM={em:.1%} F1={f1:.1%}')
" 2>/dev/null || echo "  $DATASET: (no scores)")
                echo "$SCORES"
            fi
        done
    done
fi
