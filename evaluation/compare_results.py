"""
Compare Baseline vs LuKA Results

Creates a comprehensive comparison report with:
1. Summary comparison table
2. Accuracy vs Compression scatter plot
3. Per-example heatmap
4. Best/worst case analysis

Usage:
    python -m evaluation.compare_results \
        --baseline results/baseline_simple_scores.json \
        --luka results/luka_simple_scores.json \
        --output results/comparison_report.html

    # Or compare multiple pairs
    python -m evaluation.compare_results \
        --baseline results/baseline_simple_scores.json results/baseline_sequential_scores.json \
        --luka results/luka_simple_scores.json results/luka_sequential_scores.json \
        --output results/comparison_report.html
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np


def load_scored_results(path: str) -> dict:
    """Load scored results JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def create_summary_table(baseline: dict, luka: dict) -> str:
    """Create ASCII summary comparison table."""
    baseline_agg = baseline['aggregate']
    luka_agg = luka['aggregate']

    table = []
    table.append("=" * 90)
    table.append(f"{'BASELINE vs LuKA COMPARISON':^90}")
    table.append("=" * 90)
    table.append("")
    table.append(f"{'Metric':<35} {'Baseline':<20} {'LuKA':<20} {'Δ':<15}")
    table.append("-" * 90)

    # QA Accuracy
    baseline_qa = baseline_agg['qa_accuracy']
    luka_qa = luka_agg['qa_accuracy']
    qa_diff = luka_qa - baseline_qa
    qa_diff_pct = (qa_diff / baseline_qa * 100) if baseline_qa > 0 else 0
    qa_indicator = '✓' if qa_diff >= -0.05 else '✗'  # Acceptable if < 5% drop

    table.append(f"{'QA Accuracy':<35} {baseline_qa:.3f} ({baseline_qa*100:.1f}%){'':<4} "
                 f"{luka_qa:.3f} ({luka_qa*100:.1f}%){'':<4} "
                 f"{qa_diff:+.3f} ({qa_diff_pct:+.1f}%) {qa_indicator}")

    # LuKA-specific metrics
    if 'compression_ratio' in luka_agg:
        # Boundary Detection F1
        luka_bound = luka_agg.get('boundary_f1', 0.0)
        table.append(f"{'Boundary Detection F1':<35} {'N/A':<20} "
                     f"{luka_bound:.3f} ({luka_bound*100:.1f}%){'':<4} {'N/A':<15}")

        # Compression Ratio
        comp_ratio = luka_agg['compression_ratio']
        memory_saved = (1 - 1/comp_ratio) * 100 if comp_ratio > 0 else 0
        comp_indicator = '✓' if comp_ratio > 2.0 else '○'  # Good if > 2x
        table.append(f"{'Compression Ratio':<35} {'1.00x (no compression)':<20} "
                     f"{comp_ratio:.2f}x {'':<14} {comp_indicator}")
        table.append(f"{'Memory Saved':<35} {'0%':<20} "
                     f"{memory_saved:.1f}% {'':<13}")

        # Selective Decompression
        decomp_acc = luka_agg.get('selective_decompression_accuracy', 0.0)
        table.append(f"{'Selective Decompression Acc':<35} {'N/A':<20} "
                     f"{decomp_acc:.3f} ({decomp_acc*100:.1f}%){'':<4} {'N/A':<15}")

    table.append("=" * 90)

    # Overall assessment
    table.append("")
    table.append("ASSESSMENT:")

    if 'compression_ratio' in luka_agg:
        if qa_diff >= -0.05 and comp_ratio > 2.0:
            assessment = "✓ SUCCESS: LuKA maintains accuracy while achieving good compression"
        elif qa_diff >= -0.10 and comp_ratio > 2.0:
            assessment = "○ ACCEPTABLE: Moderate accuracy drop with good compression"
        elif qa_diff < -0.10:
            assessment = "✗ NEEDS TUNING: Significant accuracy degradation"
        else:
            assessment = "○ LOW COMPRESSION: Consider adjusting LuKA parameters"
    else:
        assessment = "No compression stats available for comparison"

    table.append(f"  {assessment}")
    table.append("=" * 90)

    return "\n".join(table)


def analyze_per_example(baseline: dict, luka: dict) -> str:
    """Analyze per-example differences."""
    baseline_ex = baseline.get('per_example', [])
    luka_ex = luka.get('per_example', [])

    if not baseline_ex or not luka_ex:
        return "No per-example data available"

    # Match examples by ID
    baseline_dict = {ex['example_id']: ex for ex in baseline_ex}
    luka_dict = {ex['example_id']: ex for ex in luka_ex}

    common_ids = set(baseline_dict.keys()) & set(luka_dict.keys())

    if not common_ids:
        return "No common examples found"

    # Calculate differences
    diffs = []
    for ex_id in sorted(common_ids):
        baseline_qa = baseline_dict[ex_id]['qa_accuracy']
        luka_qa = luka_dict[ex_id]['qa_accuracy']
        luka_comp = luka_dict[ex_id].get('compression_ratio', 1.0)

        diff = luka_qa - baseline_qa
        diffs.append({
            'id': ex_id,
            'baseline_qa': baseline_qa,
            'luka_qa': luka_qa,
            'diff': diff,
            'compression': luka_comp
        })

    # Sort by difference
    diffs_sorted = sorted(diffs, key=lambda x: x['diff'])

    output = []
    output.append("\n" + "=" * 90)
    output.append("PER-EXAMPLE ANALYSIS")
    output.append("=" * 90)

    # Best cases (LuKA improves or maintains accuracy)
    best_cases = [d for d in diffs_sorted if d['diff'] >= 0][-5:]  # Top 5
    if best_cases:
        output.append("\nBEST CASES (LuKA maintains/improves accuracy):")
        output.append(f"{'ID':<6} {'Baseline QA':<12} {'LuKA QA':<12} {'Diff':<10} {'Compression':<12}")
        output.append("-" * 60)
        for d in reversed(best_cases):
            output.append(f"{d['id']:<6} {d['baseline_qa']:.3f}{'':<6} {d['luka_qa']:.3f}{'':<6} "
                         f"{d['diff']:+.3f}{'':<4} {d['compression']:.2f}x")

    # Worst cases (LuKA degrades accuracy)
    worst_cases = [d for d in diffs_sorted if d['diff'] < 0][:5]  # Bottom 5
    if worst_cases:
        output.append("\nWORST CASES (LuKA degrades accuracy):")
        output.append(f"{'ID':<6} {'Baseline QA':<12} {'LuKA QA':<12} {'Diff':<10} {'Compression':<12}")
        output.append("-" * 60)
        for d in worst_cases:
            output.append(f"{d['id']:<6} {d['baseline_qa']:.3f}{'':<6} {d['luka_qa']:.3f}{'':<6} "
                         f"{d['diff']:+.3f}{'':<4} {d['compression']:.2f}x")

    # Statistics
    all_diffs = [d['diff'] for d in diffs]
    all_comps = [d['compression'] for d in diffs]

    output.append("\n" + "-" * 90)
    output.append("STATISTICS:")
    output.append(f"  QA Diff:     Mean={np.mean(all_diffs):+.3f}, "
                 f"Std={np.std(all_diffs):.3f}, "
                 f"Min={np.min(all_diffs):+.3f}, "
                 f"Max={np.max(all_diffs):+.3f}")
    output.append(f"  Compression: Mean={np.mean(all_comps):.2f}x, "
                 f"Std={np.std(all_comps):.2f}x, "
                 f"Min={np.min(all_comps):.2f}x, "
                 f"Max={np.max(all_comps):.2f}x")
    output.append(f"  Examples with QA drop > 10%: {sum(1 for d in all_diffs if d < -0.1)}/{len(all_diffs)}")
    output.append(f"  Examples with QA drop > 5%:  {sum(1 for d in all_diffs if d < -0.05)}/{len(all_diffs)}")
    output.append("=" * 90)

    return "\n".join(output)


def create_ascii_scatter_plot(baseline: dict, luka: dict, width: int = 70, height: int = 20) -> str:
    """Create ASCII scatter plot of accuracy vs compression."""
    baseline_ex = baseline.get('per_example', [])
    luka_ex = luka.get('per_example', [])

    if not baseline_ex or not luka_ex:
        return "No data for scatter plot"

    # Match examples
    baseline_dict = {ex['example_id']: ex for ex in baseline_ex}
    luka_dict = {ex['example_id']: ex for ex in luka_ex}
    common_ids = set(baseline_dict.keys()) & set(luka_dict.keys())

    if not common_ids:
        return "No common examples"

    # Extract data
    points = []
    for ex_id in common_ids:
        luka_qa = luka_dict[ex_id]['qa_accuracy']
        luka_comp = luka_dict[ex_id].get('compression_ratio', 1.0)
        points.append((luka_comp, luka_qa))

    if not points:
        return "No data points"

    # Create plot
    comp_values = [p[0] for p in points]
    qa_values = [p[1] for p in points]

    comp_min, comp_max = min(comp_values), max(comp_values)
    qa_min, qa_max = 0.0, 1.0  # Fixed scale for QA

    # Add padding
    comp_range = comp_max - comp_min
    comp_min = max(0, comp_min - comp_range * 0.1)
    comp_max = comp_max + comp_range * 0.1

    # Create grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    # Plot points
    for comp, qa in points:
        x = int((comp - comp_min) / (comp_max - comp_min) * (width - 1))
        y = height - 1 - int((qa - qa_min) / (qa_max - qa_min) * (height - 1))

        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = '●'

    # Convert to string
    output = []
    output.append("\n" + "=" * 90)
    output.append("ACCURACY vs COMPRESSION SCATTER PLOT")
    output.append("=" * 90)
    output.append("")
    output.append(f"QA Accuracy ({qa_min:.1f} - {qa_max:.1f})")

    for i, row in enumerate(grid):
        if i == 0:
            label = f"{qa_max:.2f} |"
        elif i == height - 1:
            label = f"{qa_min:.2f} |"
        elif i == height // 2:
            label = f"{(qa_max+qa_min)/2:.2f} |"
        else:
            label = "     |"

        output.append(label + ''.join(row))

    # X-axis
    output.append("     " + "-" * width)
    output.append(f"     {comp_min:.1f}x" + " " * (width - 20) + f"{comp_max:.1f}x")
    output.append(f"     {'Compression Ratio':^{width}}")
    output.append("")
    output.append(f"Total points: {len(points)}")
    output.append("=" * 90)

    return "\n".join(output)


def create_heatmap(baseline: dict, luka: dict, width: int = 60) -> str:
    """Create ASCII heatmap of per-example performance."""
    baseline_ex = baseline.get('per_example', [])
    luka_ex = luka.get('per_example', [])

    if not baseline_ex or not luka_ex or len(baseline_ex) > 30:
        return "Heatmap skipped (too many examples or no data)"

    # Match examples
    baseline_dict = {ex['example_id']: ex for ex in baseline_ex}
    luka_dict = {ex['example_id']: ex for ex in luka_ex}
    common_ids = sorted(set(baseline_dict.keys()) & set(luka_dict.keys()))

    if not common_ids:
        return "No common examples"

    output = []
    output.append("\n" + "=" * 90)
    output.append("PER-EXAMPLE HEATMAP")
    output.append("=" * 90)
    output.append("")
    output.append(f"{'Example':<8} {'Baseline QA':<35} {'LuKA QA':<35}")
    output.append("-" * 90)

    for ex_id in common_ids:
        baseline_qa = baseline_dict[ex_id]['qa_accuracy']
        luka_qa = luka_dict[ex_id]['qa_accuracy']
        luka_comp = luka_dict[ex_id].get('compression_ratio', 1.0)

        # Create bars
        baseline_bar = '█' * int(baseline_qa * 30) + '░' * (30 - int(baseline_qa * 30))
        luka_bar = '█' * int(luka_qa * 30) + '░' * (30 - int(luka_qa * 30))

        # Indicator
        diff = luka_qa - baseline_qa
        if diff >= -0.05:
            indicator = '✓'
        elif diff >= -0.10:
            indicator = '○'
        else:
            indicator = '✗'

        output.append(f"{ex_id:<8} {baseline_bar} {baseline_qa:.2f}  {luka_bar} {luka_qa:.2f} "
                     f"({luka_comp:.1f}x) {indicator}")

    output.append("=" * 90)

    return "\n".join(output)


def generate_html_report(baseline: dict, luka: dict, output_path: str):
    """Generate HTML report with embedded visualizations."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping HTML report generation")
        return

    baseline_ex = baseline.get('per_example', [])
    luka_ex = luka.get('per_example', [])

    if not baseline_ex or not luka_ex:
        print("Warning: No per-example data, skipping HTML report")
        return

    # Match examples
    baseline_dict = {ex['example_id']: ex for ex in baseline_ex}
    luka_dict = {ex['example_id']: ex for ex in luka_ex}
    common_ids = sorted(set(baseline_dict.keys()) & set(luka_dict.keys()))

    if not common_ids:
        print("Warning: No common examples, skipping HTML report")
        return

    # Prepare data
    example_ids = []
    baseline_qa_values = []
    luka_qa_values = []
    compression_values = []
    diff_values = []

    for ex_id in common_ids:
        example_ids.append(ex_id)
        baseline_qa = baseline_dict[ex_id]['qa_accuracy']
        luka_qa = luka_dict[ex_id]['qa_accuracy']
        luka_comp = luka_dict[ex_id].get('compression_ratio', 1.0)

        baseline_qa_values.append(baseline_qa)
        luka_qa_values.append(luka_qa)
        compression_values.append(luka_comp)
        diff_values.append(luka_qa - baseline_qa)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Baseline vs LuKA Comparison', fontsize=16, fontweight='bold')

    # 1. Accuracy Comparison Bar Chart
    ax1 = axes[0, 0]
    x = np.arange(len(example_ids))
    width = 0.35
    ax1.bar(x - width/2, baseline_qa_values, width, label='Baseline', alpha=0.8)
    ax1.bar(x + width/2, luka_qa_values, width, label='LuKA', alpha=0.8)
    ax1.set_xlabel('Example ID')
    ax1.set_ylabel('QA Accuracy')
    ax1.set_title('QA Accuracy by Example')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    if len(example_ids) <= 20:
        ax1.set_xticks(x)
        ax1.set_xticklabels(example_ids)

    # 2. Accuracy vs Compression Scatter
    ax2 = axes[0, 1]
    colors = ['red' if d < -0.1 else 'orange' if d < -0.05 else 'green' for d in diff_values]
    ax2.scatter(compression_values, luka_qa_values, c=colors, alpha=0.6, s=100)
    ax2.set_xlabel('Compression Ratio')
    ax2.set_ylabel('LuKA QA Accuracy')
    ax2.set_title('Accuracy vs Compression Tradeoff')
    ax2.grid(alpha=0.3)
    ax2.axhline(y=np.mean(baseline_qa_values), color='blue', linestyle='--', label='Baseline Mean', alpha=0.5)
    ax2.legend()

    # 3. Accuracy Difference (Delta)
    ax3 = axes[1, 0]
    colors_bar = ['red' if d < -0.1 else 'orange' if d < -0.05 else 'green' for d in diff_values]
    ax3.bar(x, diff_values, color=colors_bar, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax3.axhline(y=-0.05, color='orange', linestyle='--', linewidth=0.8, alpha=0.5)
    ax3.axhline(y=-0.10, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax3.set_xlabel('Example ID')
    ax3.set_ylabel('QA Accuracy Difference (LuKA - Baseline)')
    ax3.set_title('Per-Example Accuracy Delta')
    ax3.grid(axis='y', alpha=0.3)
    if len(example_ids) <= 20:
        ax3.set_xticks(x)
        ax3.set_xticklabels(example_ids)

    # 4. Compression Distribution
    ax4 = axes[1, 1]
    ax4.hist(compression_values, bins=min(15, len(compression_values)), alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_xlabel('Compression Ratio')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Compression Ratio Distribution')
    ax4.axvline(x=np.mean(compression_values), color='red', linestyle='--', label=f'Mean: {np.mean(compression_values):.2f}x')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save figure
    plot_path = output_path.replace('.html', '_plots.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Generate HTML
    baseline_agg = baseline['aggregate']
    luka_agg = luka['aggregate']

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Baseline vs LuKA Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: auto; background: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
            h2 {{ color: #555; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .success {{ color: #4CAF50; font-weight: bold; }}
            .warning {{ color: #FF9800; font-weight: bold; }}
            .error {{ color: #F44336; font-weight: bold; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; }}
            .metric {{ font-size: 1.2em; }}
            .assessment {{ background: #e8f5e9; padding: 15px; border-left: 4px solid #4CAF50; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Baseline vs LuKA Comparison Report</h1>

            <h2>Summary Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Baseline</th>
                    <th>LuKA</th>
                    <th>Difference</th>
                </tr>
                <tr>
                    <td class="metric">QA Accuracy</td>
                    <td>{baseline_agg['qa_accuracy']:.3f} ({baseline_agg['qa_accuracy']*100:.1f}%)</td>
                    <td>{luka_agg['qa_accuracy']:.3f} ({luka_agg['qa_accuracy']*100:.1f}%)</td>
                    <td class="{'success' if luka_agg['qa_accuracy'] >= baseline_agg['qa_accuracy'] * 0.95 else 'warning' if luka_agg['qa_accuracy'] >= baseline_agg['qa_accuracy'] * 0.90 else 'error'}">
                        {luka_agg['qa_accuracy'] - baseline_agg['qa_accuracy']:+.3f}
                        ({(luka_agg['qa_accuracy'] - baseline_agg['qa_accuracy']) / baseline_agg['qa_accuracy'] * 100:+.1f}%)
                    </td>
                </tr>
                <tr>
                    <td class="metric">Compression Ratio</td>
                    <td>1.00x (no compression)</td>
                    <td class="{'success' if luka_agg.get('compression_ratio', 1.0) > 2.0 else 'warning'}">{luka_agg.get('compression_ratio', 1.0):.2f}x</td>
                    <td>Memory saved: {(1 - 1/luka_agg.get('compression_ratio', 1.0))*100:.1f}%</td>
                </tr>
                <tr>
                    <td class="metric">Boundary Detection F1</td>
                    <td>N/A</td>
                    <td>{luka_agg.get('boundary_f1', 0.0):.3f} ({luka_agg.get('boundary_f1', 0.0)*100:.1f}%)</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td class="metric">Selective Decompression Acc</td>
                    <td>N/A</td>
                    <td>{luka_agg.get('selective_decompression_accuracy', 0.0):.3f} ({luka_agg.get('selective_decompression_accuracy', 0.0)*100:.1f}%)</td>
                    <td>-</td>
                </tr>
            </table>

            <div class="assessment">
                <strong>Assessment:</strong>
                {'✓ LuKA successfully maintains accuracy while achieving compression' if luka_agg['qa_accuracy'] >= baseline_agg['qa_accuracy'] * 0.95 and luka_agg.get('compression_ratio', 1.0) > 2.0 else '○ Acceptable tradeoff between accuracy and compression' if luka_agg['qa_accuracy'] >= baseline_agg['qa_accuracy'] * 0.90 else '✗ Significant accuracy degradation - consider tuning'}
            </div>

            <h2>Visual Analysis</h2>
            <img src="{Path(plot_path).name}" alt="Comparison Plots">

            <h2>Statistics</h2>
            <p><strong>QA Accuracy Difference:</strong> Mean={np.mean(diff_values):+.3f}, Std={np.std(diff_values):.3f}, Min={np.min(diff_values):+.3f}, Max={np.max(diff_values):+.3f}</p>
            <p><strong>Compression Ratio:</strong> Mean={np.mean(compression_values):.2f}x, Std={np.std(compression_values):.2f}x, Min={np.min(compression_values):.2f}x, Max={np.max(compression_values):.2f}x</p>
            <p><strong>Examples evaluated:</strong> {len(common_ids)}</p>
            <p><strong>Examples with accuracy drop &gt; 10%:</strong> {sum(1 for d in diff_values if d < -0.1)}/{len(diff_values)}</p>
            <p><strong>Examples with accuracy drop &gt; 5%:</strong> {sum(1 for d in diff_values if d < -0.05)}/{len(diff_values)}</p>
        </div>
    </body>
    </html>
    """

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"\n✓ HTML report saved to: {output_path}")
    print(f"✓ Plots saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline and LuKA evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--baseline",
        type=str,
        nargs='+',
        required=True,
        help="Path(s) to baseline scored results JSON"
    )
    parser.add_argument(
        "--luka",
        type=str,
        nargs='+',
        required=True,
        help="Path(s) to LuKA scored results JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for comparison report (HTML)"
    )
    parser.add_argument(
        "--no-html",
        action='store_true',
        help="Skip HTML report generation (text only)"
    )

    args = parser.parse_args()

    if len(args.baseline) != len(args.luka):
        print("Error: Number of baseline and LuKA files must match", file=sys.stderr)
        sys.exit(1)

    # For now, just compare the first pair
    # TODO: Support multiple comparisons
    baseline_path = args.baseline[0]
    luka_path = args.luka[0]

    print(f"\n{'='*60}")
    print("Comparing Results")
    print(f"{'='*60}")
    print(f"Baseline: {baseline_path}")
    print(f"LuKA:     {luka_path}")

    # Load results
    baseline = load_scored_results(baseline_path)
    luka = load_scored_results(luka_path)

    # Generate ASCII report
    print(create_summary_table(baseline, luka))
    print(analyze_per_example(baseline, luka))
    print(create_ascii_scatter_plot(baseline, luka))
    print(create_heatmap(baseline, luka))

    # Generate HTML report if requested
    if not args.no_html:
        print(f"\n{'='*60}")
        print("Generating HTML Report")
        print(f"{'='*60}")
        generate_html_report(baseline, luka, args.output)


if __name__ == "__main__":
    main()
