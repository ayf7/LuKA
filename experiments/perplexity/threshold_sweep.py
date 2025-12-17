"""
Combined threshold sweep - runs both token position curves and retention analysis.

For focused analysis, use:
- token_position_curves.py - per-token perplexity at a fixed threshold
- retention_analysis.py - threshold sweep with retention/speed analysis

Run: python experiments/perplexity/threshold_sweep.py
"""

import argparse

from experiments.perplexity.token_position_curves import run as run_token_curves
from experiments.perplexity.retention_analysis import run as run_retention


def main():
    parser = argparse.ArgumentParser(description="Combined threshold sweep")
    parser.add_argument("--threshold", type=float, default=0.05,
                       help="Fixed threshold for token position curves")
    parser.add_argument("--max-tokens", type=int, default=1024,
                       help="Max new tokens to generate")
    parser.add_argument("--prompt", type=str, default="paragraphs_1",
                       help="Prompt name")
    parser.add_argument("--skip-token-curves", action="store_true",
                       help="Skip token position curve analysis")
    parser.add_argument("--skip-retention", action="store_true",
                       help="Skip retention analysis sweep")
    args = parser.parse_args()

    if not args.skip_token_curves:
        print("=" * 80)
        print("Running Token Position Curves Analysis")
        print("=" * 80)
        run_token_curves(
            threshold=args.threshold,
            max_new_tokens=args.max_tokens,
            prompt_name=args.prompt,
        )
        print()

    if not args.skip_retention:
        print("=" * 80)
        print("Running Retention Analysis Sweep")
        print("=" * 80)
        run_retention(
            max_new_tokens=args.max_tokens,
            prompt_name=args.prompt,
        )


if __name__ == "__main__":
    main()
