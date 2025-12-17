"""
Hyperparameter sweep: varying number of hybrid layers.

Tests different configurations where N layers at the beginning and end use
lined (H2O) attention, while middle layers use top-down (LuKA) attention.

Sweep values: 0, 4, 8, 12 layers from each end as lined.
  - 0: All top-down (pure LuKA)
  - 4: Layers 0-3 and 24-27 lined, 4-23 top-down
  - 8: Layers 0-7 and 20-27 lined, 8-19 top-down
  - 12: Layers 0-11 and 16-27 lined, 12-15 top-down
  - all: All lined (pure H2O)

Run:
    python experiments/perplexity/hybrid_layer_sweep.py --prompt paragraphs_3
    python experiments/perplexity/hybrid_layer_sweep.py --dataset wikitext --max-tokens 2048
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from experiments.perplexity.utils import (
    MODEL_NAME,
    get_device,
    get_tokenizer,
    get_prompt,
    get_baseline_perplexity,
    load_eval_text,
    get_output_dir,
)
from modeling.compressor import AttentionWeightedCompressor
from modeling.qwen.luka_qwen3 import load_luka_model, set_luka_kv_params


# Output paths
OUTPUT_DIR = get_output_dir("hybrid_layer_sweep")
CSV_PATH = OUTPUT_DIR / "results.csv"
PLOT_PATH = OUTPUT_DIR / "sweep.png"


def load_csv():
    """Load results from CSV file."""
    import csv as csv_module

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}. Run without --allow-skipping first.")

    results = {}
    base_ppl = None
    base_tps = None
    num_layers = 28

    with open(CSV_PATH, "r") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            name = row["config"]
            if name == "baseline":
                base_ppl = float(row["perplexity"])
                base_tps = float(row["tokens_per_sec"])
                continue

            results[name] = {
                "perplexity": float(row["perplexity"]),
                "tokens_per_sec": float(row["tokens_per_sec"]),
                "n_lined_per_end": int(row["n_lined_per_end"]),
            }

    return results, base_ppl, base_tps, num_layers


def get_sweep_configs(num_layers: int, sweep_values: list[int] = None):
    """
    Generate sweep configurations for hybrid layer experiments.

    Args:
        num_layers: Total number of transformer layers.
        sweep_values: List of N values (layers from each end to make lined).
                      Default: [0, 4, 8, 12, num_layers//2]

    Returns:
        List of config dicts.
    """
    if sweep_values is None:
        sweep_values = [0, 4, 8, 12]

    configs = []
    colors = plt.cm.viridis([i / (len(sweep_values) + 1) for i in range(len(sweep_values) + 1)])

    for i, n in enumerate(sweep_values):
        if n == 0:
            # Pure top-down
            name = "topdown_only"
            label = "Top-Down Only"
            lined_layers = set()
            use_lined = False
        else:
            # Hybrid: first N and last N are lined
            n_clamped = min(n, num_layers // 2)
            first_lined = set(range(n_clamped))
            last_lined = set(range(num_layers - n_clamped, num_layers))
            lined_layers = first_lined | last_lined

            name = f"hybrid_{n}"
            label = f"Hybrid ({n} from ends)"
            use_lined = True

        configs.append({
            "name": name,
            "label": label,
            "use_lined": use_lined,
            "lined_layers": lined_layers,
            "n_lined_per_end": n,
            "color": colors[i],
        })

    # Add pure lined (all layers)
    configs.append({
        "name": "lined_only",
        "label": "Lined Only (H2O)",
        "use_lined": True,
        "lined_layers": set(range(num_layers)),
        "n_lined_per_end": num_layers // 2,
        "color": colors[-1],
    })

    return configs


def prefill_then_decode_perplexity(model, rollout_ids, prompt_len):
    """
    Prefill with prompt, then teacher-force through the rest.
    Returns perplexity and tokens/sec.
    """
    import time

    device = rollout_ids.device
    B, T = rollout_ids.shape
    assert B == 1

    pre_ids = rollout_ids[:, :prompt_len]
    pre_mask = torch.ones_like(pre_ids)
    with torch.no_grad():
        pre_out = model(
            input_ids=pre_ids,
            attention_mask=pre_mask,
            use_cache=True,
        )
    past_key_values = pre_out.past_key_values

    nll_list = []
    total_tokens = T - prompt_len
    start_time = time.perf_counter()

    # First prediction from prefill logits
    logits = pre_out.logits[:, -1, :]
    target = rollout_ids[:, prompt_len]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    nll_list.append(nll)

    # Subsequent predictions
    for t in range(prompt_len, T - 1):
        cur_id = rollout_ids[:, t : t + 1]
        attn_mask = torch.ones(1, t + 1, device=device, dtype=rollout_ids.dtype)
        with torch.no_grad():
            out = model(
                input_ids=cur_id,
                attention_mask=attn_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        logits = out.logits[:, -1, :]
        target = rollout_ids[:, t + 1]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        nll_list.append(nll)
        past_key_values = out.past_key_values

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    tps = total_tokens / max(elapsed, 1e-8)

    nll_tensor = torch.stack(nll_list, dim=1)
    total_nll = nll_tensor.sum(dim=1) / total_tokens
    ppl = torch.exp(total_nll)[0].item()

    return ppl, tps


def run_single_config(cfg, rollout_ids, prompt_len, device, num_layers, model_name):
    """Run perplexity evaluation for a single configuration."""
    print(f"\nRunning {cfg['label']}...")
    print(f"  Lined layers: {sorted(cfg['lined_layers']) if cfg['lined_layers'] else 'None'}")

    # Set LuKA params
    set_luka_kv_params(
        default_tail_len=16,
        min_compress_chunk=16,
        max_pages=15,
        refinement_rule="top_k",
        refinement_rule_kwargs={"k": 3},
        compressor=AttentionWeightedCompressor(temperature=1.0),
        log_bias_mode="none",
        segmenter="dummy",
        segment_interval=16,
        create_pages_in_generation=(len(cfg["lined_layers"]) < num_layers),
        production_mode=True,
    )

    # Load model
    model = load_luka_model(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    ).to(device)
    model.eval()

    # Configure attention mode
    if hasattr(model, "model") and hasattr(model.model, "luka_kv_controller"):
        controller = model.model.luka_kv_controller
        controller.use_lined_attention = cfg["use_lined"]
        controller.lined_layers = cfg["lined_layers"]
        controller.grid_top_k = 96
        controller.grid_update_interval = 16
        controller.grid_decay = 0.99

    # Run evaluation
    with torch.no_grad():
        ppl, tps = prefill_then_decode_perplexity(model, rollout_ids, prompt_len)

    # Get stats
    summary_frac = None
    pages = 0
    if hasattr(model, "model") and hasattr(model.model, "luka_kv_controller"):
        controller = model.model.luka_kv_controller
        stats = controller.get_refinement_stats()
        summary_frac = stats.get("refinement_rate", 0)
        pages = stats.get("avg_pages_per_layer", 0)

    # Cleanup
    del model
    torch.cuda.empty_cache()

    print(f"  ppl={ppl:.3f}, tps={tps:.1f}")

    return {
        "perplexity": ppl,
        "tokens_per_sec": tps,
        "summary_frac": summary_frac,
        "pages": pages,
    }


def run(
    max_tokens: int = 2048,
    prompt_name: str = "paragraphs_1",
    eval_dataset: str = None,
    model_name: str = None,
    sweep_values: list[int] = None,
    allow_skipping: bool = False,
):
    if model_name is None:
        model_name = MODEL_NAME

    # Check for existing results if allow_skipping is enabled
    if allow_skipping and CSV_PATH.exists():
        print(f"Found existing CSV at {CSV_PATH}, loading cached results...")
        try:
            results, base_ppl, base_tps, num_layers = load_csv()
            configs = get_sweep_configs(num_layers, sweep_values)
            # Merge results with configs for plotting
            for cfg in configs:
                if cfg["name"] in results:
                    results[cfg["name"]].update(cfg)
            print(f"  Loaded results for {len(results)} configurations")
            print(f"  Baseline: ppl={base_ppl:.3f}, tps={base_tps:.1f}")
            generate_plot(results, configs, base_ppl, base_tps, num_layers)
            return results, configs, base_ppl, base_tps
        except Exception as e:
            print(f"  Failed to load CSV: {e}, running experiments...")

    device = get_device()
    tokenizer = get_tokenizer(model_name)

    # Load evaluation data
    if eval_dataset:
        print(f"Loading eval text from {eval_dataset} (max {max_tokens} tokens)...")
        _, rollout_ids = load_eval_text(eval_dataset, max_tokens, tokenizer=tokenizer)
        rollout_ids = rollout_ids.to(device)
        prompt_len = 1
    else:
        # Load paragraph as evaluation text (teacher-forcing)
        paragraph_text = get_prompt(prompt_name)
        print(f"Loading paragraph '{prompt_name}' for teacher-forcing evaluation...")
        rollout_ids = tokenizer(paragraph_text, return_tensors="pt")["input_ids"]
        if rollout_ids.shape[1] > max_tokens:
            rollout_ids = rollout_ids[:, :max_tokens]
        rollout_ids = rollout_ids.to(device)
        prompt_len = 1

    print(f"Total tokens: {rollout_ids.shape[1]}, prompt_len: {prompt_len} (teacher-forcing on {rollout_ids.shape[1] - prompt_len} tokens)")

    # Get number of layers
    print(f"\nLoading model to get layer count: {model_name}")
    set_luka_kv_params(use_exact_attention=True, compressor="mean", segmenter="dummy")
    temp_model = load_luka_model(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if hasattr(temp_model, "model") and hasattr(temp_model.model, "luka_kv_controller"):
        num_layers = temp_model.model.luka_kv_controller.num_layers
    else:
        num_layers = 28
    del temp_model
    torch.cuda.empty_cache()
    print(f"  num_layers: {num_layers}")

    # Get sweep configs
    configs = get_sweep_configs(num_layers, sweep_values)

    # Get baseline
    print("\nRunning baseline (raw attention)...")
    base_ppl, _, _, base_tps = get_baseline_perplexity(rollout_ids, prompt_len, device, model_name)
    print(f"  ppl={base_ppl:.3f}, tps={base_tps:.1f}")

    # Run sweep
    results = {}
    for cfg in configs:
        result = run_single_config(cfg, rollout_ids, prompt_len, device, num_layers, model_name)
        results[cfg["name"]] = {**result, **cfg}

    # Print summary
    print("\n" + "=" * 80)
    print("Hybrid Layer Sweep Summary")
    print("=" * 80)
    print(f"\n{'Config':<30} {'PPL':>8} {'Delta':>10} {'TPS':>10} {'Lined Layers':>15}")
    print("-" * 80)
    print(f"{'Baseline (raw)':<30} {base_ppl:>8.3f} {0:>+10.2f} {base_tps:>10.1f} {'N/A':>15}")

    for cfg in configs:
        r = results[cfg["name"]]
        delta = r["perplexity"] - base_ppl
        n_lined = len(cfg["lined_layers"])
        print(f"{cfg['label']:<30} {r['perplexity']:>8.3f} {delta:>+10.2f} {r['tokens_per_sec']:>10.1f} {n_lined:>15}")

    print("=" * 80)

    # Save CSV
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "n_lined_per_end", "total_lined_layers", "perplexity", "delta", "tokens_per_sec"])
        writer.writerow(["baseline", 0, 0, base_ppl, 0, base_tps])
        for cfg in configs:
            r = results[cfg["name"]]
            delta = r["perplexity"] - base_ppl
            n_lined = len(cfg["lined_layers"])
            writer.writerow([cfg["name"], cfg["n_lined_per_end"], n_lined, r["perplexity"], delta, r["tokens_per_sec"]])
    print(f"\nSaved CSV to {CSV_PATH}")

    # Generate plot
    generate_plot(results, configs, base_ppl, base_tps, num_layers)


def generate_plot(results, configs, base_ppl, base_tps, num_layers):
    """Generate bar plots comparing configurations (linear and log scale)."""

    # Prepare data
    labels = ["Baseline"] + [cfg["label"] for cfg in configs]
    ppls = [base_ppl] + [results[cfg["name"]]["perplexity"] for cfg in configs]
    tps_vals = [base_tps] + [results[cfg["name"]]["tokens_per_sec"] for cfg in configs]
    colors = ["gray"] + [cfg["color"] for cfg in configs]

    # Hatching: dotted pattern for baseline, pure top-down (0), and pure lined (all)
    # Index 0 = baseline, index 1 = top-down only (n=0), last index = lined only
    hatches = ["//"] + ["" for _ in configs]
    hatches[1] = "//"  # Top-down only (first config, n=0)
    hatches[-1] = "//"  # Lined only (last config)

    x = range(len(labels))

    # === Linear scale plot ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # PPL plot (linear)
    bars1 = ax1.bar(x, ppls, color=colors)
    for bar, hatch in zip(bars1, hatches):
        bar.set_hatch(hatch)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_ylabel("Perplexity")
    ax1.set_title("Perplexity by Hybrid Configuration")
    ax1.axhline(y=base_ppl, color="gray", linestyle="--", alpha=0.5, label="Baseline")

    # Add delta labels
    for i, (bar, ppl) in enumerate(zip(bars1, ppls)):
        delta = ppl - base_ppl
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{delta:+.2f}", ha="center", va="bottom", fontsize=9)

    # TPS plot (linear)
    bars2 = ax2.bar(x, tps_vals, color=colors)
    for bar, hatch in zip(bars2, hatches):
        bar.set_hatch(hatch)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.set_ylabel("Tokens per Second")
    ax2.set_title("Throughput by Hybrid Configuration")
    ax2.axhline(y=base_tps, color="gray", linestyle="--", alpha=0.5, label="Baseline")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"Saved plot to {PLOT_PATH}")
    plt.close()

    # === Log scale plot ===
    log_plot_path = OUTPUT_DIR / "sweep_log.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # PPL plot (log)
    bars1 = ax1.bar(x, ppls, color=colors)
    for bar, hatch in zip(bars1, hatches):
        bar.set_hatch(hatch)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_ylabel("Perplexity (log scale)")
    ax1.set_title("Perplexity by Hybrid Configuration (Log Scale)")
    ax1.set_yscale("log")
    ax1.axhline(y=base_ppl, color="gray", linestyle="--", alpha=0.5, label="Baseline")

    # TPS plot (log)
    bars2 = ax2.bar(x, tps_vals, color=colors)
    for bar, hatch in zip(bars2, hatches):
        bar.set_hatch(hatch)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.set_ylabel("Tokens per Second (log scale)")
    ax2.set_title("Throughput by Hybrid Configuration (Log Scale)")
    ax2.set_yscale("log")
    ax2.axhline(y=base_tps, color="gray", linestyle="--", alpha=0.5, label="Baseline")

    plt.tight_layout()
    plt.savefig(log_plot_path, dpi=150)
    print(f"Saved log plot to {log_plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Hybrid layer sweep for lined vs top-down attention")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Model name (default: {MODEL_NAME})")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["wikitext", "pg19", "c4"],
                        help="Dataset for evaluation")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max tokens to evaluate (default: 2048)")
    parser.add_argument("--prompt", type=str, default="paragraphs_1",
                        help="Prompt name if not using dataset")
    parser.add_argument("--sweep", type=str, default="0,4,8,12,16",
                        help="Comma-separated sweep values (layers from each end)")
    parser.add_argument("--allow-skipping", action="store_true",
                        help="Skip experiments if results already exist in CSV")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only generate plots from existing CSV")
    args = parser.parse_args()

    sweep_values = [int(x) for x in args.sweep.split(",")]

    if args.plot_only:
        print("Loading data from CSV...")
        results, base_ppl, base_tps, num_layers = load_csv()
        configs = get_sweep_configs(num_layers, sweep_values)
        for cfg in configs:
            if cfg["name"] in results:
                results[cfg["name"]].update(cfg)
        print(f"Baseline: ppl={base_ppl:.3f}, tps={base_tps:.1f}")
        generate_plot(results, configs, base_ppl, base_tps, num_layers)
    else:
        run(
            max_tokens=args.max_tokens,
            prompt_name=args.prompt,
            eval_dataset=args.dataset,
            model_name=args.model,
            sweep_values=sweep_values,
            allow_skipping=args.allow_skipping,
        )


if __name__ == "__main__":
    main()
