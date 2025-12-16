"""
Test script for LuKA with HuggingFace Transformers (new controller path).
Simple script to test boundary detection during generation.
"""

import torch
from transformers import AutoTokenizer
from modeling.qwen.luka_qwen3 import load_luka_model, set_luka_kv_params
from artifacts.prompts.prompt_loader import load_prompt

# Configuration
model_name = "Qwen/Qwen3-1.7B-Base"
device = "cuda" if torch.cuda.is_available() else "cpu"
use_trained_compressor = True

set_luka_kv_params(
    compressor="attention_weighted",
    compressor_kwargs={"temperature": 1.0},
    use_log_bias=True,
    segmenter="dummy",
    # segmenter_kwargs={"mean": 16, "std": 4},
    refine_threshold=0.01,
    segment_interval=16,
)

# Load model and tokenizer
model = load_luka_model(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

# Load prompts (batch)
prompts = [
    load_prompt("paragraphs_2"),
    load_prompt("paragraphs_1"),
]

# Tokenize with padding for batching
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(prompts, return_tensors="pt", padding=True)
if device == "cuda":
    inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

# Decode and print each result
for i, (prompt, output) in enumerate(zip(prompts, outputs)):
    generated_text = tokenizer.decode(output, skip_special_tokens=True)
    new_text = generated_text[len(prompt):]

    print("\n" + "=" * 80)
    print(f"Prompt {i}:")
    print("=" * 80)
    print(prompt)
    print("=" * 80 + "\n")

    print("Generated output:")
    print(new_text)
    print("\n" + "=" * 80)

# Print LuKA refinement statistics
if hasattr(model, "model") and hasattr(model.model, "luka_kv_controller"):
    controller = model.model.luka_kv_controller
    stats = controller.get_refinement_stats()

    print("\n" + "=" * 60)
    print("LuKA Refinement Statistics")
    print("=" * 60)
    print(f"  Layers:                   {stats['num_layers']}")
    print(f"  Pages (per layer avg):    {stats['avg_pages_per_layer']:.1f}")
    print(f"  Summary tokens (per layer): {stats['avg_summaries_per_layer']:.1f}")
    print(f"  Queries processed:        {stats['avg_queries_per_layer']:.0f}")
    print(f"  Total refinements:        {stats['total_refinements_made']}")
    print(f"  Refinement rate:          {stats['refinement_rate']:.4f} ({stats['refinement_rate']*100:.2f}%)")
    print(f"  Refinements per query:    {stats['refinements_per_query']:.2f}")
    print("=" * 60 + "\n")
