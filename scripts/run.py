"""
Test script for LuKA with HuggingFace Transformers.
Simple script to test boundary detection during generation.
"""

import torch
from transformers import AutoTokenizer
from modeling.qwen.luka_qwen3 import load_luka_model, set_luka_kv_params
from artifacts.prompts.prompt_loader import load_prompt

# Configuration
model_name = "Qwen/Qwen3-1.7B-Base"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configure LuKA cache parameters via registry (matches run_batch.py style)
set_luka_kv_params(
    compressor="mean",
    segmenter="kl",
    segmenter_kwargs={"threshold": 3, "lag": 8},
    refine_threshold=1,
)

# Load model and tokenizer
model = load_luka_model(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load prompts (single)
prompts = [load_prompt("paragraphs_1")]

# Tokenize with padding for batching
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(prompts, return_tensors="pt", padding=True)
if device == "cuda":
    inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate
outputs = model.generate(
    **inputs,
    max_new_tokens=64,
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

# Print LuKA debug summaries after generation
if hasattr(model, "model") and hasattr(model.model, "luka_kv_controller"):
    controller = model.model.luka_kv_controller
    print("\n=== LuKA Layer Summaries ===")
    for layer_idx in range(controller.num_layers):
        controller.print_layer_summary(layer_idx)
