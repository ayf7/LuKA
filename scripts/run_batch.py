"""
Test script for LuKA with HuggingFace Transformers (new controller path).
Compares topdown attention vs lined attention results.
"""

import torch
from transformers import AutoTokenizer
from modeling.qwen.luka_qwen3 import load_luka_model, set_luka_kv_params
from artifacts.prompts.prompt_loader import load_prompt

# Configuration
model_name = "Qwen/Qwen3-1.7B-Base"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load prompts (batch)
prompts = [
    load_prompt("paragraphs_2"),
    load_prompt("paragraphs_1"),
]

# Tokenize with padding for batching
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

def run_generation(mode_name, use_lined_attention=False, lined_layers=None):
    """Run generation with specified attention mode."""
    print(f"\n{'='*80}")
    print(f"Running with {mode_name.upper()} ATTENTION")
    print(f"{'='*80}\n")
    
    # Set LuKA params
    set_luka_kv_params(
        compressor="mean",
        segmenter="dummy",
        refine_threshold=1,
        segment_interval=16,
    )
    
    # Load model
    model = load_luka_model(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    
    # Configure attention mode
    if hasattr(model, "model") and hasattr(model.model, "luka_kv_controller"):
        controller = model.model.luka_kv_controller
        controller.use_lined_attention = use_lined_attention
        if lined_layers is not None:
            controller.lined_layers = set(lined_layers)
        else:
            if use_lined_attention:
                # H2O-style: protect last few layers (most sensitive to approximation)
                # Use lined attention on early layers (0..23), top-down on last layers (24..27)
                num_layers = controller.num_layers
                protect_last_n = 4  # Protect last 4 layers
                controller.lined_layers = set(range(0, max(0, num_layers - protect_last_n)))
            else:
                controller.lined_layers = set()
        
        print(f"Configuration:")
        print(f"  use_lined_attention: {controller.use_lined_attention}")
        print(f"  lined_layers: {controller.lined_layers}")
        print(f"  grid_top_k: {controller.grid_top_k}")
        print(f"  grid_update_interval: {controller.grid_update_interval}")
        print(f"  grid_decay: {controller.grid_decay}\n")
    
    # Prepare inputs
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
    
    # Decode and print results
    results = []
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        new_text = generated_text[len(prompt):]
        results.append((prompt, new_text))
        
        print(f"\n{'='*80}")
        print(f"Prompt {i}:")
        print(f"{'='*80}")
        print(prompt)
        print(f"{'='*80}\n")
        print(f"Generated output ({mode_name}):")
        print(new_text)
        print(f"{'='*80}")
    
    # Print LuKA debug summaries
    if hasattr(model, "model") and hasattr(model.model, "luka_kv_controller"):
        controller = model.model.luka_kv_controller
        print(f"\n=== {mode_name.upper()} ATTENTION - Layer Summaries ===")
        for layer_idx in range(controller.num_layers):
            print("------------------------------")
            controller.print_layer_summary(layer_idx)
    
    return results

# Run with TOP-DOWN attention (default)
topdown_results = run_generation("top-down", use_lined_attention=False)

# Run with LINED attention
lined_results = run_generation("lined", use_lined_attention=True, lined_layers=None)  # None = all layers

# Compare results
print(f"\n{'='*80}")
print("COMPARISON SUMMARY")
print(f"{'='*80}\n")

for i, (prompt, _) in enumerate(zip(prompts, topdown_results)):
    print(f"\nPrompt {i}:")
    print(f"{'-'*80}")
    print(f"Prompt: {prompt[:100]}...")
    print(f"\nTopdown output: {topdown_results[i][1][:200]}...")
    print(f"\nLined output:   {lined_results[i][1][:200]}...")
    print(f"{'-'*80}")
