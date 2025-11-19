"""
Test script for LuKA with HuggingFace Transformers.
Simple script to test boundary detection during generation.
"""

import torch
from transformers import AutoTokenizer
from modeling.qwen.luka_qwen3 import load_luka_model
from artifacts.prompts.prompt_loader import load_prompt

# Configuration
model_name = "Qwen/Qwen3-1.7B-Base"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model = load_luka_model(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load prompt
prompt = load_prompt("paragraphs_2")

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt")
if device == "cuda":
    inputs = {k: v.to(device) for k, v in inputs.items()}

outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

# Decode and print
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
new_text = generated_text[len(prompt):]

print("\n" + "=" * 80)
print("Prompt:")
print("=" * 80)
print(prompt)
print("=" * 80 + "\n")

print("Generated output:")
print(new_text)
print("\n" + "=" * 80)