"""
Simple script to test lined attention on different WikiSalad questions.
Shows question + answer pairs without verbose output.
"""

import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer

from modeling.qwen.luka_qwen3 import load_luka_model, set_luka_kv_params
from experiments.comprehensive_eval import generate_text


WIKISALAD_DIR = (
    Path(__file__).resolve().parent.parent
    / "artifacts/hugging_face_wikipedia/wikisalad_datasets"
)


def resolve_dataset_path(dataset_arg: str) -> Path:
    """Resolve dataset path."""
    repo_root = Path(__file__).resolve().parent.parent
    dataset_path = Path(dataset_arg)
    
    candidates = [
        dataset_path,
        repo_root / dataset_path,
        WIKISALAD_DIR / dataset_path.name,
    ]
    
    if dataset_path.suffix != ".json":
        candidates.append(WIKISALAD_DIR / f"{dataset_path.name}.json")
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    raise FileNotFoundError(f"Dataset not found: {dataset_arg}")


def parse_example(example):
    """Parse example into context and questions."""
    if 'prompt' in example and 'questions' in example:
        return example['prompt'], example['questions']
    elif 'context' in example and 'question' in example:
        return example['context'], [{
            'question': example['question'],
            'answers': example['answers']['text'] if isinstance(example['answers'], dict) else example['answers']
        }]
    else:
        raise ValueError(f"Unknown dataset format: {example.keys()}")


def extract_answer(text: str) -> str:
    """Extract just the answer, stopping at continuation patterns."""
    if '\n' in text:
        text = text.split('\n')[0].strip()
    
    stop_patterns = ['\n\n#', '\n#', '\n\nQuestion', '\nQuestion', '\n\nAnswer', '\nAnswer']
    for pattern in stop_patterns:
        if pattern in text:
            text = text.split(pattern)[0].strip()
            break
    
    if len(text) > 100:
        sentences = text.split('.')
        if len(sentences) > 1:
            first_sent = sentences[0].strip()
            if len(first_sent) > 10:
                text = first_sent
    
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Test lined attention on WikiSalad questions")
    parser.add_argument("--dataset", type=str, required=True, help="WikiSalad dataset path")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B-Base", help="Model name")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--max-examples", type=int, default=3, help="Max examples to test")
    parser.add_argument("--max-questions", type=int, default=4, help="Max questions per example")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate")
    
    args = parser.parse_args()
    
    device = args.device
    dataset_path = resolve_dataset_path(args.dataset)
    
    # Load model
    set_luka_kv_params(
        default_tail_len=16,
        min_compress_chunk=16,
        max_pages=15,
        refine_threshold=0.05,
        compressor=None,
        segmenter="dummy",
    )
    
    model = load_luka_model(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    ).to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configure lined attention
    controller = model.model.layers[0].self_attn.luka_kv
    controller.use_lined_attention = True
    controller.min_lined_seq_len = 384
    controller.min_lined_tail_window = 192
    controller.grid_update_interval = 4
    controller.grid_decay = 0.95
    controller.grid_min_change_ratio = 0.0
    controller.debug = False
    controller.lined_layers = set(range(controller.num_layers))
    
    # Load dataset
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    
    examples_to_eval = dataset[:args.max_examples]
    
    print(f"Testing {len(examples_to_eval)} examples with lined attention\n")
    print("="*80)
    
    for i, example in enumerate(examples_to_eval):
        context, questions = parse_example(example)
        questions_to_eval = questions[:args.max_questions]
        
        print(f"\n[Example {i+1}]")
        print("-"*80)
        
        for q_idx, question_data in enumerate(questions_to_eval):
            question = question_data['question']
            answers_data = question_data['answers']
            if isinstance(answers_data, dict) and 'text' in answers_data:
                true_answers = answers_data['text']
            else:
                true_answers = answers_data
            
            try:
                # Reset cache
                controller.reset()
                
                # Format prompt
                input_prompt = f"{context}\n\nAnswer the following question in one short phrase. Do not include any other text.\n\nQuestion: {question}\n\nAnswer:"
                
                # Generate
                predicted_answer, _ = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=input_prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.3,
                    top_p=0.9,
                )
                
                # Extract answer
                predicted_answer = extract_answer(predicted_answer)
                
                # Display
                print(f"\nQuestion {q_idx+1}: {question}")
                print(f"True Answer: {true_answers[0] if true_answers else 'N/A'}")
                print(f"Predicted:    {predicted_answer}")
                
            except Exception as e:
                print(f"\nQuestion {q_idx+1}: {question}")
                print(f"ERROR: {e}")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    main()

