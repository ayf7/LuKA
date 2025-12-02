"""
Simple Long-Form QA Benchmark Evaluator

Takes any model, feeds it questions, extracts answers, computes scores.
Supports: HuggingFace models, OpenAI API, Anthropic API, or custom models.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np


@dataclass
class EvalResult:
    """Results from evaluating a model."""
    dataset_name: str
    num_examples: int
    exact_match: float
    f1_score: float
    per_example_results: List[Dict]
    

class SimpleQAEvaluator:
    """Evaluate any model on QA benchmarks."""
    
    def __init__(self, dataset_path: str):
        """
        Load a QA dataset.

        Args:
            dataset_path: Path to JSON file with QA pairs
        """
        self.dataset_path = dataset_path  # Store for later use
        with open(dataset_path, 'r') as f:
            self.data = json.load(f)

        # Handle different dataset formats
        if isinstance(self.data, list):
            self.examples = self.data
        elif 'data' in self.data:
            self.examples = self.data['data']
        else:
            self.examples = self.data

        print(f"Loaded {len(self.examples)} examples from {dataset_path}")
    
    def evaluate(
        self,
        model_fn: Callable[[str, str], str],
        max_examples: Optional[int] = None,
        verbose: bool = True
    ) -> EvalResult:
        """
        Evaluate a model on the dataset.
        
        Args:
            model_fn: Function that takes (context, question) and returns answer string
            max_examples: Limit number of examples (for quick testing)
            verbose: Print progress
            
        Returns:
            EvalResult with scores
        """
        examples_to_eval = self.examples[:max_examples] if max_examples else self.examples
        
        exact_matches = []
        f1_scores = []
        per_example = []
        
        iterator = tqdm(examples_to_eval, desc="Evaluating") if verbose else examples_to_eval
        
        for i, example in enumerate(iterator):
            # Extract context and questions
            context, questions = self._parse_example(example)

            for q_idx, question_data in enumerate(questions):
                question = question_data['question']
                # Extract answer text (handles both dict and list formats)
                answers_data = question_data['answers']
                if isinstance(answers_data, dict) and 'text' in answers_data:
                    true_answers = answers_data['text']
                else:
                    true_answers = answers_data
                
                try:
                    # Get model prediction
                    predicted_answer = model_fn(context, question)
                    
                    # Compute metrics
                    em = self._compute_exact_match(predicted_answer, true_answers)
                    f1 = self._compute_f1(predicted_answer, true_answers)
                    
                    exact_matches.append(em)
                    f1_scores.append(f1)
                    
                    per_example.append({
                        'example_id': i,
                        'question_id': q_idx,
                        'question': question,
                        'predicted': predicted_answer,
                        'true_answers': true_answers,
                        'exact_match': em,
                        'f1': f1
                    })
                    
                except Exception as e:
                    if verbose:
                        print(f"\nError on example {i}, question {q_idx}: {e}")
                    per_example.append({
                        'example_id': i,
                        'question_id': q_idx,
                        'error': str(e)
                    })
        
        return EvalResult(
            dataset_name=Path(self.dataset_path).stem if hasattr(self, 'dataset_path') else 'unknown',
            num_examples=len(per_example),
            exact_match=float(np.mean(exact_matches)),
            f1_score=float(np.mean(f1_scores)),
            per_example_results=per_example
        )
    
    def _parse_example(self, example: Dict) -> tuple[str, List[Dict]]:
        """
        Parse example into context and questions.
        Handles WikiSalad format and standard SQuAD format.
        """
        # WikiSalad format
        if 'prompt' in example and 'questions' in example:
            return example['prompt'], example['questions']
        
        # SQuAD format
        elif 'context' in example and 'question' in example:
            return example['context'], [{
                'question': example['question'],
                'answers': example['answers']['text'] if isinstance(example['answers'], dict) else example['answers']
            }]
        
        # SQuAD nested format
        elif 'paragraphs' in example:
            all_questions = []
            context = ""
            for para in example['paragraphs']:
                context += para['context'] + "\n\n"
                for qa in para.get('qas', []):
                    all_questions.append({
                        'question': qa['question'],
                        'answers': [a['text'] for a in qa.get('answers', [])]
                    })
            return context, all_questions
        
        else:
            raise ValueError(f"Unknown dataset format: {example.keys()}")
    
    def _normalize_answer(self, text: str) -> str:
        """Normalize answer for comparison."""
        text = text.lower()
        # Remove articles
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def _compute_exact_match(self, prediction: str, true_answers: List[str]) -> float:
        """
        Check if prediction matches any true answer.
        Uses both exact match and substring containment (like SQuAD).
        """
        # Handle unanswerable questions (empty answer list)
        if not true_answers:
            return 0.0

        pred_norm = self._normalize_answer(prediction)

        for true in true_answers:
            true_norm = self._normalize_answer(true)
            # Exact match
            if pred_norm == true_norm:
                return 1.0
            # Substring match: true answer is contained in prediction
            if true_norm in pred_norm:
                return 1.0
            # Reverse substring: prediction is contained in true answer
            # (for cases where prediction is more concise)
            if pred_norm in true_norm:
                return 1.0

        return 0.0
    
    def _compute_f1(self, prediction: str, true_answers: List[str]) -> float:
        """Compute F1 score against best matching answer."""
        # Handle unanswerable questions (empty answer list)
        if not true_answers:
            return 0.0

        pred_tokens = self._normalize_answer(prediction).split()

        if len(pred_tokens) == 0:
            return 1.0 if all(len(self._normalize_answer(t)) == 0 for t in true_answers) else 0.0

        f1_scores = []
        for true in true_answers:
            true_tokens = self._normalize_answer(true).split()

            if len(true_tokens) == 0:
                f1_scores.append(0.0)
                continue

            common = set(pred_tokens) & set(true_tokens)

            if len(common) == 0:
                f1_scores.append(0.0)
                continue

            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(true_tokens)
            f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)

        return max(f1_scores) if f1_scores else 0.0
    
    def save_results(self, results: EvalResult, output_path: str):
        """Save evaluation results to JSON."""
        output = {
            'dataset': results.dataset_name,
            'num_examples': results.num_examples,
            'exact_match': results.exact_match,
            'f1_score': results.f1_score,
            'per_example': results.per_example_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
    
    def print_summary(self, results: EvalResult):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print(f"EVALUATION RESULTS: {results.dataset_name}")
        print("="*60)
        print(f"Examples evaluated: {results.num_examples}")
        print(f"Exact Match:        {results.exact_match:.1%}")
        print(f"F1 Score:           {results.f1_score:.1%}")
        print("="*60)


# ============================================================================
# Model Adapters
# ============================================================================

class HuggingFaceModelAdapter:
    """Adapter for HuggingFace models."""

    def __init__(self, model_name: str, device: str = "cuda"):
        """
        Load a HuggingFace model.

        Args:
            model_name: Model identifier (e.g., "google/flan-t5-base", "Qwen/Qwen2.5-0.5B-Instruct")
            device: Device to run on ("cuda" or "cpu")
        """
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set pad token if not set (needed for some models like Qwen)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Try seq2seq first, then causal
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
            self.model_type = "seq2seq"
        except:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            self.model_type = "causal"
        
        self.device = device
        print(f"Model loaded ({self.model_type})")
    
    def __call__(self, context: str, question: str) -> str:
        """Generate answer given context and question."""
        import torch

        # Format prompt based on model type
        if self.model_type == "seq2seq":
            # FLAN-T5 style: clear instruction first
            prompt = f"Answer the following question based on the context.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=4096,
                truncation=True
            ).to(self.device)
        else:
            # For causal models (like Qwen), try chat template first
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context. Provide concise, direct answers."},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nProvide a brief, direct answer to the question based only on the context."}
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback for models without chat template
                prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=4096,
                truncation=True
            ).to(self.device)

        # Generate with sampling for randomness across runs
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,  # Enable sampling for randomness
                temperature=0.7,  # Control randomness (0.7 is reasonable)
                top_p=0.9,  # Nucleus sampling
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # For causal models, extract just the answer part
        if self.model_type == "causal":
            # Strip chat template markers if present
            if "assistant" in answer:
                # Extract text after the last "assistant" marker
                parts = answer.split("assistant")
                answer = parts[-1].strip()

            # Remove the input prompt from the output if still present
            if prompt in answer:
                answer = answer[len(prompt):].strip()

            # Try to extract after common markers
            for marker in ["Answer:", "answer:", "A:"]:
                if marker in answer:
                    answer = answer.split(marker)[-1].strip()
                    break

        return answer


class OpenAIModelAdapter:
    """Adapter for OpenAI models."""
    
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """
        Initialize OpenAI API.
        
        Args:
            model: Model name (e.g., "gpt-3.5-turbo", "gpt-4")
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        import openai
        if api_key:
            openai.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)
    
    def __call__(self, context: str, question: str) -> str:
        """Generate answer given context and question."""
        prompt = f"""Context: {context}

Question: {question}

Answer the question based only on the context above. Keep the answer concise."""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0
        )
        
        return response.choices[0].message.content.strip()


# ============================================================================
# Example Usage
# ============================================================================

def example_huggingface():
    """Example: Evaluate a HuggingFace model."""
    print("Example 1: HuggingFace Model Evaluation")
    print("="*60)

    # Load model
    model = HuggingFaceModelAdapter("google/flan-t5-small", device="cpu")

    # Load dataset (use absolute path from project root)
    dataset_path = Path(__file__).parent.parent / "artifacts/hugging_face_wikipedia/wikisalad_datasets/easy_2topic_short.json"
    evaluator = SimpleQAEvaluator(str(dataset_path))

    # Run evaluation (limit to 10 examples for demo)
    results = evaluator.evaluate(model, max_examples=10)

    # Print results
    evaluator.print_summary(results)
    evaluator.save_results(results, "eval_results_flan_t5.json")


def example_openai():
    """Example: Evaluate OpenAI model."""
    print("Example 2: OpenAI Model Evaluation")
    print("="*60)

    # Load model (requires OPENAI_API_KEY env var)
    model = OpenAIModelAdapter("gpt-3.5-turbo")

    # Load dataset (use absolute path from project root)
    dataset_path = Path(__file__).parent.parent / "artifacts/hugging_face_wikipedia/wikisalad_datasets/easy_2topic_short.json"
    evaluator = SimpleQAEvaluator(str(dataset_path))

    # Run evaluation
    results = evaluator.evaluate(model, max_examples=10)

    # Print results
    evaluator.print_summary(results)
    evaluator.save_results(results, "eval_results_gpt35.json")


def example_custom_model():
    """Example: Evaluate a custom model."""
    print("Example 3: Custom Model Evaluation")
    print("="*60)

    # Define your custom model function
    def my_custom_model(context: str, question: str) -> str:
        """Your custom QA model here."""
        # Example: simple keyword matching (obviously bad)
        if "when" in question.lower():
            # Extract years from context
            years = re.findall(r'\b\d{4}\b', context)
            return years[0] if years else "Unknown"
        elif "who" in question.lower():
            # Extract capitalized names
            names = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', context)
            return names[0] if names else "Unknown"
        else:
            return "I don't know"

    # Load dataset (use absolute path from project root)
    dataset_path = Path(__file__).parent.parent / "artifacts/hugging_face_wikipedia/wikisalad_datasets/easy_2topic_short.json"
    evaluator = SimpleQAEvaluator(str(dataset_path))

    # Run evaluation
    results = evaluator.evaluate(my_custom_model, max_examples=10)

    # Print results
    evaluator.print_summary(results)


def example_qwen():
    """Example: Evaluate a Qwen model."""
    print("Example 4: Qwen Model Evaluation")
    print("="*60)

    # Load model - you can use any Qwen model variant:
    # - "Qwen/Qwen2.5-0.5B-Instruct" (smallest, ~500MB)
    # - "Qwen/Qwen2.5-1.5B-Instruct" (~1.5GB)
    # - "Qwen/Qwen2.5-3B-Instruct" (~3GB)
    # - "Qwen/Qwen2.5-7B-Instruct" (~7GB)
    model = HuggingFaceModelAdapter("Qwen/Qwen2.5-0.5B-Instruct", device="cpu")

    # Load dataset (use absolute path from project root)
    dataset_path = Path(__file__).parent.parent / "artifacts/hugging_face_wikipedia/wikisalad_datasets/easy_2topic_short.json"
    evaluator = SimpleQAEvaluator(str(dataset_path))

    # Run evaluation (limit to 10 examples for demo)
    results = evaluator.evaluate(model, max_examples=10)

    # Print results
    evaluator.print_summary(results)
    evaluator.save_results(results, "eval_results_qwen.json")


if __name__ == "__main__":
    print("Simple Long-Form QA Benchmark Evaluator")
    print("="*60)
    print("\nChoose an example:")
    print("1. HuggingFace model (FLAN-T5)")
    print("2. OpenAI model (GPT-3.5)")
    print("3. Custom model (keyword matching demo)")
    print("4. Qwen model (Qwen2.5-0.5B-Instruct)")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        example_huggingface()
    elif choice == "2":
        example_openai()
    elif choice == "3":
        example_custom_model()
    elif choice == "4":
        example_qwen()
    else:
        print("Running custom model example by default...")
        example_custom_model()
