"""
Example Usage of LuKA Evaluation Pipeline

Demonstrates how to:
1. Create a simple mock model
2. Run evaluations in both modes
3. Score the results

This is useful for testing the pipeline without needing actual models.
"""

from evaluation import ModelInterface, GenerationConfig, QAEvaluator
from typing import Dict, Any, Optional
import random


class MockModel(ModelInterface):
    """
    Simple mock model that generates random answers.
    Useful for testing the evaluation pipeline.
    """

    def __init__(self, name: str = "mock-model"):
        self.name = name
        self.call_count = 0

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """Generate a mock answer."""
        self.call_count += 1

        # Extract question if present
        if "Question:" in prompt:
            question = prompt.split("Question:")[-1].split("Answer:")[0].strip()

            # Generate simple mock answer based on question type
            if "what" in question.lower():
                return f"Mock answer {self.call_count}: It is a thing."
            elif "when" in question.lower():
                return f"Mock answer {self.call_count}: In the year 2000."
            elif "where" in question.lower():
                return f"Mock answer {self.call_count}: In a place."
            elif "who" in question.lower():
                return f"Mock answer {self.call_count}: A person."
            elif "why" in question.lower():
                return f"Mock answer {self.call_count}: Because of reasons."
            elif "how" in question.lower():
                return f"Mock answer {self.call_count}: By doing something."
            else:
                return f"Mock answer {self.call_count}: Yes, that is correct."
        else:
            return f"Mock answer {self.call_count}"

    def get_model_info(self) -> Dict[str, Any]:
        """Return mock model info."""
        return {
            'name': self.name,
            'type': 'mock',
            'parameters': 0,
            'calls': self.call_count
        }

    def reset(self):
        """Reset doesn't do anything for mock model."""
        pass


def main():
    """Run example evaluation with mock model."""
    print("="*60)
    print("LuKA Evaluation Pipeline - Example Usage")
    print("="*60)

    # 1. Create a mock model
    print("\n1. Creating mock model...")
    model = MockModel(name="example-mock-model")

    # 2. Load evaluator
    print("\n2. Loading evaluator...")
    dataset_path = "artifacts/hugging_face_wikipedia/wikisalad_eval_example.json"
    evaluator = QAEvaluator(dataset_path)

    # 3. Create generation config
    config = GenerationConfig(
        max_new_tokens=50,
        temperature=0.0,
        do_sample=False
    )

    # 4. Run simple Q&A evaluation (just on 2 examples for demo)
    print("\n3. Running Simple Q&A evaluation (2 examples)...")
    simple_result = evaluator.evaluate_simple_qa(
        model=model,
        config=config,
        num_examples=2
    )

    # 5. Save results
    print("\n4. Saving results...")
    output_path = "results/example_simple.json"
    evaluator.save_results(simple_result, output_path)

    # 6. Run sequential evaluation
    print("\n5. Running Sequential Multi-Answer evaluation (2 examples)...")
    model.call_count = 0  # Reset counter
    sequential_result = evaluator.evaluate_sequential_multi_answer(
        model=model,
        config=config,
        num_examples=2
    )

    # 7. Save results
    print("\n6. Saving results...")
    output_path = "results/example_sequential.json"
    evaluator.save_results(sequential_result, output_path)

    # 8. Print summary
    print("\n" + "="*60)
    print("EXAMPLE COMPLETE")
    print("="*60)
    print("\nSimple Q&A Mode:")
    print(f"  Examples: {simple_result.num_examples}")
    print(f"  Model calls: {simple_result.num_examples * len(simple_result.predictions[0]['answers'])}")
    print(f"  Sample answer: {simple_result.predictions[0]['answers'][0]}")

    print("\nSequential Mode:")
    print(f"  Examples: {sequential_result.num_examples}")
    print(f"  Model calls: {sum(len(p['answers']) for p in sequential_result.predictions)}")
    print(f"  Sample answer: {sequential_result.predictions[0]['answers'][0]}")

    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Replace MockModel with actual model (HuggingFaceModel, APIModel, etc.)")
    print("2. Run on full dataset (remove num_examples limit)")
    print("3. Score results using: python -m evaluation.score_results \\")
    print("     --results results/example_simple.json \\")
    print("     --dataset artifacts/hugging_face_wikipedia/wikisalad_eval_example.json \\")
    print("     --output results/example_simple_scores.json")


if __name__ == "__main__":
    main()
