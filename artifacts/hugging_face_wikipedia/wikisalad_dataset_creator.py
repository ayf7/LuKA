"""
WikiSaladQA Dataset Creator
Creates interlaced Wikipedia article paragraphs with corresponding Q&A pairs
for evaluating LuKA's compression and attention mechanisms.

This script creates evaluation datasets that test:
1. Semantic boundary detection (topic changes)
2. Selective compression (irrelevant context)
3. Decompression on-demand (when previously irrelevant context becomes relevant)
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from datasets import load_dataset
from tqdm import tqdm


class WikiSaladCreator:
    """Creates interlaced Wikipedia QA datasets for long-context evaluation."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.dataset = None
        
    def load_squad_dataset(self, version: str = "v2"):
        """Load SQuAD dataset from HuggingFace."""
        print(f"Loading SQuAD {version}...")
        if version == "v2":
            self.dataset = load_dataset("rajpurkar/squad_v2", split="train")
        else:
            self.dataset = load_dataset("rajpurkar/squad", split="train")
        print(f"Loaded {len(self.dataset)} examples")
        
    def group_by_article(self) -> Dict[str, List[Dict]]:
        """Group QA pairs by Wikipedia article title."""
        print("Grouping examples by article...")
        articles = {}
        
        for example in tqdm(self.dataset):
            title = example['title']
            if title not in articles:
                articles[title] = []
            articles[title].append({
                'context': example['context'],
                'question': example['question'],
                'answers': example['answers'],
                'id': example['id']
            })
        
        print(f"Found {len(articles)} unique articles")
        return articles
    
    def split_context_into_paragraphs(self, context: str) -> List[str]:
        """Split context into paragraphs, handling various paragraph delimiters."""
        # Split by double newlines or single newlines followed by capitals
        paragraphs = []
        current_para = []
        
        sentences = context.split('. ')
        for i, sent in enumerate(sentences):
            current_para.append(sent)
            # End paragraph if we see a natural break or every 3-4 sentences
            if i < len(sentences) - 1:
                current_para.append('. ')
            if (i + 1) % 3 == 0 or i == len(sentences) - 1:
                paragraphs.append(''.join(current_para).strip())
                current_para = []
        
        return [p for p in paragraphs if len(p) > 50]  # Filter short fragments
    
    def create_interlaced_prompt(
        self,
        articles: List[Tuple[str, List[Dict]]],
        interleave_pattern: str = "ABAB",
        include_questions: bool = True
    ) -> Dict:
        """
        Create an interlaced prompt from multiple articles.
        
        Args:
            articles: List of (title, qa_list) tuples
            interleave_pattern: Pattern like "ABAB" or "AABBAABB" or "ABC"
            include_questions: Whether to include questions after each segment
            
        Returns:
            Dictionary with prompt, questions, and metadata
        """
        pattern_chars = list(set(interleave_pattern))
        num_topics = len(pattern_chars)
        
        if len(articles) < num_topics:
            raise ValueError(f"Need at least {num_topics} articles for pattern {interleave_pattern}")
        
        # Map pattern characters to article indices
        article_map = {char: articles[i] for i, char in enumerate(pattern_chars)}
        
        # Build the interlaced prompt
        prompt_parts = []
        all_questions = []
        metadata = {
            'pattern': interleave_pattern,
            'articles': [title for title, _ in articles[:num_topics]],
            'segment_boundaries': []
        }
        
        current_position = 0
        for segment_idx, topic_char in enumerate(interleave_pattern):
            title, qa_list = article_map[topic_char]
            
            # Get a QA pair for this segment
            if segment_idx < len(qa_list):
                qa = qa_list[segment_idx]
            else:
                # Cycle through if we run out
                qa = qa_list[segment_idx % len(qa_list)]
            
            # Add context paragraph
            context = qa['context']
            paragraphs = self.split_context_into_paragraphs(context)
            
            # Take first paragraph or full context if short
            segment_text = paragraphs[0] if paragraphs else context
            
            prompt_parts.append(f"# Segment {segment_idx + 1}: {title}\n\n{segment_text}\n")
            
            # Track boundaries for evaluation
            start_pos = current_position
            current_position += len(segment_text)
            metadata['segment_boundaries'].append({
                'segment_id': segment_idx,
                'topic': topic_char,
                'article': title,
                'start_char': start_pos,
                'end_char': current_position
            })
            
            # Optionally add question after segment
            if include_questions:
                question_text = f"\nQuestion {segment_idx + 1}: {qa['question']}\n"
                prompt_parts.append(question_text)
                current_position += len(question_text)
            
            all_questions.append({
                'question': qa['question'],
                'answers': qa['answers'],
                'segment_id': segment_idx,
                'topic': topic_char,
                'article': title
            })
        
        return {
            'prompt': '\n'.join(prompt_parts),
            'questions': all_questions,
            'metadata': metadata
        }
    
    def create_evaluation_set(
        self,
        num_examples: int = 100,
        topics_per_example: int = 2,
        pattern: str = "AABBAABB",
        output_file: str = "wikisalad_eval.json"
    ):
        """
        Create a full evaluation dataset with multiple interlaced examples.
        
        Args:
            num_examples: Number of interlaced prompts to create
            topics_per_example: Number of distinct topics to interleave
            pattern: Interleaving pattern (e.g., "ABAB", "AABBAABB", "ABCABC")
            output_file: Where to save the dataset
        """
        if self.dataset is None:
            self.load_squad_dataset()
        
        articles = self.group_by_article()
        article_list = list(articles.items())
        
        # Filter articles with enough QA pairs
        min_qa_needed = pattern.count('A')  # Max count of any letter
        filtered_articles = [
            (title, qas) for title, qas in article_list 
            if len(qas) >= min_qa_needed
        ]
        
        print(f"Creating {num_examples} evaluation examples...")
        print(f"Pattern: {pattern}")
        print(f"Topics per example: {topics_per_example}")
        
        eval_dataset = []
        
        for i in tqdm(range(num_examples)):
            # Sample random articles for this example
            sampled_articles = random.sample(filtered_articles, topics_per_example)
            
            try:
                interlaced = self.create_interlaced_prompt(
                    sampled_articles,
                    interleave_pattern=pattern,
                    include_questions=True
                )
                
                eval_dataset.append({
                    'example_id': i,
                    'prompt': interlaced['prompt'],
                    'questions': interlaced['questions'],
                    'metadata': interlaced['metadata']
                })
            except Exception as e:
                print(f"Warning: Failed to create example {i}: {e}")
                continue
        
        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(eval_dataset, f, indent=2)
        
        print(f"\nSaved {len(eval_dataset)} examples to {output_file}")
        return eval_dataset
    
    def create_difficulty_variants(self, base_output_dir: str = "wikisalad_datasets"):
        """
        Create multiple dataset variants with increasing difficulty:
        - Easy: 2 topics, simple ABAB pattern
        - Medium: 2 topics, longer segments AABBAABB
        - Hard: 3 topics, complex ABC pattern
        - Very Hard: 3 topics with long segments AAABBBCCC
        """
        variants = [
            {
                'name': 'easy_2topic_short',
                'topics': 2,
                'pattern': 'ABAB',
                'num_examples': 100,
                'description': 'Two topics, short alternating segments'
            },
            {
                'name': 'medium_2topic_medium',
                'topics': 2,
                'pattern': 'AABBAABB',
                'num_examples': 100,
                'description': 'Two topics, medium-length segments'
            },
            {
                'name': 'hard_3topic_short',
                'topics': 3,
                'pattern': 'ABCABC',
                'num_examples': 100,
                'description': 'Three topics, short rotating segments'
            },
            {
                'name': 'very_hard_3topic_long',
                'topics': 3,
                'pattern': 'AAABBBCCCAAABBBCCC',
                'num_examples': 50,
                'description': 'Three topics, long segments with delayed returns'
            }
        ]
        
        base_path = Path(base_output_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        
        for variant in variants:
            print(f"\n{'='*60}")
            print(f"Creating variant: {variant['name']}")
            print(f"Description: {variant['description']}")
            print(f"{'='*60}")
            
            output_file = base_path / f"{variant['name']}.json"
            
            self.create_evaluation_set(
                num_examples=variant['num_examples'],
                topics_per_example=variant['topics'],
                pattern=variant['pattern'],
                output_file=str(output_file)
            )


def main():
    """Example usage and dataset creation."""
    creator = WikiSaladCreator(seed=42)
    
    # Option 1: Create a single evaluation set
    print("Creating a single evaluation dataset...")
    creator.create_evaluation_set(
        num_examples=50,
        topics_per_example=2,
        pattern="AABBAABB",
        output_file="wikisalad_eval_example.json"
    )
    
    # Option 2: Create multiple difficulty variants
    print("\n" + "="*60)
    print("Creating complete evaluation suite with multiple difficulty levels...")
    print("="*60)
    creator.create_difficulty_variants(base_output_dir="wikisalad_datasets")
    
    print("\n Dataset creation complete!")


if __name__ == "__main__":
    main()
