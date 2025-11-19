#!/usr/bin/env python3
"""
Script to scrape Wikipedia articles and interleave their paragraphs.
Creates datasets with semantically unrelated content for testing KV-cache mechanisms.
"""

import argparse
import random
import re
from typing import List, Tuple, Optional
import urllib.request
import urllib.parse
import json


class WikipediaScraper:
    """Scrapes Wikipedia articles and extracts paragraphs."""

    BASE_URL = "https://en.wikipedia.org/w/api.php"

    def __init__(self, min_paragraph_length: int = 100):
        self.min_paragraph_length = min_paragraph_length
        self.headers = {
            'User-Agent': 'LuKA-Research-Bot/1.0 (Educational research project; Python/urllib)'
        }

    def search_articles(self, query: str, limit: int = 5) -> List[str]:
        """Search for Wikipedia articles matching a query."""
        params = {
            "action": "opensearch",
            "search": query,
            "limit": limit,
            "namespace": 0,
            "format": "json"
        }

        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers=self.headers)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return data[1]

    def get_article_content(self, title: str) -> str:
        """Get the full text content of a Wikipedia article."""
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "format": "json"
        }

        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers=self.headers)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            pages = data["query"]["pages"]
            page_id = list(pages.keys())[0]
            if page_id == "-1":
                raise ValueError(f"Article '{title}' not found")
            return pages[page_id].get("extract", "")

    def extract_paragraphs(self, content: str) -> List[str]:
        """Extract meaningful paragraphs from article content."""
        # Split by double newlines (paragraph separators)
        raw_paragraphs = content.split('\n\n')

        paragraphs = []
        for para in raw_paragraphs:
            # Clean up the paragraph
            para = para.strip().replace('\n', ' ')

            # Filter out short paragraphs, section headers, and lists
            if (len(para) >= self.min_paragraph_length and
                not para.startswith('==') and
                not re.match(r'^[A-Z\s]+$', para)):
                paragraphs.append(para)

        return paragraphs

    def get_random_article(self, category: Optional[str] = None) -> Tuple[str, List[str]]:
        """Get a random article, optionally from a specific category."""
        if category:
            # Search for articles in this category
            articles = self.search_articles(category, limit=10)
            if not articles:
                raise ValueError(f"No articles found for category: {category}")
            title = random.choice(articles)
        else:
            # Get a truly random article
            params = {
                "action": "query",
                "list": "random",
                "rnnamespace": 0,
                "rnlimit": 1,
                "format": "json"
            }
            url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
                title = data["query"]["random"][0]["title"]

        content = self.get_article_content(title)
        paragraphs = self.extract_paragraphs(content)

        return title, paragraphs


class QuestionGenerator:
    """Generates or provides questions about topics."""

    QUESTION_TEMPLATES = [
        "What are the key characteristics of {topic}?",
        "How did {topic} influence modern society?",
        "What are the main principles behind {topic}?",
        "Can you explain the historical context of {topic}?",
        "What makes {topic} significant?",
        "How does {topic} relate to contemporary issues?",
        "What are the fundamental concepts in {topic}?",
        "Why is {topic} important to understand?",
        "What innovations emerged from {topic}?",
        "How has {topic} evolved over time?",
    ]

    @classmethod
    def generate_simple_questions(cls, topic: str, count: int = 3) -> List[str]:
        """Generate simple template-based questions about a topic."""
        questions = []
        templates = random.sample(cls.QUESTION_TEMPLATES, min(count, len(cls.QUESTION_TEMPLATES)))
        for template in templates:
            questions.append(template.format(topic=topic))
        return questions


class ArticleInterleaver:
    """Interleaves paragraphs from multiple articles."""

    def __init__(self, scraper: WikipediaScraper, include_questions: bool = False):
        self.scraper = scraper
        self.include_questions = include_questions
        self.question_generator = QuestionGenerator()

    def interleave_articles(
        self,
        topics: List[str],
        paragraphs_per_chunk: int = 1,
        max_chunks_per_topic: Optional[int] = None
    ) -> str:
        """
        Interleave paragraphs from multiple articles on different topics.

        Args:
            topics: List of topic names or search queries
            paragraphs_per_chunk: Number of paragraphs to group together
            max_chunks_per_topic: Maximum number of chunks to use per topic

        Returns:
            Interleaved text content
        """
        # Fetch all articles
        articles = []
        for topic in topics:
            try:
                title, paragraphs = self.scraper.get_random_article(topic)
                if paragraphs:
                    articles.append({
                        "topic": topic,
                        "title": title,
                        "paragraphs": paragraphs
                    })
                    print(f"✓ Fetched: {title} ({len(paragraphs)} paragraphs)")
                else:
                    print(f"✗ Warning: No suitable paragraphs found for {topic}")
            except Exception as e:
                print(f"✗ Error fetching {topic}: {e}")

        if len(articles) < 2:
            raise ValueError("Need at least 2 articles to interleave")

        # Create chunks from each article
        article_chunks = []
        for article in articles:
            chunks = []
            for i in range(0, len(article["paragraphs"]), paragraphs_per_chunk):
                chunk = article["paragraphs"][i:i+paragraphs_per_chunk]
                chunks.append({
                    "topic": article["topic"],
                    "title": article["title"],
                    "text": "\n\n".join(chunk)
                })

            # Limit chunks if specified
            if max_chunks_per_topic:
                chunks = chunks[:max_chunks_per_topic]

            article_chunks.append(chunks)

        # Interleave the chunks
        interleaved = []
        max_chunks = max(len(chunks) for chunks in article_chunks)

        for i in range(max_chunks):
            for j, chunks in enumerate(article_chunks):
                if i < len(chunks):
                    chunk = chunks[i]
                    interleaved.append(f"[{chunk['title']}]")
                    interleaved.append(chunk["text"])
                    interleaved.append("")

                    # Add questions about OTHER topics
                    if self.include_questions:
                        # Pick a random other topic to ask about
                        other_topics = [art for k, art in enumerate(articles) if k != j]
                        if other_topics:
                            other_topic = random.choice(other_topics)
                            questions = self.question_generator.generate_simple_questions(
                                other_topic["topic"],
                                count=random.randint(1, 2)
                            )
                            for question in questions:
                                interleaved.append(f"Q: {question}")
                            interleaved.append("")

        return "\n".join(interleaved)

    def create_dataset(
        self,
        topic_groups: List[List[str]],
        output_file: str,
        paragraphs_per_chunk: int = 1,
        max_chunks_per_topic: Optional[int] = None
    ):
        """
        Create multiple interleaved article combinations.

        Args:
            topic_groups: List of topic lists, each will create one interleaved output
            output_file: Base filename for output
            paragraphs_per_chunk: Number of paragraphs per chunk
            max_chunks_per_topic: Max chunks per topic
        """
        for idx, topics in enumerate(topic_groups, 1):
            print(f"\n--- Processing group {idx}/{len(topic_groups)} ---")
            print(f"Topics: {', '.join(topics)}")

            try:
                result = self.interleave_articles(
                    topics,
                    paragraphs_per_chunk=paragraphs_per_chunk,
                    max_chunks_per_topic=max_chunks_per_topic
                )

                filename = f"{output_file}_{idx}.md"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(result)
                print(f"✓ Saved to {filename}")

            except Exception as e:
                print(f"✗ Error processing group {idx}: {e}")


# Predefined topic combinations (unrelated topics)
TOPIC_COMBINATIONS = [
    ["Quantum mechanics", "Renaissance art", "Jazz music"],
    ["Cellular biology", "Ancient Rome", "Computer networks"],
    ["Black holes", "French Revolution", "Machine learning"],
    ["Neuroscience", "Medieval architecture", "Cryptography"],
    ["Organic chemistry", "World War I", "Game theory"],
    ["Plate tectonics", "Impressionism", "Distributed systems"],
    ["Immunology", "Industrial Revolution", "Blockchain"],
    ["Particle physics", "Classical music", "Natural language processing"],
    ["Genetics", "Ancient Egypt", "Robotics"],
    ["Astrophysics", "Baroque art", "Computer vision"],
]


def main():
    parser = argparse.ArgumentParser(
        description="Scrape and interleave Wikipedia articles for KV-cache testing"
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        help="Specific topics to interleave (space-separated)"
    )
    parser.add_argument(
        "--random",
        type=int,
        metavar="N",
        help="Use N random topic combinations from predefined list"
    )
    parser.add_argument(
        "--num-topics",
        type=int,
        default=3,
        help="Number of topics per combination (default: 3)"
    )
    parser.add_argument(
        "--paragraphs-per-chunk",
        type=int,
        default=1,
        help="Number of paragraphs per chunk (default: 1)"
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        help="Maximum chunks per topic (default: unlimited)"
    )
    parser.add_argument(
        "--questions",
        action="store_true",
        help="Include questions between paragraph chunks"
    )
    parser.add_argument(
        "--output",
        default="interleaved_articles",
        help="Output filename base (default: interleaved_articles)"
    )
    parser.add_argument(
        "--min-paragraph-length",
        type=int,
        default=100,
        help="Minimum paragraph length to include (default: 100)"
    )

    args = parser.parse_args()

    # Initialize scraper and interleaver
    scraper = WikipediaScraper(min_paragraph_length=args.min_paragraph_length)
    interleaver = ArticleInterleaver(scraper, include_questions=args.questions)

    # Determine topic groups
    topic_groups = []
    if args.topics:
        topic_groups = [args.topics]
    elif args.random:
        topic_groups = random.sample(TOPIC_COMBINATIONS, min(args.random, len(TOPIC_COMBINATIONS)))
    else:
        # Default: use one random combination
        topic_groups = [random.choice(TOPIC_COMBINATIONS)]

    # Create dataset
    interleaver.create_dataset(
        topic_groups=topic_groups,
        output_file=args.output,
        paragraphs_per_chunk=args.paragraphs_per_chunk,
        max_chunks_per_topic=args.max_chunks
    )

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
