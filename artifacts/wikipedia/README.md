# Wikipedia Article Interleaver

Script for creating datasets with interleaved paragraphs from unrelated Wikipedia articles.

### Basic Examples

```bash
# Interleave 3 specific topics
python scrape.py --topics "Algebraic topology" "World War II" "iPhone"

# Use random topics from predefined list
python scrape.py --random 5

# Add questions between paragraphs
python scrape.py --topics "Quantum mechanics" "Jazz music" --questions

# Group multiple paragraphs together
python scrape.py --topics "Biology" "Architecture" --paragraphs-per-chunk 2

# Limit chunks per topic
python scrape.py --topics "Chemistry" "History" --max-chunks 5

# Custom output filename
python scrape.py --topics "AI" "Art" --output my_dataset
```

### Options

- `--topics TOPIC [TOPIC ...]`: Specific topics to interleave
- `--random N`: Use N random topic combinations from predefined list
- `--num-topics N`: Number of topics per combination (default: 3)
- `--paragraphs-per-chunk N`: Paragraphs per chunk (default: 1)
- `--max-chunks N`: Maximum chunks per topic
- `--questions`: Include questions between chunks
- `--output FILENAME`: Output filename base (default: interleaved_articles)
- `--min-paragraph-length N`: Minimum paragraph length (default: 100)

## Output Format

The script generates markdown files with:
- Article titles in brackets: `[Article Title]`
- Paragraphs from each article
- Optional questions about other topics (when `--questions` is used)

### Example Output Pattern (with questions)

```
[Algebraic Topology Article]
Paragraph about algebraic topology...

Q: What were the key events of World War II?

[World War II Article]
Paragraph about WWII...

Q: How does algebraic topology apply to modern problems?

...
```

## Predefined Topic Combinations

The script includes 10 predefined combinations of unrelated topics:
- Quantum mechanics, Renaissance art, Jazz music
- Cellular biology, Ancient Rome, Computer networks
- Black holes, French Revolution, Machine learning
- And more...

## Notes

- Requires internet connection to fetch Wikipedia articles
- Uses Wikipedia API (no external dependencies needed)
- Automatically filters out short paragraphs and section headers
- Questions are template-based when using `--questions` flag
