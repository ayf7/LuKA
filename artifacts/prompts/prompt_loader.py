from pathlib import Path

def load_prompt(file: str) -> str:
    prompts_dir = Path(__file__).parent

    # Add .md extension if not provided
    if not file.endswith('.md'):
        file = f"{file}.md"

    file_path = prompts_dir / file

    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file '{file}' not found in {prompts_dir}")

    return file_path.read_text()