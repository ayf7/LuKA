from huggingface_hub import snapshot_download
from pathlib import Path

FILE_PATH = Path(__file__).parent.resolve()

local_dir = FILE_PATH / "model__qwen_luka"
local_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="Qwen/Qwen3-1.7B-Base",
    local_dir=str(local_dir),
)