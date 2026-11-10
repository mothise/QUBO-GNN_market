import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DIPLOMA_DATA", REPO_ROOT / "data"))
