"""Make the repo-root `evals` package importable under `uv run pytest` (console script does not
add CWD to sys.path). `fplcopilot` itself is installed into the venv and needs nothing here."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
