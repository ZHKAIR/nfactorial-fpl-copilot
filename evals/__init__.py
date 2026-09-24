"""Offline evals for the news-RAG pipeline: golden dataset, metrics, LLM judge, runner.

    uv run python -m evals.run_rag --suite all --modes dense,hybrid,hybrid_rerank --k 8

Everything here is read-only with respect to the product tables: signals are extracted with
save=False, so evals never write to player_signals. Results go to evals/results/*.json.
"""

from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
GOLDEN_DIR = EVALS_DIR / "golden"
PROMPTS_DIR = EVALS_DIR / "prompts"
RESULTS_DIR = EVALS_DIR / "results"
