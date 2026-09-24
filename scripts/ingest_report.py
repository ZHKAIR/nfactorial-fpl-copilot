"""Отчёт по новостному корпусу в Postgres.

Запуск:
    uv run python scripts/ingest_report.py
(эквивалент `uv run python -m fplcopilot.rag.ingest --report`)
"""

from fplcopilot.rag.report import build_report

if __name__ == "__main__":
    print(build_report())
