"""Streamlit UI FPL Copilot (docs/ui.md).

Принцип: в UI нет бизнес-логики — страницы вызывают `agent.tools.LiveTools` / LangGraph-агента и
форматируют результат помощниками из `app/format.py` (чистые функции, unit-тесты). Запуск:

    uv run streamlit run src/fplcopilot/app/Home.py
"""
