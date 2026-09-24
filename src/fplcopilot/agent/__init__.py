"""LangGraph-агент FPL Copilot (шаг 7): router -> signals (bounded retry) -> compute -> HITL ->
explain -> validate. См. docs/agent.md.

Импорт пакета один раз переносит настройки LangSmith в окружение (agent/tracing.py; no-op без ключа).
"""

from fplcopilot.agent import tracing  # noqa: F401  (side effect: LANGSMITH_*/LANGCHAIN_* env)
from fplcopilot.agent.graph import Agent, Deps, build_graph, live_agent, live_deps
from fplcopilot.agent.state import AgentState

__all__ = ["Agent", "AgentState", "Deps", "build_graph", "live_agent", "live_deps"]
