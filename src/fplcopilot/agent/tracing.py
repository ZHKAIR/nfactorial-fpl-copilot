"""Трейсинг LangGraph-агента в LangSmith: включается только непустым LANGSMITH_API_KEY.

LangGraph трейсится через callbacks langchain-core, которые смотрят на LANGSMITH_TRACING /
LANGCHAIN_TRACING_V2 и LANGSMITH_API_KEY / LANGCHAIN_API_KEY. Здесь мы один раз (при импорте
пакета `fplcopilot.agent`) переносим настройки из `Settings` в окружение — тем же способом, что
`rag/llm.configure_tracing`, плюс LANGCHAIN_* синонимы. Без ключа — трейсинг принудительно
выключен и ни одного предупреждения. Вызовы OpenAI уже идут через `wrap_openai` (rag/llm.py).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from typing import Any

from fplcopilot.config import settings
from fplcopilot.rag.llm import configure_tracing

log = logging.getLogger(__name__)


def setup_tracing() -> bool:
    """Выставляет LANGSMITH_* и LANGCHAIN_* из settings. True — трейсинг включён."""
    enabled = configure_tracing()
    if enabled:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key or "")
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
        log.debug("LangSmith tracing on for agent (project=%s)", settings.langsmith_project)
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    return enabled


def run_config_tags(*, strategy: str, model: str, intent: str | None = None) -> list[str]:
    tags = [f"strategy:{strategy}", f"model:{model}"]
    if intent:
        tags.append(f"intent:{intent}")
    return tags


def tag_root_run(tags: Iterable[str] = (), metadata: Mapping[str, Any] | None = None) -> None:
    """Best-effort: добавить теги/метаданные к корневому run текущего трейса (интент известен
    только после узла router, а теги invoke ставятся до запуска). Без трейсинга — no-op."""
    if os.environ.get("LANGCHAIN_TRACING_V2") != "true":
        return
    try:
        from langsmith import get_current_run_tree

        run = get_current_run_tree()
        if run is None:
            return
        while getattr(run, "parent_run", None) is not None:
            run = run.parent_run
        new_tags = [t for t in tags if t not in (run.tags or [])]
        if new_tags:
            run.add_tags(new_tags)
        if metadata:
            run.add_metadata(dict(metadata))
    except Exception:  # трейсинг никогда не должен ломать ответ
        log.debug("tag_root_run failed", exc_info=True)


# ---------- деградация при отказах LangSmith (429 / исчерпана месячная квота трейсов) ----------

RATE_LIMIT_ERRORS_BEFORE_DISABLE = 3  # временный 429: столько подряд — и трейсинг выключается
_QUOTA_MARKERS = ("usage limit", "monthly unique traces", "quota")
_RATE_MARKERS = ("429", "RateLimit", "Too Many Requests")
_state: dict[str, Any] = {"errors": 0, "disabled_reason": None}


def disable_tracing(reason: str) -> None:
    """Выключить трейсинг в этом процессе (один раз, одно предупреждение в лог): агент и чат
    продолжают работать, а клиент LangSmith больше не шлёт запросы, которые всё равно отклонят."""
    global TRACING_ENABLED
    if _state["disabled_reason"] is not None:
        return
    _state["disabled_reason"] = reason
    TRACING_ENABLED = False
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    try:
        import langsmith
        from langsmith import utils as ls_utils

        langsmith.configure(enabled=False)
        ls_utils.get_env_var.cache_clear()
    except Exception:  # старый langsmith без configure — хватит переменных окружения
        log.debug("langsmith.configure unavailable", exc_info=True)
    log.warning("LangSmith tracing disabled for this process: %s", reason)


class _LangSmithFailureGuard(logging.Filter):
    """Фильтр логгера `langsmith.client`: считает отказы загрузки трейсов. Исчерпанная квота —
    выключение сразу; 429 по частоте — после RATE_LIMIT_ERRORS_BEFORE_DISABLE. После выключения
    повторные предупреждения клиента подавляются (в логе остаётся одна строка о причине)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if _state["disabled_reason"] is not None:
            return False
        msg = record.getMessage()
        if any(m.lower() in msg.lower() for m in _QUOTA_MARKERS):
            disable_tracing("LangSmith rejected traces: monthly usage limit exceeded (429)")
            return False
        if any(m in msg for m in _RATE_MARKERS):
            _state["errors"] += 1
            if _state["errors"] >= RATE_LIMIT_ERRORS_BEFORE_DISABLE:
                disable_tracing(f"LangSmith rate limit: {_state['errors']} rejected uploads (429)")
                return False
        return True


def tracing_status() -> tuple[bool, str | None]:
    """(включён ли трейсинг сейчас, причина выключения после отказов LangSmith или None)."""
    return TRACING_ENABLED, _state["disabled_reason"]


logging.getLogger("langsmith.client").addFilter(_LangSmithFailureGuard())
TRACING_ENABLED = setup_tracing()
