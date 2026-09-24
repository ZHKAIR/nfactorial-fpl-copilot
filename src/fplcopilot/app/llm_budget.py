"""Суточный лимит запросов к LLM из интерфейса (прод): `APP_DAILY_LLM_LIMIT` запросов в сутки
(UTC) на процесс — чат (и продолжение после подтверждения), распознавание скриншота, обновление
новостей игрока и клуба, объяснение сравнения. Пусто — без лимита. Счётчик в памяти процесса:
после перезапуска контейнера обнуляется; жёсткий потолок трат — лимит в кабинете OpenAI.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime


@dataclass
class DailyBudget:
    limit: int | None
    today: Callable[[], date] = field(default=lambda: datetime.now(UTC).date())
    _day: date | None = None
    _used: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _roll(self) -> None:
        d = self.today()
        if d != self._day:
            self._day, self._used = d, 0

    def spend(self, n: int = 1) -> bool:
        """Списать n запросов; False — лимит на сегодня исчерпан (ничего не списано)."""
        if not self.limit or self.limit <= 0:
            return True
        with self._lock:
            self._roll()
            if self._used + n > self.limit:
                return False
            self._used += n
            return True

    @property
    def used(self) -> int:
        with self._lock:
            self._roll()
            return self._used

    @property
    def remaining(self) -> int | None:
        return None if not self.limit else max(0, self.limit - self.used)


def limit_message(limit: int | None) -> str:
    return (
        f"Суточный лимит запросов к ИИ исчерпан ({limit} в сутки на сервер). Он обновится в "
        "полночь по UTC; остальные страницы работают без ИИ."
    )


def get_budget() -> DailyBudget:
    """Один счётчик на процесс Streamlit."""
    import streamlit as st

    from fplcopilot.config import settings

    @st.cache_resource(show_spinner=False)
    def _budget(limit: int | None) -> DailyBudget:
        return DailyBudget(limit)

    return _budget(settings.app_daily_llm_limit)


def allow_llm() -> bool:
    """Списать один запрос; при исчерпании показать сообщение и вернуть False."""
    import streamlit as st

    b = get_budget()
    if b.spend():
        return True
    st.error(limit_message(b.limit))
    return False
