"""form_notes сохранённого сигнала (промпт v4, миграция 010) для карточки игрока и объяснителя.

Форма / роль / позиция / стандарты — контекст, НЕ доказательство доступности: в FACTS они лежат
отдельным ключом рядом с news_signal, в EVIDENCE — с пометкой "scope": "form" (как клубные
новости — "scope": "club"). Модель минут / xPts их не читает (core/signals.py).

Сигнал доходит до агента и UI через LiveTools.analyze_player_risk (PlayerRisk без form_notes),
поэтому заметки читаются отдельно — по (player_id, as_of) той же строки player_signals.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from fplcopilot.db import session_scope

log = logging.getLogger(__name__)

FORM_FACTS_MAX = 3  # заметок на игрока в FACTS
FORM_EVIDENCE_MAX = 2  # цитат на игрока в EVIDENCE
FORM_ABOUT = "form / role / set-piece context from news — not evidence of availability or fitness"
KIND_LABELS_RU = {"form": "форма", "role": "роль", "position": "позиция", "set_pieces": "стандарты"}

_FORM_NOTES = text(
    """
    SELECT form_notes FROM player_signals
    WHERE player_id = :pid AND as_of = :as_of
    ORDER BY id DESC
    LIMIT 1
    """
)


def _as_dt(value: datetime | str | None) -> datetime | None:
    if value is None or value == "":
        return None
    dt = datetime.fromisoformat(value) if isinstance(value, str) else value
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def load_form_notes(player_id: int, signal_as_of: datetime | str | None) -> list[dict[str, Any]]:
    """form_notes строки сигнала (player_id, as_of); [] — нет сигнала, v1–v3, БД недоступна."""
    as_of = _as_dt(signal_as_of)
    if as_of is None:
        return []
    try:
        with session_scope() as s:
            row = s.execute(_FORM_NOTES, {"pid": int(player_id), "as_of": as_of}).first()
    except SQLAlchemyError as exc:  # без миграции 010 / без БД — карточка и агент без заметок
        log.debug("form_notes unavailable for %s: %s", player_id, exc)
        return []
    return list(row[0] or []) if row is not None else []


def note_date(note: dict[str, Any]) -> str:
    dt = _as_dt(note.get("published_at"))
    return f"{dt.astimezone(UTC):%d.%m}" if dt else ""


def form_notes_brief(notes: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Коротко для FACTS: вид, текст, источник, дата (цитаты — в EVIDENCE)."""
    return [
        {
            "kind": n.get("kind"),
            "note": n.get("text"),
            "source": n.get("source"),
            "date": note_date(n),
        }
        for n in list(notes)[:FORM_FACTS_MAX]
    ]


def form_evidence(notes: Iterable[dict[str, Any]], player: str) -> list[dict[str, Any]]:
    """Цитаты заметок для EVIDENCE с пометкой scope=form (формат как у цитат игрока)."""
    out: list[dict[str, Any]] = []
    for n in list(notes)[:FORM_EVIDENCE_MAX]:
        out.append(
            {
                "scope": "form",
                "about": FORM_ABOUT,
                "player": player,
                "kind": n.get("kind"),
                "source": n.get("source"),
                "url": n.get("url"),
                "published_at": n.get("published_at"),
                "date": note_date(n),
                "quote": " ".join(str(n.get("quote", "")).split()),
            }
        )
    return out


def form_note_rows(notes: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Строки таблицы «Форма и контекст» карточки игрока."""
    return [
        {
            "Тип": KIND_LABELS_RU.get(str(n.get("kind")), str(n.get("kind") or "")),
            "Заметка": n.get("text") or "",
            "Цитата": " ".join(str(n.get("quote", "")).split()),
            "Источник": n.get("source") or "",
            "Дата": note_date(n),
            "Ссылка": n.get("url") or "",
        }
        for n in notes
    ]
