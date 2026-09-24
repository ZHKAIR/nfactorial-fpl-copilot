"""Фасад над player_signals (rag/extract.py) для модели минут: последний сигнал на момент as_of.

core не импортирует rag (там openai/langsmith): читаем таблицу напрямую и отдаём лёгкий
SignalLite. Срок действия сигнала определяется содержанием, а не только возрастом (signal_active):

- «мягкий» сигнал (fit / doubtful / rotation, без тура возвращения) действует SIGNAL_MAX_AGE —
  новость недельной давности про «doubtful» уже ничего не говорит о ближайшем туре;
- сигнал с явным горизонтом (injured / suspended / unavailable с return_gw — в т.ч. из текста FPL
  «Suspended until 17 Oct») действует до этого тура независимо от возраста (не дольше
  HORIZON_SIGNAL_MAX_AGE); истечение по туру — в core/minutes.estimate_minutes (status_gw >=
  return_gw -> сигнал больше не применяется);
- более свежий источник побеждает: если после сигнала статус FPL сменился на `a` (снимок
  player_status_snapshots новее сигнала), сигнал не применяется. Более свежий сигнал того же игрока
  вытесняет старый сам (берётся последний).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from fplcopilot.db import session_scope

log = logging.getLogger(__name__)

SIGNAL_MAX_AGE = timedelta(days=7)
HORIZON_SIGNAL_MAX_AGE = timedelta(days=60)  # верхняя граница для сигналов с return_gw
ZERO_MINUTES_AVAILABILITY = frozenset({"injured", "suspended", "unavailable"})

_LATEST = text(
    """
    WITH sig AS (
        SELECT DISTINCT ON (player_id)
            player_id, as_of, availability, start_probability, expected_minutes, confidence,
            summary, rotation_risk, return_gw
        FROM player_signals
        WHERE as_of <= :as_of AND as_of >= :oldest
        ORDER BY player_id, as_of DESC, id DESC
    ), st AS (
        -- время изменения — snapshot_at: FPL не обновляет news_added при снятии новости (i -> a)
        SELECT DISTINCT ON (player_id)
            player_id, status, snapshot_at AS changed_at
        FROM player_status_snapshots
        WHERE player_id IN (SELECT player_id FROM sig)
          AND snapshot_at <= :as_of
        ORDER BY player_id, snapshot_at DESC, id DESC
    )
    SELECT sig.player_id, sig.as_of, sig.availability, sig.start_probability,
           sig.expected_minutes, sig.confidence, sig.summary, sig.rotation_risk, sig.return_gw,
           st.status, st.changed_at
    FROM sig LEFT JOIN st USING (player_id)
    """
)


@dataclass(frozen=True)
class SignalLite:
    player_id: int
    as_of: datetime
    availability: str
    start_probability: float
    expected_minutes: int
    confidence: float
    summary: str = ""
    rotation_risk: str = "unknown"
    return_gw: int | None = None  # тур ожидаемого возвращения (только для injured/suspended/...)

    @property
    def usable(self) -> bool:
        """unknown = доказательств не было; такой сигнал в смешивание не идёт."""
        return self.availability != "unknown" and self.confidence > 0

    @property
    def rules_out(self) -> bool:
        return self.availability in ZERO_MINUTES_AVAILABILITY

    @property
    def is_fit(self) -> bool:
        return self.availability == "fit"

    @property
    def has_horizon(self) -> bool:
        """Жёсткий сигнал с явным туром возвращения — действует до этого тура."""
        return self.rules_out and self.return_gw is not None

    def expired_for(self, status_gw: int | None) -> bool:
        """Тур возвращения наступил: ближайший тур (к которому относится статус FPL) >= return_gw."""
        return self.has_horizon and status_gw is not None and status_gw >= int(self.return_gw or 0)


def signal_active(
    sig: SignalLite,
    as_of: datetime,
    *,
    max_age: timedelta = SIGNAL_MAX_AGE,
    fpl_status: str | None = None,
    fpl_changed_at: datetime | None = None,
) -> bool:
    """Действует ли сигнал на as_of (чистая функция, см. докстринг модуля)."""
    fpl_newer = fpl_changed_at is not None and fpl_changed_at > sig.as_of
    if fpl_status == "a" and fpl_newer and (sig.rules_out or sig.availability == "doubtful"):
        return False  # FPL новее и говорит «доступен» — побеждает FPL
    age = as_of - sig.as_of
    if age <= max_age:
        return True
    return sig.has_horizon and age <= HORIZON_SIGNAL_MAX_AGE


def latest_signals(
    as_of: datetime, *, max_age: timedelta = SIGNAL_MAX_AGE
) -> dict[int, SignalLite]:
    """player_id -> последний действующий сигнал с as_of <= as_of (signal_active). Без БД — пусто."""
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=UTC)
    oldest = as_of - max(max_age, HORIZON_SIGNAL_MAX_AGE)
    try:
        with session_scope() as s:
            rows = s.execute(_LATEST, {"as_of": as_of, "oldest": oldest}).all()
    except SQLAlchemyError as exc:
        log.warning("player_signals недоступны (%s) — модель минут без новостных сигналов", exc)
        return {}
    out: dict[int, SignalLite] = {}
    for row in rows:
        pid, sig_as_of, availability, p_start, exp_min, conf, summary, rotation, ret_gw = row[:9]
        status, changed_at = row[9], row[10]
        sig = SignalLite(
            player_id=int(pid),
            as_of=sig_as_of.astimezone(UTC),
            availability=availability,
            start_probability=float(p_start),
            expected_minutes=int(exp_min),
            confidence=float(conf),
            summary=summary or "",
            rotation_risk=rotation or "unknown",
            return_gw=int(ret_gw) if ret_gw is not None else None,
        )
        changed = changed_at.astimezone(UTC) if changed_at is not None else None
        if signal_active(sig, as_of, max_age=max_age, fpl_status=status, fpl_changed_at=changed):
            out[int(pid)] = sig
    return out
