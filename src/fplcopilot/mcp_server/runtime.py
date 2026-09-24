"""Служебный слой MCP-сервера: общий `LiveTools`, разрешение имён, контракт ошибок, компактные
payload'ы. Ничего не считает — только оборачивает agent/tools.py и agent/resolve.py.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from typing import Any, Literal

import httpx
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError

from fplcopilot.agent.resolve import PlayerResolver, Resolution
from fplcopilot.agent.tools import (
    HistoryUnavailable,
    LiveTools,
    ScenarioInfeasible,
    SquadUnavailable,
)
from fplcopilot.core.optimizer import ChipPlanError, InfeasibleError
from fplcopilot.core.strategy import PRESETS

log = logging.getLogger("fplcopilot.mcp")

StrategyName = Literal["conservative", "balanced", "aggressive"]
MAX_HORIZON = 8
MAX_LIST = 25  # элементов в любом списке payload'а
MAX_STR = 400  # символов в строке (цитаты evidence ~ 300)
ROUND = 2
_STRATEGIES = frozenset(PRESETS)


# ---------- ошибки ----------


class ToolInputError(ValueError):
    """Неверный вход инструмента (валидируем сами, чтобы ответ был структурированным)."""


class ChipPlanRejected(ChipPlanError):
    """ChipPlanError оптимизатора + фишки BB / TC, которые у менеджера есть по турам горизонта."""

    def __init__(self, message: str, chips_available_by_gw: dict[str, list[str]] | None) -> None:
        super().__init__(message)
        self.chips_available_by_gw = chips_available_by_gw


def error_payload(exc: BaseException) -> dict[str, Any]:
    """Структурированная ошибка для LLM-клиента: код + подсказка, без stack trace."""
    if isinstance(exc, ToolInputError):
        return {"error": "invalid_input", "hint": str(exc)}
    if isinstance(exc, ChipPlanError):
        payload: dict[str, Any] = {
            "error": "invalid_chip_plan",
            "hint": f"{exc}. Plannable chips: bboost (Bench Boost) and 3xc (Triple Captain), "
            "one per gameweek inside the plan horizon and only while the manager still has it; "
            "Free Hit is not modelled. Drop or move the chip, or call without `chips`.",
        }
        available = getattr(exc, "chips_available_by_gw", None)
        if available is not None:
            payload["chips_available_by_gw"] = available
        return payload
    if isinstance(exc, SquadUnavailable):
        return {
            "error": "squad_unavailable",
            "hint": f"{exc}. Squad-level tools need a manager with public picks; "
            "player-level tools (predict_player, analyze_player_risk, compare_players) still work.",
        }
    if isinstance(exc, (ScenarioInfeasible, InfeasibleError)):
        return {
            "error": "infeasible",
            "hint": f"{exc}. Relax the constraints (allow a hit, drop a forced buy/sell, or a keep).",
        }
    if isinstance(exc, HistoryUnavailable):
        return {
            "error": "history_unavailable",
            "hint": f"{exc}. Drop the gameweek window to rank by season totals instead — the "
            "window numbers are not approximated.",
        }
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return {
            "error": "fpl_api_error",
            "status": code,
            "hint": (
                "manager not found or no public picks yet"
                if code == 404
                else f"FPL API returned HTTP {code}; retry in a minute (deadline-hour instability)"
            ),
        }
    if isinstance(exc, httpx.HTTPError):
        return {"error": "fpl_api_unreachable", "hint": f"{type(exc).__name__}: {exc}"[:200]}
    if isinstance(exc, SQLAlchemyError):
        return {
            "error": "db_unavailable",
            "hint": "Postgres is not reachable: `docker compose up -d db` and check DATABASE_URL; "
            "predictions need player history tables.",
        }
    if isinstance(exc, ValueError):
        return {"error": "invalid_input", "hint": str(exc)[:300]}
    return {
        "error": "internal",
        "type": type(exc).__name__,
        "hint": str(exc)[:300] or "see server stderr log",
    }


# ---------- компактные payload'ы ----------


def compact(value: Any, *, max_list: int = MAX_LIST, max_str: int = MAX_STR) -> Any:
    """Рекурсивно: float -> 2 знака, списки -> первые `max_list` элементов (+ `<key>_omitted`),
    строки -> `max_str` символов. Числа не пересчитываются — только округляются для показа."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return round(value, ROUND)
    if isinstance(value, str):
        return value if len(value) <= max_str else value[: max_str - 1] + "…"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            if isinstance(v, (list, tuple)) and len(v) > max_list:
                out[key] = [compact(x, max_list=max_list, max_str=max_str) for x in v[:max_list]]
                out[f"{key}_omitted"] = len(v) - max_list
            else:
                out[key] = compact(v, max_list=max_list, max_str=max_str)
        return out
    if isinstance(value, (list, tuple)):
        return [compact(x, max_list=max_list, max_str=max_str) for x in value[:max_list]]
    return value


def dump(model: BaseModel | None) -> dict[str, Any] | None:
    """pydantic -> JSON-совместимый dict (alias `in`, даты ISO) -> compact."""
    if model is None:
        return None
    return compact(model.model_dump(mode="json", by_alias=True))


# ---------- валидация входов ----------


def check_strategy(strategy: str) -> str:
    if strategy not in _STRATEGIES:
        raise ToolInputError(f"unknown strategy {strategy!r}; use one of {sorted(_STRATEGIES)}")
    return strategy


def check_horizon(horizon: int, *, lo: int = 1, hi: int = MAX_HORIZON) -> int:
    if not isinstance(horizon, int) or isinstance(horizon, bool) or not lo <= horizon <= hi:
        raise ToolInputError(f"horizon must be an integer in [{lo}, {hi}], got {horizon!r}")
    return horizon


def check_manager_id(manager_id: int) -> int:
    if not isinstance(manager_id, int) or isinstance(manager_id, bool) or manager_id <= 0:
        raise ToolInputError(f"manager_id must be a positive integer, got {manager_id!r}")
    return manager_id


# ---------- разрешение имён ----------


def describe_resolution(res: Resolution, field: str) -> dict[str, Any]:
    """Ошибка разрешения имени в форме контракта MCP: `ambiguous` (с кандидатами) или
    `unknown_player`. Никогда не угадываем."""
    if res.status == "ambiguous":
        return {
            "error": "ambiguous",
            "field": field,
            "mention": res.mention,
            "candidates": [
                {
                    k: c.get(k)
                    for k in ("id", "name", "full_name", "team", "position", "price", "ownership")
                    if k in c
                }
                | ({"in_squad": True} if c.get("in_squad") else {})
                for c in res.candidates[:8]
            ],
            "hint": (res.note or "several players match")
            + "; ask the user which one and retry with the full name or the numeric id",
        }
    return {
        "error": "unknown_player",
        "field": field,
        "mention": res.mention,
        "hint": (res.note or f"'{res.mention}' not found in FPL bootstrap")
        + "; check the spelling (web name like 'Haaland' or full name like 'João Pedro')",
    }


def resolve_players(
    resolver: PlayerResolver,
    mentions: Sequence[str | int],
    *,
    field: str,
) -> tuple[list[int], list[str], dict[str, Any] | None]:
    """(ids, notes, error). Числовая строка / int — это id из bootstrap; иначе — детерминированный
    резолвер агента (web name, полное имя, контекст клуба, состав менеджера, доминирование по
    владению). Первая неоднозначность/неизвестное имя -> ошибка (всё или ничего)."""
    ids: list[int] = []
    notes: list[str] = []
    for mention in mentions:
        text = str(mention).strip()
        if not text:
            continue
        if text.isdigit():
            pid = int(text)
            try:
                player = resolver.bs.player(pid)
            except KeyError:
                return (
                    [],
                    notes,
                    {
                        "error": "unknown_player",
                        "field": field,
                        "mention": text,
                        "hint": f"no player with id {pid} in FPL bootstrap",
                    },
                )
            if pid not in ids:
                ids.append(pid)
            notes.append(f"id {pid} = {player.web_name}")
            continue
        res = resolver.resolve(text, sure=resolver.sure_in_text(text))
        if res.status != "resolved" or res.player is None:
            return [], notes, describe_resolution(res, field)
        pid = int(res.player["id"])
        if pid not in ids:
            ids.append(pid)
        if res.note:
            notes.append(res.note)
    return ids, notes, None


# ---------- общий LiveTools ----------


class Runtime:
    """Один `LiveTools` на процесс (ленивый), сериализованный доступ (sync-инструменты MCP
    выполняются в worker-потоках), кэш резолверов по составу, закрытие в lifespan."""

    def __init__(self, factory: Callable[[], LiveTools] | None = None) -> None:
        self._factory = factory or LiveTools
        self._tools: LiveTools | None = None
        self._resolvers: dict[tuple[int, ...], PlayerResolver] = {}
        self.lock = threading.RLock()
        self.calls = 0

    @property
    def tools(self) -> LiveTools:
        if self._tools is None:
            started = time.perf_counter()
            self._tools = self._factory()
            log.info("LiveTools created in %.0f ms", (time.perf_counter() - started) * 1000)
        return self._tools

    def now(self) -> datetime:
        return self.tools.now()

    def gw(self) -> int:
        return self.tools.next_gw()

    def resolver(self, squad_ids: Iterable[int] = ()) -> PlayerResolver:
        key = tuple(sorted(set(squad_ids)))
        if key not in self._resolvers:
            self._resolvers[key] = PlayerResolver(self.tools.bootstrap, squad_ids=key)
        return self._resolvers[key]

    def squad_ids(self, manager_id: int, gw: int, strategy: str = "balanced") -> list[int]:
        """Состав менеджера для резолвера («Palmer» -> тот, кем владеешь); без состава — []."""
        try:
            return list(self.tools.inputs(manager_id, gw, 3, strategy).squad)
        except (SquadUnavailable, httpx.HTTPError, SQLAlchemyError) as exc:
            log.info("squad for resolver unavailable (%s): %s", manager_id, exc)
            return []

    def close(self) -> None:
        if self._tools is not None:
            try:
                self._tools.close()
            finally:
                self._tools = None
                self._resolvers.clear()

    def reset(self) -> None:
        """Для тестов: подменить фабрику/экземпляр без перезапуска процесса."""
        self.close()

    # ---- вызов с таймингом и контрактом ошибок ----

    def call(self, name: str, args: dict[str, Any], fn: Callable[[], Any]) -> Any:
        """Выполнить `fn` под общим замком; тайминг — в stderr-лог; исключение -> error payload."""
        started = time.perf_counter()
        shown = {k: v for k, v in args.items() if v not in (None, [], "")}
        with self.lock:
            self.calls += 1
            try:
                out = fn()
            except Exception as exc:  # контракт: структурированная ошибка, не stack trace
                ms = round((time.perf_counter() - started) * 1000)
                expected = isinstance(
                    exc, (ToolInputError, SquadUnavailable, ScenarioInfeasible, ChipPlanError)
                )
                log.log(
                    logging.INFO if expected else logging.WARNING,
                    "tool %s %s FAILED in %d ms: %s: %s",
                    name,
                    shown,
                    ms,
                    type(exc).__name__,
                    exc,
                    exc_info=not expected,
                )
                payload = error_payload(exc)
                payload.update({"tool": name, "latency_ms": ms})
                return payload
        ms = round((time.perf_counter() - started) * 1000)
        log.info("tool %s %s ok in %d ms", name, shown, ms)
        return out


def iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat(timespec="seconds")
