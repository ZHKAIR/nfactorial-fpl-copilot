"""Общий слой Streamlit-страниц: кэшированные ресурсы, сайдбар, источник состава, ошибки.

- один `LiveTools` на процесс (`st.cache_resource`), агент поверх него;
- выходы инструментов кэшируются `st.cache_data(ttl=600)` по ключу (менеджер, GW, стратегия,
  отпечаток состава со скриншота, параметры);
- состав со скриншота живёт в `st.session_state["squad_override"]` (`SquadOverride`) и передаётся
  каждому инструменту уровня состава через `squad_override=`;
- ошибки инфраструктуры (БД, FPL API, OpenAI, нет picks) показываются `st.error` с подсказкой.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
import streamlit as st
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from fplcopilot.agent.graph import Agent, live_agent
from fplcopilot.agent.tools import (
    BuildPlanInput,
    GameweekContext,
    GameweekContextInput,
    LineupOut,
    LiveTools,
    OptimizeTeamInput,
    PlanOut,
    PlayerPrediction,
    PlayerRisk,
    PlayerRiskInput,
    PredictPlayerInput,
    RecommendTransfersInput,
    RoutesOut,
    ScenarioInfeasible,
    SquadOverride,
    SquadUnavailable,
)
from fplcopilot.app import format as fmt
from fplcopilot.app import manager_state
from fplcopilot.config import settings
from fplcopilot.core.optimizer import InfeasibleError
from fplcopilot.db import session_scope

log = logging.getLogger(__name__)

SESSION_OVERRIDE = "squad_override"  # SquadOverride | None
SESSION_OVERRIDE_INFO = "squad_override_info"  # str: откуда состав (имя файла)
CACHE_TTL = 600


# ---------- ресурсы ----------


@st.cache_resource(show_spinner="Загружаю FPL API, прогнозы и оптимизатор…")
def get_tools() -> LiveTools:
    return LiveTools()


@st.cache_resource(show_spinner="Собираю LangGraph-агента…")
def get_agent() -> Agent:
    return live_agent(tools=get_tools())


def openai_ready() -> bool:
    return bool(settings.openai_api_key and settings.openai_api_key.strip())


def table_height(n_rows: int, *, row_px: int = 35, max_px: int = 620) -> int:
    """Высота st.dataframe, чтобы 15 игроков были видны без прокрутки."""
    return min(max_px, row_px * (n_rows + 1) + 3)


def current_override() -> SquadOverride | None:
    return st.session_state.get(SESSION_OVERRIDE)


def override_fp(override: SquadOverride | None) -> str | None:
    return None if override is None else override.fingerprint()


def squad_call(fn: Callable[..., Any], inp: Any, override: SquadOverride | None) -> Any:
    """Инструмент уровня состава: kwarg squad_override передаём только когда он есть
    (фейки в тестах его не знают)."""
    return fn(inp, squad_override=override) if override is not None else fn(inp)


# ---------- кэшированные выходы инструментов (pydantic-модели пиклятся) ----------


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_context(
    manager_id: int | None, strategy: str, fp: str | None, _override: SquadOverride | None
) -> GameweekContext:
    tools = get_tools()
    inp = GameweekContextInput(manager_id=manager_id, strategy=strategy)
    return squad_call(tools.get_gameweek_context, inp, _override)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_predictions(
    gw: int, horizon: int, player_ids: tuple[int, ...]
) -> dict[int, PlayerPrediction]:
    tools = get_tools()
    out: dict[int, PlayerPrediction] = {}
    for pid in player_ids:
        try:
            out[pid] = tools.predict_player(
                PredictPlayerInput(player_id=pid, gw=gw, horizon=horizon)
            )
        except Exception:  # один игрок без прогноза не должен ломать таблицу
            log.warning("predict_player failed for %s", pid, exc_info=True)
    return out


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_why_facts(
    gw: int, player_ids: tuple[int, ...], strategy: str, why_ver: int = 2
) -> dict[int, list[dict[str, Any]]]:
    """Детерминированные факты «Почему» по id игрока. Сеть/БД не роняют страницу.

    why_ver — ключ кэша при смене сборщика фактов (раскладка очков / метка удачи).
    """
    from fplcopilot.core.why_facts import collect_facts_for_players, try_why_data

    if not player_ids:
        return {}
    try:
        data = try_why_data(get_tools(), gw, player_ids, strategy)
        raw = collect_facts_for_players(player_ids, data)
    except Exception:
        log.warning("cached_why_facts failed", exc_info=True)
        return {}
    return {
        pid: [
            {
                "kind": f.kind,
                "clause": f.clause,
                "role": f.role,
                "priority": f.priority,
                "player_id": f.player_id,
                "points": f.points,
            }
            for f in facts
        ]
        for pid, facts in raw.items()
    }


@st.cache_data(ttl=60, show_spinner=False)
def cached_signals(player_ids: tuple[int, ...], as_of_minute: str) -> dict[int, PlayerRisk]:
    """Сохранённые сигналы (cached_only — без LLM); as_of_minute — ключ обновления раз в минуту."""
    tools = get_tools()
    as_of = datetime.fromisoformat(as_of_minute)
    out: dict[int, PlayerRisk] = {}
    for pid in player_ids:
        try:
            out[pid] = tools.analyze_player_risk(
                PlayerRiskInput(player_id=pid, as_of=as_of, cached_only=True)
            )
        except Exception:
            log.warning("analyze_player_risk(cached_only) failed for %s", pid, exc_info=True)
    return out


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_lineup(
    manager_id: int, gw: int, strategy: str, fp: str | None, _override: SquadOverride | None
) -> LineupOut:
    tools = get_tools()
    inp = OptimizeTeamInput(manager_id=manager_id, gw=gw, strategy=strategy)
    return squad_call(tools.optimize_team, inp, _override)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_routes(
    manager_id: int,
    gw: int,
    strategy: str,
    horizon: int,
    allow_hit: bool,
    fp: str | None,
    _override: SquadOverride | None,
) -> RoutesOut:
    tools = get_tools()
    inp = RecommendTransfersInput(
        manager_id=manager_id, gw=gw, horizon=horizon, strategy=strategy, allow_hit=allow_hit
    )
    return squad_call(tools.recommend_transfers, inp, _override)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_plan(
    manager_id: int,
    gw: int,
    strategy: str,
    horizon: int,
    allow_hits: bool,
    fp: str | None,
    _override: SquadOverride | None,
) -> PlanOut:
    tools = get_tools()
    inp = BuildPlanInput(
        manager_id=manager_id,
        gw=gw,
        horizon=horizon,
        strategy=strategy,
        allow_hits=allow_hits,
        save=False,
    )
    return squad_call(tools.build_gameweek_plan, inp, _override)


def save_plan_snapshot(
    manager_id: int, gw: int, strategy: str, horizon: int, allow_hits: bool
) -> PlanOut:
    """Пересчёт с save=True (снимок в plan_snapshots) — без кэша."""
    tools = get_tools()
    inp = BuildPlanInput(
        manager_id=manager_id,
        gw=gw,
        horizon=horizon,
        strategy=strategy,
        allow_hits=allow_hits,
        save=True,
    )
    return squad_call(tools.build_gameweek_plan, inp, current_override())


@st.cache_data(ttl=60, show_spinner=False)
def freshness(gw: int) -> dict[str, Any]:
    """Свежесть данных (3 SQL-запроса): последняя статья, последний сигнал, есть ли сохранённый
    прогноз на следующий тур."""
    out: dict[str, Any] = {"news_at": None, "signal_at": None, "xpts_rows": None, "error": None}
    try:
        with session_scope() as s:
            out["news_at"] = s.execute(text("SELECT max(fetched_at) FROM news_articles")).scalar()
            out["signal_at"] = s.execute(text("SELECT max(as_of) FROM player_signals")).scalar()
            out["xpts_rows"] = s.execute(
                text("SELECT count(*) FROM xpts_predictions WHERE gw = :gw"), {"gw": gw}
            ).scalar()
    except SQLAlchemyError as exc:
        out["error"] = f"{type(exc).__name__}"
    return out


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_entry(manager_id: int) -> dict[str, Any] | None:
    """Карточка менеджера из FPL API `entry/{id}` (команда, имя, очки, ранг) — подтверждение
    применённого ID в сайдбаре. None — 404 / сеть / клиент без entry(); отрицательный результат
    тоже кэшируется на CACHE_TTL, чтобы не дёргать API на каждом rerun."""
    tools = get_tools()
    try:
        e = tools.client.entry(int(manager_id))
    except Exception:  # любая ошибка (404, сеть) = «не найден», детали в логе
        log.warning("entry(%s) failed", manager_id, exc_info=True)
        return None
    return {
        "team": e.name,
        "points": e.summary_overall_points,
        "rank": e.summary_overall_rank,
        "first_name": e.player_first_name,
    }


CHAT_PAGE = "views/5_chat.py"  # путь относительно Home.py (st.switch_page)


def ask_assistant(prompt: str) -> None:
    """Вопрос в чат с любой страницы: чат подхватывает `queued_prompt` при открытии."""
    st.session_state["queued_prompt"] = prompt
    st.switch_page(CHAT_PAGE)


def go(page: str, **query: str) -> None:
    """Переход на страницу приложения (путь относительно Home.py) с query-параметрами;
    manager подставит сайдбар."""
    st.switch_page(page, query_params=query or None)


def page_head(eyebrow: str, title: str, lead: str | None = None, aside: str | None = None) -> None:
    """Шапка страницы по референсу: подпись-«бровь», H1 (st.title — его читают тесты), одна
    фраза под ним и карточка справа (дедлайн)."""
    from fplcopilot.app import ui_kit

    if aside:
        left, right = st.columns([3, 1], vertical_alignment="bottom")
    else:
        left, right = st.container(), None
    with left:
        st.markdown(ui_kit.eyebrow(eyebrow), unsafe_allow_html=True)
        st.title(title)
        if lead:
            st.markdown(ui_kit.lead(lead), unsafe_allow_html=True)
    if right is not None and aside:
        right.markdown(aside, unsafe_allow_html=True)


def deadline_aside(ui: UI) -> str | None:
    from fplcopilot.app import ui_kit

    if ui.ctx is None:
        return None
    return ui_kit.deadline_card(
        ui.ctx.gw,
        fmt.deadline_countdown(ui.ctx.deadline, ui.now, short=True),
        fmt.deadline_text(ui.ctx.deadline),
    )


def refresh_all() -> None:
    """Сбросить кэши: прогнозы/входы оптимизатора, bootstrap FPL, кэш страниц."""
    tools = get_tools()
    tools.invalidate()
    try:
        tools.client.bootstrap(refresh=True)
    except Exception:  # сеть недоступна — остаёмся на дисковом кэше
        log.warning("bootstrap refresh failed", exc_info=True)
    st.cache_data.clear()


# ---------- ошибки ----------


def explain_error(exc: BaseException) -> tuple[str, str]:
    """(заголовок, подсказка) для st.error — без stack trace."""
    if isinstance(exc, SquadUnavailable):
        hint = (
            f"{exc}. Загрузите скриншот Pick Team на странице «Мой состав» — он будет использован "
            "на всех страницах."
        )
        return ("Состав менеджера недоступен", hint)
    if isinstance(exc, (ScenarioInfeasible, InfeasibleError)):
        return ("Оптимизатор не нашёл допустимого решения", f"{exc}. Ослабьте ограничения.")
    if isinstance(exc, SQLAlchemyError):
        hint = (
            "Запустите `docker compose up -d db` и `uv run python scripts/migrate.py`; проверьте "
            "DATABASE_URL в .env. Без Postgres недоступны xPts, оптимизатор, сигналы новостей, "
            "планы и база знаний (история игроков читается из БД); работают только просмотр "
            "состава из FPL API и отказы чата (ставки / не по теме)."
        )
        return ("База данных недоступна", hint)
    if isinstance(exc, httpx.HTTPStatusError):
        return (
            f"FPL API вернул {exc.response.status_code}",
            "Проверьте ID менеджера; в час дедлайна API нестабилен — ответы кэшируются на диске (.cache/fpl).",
        )
    if isinstance(exc, httpx.HTTPError):
        return ("FPL API недоступен", f"{type(exc).__name__}: проверьте сеть.")
    name = type(exc).__name__
    if "openai" in type(exc).__module__ or "OpenAI" in name or "Authentication" in name:
        hint = (
            "Проверьте OPENAI_API_KEY в .env (нужен для чата, извлечения сигналов и распознавания "
            "скриншота)."
        )
        return (f"OpenAI: {name}", hint)
    return (name, str(exc)[:400])


def show_error(exc: BaseException) -> None:
    title, hint = explain_error(exc)
    log.warning("UI error: %s", exc, exc_info=exc)
    st.error(f"**{title}**")
    st.caption(hint)


def guarded(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Вызов с показом ошибки вместо traceback; None при ошибке."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 — любая ошибка инфраструктуры показывается пользователю
        show_error(exc)
        return None


# ---------- сайдбар ----------


@dataclass
class UI:
    manager_id: int | None
    strategy: str
    override: SquadOverride | None
    ctx: GameweekContext | None
    now: datetime

    @property
    def gw(self) -> int | None:
        return self.ctx.gw if self.ctx else None

    @property
    def has_squad(self) -> bool:
        return bool(self.ctx and self.ctx.squad)

    @property
    def fp(self) -> str | None:
        return override_fp(self.override)

    @property
    def squad_manager_id(self) -> int:
        """ID для инструментов уровня состава: 0, если состав со скриншота без менеджера."""
        return int(self.manager_id or 0)

    @property
    def as_of_minute(self) -> str:
        return self.now.replace(second=0, microsecond=0).isoformat()


def set_override(override: SquadOverride | None, info: str | None = None) -> None:
    if override is None:
        st.session_state.pop(SESSION_OVERRIDE, None)
        st.session_state.pop(SESSION_OVERRIDE_INFO, None)
    else:
        st.session_state[SESSION_OVERRIDE] = override
        st.session_state[SESSION_OVERRIDE_INFO] = info or override.source


def _persist(key: str, default: Any) -> None:
    """Значение виджета должно жить при переходе между страницами: id виджета включает hash
    скрипта страницы, поэтому «трогаем» ключ в session_state до создания виджета."""
    st.session_state[key] = st.session_state.get(key, default)


MANAGER_KEY = manager_state.MANAGER_KEY  # int в session_state: применённый ID (контракт)
MANAGER_TEXT_KEY = manager_state.MANAGER_TEXT_KEY  # str: содержимое поля ввода


def sidebar() -> UI:
    """Общий сайдбар — панель управления, объяснения только в help (?):
    ID менеджера (`manager_state`: адрес `?manager=` → сессия → cookie устройства → пусто; поле
    в форме, Enter / «Сохранить» — проверить и запомнить, «Забыть») + подтверждение
    «<команда> · очки · ранг» из FPL API entry/{id}; стратегия + «Применено: …»; GW и дедлайн;
    короткий «Состав: …»; две строки свежести (новости, разбор новостей об игроках);
    «Обновить данные»; LangSmith. Логика инструментов и override не меняется."""
    _persist("strategy", "balanced")
    manager_id = manager_state.sidebar_input(
        entry=cached_entry, fetch=lambda mid: get_tools().client.entry(mid)
    )
    st.sidebar.markdown(
        fmt.label_with_tip("Стратегия", fmt.STRATEGY_HELP),
        unsafe_allow_html=True,
    )
    strategy = st.sidebar.selectbox(
        "Стратегия",
        fmt.STRATEGIES,
        format_func=fmt.strategy_label,
        key="strategy",
        label_visibility="collapsed",
    )
    st.sidebar.caption(f"Применено: {fmt.strategy_label(strategy).lower()}")
    override = current_override()
    now = datetime.now(UTC)
    ctx: GameweekContext | None = None
    try:
        ctx = cached_context(manager_id, strategy, override_fp(override), override)
    except Exception as exc:  # noqa: BLE001
        with st.sidebar:
            show_error(exc)

    if ctx is not None:
        st.sidebar.markdown(
            f"**GW{ctx.gw}** · дедлайн {fmt.deadline_text(ctx.deadline)} · через "
            f"**{fmt.deadline_countdown(ctx.deadline, now, short=True)}**"
        )
        source_text, kind, source_help = fmt.squad_source_text(
            ctx, override.gw if override else None, has_manager=manager_id is not None
        )
        if kind == "warning":
            st.sidebar.warning(source_text)
        elif source_help:
            st.sidebar.markdown(
                fmt.label_with_tip(source_text, source_help),
                unsafe_allow_html=True,
            )
        else:
            st.sidebar.markdown(source_text)
        if override is not None:
            source = st.session_state.get(SESSION_OVERRIDE_INFO, "скриншот")
            bcol, tcol = st.sidebar.columns([4, 1])
            reset = bcol.button("Сбросить", key="reset_override")
            tcol.markdown(
                fmt.TIP_WIDGET_CSS
                + fmt.tip(f"Вернуться к составу из FPL. Сейчас: {source}"),
                unsafe_allow_html=True,
            )
            if reset:
                set_override(None)
                st.rerun()
        fr = freshness(ctx.gw)
        if fr.get("error"):
            st.sidebar.warning(f"База данных недоступна ({fr['error']})")
        else:
            lines = fmt.freshness_lines(fr, now)
            st.sidebar.caption(lines["news"])
            st.sidebar.markdown(
                fmt.label_with_tip(lines["signals"], fmt.SIGNALS_HELP),
                unsafe_allow_html=True,
            )
            if lines["xpts_missing"]:
                st.sidebar.error(f"Прогноз на GW{ctx.gw} не рассчитан")
    if not openai_ready():
        st.sidebar.warning(
            "OPENAI_API_KEY не задан: чат, разбор новостей об игроках и скриншот недоступны"
        )
    bcol, tcol = st.sidebar.columns([4, 1])
    refresh = bcol.button("Обновить данные", key="refresh_all")
    tcol.markdown(
        fmt.TIP_WIDGET_CSS + fmt.tip("Перечитать данные FPL и новости"),
        unsafe_allow_html=True,
    )
    if refresh:
        refresh_all()
        st.rerun()
    st.sidebar.caption(langsmith_status_text())
    return UI(manager_id=manager_id, strategy=strategy, override=override, ctx=ctx, now=now)


def require_squad(ui: UI) -> bool:
    """True, если состав есть; иначе объясняет, почему нет, и как загрузить скриншот."""
    if ui.ctx is None:
        return False
    if ui.has_squad:
        return True
    st.warning(f"Состав недоступен: {ui.ctx.squad_note}")
    st.info(
        "Публичный API отдаёт picks только за завершённые туры. Загрузите скриншот экрана "
        "Pick Team на странице «Мой состав» — распознанный состав будет использован здесь."
    )
    return False



def langsmith_status_text() -> str:
    """Честный статус трейсинга: ключ есть, но LangSmith отклоняет трейсы -> «выключен (…)»."""
    from fplcopilot.agent.tracing import tracing_status

    enabled, reason = tracing_status()
    if enabled:
        return "LangSmith: включён"
    if reason and "usage limit" in reason:
        return "LangSmith: выключен — месячная квота трейсов исчерпана"
    if reason:
        return "LangSmith: выключен — LangSmith отклоняет трейсы (429)"
    return "LangSmith: выключен"
