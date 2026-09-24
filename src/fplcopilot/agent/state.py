"""Состояние LangGraph-агента (шаг 7): один TypedDict на весь граф.

Принцип: в состоянии лежат только JSON-совместимые значения (dict/list/str/int/float/bool/None,
даты — ISO-строки), чтобы SQLite-checkpointer (`AGENT_CHECKPOINT_PATH`) и `--json` в CLI
работали без сюрпризов. Pydantic-модели инструментов складываются через `model_dump(mode="json")`.

Два списка — `tool_log` и `llm_calls` — имеют reducer `operator.add`: узлы возвращают только
свои новые записи, LangGraph склеивает. Всё остальное узлы перезаписывают целиком.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

Intent = Literal[
    "player_status",
    "compare_players",
    "transfer",
    "captain",
    "lineup",
    "plan",
    "what_if",
    "player_ranking",
    "strategy_question",
    "off_topic",
    "betting",
    # v4: оценка состава, календарь клубов, фишки менеджера, общий FPL-вопрос с частичным ответом
    "squad_review",
    "fixtures",
    "chips",
    "general_fpl",
    "gw_review",  # v4: разбор завершённого тура по фактам (очки, капитан, прогноз vs факт)
]
INTENTS: tuple[str, ...] = (
    "player_status",
    "compare_players",
    "transfer",
    "captain",
    "lineup",
    "plan",
    "what_if",
    "player_ranking",
    "strategy_question",
    "off_topic",
    "betting",
    "squad_review",
    "fixtures",
    "chips",
    "general_fpl",
    "gw_review",
)
# Интенты, для которых без состава менеджера полноценный ответ невозможен.
SQUAD_INTENTS: frozenset[str] = frozenset(
    {"transfer", "captain", "lineup", "plan", "what_if", "squad_review", "chips", "gw_review"}
)
# Интенты, где нужны новостные сигналы игроков с проблемами в составе (не только упомянутых).
# Разбор состава («посмотри мой состав, скажи слабые места»): роутер v3 относит его к transfer,
# роутер v4 — к squad_review (проблемы состава + лучший XI + маршруты, которые их чинят).
SIGNAL_SQUAD_INTENTS: frozenset[str] = frozenset(
    {"transfer", "plan", "lineup", "captain", "squad_review"}
)
# Интенты, где после compute подтягиваются новости по игрокам, которых рекомендует / ранжирует /
# сравнивает ответ (узел candidate_news), и по их клубам.
NEWS_INTENTS: frozenset[str] = frozenset(
    {"transfer", "plan", "what_if", "player_ranking", "compare_players", "player_status",
     "squad_review"}
)

Decision = Literal["confirm", "reject"]


class AgentState(TypedDict, total=False):
    # --- вход ---
    query: str
    # предыдущие реплики чата (старые первыми): {role: user|assistant, content, meta?}; meta —
    # agent.chat.turn_meta предыдущего хода (интент, игроки, тур, фишки, ограничения)
    history: list[dict[str, Any]]
    thread_id: str
    manager_id: int | None
    strategy: str
    # состав не из API (скриншот): agent.tools.SquadOverride.model_dump(mode="json"); None — picks
    squad_override: dict[str, Any] | None

    # --- контекст тура (load_context) ---
    gw: int | None
    current_gw: int | None
    deadline: str | None  # ISO UTC
    as_of: str | None  # ISO UTC: момент «знания» для сигналов и прогнозов
    squad_summary: dict[str, Any] | None  # GameweekContext без issues; None — состава нет
    squad_note: str | None  # почему состава нет / откуда он взят
    issues: list[dict[str, Any]]  # SquadIssue.model_dump()

    # --- маршрутизация (router) ---
    intent: str | None
    router_raw: dict[str, Any] | None  # ответ LLM как есть (для отладки и трейсинга)
    players: list[dict[str, Any]]  # разрешённые упоминания: {id, name, team, position, ...}
    unresolved: list[str]  # упоминания, которых нет в bootstrap
    # {mentions: [{mention, note, candidates: [...], roles: [sell|buy|keep]}]} -> узел clarify
    clarification: dict[str, Any] | None
    horizon: int | None
    language: str | None  # ISO 639-1 язык вопроса (роутер v2 / детектор) -> язык ответа
    scenario: dict[str, Any]  # {sell: [id], buy: [id], keep: [id], allow_hit, use_wildcard}
    # player_ranking: {metric, position, max_price, min_price, limit, gw_from, gw_to, last_n_gws}
    # (роутер v2 + regex-страховка; окно туров None -> весь сезон)
    ranking: dict[str, Any] | None
    needs_squad: bool
    # v4: вопрос, переписанный роутером в самостоятельный (с учётом истории), целевой тур,
    # фишки по турам [{gw, chip}], клубы как написаны (fixtures)
    standalone_query: str | None
    target_gw: int | None
    chips_asked: list[dict[str, Any]]
    team_mentions: list[str]
    not_in_squad: list[str]  # v4: keep / sell про игроков не из состава (условие не применено)

    # --- сигналы (ensure_signals / grade_signals / rewrite_retry) ---
    relevant_players: list[int]
    signals: dict[str, dict[str, Any]]  # str(player_id) -> PlayerRisk.model_dump()
    evidence: list[dict[str, Any]]  # плоский список цитат по всем игрокам
    fresh_signals: bool  # в этом прогоне были новые извлечения -> прогнозы пересчитать
    retries: int
    grade: dict[str, Any] | None  # {insufficient: [ids], reasons: {...}, sufficient: bool}

    # --- вычисления (compute) ---
    predictions: dict[str, dict[str, Any]]  # str(player_id) -> PlayerPrediction.model_dump()
    lineup: dict[str, Any] | None
    routes: list[dict[str, Any]]
    plan: dict[str, Any] | None
    scenario_result: dict[str, Any] | None
    facts: dict[str, Any]  # компактный JSON фактов для explain (числа уже округлены)
    caveats: list[str]

    # --- новости кандидатов (candidate_news, после compute) ---
    # [{id, name, role, primary, forced, club_news}] — кого рекомендует / ранжирует / сравнивает compute
    news_targets: list[dict[str, Any]]
    candidate_signals: dict[str, dict[str, Any]]  # str(player_id) -> PlayerRisk.model_dump()
    team_news: dict[str, dict[str, Any]]  # str(team_id) -> TeamNewsOut.model_dump()
    news_llm_used: int  # извлечений (игроки + клубы) в candidate_news за запрос (бюджет)
    news_exclude: list[int]  # покупки, исключённые после новости injured/... -> exclude в compute
    news_reoptimize: bool  # candidate_news -> compute: один пересчёт без исключённых
    news_reoptimized: bool  # пересчёт уже был (второго не будет)

    # --- human-in-the-loop (check_action / confirm_action) ---
    pending_action: dict[str, Any] | None  # {kind: hit|wildcard, detail, cost}
    user_decision: str | None  # confirm | reject
    action_history: list[dict[str, Any]]
    # --- human-in-the-loop (clarify / resolve_clarification): неоднозначное имя ---
    # {mention, note, roles, candidates: [{player_id, name, full_name, team, position, price,
    #  ownership, status, in_squad}]}; граф прерывается перед resolve_clarification
    pending_clarification: dict[str, Any] | None
    clarification_choice: dict[str, Any] | None  # {player_id, decision: choose|cancel} от resume
    clarification_history: list[dict[str, Any]]

    # --- ответ ---
    answer: str | None
    validation: dict[str, Any] | None
    explain_attempts: int

    # --- учёт ---
    llm_calls: Annotated[list[dict[str, Any]], operator.add]
    tool_log: Annotated[list[dict[str, Any]], operator.add]
    cost_estimate: float


def initial_state(
    query: str,
    *,
    thread_id: str,
    manager_id: int | None,
    strategy: str,
    squad_override: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> AgentState:
    return AgentState(
        query=query,
        history=list(history or []),
        standalone_query=None,
        target_gw=None,
        chips_asked=[],
        team_mentions=[],
        not_in_squad=[],
        thread_id=thread_id,
        manager_id=manager_id,
        strategy=strategy,
        squad_override=squad_override,
        issues=[],
        players=[],
        unresolved=[],
        clarification=None,
        language=None,
        scenario={"sell": [], "buy": [], "keep": [], "allow_hit": None, "use_wildcard": None},
        ranking=None,
        needs_squad=False,
        relevant_players=[],
        signals={},
        evidence=[],
        fresh_signals=False,
        retries=0,
        grade=None,
        predictions={},
        lineup=None,
        routes=[],
        plan=None,
        scenario_result=None,
        facts={},
        caveats=[],
        news_targets=[],
        candidate_signals={},
        team_news={},
        news_llm_used=0,
        news_exclude=[],
        news_reoptimize=False,
        news_reoptimized=False,
        pending_action=None,
        user_decision=None,
        action_history=[],
        pending_clarification=None,
        clarification_choice=None,
        clarification_history=[],
        answer=None,
        validation=None,
        explain_attempts=0,
        llm_calls=[],
        tool_log=[],
        cost_estimate=0.0,
    )
