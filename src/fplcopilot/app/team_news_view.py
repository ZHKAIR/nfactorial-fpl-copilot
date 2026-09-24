"""Блок «Новости клуба» карточки игрока (app/views/4_player.py): недоступные игроки клуба по данным
FPL (тур возвращения по календарю) + мягкий дайджест (тренер, ротация, форма) с цитатами.

Та же политика, что у «Новостей об игроке»: при открытии — сохранённый дайджест любого возраста
(cached_only, без LLM), кнопка — принудительное извлечение. Клубный контекст — не доказательство
доступности самого игрока, поэтому он показан отдельно.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import streamlit as st

from fplcopilot.agent.llm import estimate_cost
from fplcopilot.agent.tools import TeamNewsInput, TeamNewsOut
from fplcopilot.app import format as fmt
from fplcopilot.app import llm_budget

KIND_RU = {"rotation": "Ротация", "manager_quote": "Слова тренера", "form_context": "Форма"}
SESSION_KEY = "fresh_team_news"


def absence_rows(out: TeamNewsOut) -> list[dict[str, Any]]:
    return [
        {
            "Игрок": a.player,
            "Поз.": a.position,
            "Статус FPL": fmt.STATUS_RU.get(a.status, a.status_label)
            + (f" ({a.chance_next}%)" if a.chance_next is not None else ""),
            "Вернётся": f"GW{a.return_gw}" if a.return_gw else "—",
            "Новость FPL": a.fpl_news or "—",
            "Сигнал новостей": (
                fmt.AVAILABILITY_RU.get(a.news_availability, a.news_availability)
                if a.news_availability
                else "—"
            ),
        }
        for a in out.absences
    ]


def item_rows(out: TeamNewsOut) -> list[dict[str, Any]]:
    return [
        {
            "Тип": KIND_RU.get(i.kind, i.kind),
            "Кратко": i.claim,
            "Цитата": i.quote,
            "Источник": i.source,
            "Дата": i.date,
            "Ссылка": i.url,
        }
        for i in out.items
    ]


def render(tools: Any, player: Any, as_of: datetime, *, openai_ready: bool, guarded: Any) -> None:
    fn = getattr(tools, "team_news", None)
    if fn is None:  # инструменты без дайджеста клуба (фейки UI-тестов)
        return
    team = tools.bootstrap.team(player.team)
    st.subheader("Новости клуба")
    st.caption(
        f"Контекст {team.name}: кто недоступен по данным FPL и что говорят о ротации и форме. "
        "Это не доказательство доступности самого игрока."
    )
    clicked = st.button(
        "Обновить новости клуба", key="refresh_team_news", disabled=not openai_ready
    )
    out: TeamNewsOut | None = None
    if clicked and llm_budget.allow_llm():
        with st.spinner("Ищу новости клуба и разбираю…"):
            out = guarded(fn, TeamNewsInput(team_id=team.id, as_of=as_of, force=True))
        if out is not None:
            st.session_state[SESSION_KEY] = out
    fresh = st.session_state.get(SESSION_KEY)
    if out is None and fresh is not None and fresh.team_id == team.id:
        out = fresh
    if out is None:
        out = guarded(fn, TeamNewsInput(team_id=team.id, as_of=as_of, cached_only=True))
    if out is None:
        return
    if out.origin == "extracted":
        cost = estimate_cost(out.model or "gpt-4o-mini", out.prompt_tokens, out.completion_tokens)
        st.success(f"Дайджест сделан сейчас: {out.latency_ms / 1000:.1f} с, ≈ ${cost:.4f}")
    elif out.origin == "cached":
        age = f"{out.age_h:.0f} ч назад" if out.age_h is not None else "ранее"
        note = fmt.template_ru(out.note)
        st.caption(f"Сохранённый дайджест, {age}" + (f" — {note}" if note else ""))
    else:
        st.caption("Сохранённого дайджеста нет — нажмите «Обновить новости клуба».")
    if out.absences:
        st.markdown("**Недоступны или под вопросом (данные FPL)**")
        st.dataframe(absence_rows(out), hide_index=True, width="stretch")
    else:
        st.caption(
            "По данным FPL в клубе нет травмированных, дисквалифицированных или под вопросом."
        )
    if out.summary and out.origin != "unavailable":
        st.write(fmt.template_ru(out.summary))
    if out.items:
        st.dataframe(
            item_rows(out),
            hide_index=True,
            width="stretch",
            column_config={"Ссылка": st.column_config.LinkColumn("Ссылка")},
        )
