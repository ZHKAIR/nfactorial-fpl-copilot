"""Подблок «Форма и контекст» в «Новостях об игроке» карточки (app/views/4_player.py).

Форма / роль / позиция / стандарты из того же сигнала (промпт v4, player_signals.form_notes) с
цитатами и ссылками. Показывается отдельно от доказательств доступности: это контекст, он не
влияет ни на вердикт доступности, ни на прогноз минут / xPts. У сигналов v1–v3 заметок нет —
подблок не рисуется.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from fplcopilot.rag.form_notes import form_note_rows, load_form_notes

CAPTION = (
    "Форма, роль и стандарты из тех же новостей — контекст, не доказательство доступности; "
    "в вердикт, прогноз минут и xPts не входит."
)


def render(risk: Any) -> None:
    notes = load_form_notes(risk.player_id, getattr(risk, "signal_as_of", None))
    if not notes:
        return
    st.markdown("**Форма и контекст**")
    st.caption(CAPTION)
    st.dataframe(
        form_note_rows(notes),
        hide_index=True,
        width="stretch",
        column_config={"Ссылка": st.column_config.LinkColumn("Ссылка")},
    )
