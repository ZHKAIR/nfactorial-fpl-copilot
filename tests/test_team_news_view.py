"""Блок «Новости клуба» карточки игрока (app/team_news_view.py): таблицы без запуска Streamlit."""

from __future__ import annotations

from fplcopilot.agent.tools import ClubAbsenceOut, TeamNewsOut
from fplcopilot.app.team_news_view import absence_rows


def test_team_news_absence_rows_are_russian():
    out = TeamNewsOut(
        team_id=1,
        team="ARS",
        team_name="Arsenal",
        absences=[
            ClubAbsenceOut(
                player_id=5,
                player="Havertz",
                position="FWD",
                status="i",
                status_label="injured",
                fpl_news="Knee injury",
                news_availability="injured",
            ),
            ClubAbsenceOut(
                player_id=6,
                player="White",
                position="DEF",
                status="d",
                status_label="doubtful",
                chance_next=75,
            ),
        ],
    )
    rows = absence_rows(out)
    assert rows[0]["Статус FPL"] == "травма" and rows[0]["Сигнал новостей"] == "травма"
    assert rows[1]["Статус FPL"] == "под вопросом (75%)" and rows[1]["Сигнал новостей"] == "—"
