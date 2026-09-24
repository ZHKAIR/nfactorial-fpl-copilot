"""«Форма и контекст» (form_notes сигнала v4) в карточке и у объяснителя: строки таблицы, FACTS /
EVIDENCE со scope=form, цитаты строки NEWS без формы, валидатор ответа проходит. db-тест —
round-trip колонки form_notes (миграция 010)."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from fplcopilot.agent.graph import signal_brief
from fplcopilot.agent.llm import _cites
from fplcopilot.agent.validate import validate_answer
from fplcopilot.rag import form_notes as fn
from fplcopilot.rag.extract import Evidence, FormNote, PlayerSignal, save_signal

NOTES = [
    {
        "kind": "form",
        "text": "Rayan Cherki scored in his side's 5-3 Premier League win over Sunderland.",
        "quote": 'Rayan Cherki a "genius" after the forward scored in his side\'s 5-3 Premier '
        "League win over Sunderland.",
        "chunk_id": 3591,
        "source": "bbc_football",
        "url": "https://www.bbc.co.uk/sport/football/articles/x",
        "published_at": "2026-09-20T18:00:00+00:00",
    },
    {
        "kind": "role",
        "text": "Cherki is more of an inside forward, as a number 10 or playing narrow.",
        "quote": "Cherki's more of an inside forward, as a number 10 or playing narrow",
        "chunk_id": 3463,
        "source": "bbc_football",
        "url": "https://www.bbc.co.uk/sport/football/articles/y",
        "published_at": "2026-09-17T09:00:00+00:00",
    },
]


def test_card_rows_have_kind_quote_and_link():
    rows = fn.form_note_rows(NOTES)
    assert [r["Тип"] for r in rows] == ["форма", "роль"]
    assert rows[0]["Дата"] == "20.09" and rows[0]["Ссылка"].startswith("https://")
    assert rows[1]["Цитата"].startswith("Cherki's more of an inside forward")


def test_facts_brief_and_evidence_are_scoped_as_form():
    brief = fn.form_notes_brief(NOTES)
    assert brief[0] == {
        "kind": "form",
        "note": NOTES[0]["text"],
        "source": "bbc_football",
        "date": "20.09",
    }
    ev = fn.form_evidence(NOTES, "Cherki")
    assert {e["scope"] for e in ev} == {"form"} and ev[0]["player"] == "Cherki"
    assert "not evidence of availability" in ev[0]["about"]


def test_signal_brief_keeps_form_next_to_availability():
    sig = {"availability": "fit", "confidence": 0.8, "evidence": [{}], "form_notes": NOTES}
    brief = signal_brief(sig)
    assert brief["availability"] == "fit" and brief["evidence_count"] == 1
    assert [n["kind"] for n in brief["form_and_context"]] == ["form", "role"]
    assert "form_and_context" not in signal_brief({"availability": "fit", "evidence": []})


def test_news_line_citations_skip_form_quotes():
    evidence = [
        *fn.form_evidence(NOTES, "Cherki"),
        {"player": "Cherki", "source": "sky_football", "date": "17.09", "quote": "rested"},
    ]
    assert _cites(evidence, player="Cherki") == ["[sky_football, 17.09]"]


def test_answer_with_form_context_passes_validator():
    facts = {
        "players": {"Cherki": {"news_signal": {"availability": "fit", "confidence": 0.8}}},
        "form_and_context": {"about": fn.FORM_ABOUT, "notes": fn.form_notes_brief(NOTES)},
    }
    evidence = fn.form_evidence(NOTES, "Cherki")
    answer = (
        "**Verdict:** Cherki is available (FPL status a).\n"
        "**Why:**\n- Form context: he scored in the 5-3 Premier League win over Sunderland "
        "[bbc_football, 20.09]; he plays as an inside forward or number 10.\n"
    )
    res = validate_answer(answer, facts, evidence)
    assert res.passed, res.as_dict()


def test_load_form_notes_without_signal_or_db(monkeypatch):
    assert fn.load_form_notes(1, None) == []

    def boom():
        raise OperationalError("SELECT", {}, Exception("db down"))

    monkeypatch.setattr(fn, "session_scope", boom)
    assert fn.load_form_notes(1, "2026-09-23T18:00:00+00:00") == []


@pytest.mark.db
def test_form_notes_round_trip_in_player_signals():
    from fplcopilot.db import ping, session_scope

    if not ping():
        pytest.skip("Postgres недоступен")
    now = datetime.now(UTC).replace(microsecond=0)
    note = FormNote(**{**NOTES[0], "published_at": datetime(2026, 9, 20, 18, tzinfo=UTC)})
    sig = PlayerSignal(
        player_id=999_998,
        player_name="Test Player",
        as_of=now,
        availability="fit",
        start_probability=0.85,
        expected_minutes=68,
        rotation_risk="low",
        confidence=0.8,
        summary="Test signal.",
        evidence=[
            Evidence(chunk_id=1, source="test", url="test://x", published_at=now, quote="rested")
        ],
        fpl_status="a",
        model="test-model",
        prompt_version="v4",
        retrieved_chunk_ids=[1],
        mode="hybrid_rerank",
        form_notes=[note],
    )
    row_id = save_signal(sig)
    try:
        loaded = fn.load_form_notes(999_998, now.isoformat())
        assert [n["kind"] for n in loaded] == ["form"]
        assert loaded[0]["quote"] == NOTES[0]["quote"]
    finally:
        with session_scope() as s:
            s.execute(text("DELETE FROM player_signals WHERE id = :id"), {"id": row_id})
