"""Сбор фактов сравнения двух игроков (app/compare.py): сценарий «побеждает на 1 тур,
проигрывает на 3 из-за календаря», тексты без жаргона, новости не выдумываются.
LLM — под маркером llm / с фейковым клиентом.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from fplcopilot.agent.tools import (
    EvidenceItem,
    FixtureBrief,
    GWPrediction,
    PlayerPrediction,
    PlayerRef,
    PlayerRisk,
)
from fplcopilot.app import compare as cmp
from fplcopilot.data.schemas import Bootstrap, Event, Player, Team

NOW = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


def pl(pid, web, team, pos, cost, own, **stats) -> Player:
    base = {
        "id": pid,
        "web_name": web,
        "first_name": web,
        "second_name": "Test",
        "team": team,
        "element_type": pos,
        "now_cost": cost,
        "selected_by_percent": own,
        "minutes": 450,
        "starts": 5,
        "total_points": 20,
        "form": 5.0,
        "points_per_game": 4.0,
    }
    base.update(stats)
    return Player(**base)


def make_bs() -> Bootstrap:
    teams = [
        Team(id=1, name="Arsenal", short_name="ARS"),
        Team(id=2, name="Chelsea", short_name="CHE"),
    ]
    players = [
        pl(10, "Saka", 1, 3, 100, 40.0, total_points=50, form=6.5, points_per_game=6.0),
        pl(355, "Palmer", 2, 3, 105, 45.0, total_points=55, form=7.0, points_per_game=6.5),
    ]
    events = [Event(id=6, name="Gameweek 6", deadline_time=NOW, is_next=True)]
    return Bootstrap(events=events, teams=teams, elements=players)


BS = make_bs()


def fx(opp: str, home: bool, fsi: int) -> FixtureBrief:
    return FixtureBrief(
        opponent=opp, is_home=home, fsi=fsi, xg_for=1.5, xg_against=1.2, clean_sheet_prob=0.3
    )


def pred(
    pid: int,
    gws: dict[int, tuple[float, list[FixtureBrief]]],
    *,
    p_start: float = 0.9,
    exp_minutes: float = 85,
) -> PlayerPrediction:
    p = BS.player(pid)
    by_gw = [
        GWPrediction(
            gw=g,
            xpts=x,
            sd=2.0,
            p_start=p_start,
            exp_minutes=exp_minutes,
            components={"appearance": 1.9},
            fixtures=fixtures,
        )
        for g, (x, fixtures) in sorted(gws.items())
    ]
    return PlayerPrediction(
        player=PlayerRef(
            id=p.id,
            name=p.web_name,
            full_name=p.full_name,
            team=BS.team(p.team).short_name,
            position=p.position.short,
            price=p.price,
            status=p.status,
            chance=None,
            ownership=float(p.selected_by_percent or 0),
        ),
        by_gw=by_gw,
        total_xpts=round(sum(g.xpts for g in by_gw), 2),
    )


def student_scenario_preds() -> tuple[PlayerPrediction, PlayerPrediction]:
    """Saka выигрывает GW6 (лёгкий матч), но проигрывает на 3 тура: GW7–8 тяжёлые.
    Palmer — наоборот: тяжёлый ближайший, затем два лёгких."""
    saka = pred(
        10,
        {
            6: (6.0, [fx("BUR", True, 1)]),  # лёгкий
            7: (2.5, [fx("LIV", False, 5)]),  # тяжёлый
            8: (2.0, [fx("MCI", False, 5)]),  # тяжёлый
            9: (4.0, [fx("WOL", True, 2)]),
            10: (4.5, [fx("EVE", True, 2)]),
            11: (3.5, [fx("BHA", False, 3)]),
        },
    )
    palmer = pred(
        355,
        {
            6: (3.5, [fx("ARS", False, 4)]),  # тяжёлый
            7: (5.5, [fx("IPS", True, 1)]),  # лёгкий
            8: (5.0, [fx("LEI", True, 2)]),  # лёгкий
            9: (4.0, [fx("TOT", False, 3)]),
            10: (3.5, [fx("NEW", False, 4)]),
            11: (4.0, [fx("SOU", True, 2)]),
        },
    )
    # 3GW: Saka 6+2.5+2=10.5; Palmer 3.5+5.5+5=14.0
    return saka, palmer


def test_student_scenario_wins_next_loses_on_3gw():
    saka_p, palmer_p = student_scenario_preds()
    saka, palmer = BS.player(10), BS.player(355)
    facts = cmp.build_compare_facts(
        saka,
        palmer,
        saka_p,
        palmer_p,
        None,
        None,
        bs=BS,
        frr_a=4,
        frr_b=12,
        strategy="balanced",
    )
    assert facts["next_gw"]["winner"] == "a"
    assert facts["next_gw"]["leader"] == "Saka"
    assert facts["next_gw"]["margin_points"] == pytest.approx(2.5)
    assert facts["next_gw"]["a_difficulty"] == 1
    assert facts["next_gw"]["b_difficulty"] == 4
    assert facts["horizon_3"]["winner"] == "b"
    assert facts["horizon_3"]["leader"] == "Palmer"
    assert facts["horizon_3"]["a_xpts"] == pytest.approx(10.5)
    assert facts["horizon_3"]["b_xpts"] == pytest.approx(14.0)
    # календарь: у Saka легче GW6, у Palmer — GW7 и GW8
    by_gw = {f["gw"]: f for f in facts["fixtures"]}
    assert by_gw[6]["delta_steps"] == 1 - 4  # Saka легче
    assert by_gw[7]["delta_steps"] == 5 - 1  # Palmer легче
    assert by_gw[8]["delta_steps"] == 5 - 2
    assert "Saka" in facts["calendar_summary"]
    assert "Palmer" in facts["calendar_summary"]
    assert "GW6" in facts["calendar_summary"]
    assert "GW7" in facts["calendar_summary"]


def test_facts_prose_has_no_forbidden_jargon():
    saka_p, palmer_p = student_scenario_preds()
    facts = cmp.build_compare_facts(
        BS.player(10),
        BS.player(355),
        saka_p,
        palmer_p,
        None,
        None,
        bs=BS,
        frr_a=4,
        frr_b=12,
    )
    bad = cmp.assert_facts_prose_clean(facts)
    assert bad == [], bad
    assert "сложность" in (facts["next_gw"].get("fixture_note") or "")
    assert "очк" in facts["next_gw"]["text"]
    assert cmp.prose_is_clean("плюс 1.4 очка за 3 тура")
    assert not cmp.prose_is_clean("Δ xPts → FSI 2")


def test_news_not_invented_when_signal_missing():
    saka_p, palmer_p = student_scenario_preds()
    unavailable = PlayerRisk(
        player_id=10,
        player="Saka",
        fpl_status="a",
        fpl_chance=100,
        origin="unavailable",
        note="no saved news signal (cached_only)",
    )
    facts = cmp.build_compare_facts(
        BS.player(10),
        BS.player(355),
        saka_p,
        palmer_p,
        unavailable,
        None,
        bs=BS,
        frr_a=1,
        frr_b=2,
    )
    assert facts["news_a"]["has_signal"] is False
    assert facts["news_a"]["note"] == "разбора новостей нет"
    assert facts["news_a"]["evidence"] == []
    assert facts["news_b"]["has_signal"] is False
    lines = " ".join(facts["news_lines"])
    assert "разбора новостей нет" in lines
    assert "ротац" not in lines.lower() or "разбора" in lines  # нет выдуманной ротации
    assert any("обоим" in c or "обоим" in c for c in facts["caveats_seed"]) or any(
        "разбора новостей" in c for c in facts["caveats_seed"]
    )


def test_news_facts_copy_evidence_literally():
    risk = PlayerRisk(
        player_id=10,
        player="Saka",
        fpl_status="a",
        fpl_chance=100,
        origin="cached",
        availability="fit",
        start_probability=0.85,
        expected_minutes=80,
        rotation_risk="low",
        summary="Fit and starting",
        evidence=[
            EvidenceItem(
                player_id=10,
                player="Saka",
                source="bbc_football",
                url="https://example.com/saka",
                published_at="2026-09-20T12:00:00+00:00",
                date="20.09",
                quote="Saka trained fully and is expected to start",
            )
        ],
        age_h=2.0,
    )
    nf = cmp.news_facts(risk, "Saka")
    assert nf["has_signal"] is True
    assert nf["evidence"][0]["quote"] == "Saka trained fully and is expected to start"
    assert nf["evidence"][0]["source"] == "bbc_football"
    assert nf["evidence"][0]["date"] == "20.09"
    lines = cmp.news_plain_lines(nf)
    assert any("bbc_football" in ln and "20.09" in ln for ln in lines)


def test_metric_rows_highlight_winner():
    saka_p, palmer_p = student_scenario_preds()
    rows = {r.key: r for r in cmp.metric_rows(
        BS.player(10), BS.player(355), saka_p, palmer_p, frr_a=4, frr_b=12, gw=6
    )}
    assert rows["xpts"].winner == "a"  # Saka 6.0 > Palmer 3.5
    assert rows["xpts3"].winner == "b"  # Palmer 14 > Saka 10.5
    assert rows["frr"].winner == "a"  # lower FRR better
    html = cmp.compare_table_html(
        list(rows.values()), "Saka", "Palmer", team_a="ARS", team_b="CHE", pos_a="MID", pos_b="MID"
    )
    assert "Saka" in html and "Palmer" in html
    assert "win" in html


def test_fixture_compare_delta_text():
    saka_p, palmer_p = student_scenario_preds()
    rows = cmp.fixture_compare_rows(saka_p, palmer_p, "Saka", "Palmer")
    assert rows[0].delta_text == "у Saka легче на 3 ступени"
    assert "ступен" in rows[1].delta_text
    # 1 ступень — правильное склонение
    one = cmp.steps_word(1)
    assert one == "1 ступень"
    html = cmp.fixture_compare_html(rows, "Saka", "Palmer")
    assert "GW6" in html
    assert "BUR (д)" in html or "BUR" in html


def test_explain_compare_with_fake_client():
    """Детерминированный путь: фейковый клиент, без сети."""
    saka_p, palmer_p = student_scenario_preds()
    facts = cmp.build_compare_facts(
        BS.player(10),
        BS.player(355),
        saka_p,
        palmer_p,
        None,
        None,
        bs=BS,
        frr_a=4,
        frr_b=12,
    )
    parsed = cmp.CompareExplanation(
        verdict_next_gw=(
            "Saka лучше на ближайший тур: ближайший матч легче (сложность 1 из 5 против 4 из 5), "
            "плюс 2.5 очка."
        ),
        verdict_3gw=(
            "На 3 туров лидирует Palmer: после лёгкого старта у Saka идут два тяжёлых матча, "
            "итого минус 3.5 очка к Palmer."
        ),
        why=[
            "На ближайший тур у Saka сложность 1 из 5, у Palmer 4 из 5.",
            "За 3 тура у Palmer 14.00 очков против 10.50 у Saka.",
        ],
        news_points=["По игроку Saka разбора новостей нет.", "По игроку Palmer разбора новостей нет."],
        caveats=["Прогноз очков — модель, а не гарантия результата матча."],
    )

    class _Msg:
        def __init__(self) -> None:
            self.parsed = parsed
            self.refusal = None

    class _Completions:
        def parse(self, **kwargs: Any) -> Any:
            assert kwargs["temperature"] == cmp.COMPARE_TEMPERATURE
            assert kwargs["model"] == cmp.COMPARE_MODEL
            return SimpleNamespace(
                choices=[SimpleNamespace(message=_Msg())],
                usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    expl, usage = cmp.explain_compare(facts, client=client)
    assert expl.verdict_next_gw.startswith("Saka")
    assert "Palmer" in expl.verdict_3gw
    assert usage.prompt_tokens == 100
    assert usage.cost_usd >= 0
    for text in [expl.verdict_next_gw, expl.verdict_3gw, *expl.why, *expl.news_points]:
        assert cmp.prose_is_clean(text), text


@pytest.mark.llm
@pytest.mark.network
def test_explain_compare_live_llm():
    """Живой вызов gpt-4o-mini — только с -m llm."""
    saka_p, palmer_p = student_scenario_preds()
    facts = cmp.build_compare_facts(
        BS.player(10),
        BS.player(355),
        saka_p,
        palmer_p,
        None,
        None,
        bs=BS,
        frr_a=4,
        frr_b=12,
    )
    expl, usage = cmp.explain_compare(facts)
    assert expl.verdict_next_gw
    assert expl.verdict_3gw
    assert expl.why
    assert expl.caveats
    assert usage.prompt_tokens > 0
    blob = " ".join(
        [expl.verdict_next_gw, expl.verdict_3gw, *expl.why, *expl.news_points, *expl.caveats]
    )
    assert cmp.prose_is_clean(blob)
    assert "Saka" in blob or "Palmer" in blob
