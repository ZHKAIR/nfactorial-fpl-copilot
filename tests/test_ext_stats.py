"""Внешняя стата: парсер Understat, маппинг имён, blend ±15 %, бонус пенальтиста, fallback без сети."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from fplcopilot.config import settings
from fplcopilot.core.ext_stats import (
    blend_rate,
    blend_rates_with_understat,
    build_index,
    load_ext_index,
    player_advanced_stats,
    player_ext,
    reset_ext_index,
    setpiece_bonus_xpts,
    setpiece_phrase,
    team_defensive_profile,
)
from fplcopilot.core.scoring import setpiece_xpts
from fplcopilot.core.understat import (
    UnderstatClient,
    UnderstatSnapshot,
    extract_embedded_json,
    match_players,
    parse_html,
)
from fplcopilot.core.xpts import Rates, XPtsComponents, make_context, predict_player
from fplcopilot.data.schemas import Bootstrap, Player, Team
from tests.test_core_xpts import DEADLINE, NOW, _bootstrap, _fixtures, _history

FIXTURE_HTML = Path(__file__).parent / "fixtures" / "understat_league_snippet.html"


@pytest.fixture(autouse=True)
def _no_live_understat(monkeypatch):
    monkeypatch.setattr(settings, "understat_enabled", False)
    monkeypatch.setattr(settings, "xpts_fixture_provider", "team_rating")
    monkeypatch.setattr(settings, "xpts_setpiece_weight", 0.12)
    monkeypatch.setattr(settings, "xpts_freekick_weight", 0.03)
    monkeypatch.setattr(settings, "xpts_understat_blend", 0.25)
    monkeypatch.setattr(settings, "xpts_understat_shift_cap", 0.15)
    reset_ext_index()
    yield
    reset_ext_index()


def _fpl_named() -> Bootstrap:
    """Мини-bootstrap с проблемными именами (João Pedro, Ødegaard, Gvardiol)."""
    return Bootstrap.model_validate(
        {
            "events": [{"id": 1, "name": "GW1", "deadline_time": "2026-08-21T17:30:00Z"}],
            "teams": [
                {"id": 1, "name": "Arsenal", "short_name": "ARS"},
                {"id": 3, "name": "Chelsea", "short_name": "CHE"},
                {"id": 8, "name": "Manchester City", "short_name": "MCI"},
            ],
            "elements": [
                {
                    "id": 17,
                    "web_name": "Ødegaard",
                    "first_name": "Martin",
                    "second_name": "Ødegaard",
                    "team": 1,
                    "element_type": 3,
                    "now_cost": 80,
                    "minutes": 391,
                    "penalties_order": None,
                    "direct_freekicks_order": 2,
                },
                {
                    "id": 29,
                    "web_name": "João Pedro",
                    "first_name": "João",
                    "second_name": "Pedro Junqueira de Jesus",
                    "team": 3,
                    "element_type": 4,
                    "now_cost": 75,
                    "minutes": 360,
                    "penalties_order": 2,
                },
                {
                    "id": 44,
                    "web_name": "Gvardiol",
                    "first_name": "Joško",
                    "second_name": "Gvardiol",
                    "team": 8,
                    "element_type": 2,
                    "now_cost": 60,
                    "minutes": 436,
                    "penalties_order": 1,
                },
                {
                    "id": 99,
                    "web_name": "Nobody",
                    "first_name": "No",
                    "second_name": "Match",
                    "team": 1,
                    "element_type": 3,
                    "now_cost": 45,
                    "minutes": 10,
                },
            ],
        }
    )


def test_parse_understat_html_fixture():
    html = FIXTURE_HTML.read_text()
    extracted = extract_embedded_json(html)
    assert set(extracted) >= {"players", "teams"}
    players, teams = parse_html(html)
    assert {p.player_name for p in players} == {"João Pedro", "Martin Odegaard", "Josko Gvardiol"}
    pedro = next(p for p in players if "Pedro" in p.player_name)
    assert pedro.shots == 11 and pedro.minutes == 360
    assert pedro.xg90 == pytest.approx(2.5174 * 90 / 360)
    assert len(teams) == 3
    arsenal = next(t for t in teams if t.title == "Arsenal")
    assert arsenal.xga_per_game == pytest.approx(0.56)


def test_name_mapping_aliases_and_unmatched():
    bs = _fpl_named()
    players, _teams = parse_html(FIXTURE_HTML.read_text())
    matched, unmatched = match_players(bs.elements, players, bs.teams)
    assert set(matched) == {17, 29, 44}
    assert matched[17].player_name == "Martin Odegaard"
    assert matched[29].player_name == "João Pedro"
    assert matched[44].player_name == "Josko Gvardiol"
    assert unmatched == []
    assert 99 not in matched


def test_setpiece_bonus_only_order_1():
    taker = Player(
        id=1,
        web_name="Calafiori",
        first_name="Riccardo",
        second_name="Calafiori",
        team=1,
        element_type=2,
        now_cost=60,
        penalties_order=1,
    )
    backup = taker.model_copy(update={"id": 2, "penalties_order": 2})
    none = taker.model_copy(update={"id": 3, "penalties_order": None})
    fk = taker.model_copy(
        update={"id": 4, "penalties_order": None, "direct_freekicks_order": 1}
    )
    assert setpiece_xpts(
        penalty_order=1, dfk_order=None, penalty_weight=0.10, dfk_weight=0.03
    ) == pytest.approx(0.10)
    assert setpiece_bonus_xpts(taker) == pytest.approx(settings.xpts_setpiece_weight)
    assert setpiece_bonus_xpts(backup) == 0.0
    assert setpiece_bonus_xpts(none) == 0.0
    assert setpiece_bonus_xpts(fk) == pytest.approx(settings.xpts_freekick_weight)
    team = Team(id=1, name="Arsenal", short_name="ARS")
    assert setpiece_phrase(taker, team) == "Calafiori бьёт пенальти Arsenal"
    assert setpiece_phrase(backup, team) is None
    assert setpiece_phrase(fk, team) == "Calafiori бьёт штрафные Arsenal"


def test_penalty_bonus_applies_in_xpts_only_for_order_1():
    bs = _bootstrap()
    ctx = make_context(
        2, bs=bs, fixtures=_fixtures(), history_rows=_history(), as_of=DEADLINE[2], live=False, now=NOW
    )
    mid = predict_player(3, 2, ctx=ctx)  # penalties_order=1
    dfn = predict_player(2, 2, ctx=ctx)  # нет очереди
    assert mid.setpiece_bonus == pytest.approx(
        settings.xpts_setpiece_weight * min(1.0, mid.exp_minutes / 90.0)
    )
    assert mid.components.setpiece == pytest.approx(mid.setpiece_bonus, abs=1e-9)
    assert any(n.startswith("setpiece_bonus=") for n in mid.notes)
    assert dfn.setpiece_bonus == 0.0 and dfn.components.setpiece == 0.0
    assert pred_components_sum(mid)


def pred_components_sum(pred) -> bool:
    assert pred.components.total() == pytest.approx(pred.xpts)
    return True


def test_blank_gw_gets_no_setpiece():
    ctx = make_context(
        2,
        bs=_bootstrap(),
        fixtures=_fixtures(),
        history_rows=_history(),
        as_of=DEADLINE[2],
        live=False,
        now=NOW,
    )
    pred = predict_player(4, 2, ctx=ctx)
    assert pred.xpts == 0.0
    assert pred.setpiece_bonus == 0.0
    assert pred.components == XPtsComponents()


def test_understat_blend_caps_shift_and_xpts_delta():
    ours = 0.40
    insane = 4.0
    blended = blend_rate(ours, insane, weight=0.25, cap=0.15)
    assert blended == pytest.approx(ours * 1.15)
    assert abs(blended - ours) <= ours * 0.15 + 1e-12
    rates = Rates(xg=0.40, xa=0.10, saves=0.0, bps=20.0, cbit=3.0, cbirt=6.0, yellow=0.2)
    players, _ = parse_html(FIXTURE_HTML.read_text())
    pedro = next(p for p in players if "Pedro" in p.player_name)
    out = blend_rates_with_understat(rates, pedro)
    assert abs(out.xg - rates.xg) <= rates.xg * 0.15 + 1e-12
    # абсурдный xG упирается в +15 %
    object.__setattr__(pedro, "xg", 40.0)
    object.__setattr__(pedro, "minutes", 90)
    capped = blend_rates_with_understat(rates, pedro)
    assert capped.xg == pytest.approx(0.46)
    assert out.saves == rates.saves


def test_blend_does_not_move_xpts_more_than_threshold():
    """Даже абсурдный Understat xG сдвигает xPts меньше чем на 0.5 (cap 15 % × pts гола)."""
    bs = _bootstrap()
    players, teams = parse_html(FIXTURE_HTML.read_text())
    # подсовываем игроку 3 гигантский xG90
    fake = players[0]
    object.__setattr__(fake, "xg", 80.0)
    object.__setattr__(fake, "xa", 80.0)
    object.__setattr__(fake, "minutes", 90)
    snap = UnderstatSnapshot(season=2026, players=[fake], teams=teams, source="html")
    # id 3 — MID с минутами; маппим вручную
    index = build_index(bs, snap)
    index.by_player = {3: fake}
    base = make_context(
        2, bs=bs, fixtures=_fixtures(), history_rows=_history(), as_of=DEADLINE[2], live=False, now=NOW
    )
    blended = make_context(
        2,
        bs=bs,
        fixtures=_fixtures(),
        history_rows=_history(),
        as_of=DEADLINE[2],
        live=False,
        now=NOW,
        ext=index,
    )
    a = predict_player(3, 2, ctx=base)
    b = predict_player(3, 2, ctx=blended)
    assert abs(b.xpts - a.xpts) < 0.50
    assert b.understat_blend == pytest.approx(settings.xpts_understat_blend)
    assert any("understat_blend=" in n for n in b.notes)


def test_fallback_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "understat_enabled", True)
    monkeypatch.setattr(settings, "understat_cache_ttl", 3600)

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    client = UnderstatClient(
        cache_dir=tmp_path,
        cache_ttl=3600,
        transport=httpx.MockTransport(boom),
    )
    snap = client.snapshot()
    assert snap.empty and snap.source == "empty"
    assert "FPL" in snap.note
    bs = _fpl_named()
    index = load_ext_index(bs, snapshot=snap)
    ext = player_ext(bs.player(29), index)
    assert ext.understat is None and "fpl" in ext.sources
    stats = player_advanced_stats(bs.player(29), bs, index=index)
    assert stats["understat"] is None and stats["matched"] is False
    assert stats["fpl"]["penalties_order"] == 2


def test_client_uses_cache_not_network(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "understat_enabled", True)
    payload = extract_embedded_json(FIXTURE_HTML.read_text())
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=payload)

    client = UnderstatClient(cache_dir=tmp_path, cache_ttl=99_000, transport=httpx.MockTransport(handler))
    first = client.snapshot()
    second = client.snapshot()
    assert first.players and second.players
    assert calls["n"] == 1
    assert second.source == "cache"


def test_team_defensive_profile_marks_low_xga():
    bs = _fpl_named()
    players, teams = parse_html(FIXTURE_HTML.read_text())
    snap = UnderstatSnapshot(season=2026, players=players, teams=teams, source="html")
    index = build_index(bs, snap)
    ars = team_defensive_profile(bs.team(1), bs, index=index)
    assert ars["understat"]["xga_per_game"] == pytest.approx(0.56)
    assert ars["concedes_little"] is True
    assert "understat" in ars["sources"]
    assert "FBref" in ars["note"]
    assert "pens_won" in ars and "pens_conceded" in ars


def test_player_card_setpiece_and_understat_sections():
    from fplcopilot.app import player_card as card

    bs = _fpl_named()
    p = bs.player(44)
    section = card.setpiece_section(p)
    assert section.title == "Set pieces"
    labels = [m.label for m in section.metrics]
    assert labels == ["Pens", "Direct FK", "Corners", "Saves"]
    assert next(m for m in section.metrics if m.key == "pen_order").text == "№1"
    assert all(any("а" <= ch <= "я" for ch in m.help.lower()) for m in section.metrics)
    players, _ = parse_html(FIXTURE_HTML.read_text())
    u = card.understat_section(p, players[2])
    assert u is not None and u.title == "Understat"
    assert card.understat_section(p, None) is None
