"""xPts v0 на синтетическом bootstrap: компоненты, blank GW, сила фикстур, утечка."""

import math
from datetime import UTC, datetime

import pytest

from fplcopilot.config import settings
from fplcopilot.core.fixtures import (
    LEAGUE_AVG_XG,
    TeamRatingProvider,
    fixture_strength_index,
    fixture_views,
)
from fplcopilot.core.history import TeamMatchStat
from fplcopilot.core.xpts import (
    FALLBACK_PRIORS,
    Rates,
    Totals,
    XPtsComponents,
    make_context,
    predict_all,
    predict_player,
    shrink_rates,
)
from fplcopilot.data import Position
from fplcopilot.data.schemas import Bootstrap, Fixture, PlayerGWHistory

DEADLINE = {
    1: datetime(2026, 8, 21, 17, 30, tzinfo=UTC),
    2: datetime(2026, 8, 28, 17, 30, tzinfo=UTC),
    3: datetime(2026, 9, 4, 17, 30, tzinfo=UTC),
}
NOW = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _team_rating_provider(monkeypatch):
    """В .env проекта может стоять XPTS_FIXTURE_PROVIDER=odds — тесты ядра должны быть
    детерминированы и без сети/кэша коэффициентов / Understat."""
    monkeypatch.setattr(settings, "xpts_fixture_provider", "team_rating")
    monkeypatch.setattr(settings, "understat_enabled", False)
    monkeypatch.setattr(settings, "xpts_setpiece_weight", 0.10)
    monkeypatch.setattr(settings, "xpts_freekick_weight", 0.03)
    monkeypatch.setattr(settings, "xpts_understat_blend", 0.25)
    monkeypatch.setattr(settings, "xpts_understat_shift_cap", 0.15)


def _bootstrap() -> Bootstrap:
    def player(pid, pos, team, **kw):
        d = {"id": pid, "web_name": f"P{pid}", "team": team, "element_type": pos, "now_cost": 50}
        d.update(kw)
        return d

    return Bootstrap.model_validate(
        {
            "events": [
                {"id": 1, "name": "GW1", "deadline_time": DEADLINE[1], "finished": True},
                {"id": 2, "name": "GW2", "deadline_time": DEADLINE[2], "is_next": True},
                {"id": 3, "name": "GW3", "deadline_time": DEADLINE[3]},
            ],
            "teams": [
                {
                    "id": 1,
                    "name": "Alpha",
                    "short_name": "ALP",
                    "strength_overall_home": 4,
                    "strength_overall_away": 4,
                },
                {
                    "id": 2,
                    "name": "Beta",
                    "short_name": "BET",
                    "strength_overall_home": 3,
                    "strength_overall_away": 3,
                },
                {
                    "id": 3,
                    "name": "Gamma",
                    "short_name": "GAM",
                    "strength_overall_home": 2,
                    "strength_overall_away": 2,
                },
            ],
            "elements": [
                player(1, 1, 1, ep_next=4.0),
                player(2, 2, 1),
                player(3, 3, 2, penalties_order=1),
                player(4, 4, 3),
                player(5, 3, 3, status="i", chance_of_playing_next_round=0),
            ],
        }
    )


def _fixtures() -> list[Fixture]:
    return [
        Fixture.model_validate(
            {"id": 1, "event": 1, "team_h": 1, "team_a": 2, "kickoff_time": "2026-08-22T14:00:00Z"}
        ),
        Fixture.model_validate(
            {"id": 2, "event": 1, "team_h": 3, "team_a": 1, "kickoff_time": "2026-08-23T14:00:00Z"}
        ),
        Fixture.model_validate(
            {"id": 3, "event": 2, "team_h": 2, "team_a": 1, "kickoff_time": "2026-08-29T14:00:00Z"}
        ),
        Fixture.model_validate(
            {"id": 4, "event": 3, "team_h": 3, "team_a": 2, "kickoff_time": "2026-09-05T14:00:00Z"}
        ),
        Fixture.model_validate(
            {"id": 5, "event": 3, "team_h": 1, "team_a": 3, "kickoff_time": "2026-09-05T16:30:00Z"}
        ),
    ]


def _row(element, fixture, rnd, home, **kw) -> PlayerGWHistory:
    d = {
        "element": element,
        "fixture": fixture,
        "round": rnd,
        "was_home": home,
        "starts": 1,
        "minutes": 90,
    }
    d.update(kw)
    return PlayerGWHistory.model_validate(d)


def _history() -> list[PlayerGWHistory]:
    return [
        _row(1, 1, 1, True, saves=4, clean_sheets=1, bps=25, recoveries=8),
        _row(2, 1, 1, True, clearances_blocks_interceptions=8, tackles=3, clean_sheets=1, bps=30),
        _row(3, 1, 1, False, expected_goals=0.6, expected_assists=0.3, bps=35, goals_scored=1),
        _row(4, 2, 1, True, expected_goals=0.5, bps=20),
        _row(5, 2, 1, True, minutes=0, starts=0),
        # утечка: раунд 2 уже «сыгран» с огромными числами — не должен попасть в контекст GW2
        _row(3, 3, 2, True, expected_goals=5.0, expected_assists=5.0, bps=200),
    ]


@pytest.fixture
def ctx_gw2():
    return make_context(
        2,
        bs=_bootstrap(),
        fixtures=_fixtures(),
        history_rows=_history(),
        as_of=DEADLINE[2],
        live=False,
        now=NOW,
    )


def test_context_is_leak_free(ctx_gw2):
    assert all(r.round < 2 for rows in ctx_gw2.history.values() for r in rows)
    assert ctx_gw2.n_rounds == 1
    assert ctx_gw2.as_of == DEADLINE[2]
    assert ctx_gw2.rates[3].xg < 1.0  # раунд-2 строка с xG=5 отфильтрована


def test_components_sum_to_total_for_every_player(ctx_gw2):
    preds = predict_all(2, ctx=ctx_gw2)
    assert len(preds) == 5
    for p in preds:
        assert math.isclose(p.components.total(), p.xpts, abs_tol=1e-9)
        assert p.variance >= 0
        assert 0 <= p.p60 <= p.p_appear <= 1
    assert preds == sorted(preds, key=lambda p: -p.xpts)


def test_blank_gameweek_gives_zero(ctx_gw2):
    pred = predict_player(4, 2, ctx=ctx_gw2)  # команда 3 в GW2 не играет
    assert pred.xpts == 0.0
    assert pred.variance == 0.0
    assert pred.fixtures == []
    assert pred.components == XPtsComponents()
    assert "blank gameweek: no fixture" in pred.notes


def test_position_specific_components(ctx_gw2):
    gk = predict_player(1, 2, ctx=ctx_gw2)
    dfn = predict_player(2, 2, ctx=ctx_gw2)
    mid = predict_player(3, 2, ctx=ctx_gw2)
    assert gk.components.saves > 0 and gk.components.defcon == 0
    assert gk.components.clean_sheet > 0 and gk.components.goals_conceded < 0
    assert dfn.components.defcon > 0 and dfn.components.goals_conceded < 0
    assert dfn.components.clean_sheet > mid.components.clean_sheet > 0  # 4 очка против 1
    assert mid.components.goals_conceded == 0 and mid.components.saves == 0
    assert mid.components.goals > dfn.components.goals
    assert gk.ep_next == 4.0 and mid.ep_next is None  # ep_next только из bootstrap
    assert len(mid.fixtures) == 1 and mid.fixtures[0].opponent == "ALP"


def test_live_status_zeroes_injured_player():
    ctx = make_context(
        3,
        bs=_bootstrap(),
        fixtures=_fixtures(),
        history_rows=_history(),
        as_of=NOW,
        live=True,
        now=NOW,
    )
    assert ctx.as_of == DEADLINE[3]  # as_of не может быть позже дедлайна тура
    injured = predict_player(5, 3, ctx=ctx)
    assert injured.xpts == 0.0 and injured.status == "i"
    fit = predict_player(4, 3, ctx=ctx)
    assert fit.xpts > 0


def test_status_overrides_apply_only_when_not_live():
    ctx = make_context(
        2,
        bs=_bootstrap(),
        fixtures=_fixtures(),
        history_rows=_history(),
        as_of=DEADLINE[2],
        live=False,
        now=NOW,
        status_overrides={3: ("u", None)},
    )
    assert predict_player(3, 2, ctx=ctx).xpts == 0.0
    assert predict_player(2, 2, ctx=ctx).xpts > 0


def test_shrinkage_moves_rates_towards_prior():
    prior = FALLBACK_PRIORS[Position.MID]
    hot = Totals(minutes=180, xg=1.6, xa=0.2, bps=60)  # 0.8 xG/90 за два матча
    shrunk = shrink_rates(hot, prior)
    assert prior.xg < shrunk.xg < 0.8
    assert shrink_rates(Totals(), prior) == prior
    assert Totals(minutes=0).per90() is None
    blended = Rates(
        *(
            0.5 * a + 0.5 * b
            for a, b in zip(prior.__dict__.values(), prior.__dict__.values(), strict=True)
        )
    )
    assert blended == prior


def test_team_rating_provider_calibration_and_ordering():
    bs = _bootstrap()
    fixtures = _fixtures()
    provider = TeamRatingProvider(bs.teams, None, calibration_fixtures=fixtures)
    lams = [x for f in fixtures for x in provider.expected_goals(f)]
    assert math.isclose(sum(lams) / len(lams), LEAGUE_AVG_XG, rel_tol=0.01)
    strong_home, weak_away = provider.expected_goals(fixtures[4])  # Alpha (4) дома против Gamma (2)
    assert strong_home > weak_away
    views = fixture_views(provider, fixtures, 3)
    assert set(views) == {1, 2, 3}
    alpha = views[1][0]
    assert alpha.is_home and alpha.opponent_id == 3
    assert math.isclose(alpha.clean_sheet_prob, math.exp(-alpha.xg_against))
    assert 1 <= alpha.fixture_strength_index <= 5
    assert views[2] == [] or views[2][0].team_id == 2


def test_season_data_shrinks_ratings_towards_observed_xg():
    bs = _bootstrap()
    fixtures = _fixtures()
    # Gamma (слабый рейтинг) создаёт много xG в каждом матче
    stats = {
        (10 + i, 3): TeamMatchStat(
            10 + i, 1, 3, 1, True, xg_for=3.0, xg_against=0.5, goals_for=3, goals_against=0
        )
        for i in range(4)
    }
    for i in range(4):
        stats[(10 + i, 1)] = TeamMatchStat(10 + i, 1, 1, 3, False, 0.5, 3.0, 0, 3)
    plain = TeamRatingProvider(bs.teams, None, calibration_fixtures=fixtures)
    with_data = TeamRatingProvider(bs.teams, stats, calibration_fixtures=fixtures)
    assert with_data.ratings[3].attack > plain.ratings[3].attack
    assert with_data.ratings[3].matches == 4 and plain.ratings[3].matches == 0
    assert with_data.ratings[1].defence > plain.ratings[1].defence  # пропускала много


def test_fixture_strength_index_edges():
    assert fixture_strength_index(2.5, 0.8) == 1
    assert fixture_strength_index(1.8, 1.3) == 2
    assert fixture_strength_index(1.4, 1.4) == 3
    assert fixture_strength_index(1.1, 1.7) == 4
    assert fixture_strength_index(0.7, 2.2) == 5


def test_setpiece_bonus_only_when_penalties_order_is_one(ctx_gw2):
    mid = predict_player(3, 2, ctx=ctx_gw2)  # penalties_order=1
    dfn = predict_player(2, 2, ctx=ctx_gw2)
    assert mid.components.setpiece > 0
    assert mid.setpiece_bonus == pytest.approx(mid.components.setpiece)
    assert dfn.components.setpiece == 0.0
    assert mid.fixtures[0].setpiece_bonus == pytest.approx(mid.setpiece_bonus)
    assert math.isclose(mid.components.total(), mid.xpts, abs_tol=1e-9)
