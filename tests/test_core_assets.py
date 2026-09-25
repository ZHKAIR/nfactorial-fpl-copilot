"""Правило «ценный актив с короткой травмой» (core/assets.py): придержать своего, купить чужого на
спаде; тур возвращения из новости FPL; минуты «если здоров»; бонус в цели оптимизатора."""

from __future__ import annotations

from datetime import date

import pytest
from test_core_optimizer import as_pool, base_squad, cand, ids

from fplcopilot.core.assets import asset_flags, tail_factor
from fplcopilot.core.candidates import Candidate, return_gw_from_news
from fplcopilot.core.minutes import estimate_minutes, fit_minutes
from fplcopilot.core.optimizer import recommend_route, single_transfer
from fplcopilot.core.strategy import get_strategy
from fplcopilot.data import Position
from fplcopilot.data.schemas import PlayerGWHistory

GWS = [5, 6, 7]


def player(
    pid: int,
    pos: int,
    price: float,
    xp: float,
    *,
    fit: float | None = None,
    status: str = "a",
    chance: int | None = None,
    return_gw: int | None = None,
    p_start: float = 0.9,
    change: float = 0.0,
    net: int = 0,
) -> Candidate:
    return Candidate(
        player_id=pid,
        name=f"P{pid}",
        position=pos,
        team_id=pid,
        price=price,
        xpts_by_gw={g: xp for g in GWS},
        xpts_fit_by_gw={g: fit for g in GWS} if fit is not None else {},
        status=status,
        chance=chance,
        return_gw=return_gw,
        p_start=p_start,
        price_change_start=change,
        transfers_net=net,
    )


def league() -> dict[int, Candidate]:
    """По 10 регулярных стартеров на позицию с очками 2.0…6.5 (медиана 4.25, P75 5.5)."""
    out = {}
    for pos in (1, 2, 3, 4):
        for i in range(10):
            pid = pos * 100 + i
            out[pid] = player(pid, pos, 4.5 + i * 0.5, 2.0 + i * 0.5)
    return out


def test_asset_flags_hold_buy_low_and_exclusions():
    c = league()
    specials = [
        player(901, 3, 9.0, 3.0, fit=7.0, status="d", chance=75, p_start=0.7),  # свой, ценный
        player(902, 3, 7.0, 0.0, fit=6.8, status="i", return_gw=6, p_start=0.0, change=-0.3, net=-40_000),
        player(903, 3, 7.0, 0.0, fit=6.8, status="i", return_gw=6, p_start=0.0, change=0.1, net=5_000),
        player(904, 3, 9.0, 0.0, fit=7.0, status="i", return_gw=9, p_start=0.0),  # долго
        player(905, 3, 9.0, 0.0, fit=7.0, status="u", p_start=0.0),
        player(906, 3, 5.0, 2.0, fit=3.0, status="d", chance=75, p_start=0.7),  # не топ
    ]
    c.update({s.player_id: s for s in specials})
    S = get_strategy("balanced")
    flags = asset_flags(c, squad_ids=[901, 904, 905, 906], gws=GWS, strategy=S)
    assert set(flags) == {901, 902}
    hold, buy = flags[901], flags[902]
    assert hold.kind == "hold" and buy.kind == "buy_low"
    assert hold.fit_xpts == 7.0 and hold.position_median == 4.25 and hold.top_share == 1.0
    assert hold.bonus == pytest.approx(0.5 * (7.0 - 4.25) * tail_factor(S, 3), abs=1e-3)
    assert tail_factor(S, 3) == pytest.approx(0.84**3 + 0.84**4)
    assert buy.return_gw == 6 and buy.price_change_start == -0.3
    # в составе нет регулярных — уровни позиции не считаются, флагов нет
    assert asset_flags({901: specials[0]}, [901], GWS, S) == {}


def test_return_gw_from_fpl_news():
    kick = [(date(2026, 9, 26), 6), (date(2026, 10, 3), 7), (date(2026, 10, 17), 8)]
    assert return_gw_from_news("Knee injury - Expected back 03 Oct", kick, 8) == 7
    assert return_gw_from_news("Hamstring - Expected back 29 Sep", kick, 8) == 7
    assert return_gw_from_news("Suspended until 20 Oct", kick, 8) == 9  # за горизонтом
    assert return_gw_from_news("Knee injury - 75% chance of playing", kick, 8) is None
    assert return_gw_from_news("Expected back 03 Oct", [], 8) is None


def test_fit_minutes_ignores_current_absence_streak():
    def row(rnd: int, minutes: int) -> PlayerGWHistory:
        return PlayerGWHistory(
            element=1, fixture=rnd, round=rnd, was_home=True, minutes=minutes, starts=int(minutes > 0)
        )

    rows = [row(1, 90), row(2, 90), row(3, 90), row(4, 0), row(5, 0)]  # травма в 4–5 туре
    injured = estimate_minutes(rows, position=Position.MID, status="i")
    fit = fit_minutes(rows, position=Position.MID)
    assert injured.p_start == 0.0
    assert fit.p_start > estimate_minutes(rows, position=Position.MID).p_start
    assert fit.p_start > 0.7


def test_asset_bonus_stops_selling_injured_star_and_buys_on_the_dip():
    squad = base_squad()
    squad[8] = cand(9, 3, 9, 8.0, {5: 0.0, 6: 2.0, 7: 2.0})  # травмированная звезда (MID £8)
    pool = as_pool(squad, [cand(30, 3, 30, 8.0, 3.0)])
    routes = single_transfer(ids(squad), 0.0, 1, pool, 5, 3, "balanced")
    assert recommend_route(routes).out == [9]  # без правила — продать
    held = single_transfer(ids(squad), 0.0, 1, pool, 5, 3, "balanced", asset_bonus={9: 6.0})
    pick = recommend_route(held)
    assert pick is None or 9 not in pick.out  # с бонусом — держим

    # покупка на спаде: 31 сейчас не играет (0 очков), но бонус актива перевешивает FT
    pool = as_pool(base_squad(), [cand(31, 2, 31, 4.0, 0.0)])
    routes = single_transfer(ids(base_squad()), 0.0, 1, pool, 5, 3, "balanced")
    assert recommend_route(routes) is None
    dip = single_transfer(ids(base_squad()), 0.0, 1, pool, 5, 3, "balanced", asset_bonus={31: 4.0})
    assert recommend_route(dip).in_ == [31]
