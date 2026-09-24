"""agent/tools.constrained_routes на игрушечных данных: обязательные покупки/продажи, хиты."""

from __future__ import annotations

import pytest

from fplcopilot.agent.tools import ScenarioInfeasible, constrained_routes, route_out
from fplcopilot.core.candidates import Candidate
from fplcopilot.core.optimizer import ManagerInputs

GWS = [5, 6, 7]


def cand(pid: int, pos: int, team: int, price: float, xp: float, *, own: float = 10.0) -> Candidate:
    return Candidate(
        player_id=pid,
        name=f"P{pid}",
        position=pos,
        team_id=team,
        price=price,
        xpts_by_gw={g: xp for g in GWS},
        variance_by_gw={g: 5.0 for g in GWS},
        ownership=own,
    )


def toy_inputs(*, free_transfers: int = 1, bank: float = 1.0) -> ManagerInputs:
    squad = [
        cand(1, 1, 1, 5.0, 4.0),
        cand(2, 1, 2, 4.0, 2.0),
        cand(3, 2, 3, 6.0, 4.5),
        cand(4, 2, 4, 5.0, 4.0),
        cand(5, 2, 5, 4.5, 3.5),
        cand(6, 2, 6, 4.0, 2.0),
        cand(7, 2, 7, 4.0, 1.5),
        cand(8, 3, 8, 12.0, 7.0),
        cand(9, 3, 9, 8.0, 5.5),
        cand(10, 3, 10, 7.0, 5.0),
        cand(11, 3, 11, 6.0, 4.0),
        cand(12, 3, 12, 4.5, 2.0),
        cand(13, 4, 13, 15.0, 7.5),
        cand(14, 4, 14, 7.0, 4.5),
        cand(15, 4, 15, 5.0, 1.0),
    ]
    market = [
        cand(16, 3, 16, 9.0, 8.0, own=3.0),  # MID, лучше любого своего полузащитника
        cand(17, 2, 17, 5.5, 6.0),  # DEF
        cand(18, 4, 18, 6.0, 5.0),  # FWD
        cand(19, 3, 19, 4.5, 3.0),  # дешёвый MID
    ]
    cands = {c.player_id: c for c in squad + market}
    for c in squad:
        c.in_squad = True
    return ManagerInputs(
        manager_id=1,
        from_gw=5,
        gws=GWS,
        squad=[c.player_id for c in squad],
        bank=bank,
        free_transfers=free_transfers,
        chips_available=[],
        cands=cands,
        pool=dict(cands),
        issues=[],
        store=None,  # type: ignore[arg-type]
        squad_gw=4,
    )


def test_forced_buy_is_in_every_route_and_free_when_one_ft_suffices():
    routes, meta = constrained_routes(toy_inputs(), 5, 3, "balanced", force_in=[16])
    assert routes and all(16 in r.in_ for r in routes)
    assert meta["cap"] == 1 and meta["free_cap"] == 1
    first = routes[0]
    assert first.hit_cost == 0 and len(first.in_) == 1
    assert first.out[0] in {8, 9, 10, 11, 12}  # продаётся полузащитник (квоты состава)
    assert first.expected_gain_horizon > 0 and first.verdict == "go"
    out = route_out(first, 1)
    assert out.in_ == ["P16"] and out.rank == 1


def test_two_forced_buys_with_one_ft_imply_a_hit_and_carry_verdict():
    routes, meta = constrained_routes(toy_inputs(), 5, 3, "balanced", force_in=[16, 17])
    assert meta["cap"] == 2 and routes
    r = routes[0]
    assert set(r.in_) >= {16, 17} and r.hit_cost == 4
    assert r.hit_marginal_gain is not None and r.verdict in {"go", "hit_not_worth"}
    assert r.worth_hit == (r.verdict == "go")


def test_forced_buys_beyond_free_transfers_infeasible_when_hits_rejected():
    with pytest.raises(ScenarioInfeasible):
        constrained_routes(toy_inputs(), 5, 3, "balanced", force_in=[16, 17], allow_hit=False)


def test_forced_sell_is_in_every_route_and_alternatives_differ_by_buys():
    routes, _ = constrained_routes(toy_inputs(free_transfers=2), 5, 3, "balanced", force_out=[8])
    assert routes and all(8 in r.out for r in routes)
    buys = [tuple(sorted(r.in_)) for r in routes]
    assert len(set(buys)) == len(buys)  # no-good cuts: маршруты различаются покупками


def test_allow_hit_reference_is_best_free_route_under_same_constraints():
    routes, meta = constrained_routes(
        toy_inputs(free_transfers=1), 5, 3, "balanced", force_in=[16], allow_hit=True
    )
    assert meta["cap"] == 2 and meta["free_route"] is not None
    assert 16 in meta["free_route"].in_ and meta["free_route"].hit_cost == 0
    for r in routes:
        if r.hit_cost:
            assert r.hit_marginal_gain is not None
