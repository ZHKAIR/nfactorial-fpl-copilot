"""MILP-оптимизатор на игрушечных данных: без БД и сети, каждый solve < 1 с."""

import time

import pytest

from fplcopilot.core.candidates import Candidate, SquadIssue, build_pool, wildcard_available
from fplcopilot.core.optimizer import (
    InfeasibleError,
    best_xi,
    make_solver,
    plan_transfers,
    single_transfer,
)
from fplcopilot.core.strategy import FORMATION_BOUNDS, POSITION_QUOTA
from fplcopilot.data.schemas import Bootstrap

GWS = [5, 6, 7]


def cand(
    pid: int,
    pos: int,
    team: int,
    price: float,
    xp: float | dict[int, float],
    *,
    var: float = 5.0,
    own: float = 10.0,
    status: str = "a",
) -> Candidate:
    xpts = {g: xp for g in GWS} if isinstance(xp, (int, float)) else dict(xp)
    return Candidate(
        player_id=pid,
        name=f"P{pid}",
        position=pos,
        team_id=team,
        price=price,
        xpts_by_gw=xpts,
        variance_by_gw={g: var for g in GWS},
        ownership=own,
        status=status,
    )


def base_squad() -> list[Candidate]:
    """2 GK / 5 DEF / 5 MID / 3 FWD, все из разных клубов, £98.5 суммарно."""
    return [
        cand(1, 1, 1, 5.0, 4.0),
        cand(2, 1, 2, 4.0, 2.0),
        cand(3, 2, 3, 6.0, 4.5),
        cand(4, 2, 4, 5.0, 4.0),
        cand(5, 2, 5, 4.5, 3.5),
        cand(6, 2, 6, 4.0, 2.0),
        cand(7, 2, 7, 4.0, 1.5),
        cand(8, 3, 8, 12.0, 7.0, var=16.0),
        cand(9, 3, 9, 8.0, 5.5),
        cand(10, 3, 10, 7.0, 5.0),
        cand(11, 3, 11, 6.0, 4.0),
        cand(12, 3, 12, 4.5, 2.0),
        cand(13, 4, 13, 15.0, 7.5, var=18.0),
        cand(14, 4, 14, 7.0, 4.5),
        cand(15, 4, 15, 5.0, 1.0),
    ]


def as_pool(*groups: list[Candidate]) -> dict[int, Candidate]:
    return {c.player_id: c for g in groups for c in g}


def ids(cands: list[Candidate]) -> list[int]:
    return [c.player_id for c in cands]


def check_formation(pool: dict[int, Candidate], starters: list[int]) -> None:
    assert len(starters) == 11
    counts = {pos: sum(pool[p].position == pos for p in starters) for pos in POSITION_QUOTA}
    for pos, (lo, hi) in FORMATION_BOUNDS.items():
        assert lo <= counts[pos] <= hi, (pos, counts)


# ---------- солвер ----------


def test_solver_available_and_fast():
    solver, name = make_solver(time_limit=5)
    assert name in {"HiGHS", "CBC"}
    assert solver.available()


# ---------- best_xi ----------


def test_best_xi_respects_formation_captain_and_bench_order():
    squad = base_squad()
    pool = as_pool(squad)
    t = time.perf_counter()
    lu = best_xi(squad, 5, "balanced")
    assert time.perf_counter() - t < 1.0
    check_formation(pool, lu.starters)
    assert (
        lu.formation == "3-5-2"
    )  # 5 полузащитников (7, 5.5, 5, 4, 2) сильнее 4-го защитника (2.0)
    assert lu.captain == 13  # максимальный xPts среди стартеров
    assert lu.vice == 8 and lu.vice != lu.captain
    assert lu.bench_order[0] == 2  # слот 12 — запасной вратарь
    bench_out = lu.bench_order[1:]
    xp = [pool[p].xpts(5) for p in bench_out]
    assert xp == sorted(xp, reverse=True) and set(bench_out) == {6, 7, 15}
    assert lu.expected_points == pytest.approx(
        sum(pool[p].xpts(5) for p in lu.starters) + pool[13].xpts(5)
    )
    assert set(lu.starters) | set(lu.bench_order) == set(ids(squad))


def test_best_xi_variance_penalty_changes_pick_between_presets():
    """Два равных по xPts полузащитника: консервативная берёт спокойного, агрессивная — с дисперсией."""
    squad = base_squad()
    squad[11] = cand(12, 3, 12, 4.5, 4.0, var=1.0)  # спокойный, xPts 4.0
    squad[10] = cand(11, 3, 11, 6.0, 4.0, var=30.0)  # лотерея, xPts 4.0
    squad[6] = cand(7, 2, 7, 4.0, 4.0, var=5.0)  # ещё два защитника по 4.0/4.3: ровно один из
    squad[4] = cand(5, 2, 5, 4.5, 4.3, var=5.0)  # четырёх игроков с xPts 4.0 сядет на скамейку
    pool = as_pool(squad)
    cons = best_xi(squad, 5, "conservative")
    aggr = best_xi(squad, 5, "aggressive")
    check_formation(pool, cons.starters)
    check_formation(pool, aggr.starters)
    assert 12 in cons.starters and 11 not in cons.starters
    assert 11 in aggr.starters and 12 not in aggr.starters
    assert cons.expected_points == pytest.approx(aggr.expected_points)  # чистые xPts равны


def test_best_xi_rejects_malformed_squad():
    with pytest.raises(ValueError):
        best_xi(base_squad()[:14], 5)
    squad = base_squad()
    squad[1] = cand(2, 2, 2, 4.0, 2.0)  # 1 вратарь и 6 защитников — квоты нарушены
    with pytest.raises(InfeasibleError):
        best_xi(squad, 5)


# ---------- single_transfer ----------


def test_single_transfer_rejects_unaffordable_and_respects_club_limit_keep_exclude():
    squad = base_squad()
    # три игрока клуба 3: 3 (DEF) + два лучших MID переведём в клуб 3
    squad[8] = cand(9, 3, 3, 8.0, 5.5)
    squad[9] = cand(10, 3, 3, 7.0, 5.0)
    extra = [
        cand(30, 4, 30, 20.0, 12.0),  # мечта, но £20 при банке 0.5 и продаже максимум £15 -> нельзя
        cand(31, 4, 3, 5.5, 6.0),  # 4-й из клуба 3: можно только продав одноклубника
        cand(32, 2, 32, 4.5, 4.2),
        cand(33, 3, 33, 5.0, 4.6),
        cand(34, 4, 34, 5.5, 4.8),
    ]
    pool = as_pool(squad, extra)
    routes = single_transfer(ids(squad), 0.5, 1, pool, 5, 3, "balanced", top=5)
    assert routes
    for r in routes:
        assert 30 not in r.in_
        assert r.hit_cost == 0 and len(r.in_) == 1  # FT = 1 и без allow_hit -> ровно один трансфер
        squad_after = (set(ids(squad)) - set(r.out)) | set(r.in_)
        assert sum(pool[p].team_id == 3 for p in squad_after) <= 3
        assert sum(pool[p].price for p in squad_after) <= sum(c.price for c in squad) + 0.5 + 1e-9
        if 31 in r.in_:
            assert pool[r.out[0]].team_id == 3
    ins = [tuple(r.in_) for r in routes]
    assert len(set(ins)) == len(ins)  # маршруты различаются покупками

    kept = single_transfer(ids(squad), 0.5, 1, pool, 5, 3, "balanced", keep=[15, 7], top=3)
    assert all(15 not in r.out and 7 not in r.out for r in kept)
    excluded = single_transfer(ids(squad), 0.5, 1, pool, 5, 3, "balanced", exclude=[34, 33], top=3)
    assert all(34 not in r.in_ and 33 not in r.in_ for r in excluded)


def test_single_transfer_hit_only_when_gain_clears_threshold():
    squad = base_squad()
    # базовые 11 замыкает игрок с 2.0; 40 (7.0) даёт +5.0/тур бесплатно, 41 (6.5 за £4.0) ещё
    # +3.0/тур вместо защитника 3.5 -> за хит: 3.0 · 2.55 − 4 = +3.6 >= порога 1.0
    extra = [cand(40, 4, 40, 5.0, 7.0), cand(41, 2, 41, 4.0, 6.5)]
    pool = as_pool(squad, extra)
    no_hit = single_transfer(ids(squad), 0.0, 1, pool, 5, 3, "balanced")
    assert all(r.hit_cost == 0 and len(r.in_) == 1 for r in no_hit)
    with_hit = single_transfer(ids(squad), 0.0, 1, pool, 5, 3, "balanced", allow_hit=True)
    best = with_hit[0]
    assert best.hit_cost == 4 and set(best.in_) == {40, 41}
    assert best.worth_hit is True and best.verdict == "go"
    assert best.hit_marginal_gain is not None and best.hit_marginal_gain >= 1.0
    assert best.hit_marginal_gain == pytest.approx(3.0 * (1 + 0.84 + 0.84**2) - 4, abs=0.05)
    # второй апгрейд +0.5/тур (4.0 вместо 3.5): 0.5 · 2.55 − 4 < 0 -> хит не оправдан
    pool_small = as_pool(squad, [cand(40, 4, 40, 5.0, 7.0), cand(42, 2, 42, 4.0, 4.0)])
    routes = single_transfer(ids(squad), 0.0, 1, pool_small, 5, 3, "balanced", allow_hit=True)
    assert routes[0].hit_cost == 0 and routes[0].in_ == [40]  # оптимум — один бесплатный трансфер
    hit_routes = [r for r in routes if r.hit_cost > 0]
    assert hit_routes, "среди top-3 должен быть маршрут с хитом (для вердикта)"
    assert all(r.verdict == "hit_not_worth" and r.worth_hit is False for r in hit_routes)
    assert all(r.hit_marginal_gain < 0 for r in hit_routes)
    # порог стратегии: +2.1/тур сверх бесплатного даёт 2.1 · 2.55 − 4 = 1.35 -> balanced (1.0)
    # принимает хит, conservative (2.0) отвергает
    pool_mid = as_pool(squad, [cand(40, 4, 40, 5.0, 7.0), cand(43, 2, 43, 4.0, 5.6)])
    bal = single_transfer(ids(squad), 0.0, 1, pool_mid, 5, 3, "balanced", allow_hit=True)
    con = single_transfer(ids(squad), 0.0, 1, pool_mid, 5, 3, "conservative", allow_hit=True)
    bal_hit = next(r for r in bal if r.hit_cost > 0)
    con_hit = next(r for r in con if r.hit_cost > 0)
    assert bal_hit.verdict == "go" and con_hit.verdict == "hit_not_worth"


def test_single_transfer_hold_verdict_when_no_upgrade_beats_ft_value():
    squad = base_squad()
    pool = as_pool(squad, [cand(50, 4, 50, 5.0, 1.2)])  # апгрейд на +0.2/тур не стоит FT (1.5)
    routes = single_transfer(ids(squad), 0.0, 1, pool, 5, 3, "balanced")
    assert routes and routes[0].verdict == "hold" and routes[0].objective_gain < 0


# ---------- plan_transfers ----------


def test_plan_rolls_free_transfers_and_keeps_continuity():
    squad = base_squad()
    pool = as_pool(squad, [cand(60, 4, 60, 5.0, 1.1)])  # ничего стоящего в пуле
    t = time.perf_counter()
    plan = plan_transfers(ids(squad), 0.0, 1, [], 5, 3, "balanced", pool=pool)
    assert plan.runtime_s > 0 and time.perf_counter() - t < 3.0
    assert plan.recommendation == "hold"
    assert all(not moves for moves in plan.moves_by_gw.values())
    assert plan.ft_by_gw == {5: 1, 6: 2, 7: 3, 8: 4}  # FT копятся: 1 -> 2 -> 3 -> 4 после горизонта
    assert plan.expected_total == pytest.approx(plan.baseline_total)
    assert plan.target_squad == sorted(ids(squad))
    assert plan.wildcard_alternative is None


def test_plan_continuity_and_ft_after_transfer():
    squad = base_squad()
    extra = [cand(61, 4, 61, 5.0, 4.8), cand(62, 2, 62, 4.0, 4.2)]
    pool = as_pool(squad, extra)
    plan = plan_transfers(ids(squad), 0.0, 1, [], 5, 3, "balanced", pool=pool)
    assert plan.recommendation == "transfers"
    current = set(ids(squad))
    for gw in GWS:
        moves = plan.moves_by_gw[gw]
        current = (current - {m.out for m in moves}) | {m.in_ for m in moves}
        lu = plan.lineups_by_gw[gw]
        assert set(lu.starters) | set(lu.bench_order) == current
        assert len(current) == 15
    assert current == set(plan.target_squad)
    # два апгрейда по одному FT: GW5 и GW6, без хитов; в GW7 FT снова 1, после горизонта 2
    assert sum(len(m) for m in plan.moves_by_gw.values()) == 2
    assert all(h == 0 for h in plan.hits_by_gw.values())
    assert plan.ft_by_gw[5] == 1 and plan.ft_by_gw[8] == 2
    assert plan.expected_total > plan.baseline_total


def squads_by_gw(plan) -> dict[int, set[int]]:
    return {gw: set(lu.starters) | set(lu.bench_order) for gw, lu in plan.lineups_by_gw.items()}


def test_plan_sells_injured_player_depending_on_return_gw():
    """Травмированный премиум (8.0 xPts после возвращения) при 1 FT/тур: с return_gw=7 его нет в
    составе GW5–6 (продан сразу), с return_gw=6 он в составе в GW6 и GW7 (модель его не теряет)."""
    squad = base_squad()
    replacement = cand(70, 3, 70, 12.0, 5.0)
    late = squad.copy()
    late[7] = cand(8, 3, 8, 12.0, {5: 0.0, 6: 0.0, 7: 8.0}, status="i")
    plan = plan_transfers(
        ids(late), 0.0, 1, [], 5, 3, "balanced", pool=as_pool(late, [replacement])
    )
    assert [m.out for m in plan.moves_by_gw[5]] == [8]
    assert plan.moves_by_gw[5][0].in_ == 70 and plan.moves_by_gw[5][0].out_problem is None
    squads = squads_by_gw(plan)
    assert 8 not in squads[5] and 8 not in squads[6]
    early = squad.copy()
    early[7] = cand(8, 3, 8, 12.0, {5: 0.0, 6: 8.0, 7: 8.0}, status="i")
    plan2 = plan_transfers(
        ids(early), 0.0, 1, [], 5, 3, "balanced", pool=as_pool(early, [replacement])
    )
    squads2 = squads_by_gw(plan2)
    assert 8 in squads2[6] and 8 in squads2[7]
    assert 8 in plan2.lineups_by_gw[6].starters and plan2.lineups_by_gw[6].captain == 8
    # проблема из diagnose попадает в rationale хода
    issues = [SquadIssue(player_id=8, name="P8", kind="injured", severity=3, detail="i", gw=5)]
    plan3 = plan_transfers(
        ids(late), 0.0, 1, [], 5, 3, "balanced", pool=as_pool(late, [replacement]), issues=issues
    )
    assert plan3.moves_by_gw[5][0].out_problem == "injured"


def test_plan_allow_hits_false_forbids_paid_transfers_in_every_gw():
    """Два апгрейда (+8 и +6 xPts/тур) при 1 FT: с хитами оптимум берёт оба в GW5 (−4 окупается:
    10 + 14·0.84 + 14·0.84² против 8 + 14·0.84 + 14·0.84²); с allow_hits=False ни одного платного
    трансфера ни в одном туре — ходы раскладываются по FT (GW5 и GW6)."""
    squad = base_squad()
    extra = [cand(40, 4, 40, 5.0, 9.0), cand(41, 2, 41, 4.0, 8.0)]
    pool = as_pool(squad, extra)
    with_hits = plan_transfers(ids(squad), 0.0, 1, [], 5, 3, "balanced", pool=pool)
    assert with_hits.allow_hits is True
    assert with_hits.hits_by_gw[5] == 4 and len(with_hits.moves_by_gw[5]) == 2
    assert sum(m.paid for m in with_hits.moves_by_gw[5]) == 1

    no_hits = plan_transfers(ids(squad), 0.0, 1, [], 5, 3, "balanced", pool=pool, allow_hits=False)
    assert no_hits.allow_hits is False
    assert all(h == 0 for h in no_hits.hits_by_gw.values())
    assert all(not m.paid for ms in no_hits.moves_by_gw.values() for m in ms)
    # оба апгрейда всё равно сделаны, но по одному в тур — в пределах FT
    assert [len(no_hits.moves_by_gw[g]) for g in GWS] == [1, 1, 0]
    assert {m.in_ for ms in no_hits.moves_by_gw.values() for m in ms} == {40, 41}
    assert no_hits.moves_by_gw[5][0].in_ == 40  # сначала больший апгрейд (+8/тур)
    assert no_hits.ft_by_gw == {5: 1, 6: 1, 7: 1, 8: 2}
    # запрет — сужение допустимого множества: цель не выше, чем с хитами, но план всё ещё
    # лучше «ничего не делать»
    assert no_hits.objective <= with_hits.objective
    assert no_hits.expected_total <= with_hits.expected_total
    assert no_hits.expected_total > no_hits.baseline_total
    assert set(no_hits.target_squad) == set(with_hits.target_squad)
    # WC-альтернатива тоже строится без хитов после тура wildcard
    with_wc = plan_transfers(
        ids(squad), 0.0, 1, ["wildcard"], 5, 3, "balanced", pool=pool, allow_hits=False
    )
    assert with_wc.wildcard_alternative is not None
    assert all(h == 0 for h in with_wc.hits_by_gw.values())


def test_plan_prefers_wildcard_when_squad_is_dead():
    squad = base_squad()
    dead = {5, 6, 7, 12}
    for i, c in enumerate(squad):
        if c.player_id in dead:
            squad[i] = cand(c.player_id, c.position, c.team_id, c.price, 0.0, status="i")
    extra = [cand(80 + k, 2, 80 + k, 4.5, 4.0) for k in range(4)] + [
        cand(90 + k, 3, 90 + k, 5.0, 4.5) for k in range(3)
    ]
    pool = as_pool(squad, extra)
    issues = [
        SquadIssue(player_id=p, name=f"P{p}", kind="injured", severity=3, detail="i", gw=5)
        for p in dead
    ]
    plan = plan_transfers(
        ids(squad), 0.0, 1, ["wildcard"], 5, 3, "balanced", pool=pool, issues=issues
    )
    assert plan.wildcard_alternative is not None
    wc = plan.wildcard_alternative
    assert wc.gw == 5 and not (set(wc.squad) & dead)
    assert wc.expected_total - plan.expected_total > 2.0
    assert plan.recommendation == "wildcard"
    assert plan.solver in {"HiGHS", "CBC"} and plan.runtime_s > 0
    # без WC в наличии — обычный план трансферов
    no_wc = plan_transfers(ids(squad), 0.0, 1, [], 5, 3, "balanced", pool=pool, issues=issues)
    assert no_wc.wildcard_alternative is None and no_wc.recommendation == "transfers"


# ---------- пул и чипы ----------


def test_build_pool_keeps_squad_prunes_unavailable_and_covers_positions():
    squad = base_squad()
    cands = as_pool(squad)
    for k in range(60):
        pos = 1 + k % 4
        cands[100 + k] = cand(100 + k, pos, 20 + k % 10, 4.0 + (k % 12) * 0.5, 6.0 - k * 0.05)
    cands[200] = cand(200, 4, 21, 5.0, 9.0, status="i")  # травмирован -> не покупаем
    cands[201] = cand(201, 4, 22, 15.6, 9.0)  # дороже банка + самого дорогого (15.0 + 0.5)
    pool = build_pool(cands, squad_ids=ids(squad), gws=GWS, size=20, bank=0.5, exclude=[100])
    assert set(ids(squad)) <= set(pool)
    assert 200 not in pool and 201 not in pool and 100 not in pool
    assert all(
        any(c.position == pos for c in pool.values() if not c.in_squad) for pos in (1, 2, 3, 4)
    )
    assert len(pool) <= 15 + 20 + 4 * 6


def test_wildcard_available_by_window_and_usage():
    bs = Bootstrap.model_validate(
        {
            "events": [],
            "teams": [],
            "elements": [],
            "chips": [
                {"name": "wildcard", "start_event": 2, "stop_event": 19},
                {"name": "wildcard", "start_event": 20, "stop_event": 38},
                {"name": "freehit", "start_event": 2, "stop_event": 19},
            ],
        }
    )
    assert wildcard_available(bs, [], 5)
    assert not wildcard_available(bs, ["wildcard"], 5)
    assert not wildcard_available(bs, ["bboost", "wildcard"], 19)
    assert wildcard_available(bs, ["wildcard"], 20)  # второй WC после GW19
    assert not wildcard_available(bs, ["wildcard", "wildcard"], 25)
    assert not wildcard_available(bs, [], 1)  # до окна
