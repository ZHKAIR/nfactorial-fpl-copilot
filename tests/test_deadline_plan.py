"""«План на тур» (app/deadline_plan.py) и цены продажи FPL (data/prices.py) — без сети и LLM."""

from __future__ import annotations

from datetime import timedelta

from test_agent_fakes import BS, DEADLINE, NOW, route

from fplcopilot.agent.tools import (
    GameweekContext,
    GWPrediction,
    LineupOut,
    LineupPlayer,
    PlayerPrediction,
    PlayerRef,
    RouteLineup,
    RoutesOut,
    SquadPlayerRow,
)
from fplcopilot.app import deadline_plan, ui_kit
from fplcopilot.app import format as fmt
from fplcopilot.core.assets import AssetFlag
from fplcopilot.data.prices import purchase_prices, selling_price
from fplcopilot.data.schemas import SalePrice, TransferRow

# Фейковый bootstrap: 301 Palmer (GKP), 4 Gabriel / 388 Guéhi / 331 Gudmundsson / 611 Wan-Bissaka
# (DEF), 154 Palmer / 12 Saka (MID), 411 Haaland / 165 João Pedro (FWD, под вопросом 75 %) / 27.
BASE_XI = [301, 4, 388, 154, 411, 165]
BASE_BENCH = [331, 611]


def lp(pid: int, xpts: float) -> LineupPlayer:
    p = BS.player(pid)
    return LineupPlayer(
        id=pid,
        name=p.web_name,
        team="ARS",
        position=p.position.short,
        price=p.price,
        xpts=xpts,
        sd=2.0,
        p_start=0.9,
        ownership=10.0,
        fixture="vTOT (FSI 2)",
    )


def base_lineup() -> LineupOut:
    xp = {301: 3.0, 4: 4.6, 388: 5.9, 154: 4.8, 411: 7.1, 165: 3.49, 331: 2.0, 611: 1.5}
    return LineupOut(
        gw=5,
        formation="2-1-2",
        starters=[lp(p, xp[p]) for p in BASE_XI],
        bench=[lp(p, xp[p]) for p in BASE_BENCH],
        captain="Haaland",
        vice="Palmer",
        expected_points=40.0,
        captain_options=[],
    )


def context(captain: int = 154) -> GameweekContext:
    rows = []
    for pid in BASE_XI + BASE_BENCH:
        p = BS.player(pid)
        rows.append(
            SquadPlayerRow(
                id=pid,
                name=p.web_name,
                team="ARS",
                position=p.position.short,
                price=p.price,
                status=p.status,
                chance=p.chance_of_playing_next_round,
                news="Knee injury - 75% chance of playing" if pid == 165 else "",
                is_starting=pid in BASE_XI,
                is_captain=pid == captain,
            )
        )
    return GameweekContext(gw=5, current_gw=4, deadline=DEADLINE, as_of=NOW, squad=rows)


def pred(pid: int, xpts: float) -> PlayerPrediction:
    p = BS.player(pid)
    ref = PlayerRef(id=pid, name=p.web_name, team="ARS", position=p.position.short, price=p.price)
    gp = GWPrediction(gw=5, xpts=xpts, sd=2.0, p_start=0.95, exp_minutes=80, components={}, fixtures=[])
    return PlayerPrediction(player=ref, by_gw=[gp], total_xpts=xpts * 3)


def routes() -> RoutesOut:
    """Вариант 1: бенч-игрок Wan-Bissaka → Saka, Saka встаёт в старт вместо João Pedro;
    вариант 2: продать самого João Pedro."""
    r1 = route(1, ["Wan-Bissaka"], ["Saka"]).model_copy(
        update={
            "out_ids": [611],
            "in_ids": [12],
            "gain_next_gw": 0.87,
            "gain_horizon": 3.7,
            "new_bank": 0.0,
            "lineup_after": RouteLineup(
                formation="2-2-1",
                starter_ids=[301, 4, 388, 154, 12, 411],
                bench_ids=[165, 331],
                captain_id=411,
                vice_id=154,
                expected_points=41.0,
            ),
        }
    )
    r2 = route(2, ["João Pedro"], ["G.Jesus"]).model_copy(
        update={"out_ids": [165], "in_ids": [27], "gain_horizon": 2.9}
    )
    return RoutesOut(
        gw=5,
        horizon=3,
        strategy="balanced",
        free_transfers=1,
        bank=0.0,
        routes=[r1, r2],
        recommended_rank=1,
        recommendation="route 1",
        baseline_xi_points=40.0,
    )


def test_selling_price_and_purchase_prices():
    assert selling_price(75, 77) == 76  # +0.2 -> половина прироста
    assert selling_price(75, 78) == 76  # +0.3 -> половина вниз
    assert selling_price(75, 73) == 73  # упал — по текущей
    t0 = NOW - timedelta(days=20)
    transfers = [
        TransferRow(element_in=12, element_in_cost=90, element_out=18, element_out_cost=65, event=2, time=t0),
        TransferRow(element_in=12, element_in_cost=95, element_out=27, element_out_cost=68, event=4, time=t0 + timedelta(days=10)),
        TransferRow(element_in=411, element_in_cost=160, element_out=27, element_out_cost=68, event=3, time=t0 + timedelta(days=5)),
    ]
    got = purchase_prices(
        [12, 165, 411, 999], transfers, {165: 75, 411: 150, 999: None}, freehit_events=[3]
    )
    assert got == {12: 95, 165: 75, 411: 150}  # последняя покупка; Free Hit не в счёт; без цены — нет


def test_plan_explains_bench_sale_and_displaced_player():
    sale = {165: SalePrice(purchase=7.5, selling=7.6, now=7.8)}
    plan = deadline_plan.build_plan(
        base_lineup(), routes(), context(), BS, {12: pred(12, 4.36)}, sale, horizon=3
    )
    assert plan.title == "Wan-Bissaka → Saka, João Pedro — на скамейку"
    assert plan.gain == ("+3.7", "очка за 3 тура · +0.9 очка в GW5")
    first, second, *rest = plan.steps
    assert first.title == "Трансфер: Wan-Bissaka → Saka"
    assert "Бесплатный, в банке останется £0.0." in first.body
    assert "Wan-Bissaka сидит на скамейке" in first.body and "Saka сразу выходит в старт" in first.body
    assert second.title == "João Pedro — на скамейку, не продаём" and second.tone == "warn"
    assert "Под вопросом (75 %), колено." in second.body
    assert "у Saka 4.4 очка, у João Pedro — 3.5 очка" in second.body
    assert "(João Pedro → G.Jesus) менее выгодно: +2.9 очка за 3 тура против +3.7 очка" in second.body
    assert "только за £7.6, а выкупить обратно — уже за £7.8" in second.body
    assert "подстраховка" in second.body
    schema = rest[-1]
    assert schema.title == "Схема 2-2-1"  # нынешняя схема — только при полных 11 в старте
    assert schema.body == "Капитан — Haaland, вице — Palmer. Сейчас капитан Palmer — поменяйте."
    lines = dict(plan.lines)
    assert [(p.name, p.mark) for p in lines["Полузащита"]] == [("Palmer", ""), ("Saka", "new")]
    assert [p.role for p in lines["Нападение"]] == ["C"]
    assert [(p.name, p.mark) for p in plan.bench] == [("João Pedro", "out"), ("Gudmundsson", "")]


def test_plan_when_displaced_player_already_on_bench():
    ctx = context()
    swap = {165: False, 331: True}  # João Pedro уже на скамейке, Gudmundsson в старте
    ctx.squad = [
        r.model_copy(update={"is_starting": swap[r.id]}) if r.id in swap else r for r in ctx.squad
    ]
    plan = deadline_plan.build_plan(base_lineup(), routes(), ctx, BS, {}, {}, horizon=3)
    assert plan.title == "Wan-Bissaka → Saka"
    assert plan.steps[1].title == "João Pedro — остаётся на скамейке, не продаём"
    change = next(s for s in plan.steps if s.title == "Поменять старт")
    assert change.body == "На скамейку: Gudmundsson."
    assert [(p.name, p.mark) for p in plan.bench] == [("João Pedro", ""), ("Gudmundsson", "out")]


def test_plan_explains_hold_and_buy_low_assets():
    hold_jp = AssetFlag(
        player_id=165, name="João Pedro", kind="hold", bonus=1.1, fit_xpts=5.37,
        position_median=3.4, top_share=0.94, status="d", chance=75, price=7.8,
    )
    hold_palmer = AssetFlag(
        player_id=154, name="Palmer", kind="hold", bonus=1.3, fit_xpts=5.53,
        position_median=3.2, top_share=0.98, status="d", chance=75, price=9.7,
    )
    buy = AssetFlag(
        player_id=12, name="Saka", kind="buy_low", bonus=0.6, fit_xpts=6.1, position_median=3.2,
        top_share=0.99, status="i", return_gw=6, price=9.5, price_change_start=-0.2,
        transfers_net=-40_000,
    )
    rt = routes().model_copy(update={"assets": [hold_jp, hold_palmer, buy]})
    ctx = context(411)
    ctx.squad = [
        r.model_copy(update={"status": "d", "chance": 75}) if r.id == 154 else r for r in ctx.squad
    ]
    sale = {154: SalePrice(purchase=9.5, selling=9.6, now=9.7)}
    plan = deadline_plan.build_plan(base_lineup(), rt, ctx, BS, {12: pred(12, 4.36)}, sale)
    first, jp, palmer = plan.steps[:3]
    assert (
        "Saka сейчас травма, вернётся к GW6, но ненадолго; его сбрасывают "
        "(чистые продажи — 40 000 за тур), цена уже ниже стартовой на £0.2. Берём, пока дёшево: "
        "здоровым даёт 6.1 очка за тур — больше, чем у 99 % игроков позиции." in first.body
    )
    assert "Но продавать не стоит: здоровым он даёт 5.4 очка за тур — больше, чем у 94 %" in jp.body
    assert "скорее всего, снова в старте и подорожает" in jp.body
    assert "подстраховка" not in jp.body and "менее выгодно" in jp.body
    assert palmer.title == "Не продаём: Palmer" and palmer.tone == "warn"
    assert palmer.body.startswith("Palmer под вопросом (75 %), но здоровым даёт 5.5 очка")
    assert "продать можно только за £9.6, а выкупить обратно — уже за £9.7" in palmer.body


def test_plan_skips_price_line_when_no_loss_and_hold_case():
    sale = {165: SalePrice(purchase=7.8, selling=7.8, now=7.8)}
    plan = deadline_plan.build_plan(base_lineup(), routes(), context(411), BS, {}, sale)
    body = plan.steps[1].body
    assert "выкупить" not in body and "Прогноз на GW" not in body  # нет прогноза покупки
    assert plan.steps[-1].body == "Капитан — Haaland, вице — Palmer."
    hold = routes().model_copy(update={"recommended_rank": None, "recommendation": "hold"})
    plan = deadline_plan.build_plan(base_lineup(), hold, context(411), BS)
    assert plan.title == "Без трансфера" and plan.gain == ("40.0", "очков — прогноз состава")
    assert plan.steps[0].title == "Трансфер не делаем"
    assert "перейдёт на следующий тур" in plan.steps[0].body
    assert [p.name for p in plan.bench] == ["Gudmundsson", "Wan-Bissaka"]


def test_plan_html_marks_and_links():
    plan = deadline_plan.build_plan(
        base_lineup(), routes(), context(), BS, {12: pred(12, 4.36)}, {}, horizon=3
    )
    html = ui_kit.deadline_plan_html(plan, fmt.PLAYER_URL)
    assert '<b class="n">1</b>' in html and 'class="fpl-plan"' in html
    assert '<a class="chip new" href="/player?pid=12" target="_self">Saka<em>новый</em></a>' in html
    assert 'class="chip out"' in html and "на скамейку</em>" in html
    assert '<b class="role">C</b>' in html and "Состав на тур · 2-2-1" in html
    assert '<div class="gain good"><b>+3.7</b>' in html
