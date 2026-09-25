"""«План на тур» на странице «К дедлайну»: что сделать и почему — из выходов инструментов.

Трансфер берётся из рекомендованного маршрута recommend_transfers, состав после него — из
того же решения оптимизатора (RouteOut.lineup_after); игрок, которого покупка вытесняет из
старта, объясняется фактами: статус FPL, прогноз на тур, вариант с его продажей и цена
продажи против цены обратного выкупа. Нет данных — нет фразы. Без LLM.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from fplcopilot.agent.tools import (
    GameweekContext,
    LineupOut,
    PlayerPrediction,
    RouteOut,
    RoutesOut,
    SquadPlayerRow,
)
from fplcopilot.app import format as fmt
from fplcopilot.app.briefing import injury_ru
from fplcopilot.core.assets import AssetFlag
from fplcopilot.data.schemas import Bootstrap, SalePrice

POSITIONS = (("GKP", "Вратарь"), ("DEF", "Защита"), ("MID", "Полузащита"), ("FWD", "Нападение"))
MAX_FREE_TRANSFERS = 5


@dataclass(frozen=True)
class PlanStep:
    title: str
    body: str
    tone: str = ""  # accent | warn | good


@dataclass(frozen=True)
class PlanPlayer:
    id: int
    name: str
    mark: str = ""  # new — покупка, in — выходит в старт, out — уходит на скамейку
    role: str = ""  # C | V


@dataclass
class DeadlinePlan:
    title: str
    gain: tuple[str, str] | None
    steps: list[PlanStep]
    formation: str
    lines: list[tuple[str, list[PlanPlayer]]] = field(default_factory=list)
    bench: list[PlanPlayer] = field(default_factory=list)


def recommended(rt: RoutesOut | None) -> RouteOut | None:
    if rt is None or rt.recommended_rank is None:
        return None
    return next((r for r in rt.routes if r.rank == rt.recommended_rank), None)


def _formation(ids: list[int], bs: Bootstrap) -> str:
    n = {pos: 0 for pos, _ in POSITIONS}
    for pid in ids:
        n[bs.player(pid).position.short] += 1
    return f"{n['DEF']}-{n['MID']}-{n['FWD']}"


def _is_doubtful(row: SquadPlayerRow | None) -> bool:
    return row is not None and (
        row.status != "a" or (row.chance is not None and row.chance < 100)
    )


def _join(names: list[str]) -> str:
    return ", ".join(names) if names else "—"


def _tours(n: int) -> str:
    return fmt.plural(n, "тур", "тура", "туров")


def _status_phrase(row: SquadPlayerRow | None, asset: AssetFlag | None = None) -> str:
    """«под вопросом (75 %), колено» / «травма, вернётся к GW7» — без точки и заглавной."""
    status = row.status if row is not None else (asset.status if asset else "a")
    chance = row.chance if row is not None else (asset.chance if asset else None)
    text = fmt.status_text(status, chance)
    where = injury_ru(row.news) if row is not None else None
    if where:
        text += f", {where}"
    if asset is not None and asset.return_gw is not None and status != "d":
        text += f", вернётся к GW{asset.return_gw}"
    return text


def _asset_value(a: AssetFlag) -> str:
    return (
        f"здоровым даёт {fmt.points_text(a.fit_xpts)} за тур — больше, чем у "
        f"{round(a.top_share * 100)} % игроков позиции"
    )


def _price_phrase(sp: SalePrice | None) -> str:
    if sp is None or sp.selling >= sp.now - 1e-9:
        return ""
    return f"продать можно только за £{sp.selling:.1f}, а выкупить обратно — уже за £{sp.now:.1f}"


HOLD_TAIL = "травма короткая — после неё он, скорее всего, снова в старте и подорожает"


def _displaced_step(
    pid: int,
    rec: RouteOut,
    rt: RoutesOut,
    lu: LineupOut,
    rows: Mapping[int, SquadPlayerRow],
    preds: Mapping[int, PlayerPrediction],
    sale: Mapping[int, SalePrice],
    starters_after: set[int],
    current_start: set[int],
    bs: Bootstrap,
    gw: int,
    horizon: int,
    asset: AssetFlag | None = None,
) -> PlanStep:
    name = bs.player(pid).web_name
    where_now = "на скамейку" if pid in current_start else "остаётся на скамейке"
    row = rows.get(pid)
    parts: list[str] = []
    if _is_doubtful(row):
        parts.append(_status_phrase(row).capitalize() + ".")
    base = next((p for p in lu.starters if p.id == pid), None)
    newcomer = next(
        (
            (n, preds[i].by_gw[0].xpts)
            for i, n in zip(rec.in_ids, rec.in_, strict=False)
            if i in starters_after and i in preds and preds[i].by_gw
        ),
        None,
    )
    if base is not None and newcomer is not None:
        parts.append(
            f"Прогноз на GW{gw}: у {newcomer[0]} {fmt.points_text(newcomer[1])}, "
            f"у {name} — {fmt.points_text(base.xpts)}."
        )
    alt = next((r for r in rt.routes if pid in r.out_ids and r.rank != rec.rank), None)
    if alt is not None and alt.gain_horizon < rec.gain_horizon:
        parts.append(
            f"Продать его ({name} → {_join(list(alt.in_))}) менее выгодно: "
            f"{fmt.points_text(alt.gain_horizon, signed=True)} за {_tours(horizon)} против "
            f"{fmt.points_text(rec.gain_horizon, signed=True)}."
        )
    if asset is not None and asset.kind == "hold":
        parts.append(
            f"Но продавать не стоит: {_asset_value(asset).replace('даёт', 'он даёт', 1)}; "
            f"{HOLD_TAIL}."
        )
    price = _price_phrase(sale.get(pid))
    if price:
        parts.append(price[0].upper() + price[1:] + ".")
    if asset is None or asset.kind != "hold":
        parts.append(
            "Пусть посидит на скамейке как подстраховка."
            if _is_doubtful(row)
            else "Остаётся в составе на скамейке."
        )
    return PlanStep(f"{name} — {where_now}, не продаём", " ".join(parts), "warn")


def build_plan(
    lu: LineupOut,
    rt: RoutesOut | None,
    ctx: GameweekContext,
    bs: Bootstrap,
    preds: Mapping[int, PlayerPrediction] | None = None,
    sale: Mapping[int, SalePrice] | None = None,
    *,
    horizon: int = 3,
) -> DeadlinePlan:
    preds = preds or {}
    sale = sale or {}
    gw = lu.gw
    rows = {p.id: p for p in ctx.squad or []}
    current_start = {p.id for p in ctx.squad or [] if p.is_starting}
    base_ids = [p.id for p in lu.starters]
    rec = recommended(rt)
    after = rec.lineup_after if rec is not None else None

    if after is not None:
        starters, bench = list(after.starter_ids), list(after.bench_ids)
        cap_id, vice_id, formation = after.captain_id, after.vice_id, after.formation
    else:
        starters, bench = base_ids, [p.id for p in lu.bench]
        by_name = {p.name: p.id for p in lu.starters}
        cap_id, vice_id = by_name.get(lu.captain, 0), by_name.get(lu.vice, 0)
        formation = lu.formation
    starters_set = set(starters)

    def name(pid: int) -> str:
        return bs.player(pid).web_name

    steps: list[PlanStep] = []
    displaced: list[int] = []
    assets = {a.player_id: a for a in (rt.assets if rt is not None else [])}
    if rec is not None:
        outs, ins = _join(list(rec.out)), _join(list(rec.in_))
        cost = f"Платный, −{rec.hit_cost} очка" if rec.hit_cost else "Бесплатный"
        why = ""
        if after is not None:
            sold_on_bench = all(pid not in base_ids for pid in rec.out_ids)
            bought_start = [
                n for i, n in zip(rec.in_ids, rec.in_, strict=False) if i in starters_set
            ]
            if bought_start and sold_on_bench:
                why = (
                    f" {outs} сидит на скамейке — его продажа не стоит очков, "
                    f"а {_join(bought_start)} сразу выходит в старт."
                )
            elif bought_start:
                why = f" {_join(bought_start)} выходит в старт вместо {outs}."
            else:
                why = f" {ins} пока на скамейку — задел на следующие туры."
        for pid in rec.in_ids:
            a = assets.get(pid)
            if a is not None and a.kind == "buy_low":
                net = f"{abs(a.transfers_net):,}".replace(",", " ")
                sold = f"его сбрасывают (чистые продажи — {net} за тур)"
                if a.price_change_start < 0:
                    sold += f", цена уже ниже стартовой на £{-a.price_change_start:.1f}"
                why += (
                    f" {a.name} сейчас {_status_phrase(None, a)}, но ненадолго; {sold}. "
                    f"Берём, пока дёшево: {_asset_value(a)}."
                )
        steps.append(
            PlanStep(
                f"Трансфер: {outs} → {ins}",
                f"{cost}, в банке останется £{rec.new_bank:.1f}.{why}",
                "accent",
            )
        )
        if after is not None:
            assert rt is not None
            displaced = [
                pid for pid in base_ids if pid not in starters_set and pid not in rec.out_ids
            ]
            for pid in displaced:
                steps.append(
                    _displaced_step(
                        pid,
                        rec,
                        rt,
                        lu,
                        rows,
                        preds,
                        sale,
                        starters_set,
                        current_start,
                        bs,
                        gw,
                        horizon,
                        assets.get(pid),
                    )
                )
    elif rt is not None:
        ft = int(rt.free_transfers or 0)
        body = "Выгодного варианта нет — бесплатный трансфер перейдёт на следующий тур."
        if ft >= MAX_FREE_TRANSFERS:
            body = (
                f"Выгодного варианта нет. Бесплатных трансферов уже {ft} — больше не копится, "
                "так что этот можно потратить без потерь."
            )
        steps.append(PlanStep("Трансфер не делаем", body, "accent"))

    sold_ids = set(rec.out_ids) if rec is not None else set()
    holds = [
        a
        for a in assets.values()
        if a.kind == "hold" and a.player_id in rows and a.player_id not in sold_ids
        and a.player_id not in displaced
    ]
    if holds:
        parts = []
        for a in holds:
            price = _price_phrase(sale.get(a.player_id))
            parts.append(
                f"{a.name} {_status_phrase(rows.get(a.player_id), a)}, но {_asset_value(a)}"
                + (f"; {price}" if price else "")
                + "."
            )
        tail = (
            "Травма короткая — после неё он, скорее всего, снова в старте и подорожает."
            if len(holds) == 1
            else "Травмы короткие — после них они, скорее всего, снова в старте и подорожают."
        )
        steps.append(
            PlanStep(
                f"Не продаём: {_join([a.name for a in holds])}",
                " ".join(parts) + " " + tail,
                "warn",
            )
        )

    known_xi = rec is None or after is not None
    bought = set(rec.in_ids) if rec is not None else set()
    moved_in = [name(pid) for pid in starters if pid in rows and pid not in current_start]
    moved_out = [name(pid) for pid in bench if pid in current_start and pid not in displaced]
    if known_xi and (moved_in or moved_out):
        parts = []
        if moved_in:
            parts.append(f"В старт: {_join(moved_in)}.")
        if moved_out:
            parts.append(f"На скамейку: {_join(moved_out)}.")
        steps.append(PlanStep("Поменять старт", " ".join(parts)))

    current_ids = [pid for pid in current_start if pid in rows]
    current_form = _formation(current_ids, bs) if len(current_ids) == 11 else ""
    cap, vice = (name(cap_id) if cap_id else lu.captain), (name(vice_id) if vice_id else lu.vice)
    cur_cap = next((p.name for p in ctx.squad or [] if p.is_captain), None)
    title = f"Схема {formation}" + (
        f" вместо {current_form}" if current_form and current_form != formation else ""
    )
    body = f"Капитан — {cap}, вице — {vice}."
    if cur_cap and cur_cap != cap:
        body += f" Сейчас капитан {cur_cap} — поменяйте."
    if known_xi:
        steps.append(PlanStep(title, body))

    if rec is not None:
        number, word = fmt.points_text(rec.gain_horizon, signed=True).split(" ", 1)
        gain: tuple[str, str] | None = (
            number,
            f"{word} за {_tours(horizon)} · {fmt.points_text(rec.gain_next_gw, signed=True)} в GW{gw}",
        )
        head = f"{_join(list(rec.out))} → {_join(list(rec.in_))}"
        benched = [name(p) for p in displaced if p in current_start]
        if benched:
            head += f", {_join(benched)} — на скамейку"
    else:
        gain = (f"{lu.expected_points:.1f}", "очков — прогноз состава")
        head = "Без трансфера"

    def player(pid: int, on_bench: bool) -> PlanPlayer:
        if pid in bought:
            mark = "new"
        elif not on_bench and pid in rows and pid not in current_start:
            mark = "in"
        elif on_bench and pid in current_start:
            mark = "out"
        else:
            mark = ""
        role = "C" if pid == cap_id else "V" if pid == vice_id else ""
        return PlanPlayer(pid, name(pid), mark, role)

    lines = []
    for pos, label in POSITIONS:
        group = [player(pid, False) for pid in starters if bs.player(pid).position.short == pos]
        if group and known_xi:
            lines.append((label, group))
    return DeadlinePlan(
        title=head,
        gain=gain,
        steps=steps,
        formation=formation,
        lines=lines,
        bench=[player(pid, True) for pid in bench] if known_xi else [],
    )
