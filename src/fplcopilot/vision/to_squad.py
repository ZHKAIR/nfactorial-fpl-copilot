"""ParsedSquad -> fplcopilot.data.schemas.Squad — тот же объект, что отдаёт FPLClient.squad(),
поэтому оптимизатор и агент потребляют состав со скриншота без изменений.

Правила:
- picks 1..11 — старт в порядке FPL (GKP, DEF, MID, FWD; внутри позиции — порядок на экране),
  12..15 — скамейка в порядке bench_order (12 — запасной вратарь);
- скамейка не распознана (экран трансферов) -> choose_default_lineup: запасной вратарь — более
  дешёвый GKP, на скамейку уходят самые дешёвые полевые, пока схема остаётся валидной (3-2-1 минимум);
- multiplier: 2 капитан, 1 старт, 0 скамейка. Капитан на скамейке/отсутствует -> без капитана
  (оптимизатор выберет сам); вице никогда не совпадает с капитаном;
- bank — с экрана (иначе 0.0); team_value — сумма цен (с экрана, иначе now_cost) + bank;
- gw — parsed.gw (ближайший непрошедший дедлайн); manager_id — 0, если не задан.
"""

from __future__ import annotations

from collections import Counter

from fplcopilot.data.schemas import Bootstrap, Pick, Position, Squad, SquadPlayer
from fplcopilot.vision.schemas import FORMATION_LIMITS, POSITION_ORDER, ParsedSquad, ResolvedPlayer

POSITION_RANK = {pos: i for i, pos in enumerate(POSITION_ORDER)}
BENCH_OUTFIELD = 3


def _price(p: ResolvedPlayer, bs: Bootstrap) -> float:
    if p.price is not None:
        return p.price
    return bs.player(p.player_id).price  # type: ignore[arg-type]


def choose_default_lineup(
    players: list[ResolvedPlayer], bs: Bootstrap
) -> tuple[list[ResolvedPlayer], list[ResolvedPlayer]]:
    """Детерминированный старт, когда скамейка на экране не показана.

    Запасной вратарь — более дешёвый GKP; далее на скамейку — самые дешёвые полевые, пока
    в старте остаётся не меньше минимума схемы по позиции (DEF 3, MID 2, FWD 1). Скамейка:
    вратарь, затем полевые по убыванию цены (более сильный — первый на замену).
    """
    gks = [p for p in players if p.position == "GKP"]
    bench_gk = sorted(gks, key=lambda p: _price(p, bs))[:1] if len(gks) > 1 else []
    outfield = [p for p in players if p.position != "GKP"]
    counts = Counter(p.position for p in outfield)
    bench_out: list[ResolvedPlayer] = []
    for p in sorted(outfield, key=lambda p: _price(p, bs)):
        if len(bench_out) == BENCH_OUTFIELD:
            break
        lo = FORMATION_LIMITS.get(p.position or "", (0, 0))[0]
        if counts[p.position] - 1 >= lo:
            bench_out.append(p)
            counts[p.position] -= 1
    benched = {id(p) for p in (*bench_gk, *bench_out)}
    starters = [p for p in players if id(p) not in benched]
    bench = [*bench_gk, *sorted(bench_out, key=lambda p: -_price(p, bs))]
    return starters, bench


def to_squad(
    parsed: ParsedSquad,
    bs: Bootstrap,
    *,
    manager_id: int = 0,
    gw: int | None = None,
    allow_invalid: bool = False,
) -> Squad:
    """Валидный ParsedSquad -> Squad. С blocking-нарушениями — ValueError, если не allow_invalid."""
    if not parsed.is_valid and not allow_invalid:
        details = "; ".join(str(i) for i in parsed.blocking_issues)
        raise ValueError(f"parsed squad has blocking issues: {details}")

    seen: set[int] = set()
    players: list[ResolvedPlayer] = []
    for p in parsed.players:
        if p.resolved and p.player_id not in seen:
            seen.add(p.player_id)  # type: ignore[arg-type]
            players.append(p)

    if parsed.has_bench:
        starters = [p for p in players if not p.is_bench]
        bench = sorted(
            (p for p in players if p.is_bench),
            key=lambda p: (p.bench_order is None, p.bench_order or 0),
        )
    else:
        starters, bench = choose_default_lineup(players, bs)
    starters = sorted(starters, key=lambda p: POSITION_RANK.get(p.position or "", 9))  # stable

    captain_id = next((p.player_id for p in starters if p.is_captain), None)
    vice_id = next(
        (p.player_id for p in starters if p.is_vice_captain and p.player_id != captain_id), None
    )

    picks: list[Pick] = []
    for slot, p in enumerate(starters, start=1):
        is_c = p.player_id == captain_id
        picks.append(
            Pick(
                element=p.player_id,  # type: ignore[arg-type]
                position=slot,
                multiplier=2 if is_c else 1,
                is_captain=is_c,
                is_vice_captain=p.player_id == vice_id,
                element_type=Position[p.position] if p.position else None,
            )
        )
    for slot, p in enumerate(bench, start=len(starters) + 1):
        picks.append(
            Pick(
                element=p.player_id,  # type: ignore[arg-type]
                position=slot,
                multiplier=0,
                element_type=Position[p.position] if p.position else None,
            )
        )

    squad_players = [
        SquadPlayer(pick=pk, player=bs.player(pk.element), team=bs.team(bs.player(pk.element).team))
        for pk in picks
    ]
    bank = parsed.bank if parsed.bank is not None else 0.0
    prices = sum(
        p.price_as_shown if p.price_as_shown is not None else _price(p, bs) for p in players
    )
    if gw is None:
        gw = parsed.gw
    if gw is None:
        nxt = bs.next_event
        gw = nxt.id if nxt else (bs.current_event.id + 1 if bs.current_event else 1)
    return Squad(
        manager_id=manager_id,
        gw=gw,
        players=squad_players,
        bank=round(bank, 1),
        team_value=round(prices + bank, 1),
        active_chip=None,
        free_transfers=parsed.free_transfers,
        chips_used=[],
    )
