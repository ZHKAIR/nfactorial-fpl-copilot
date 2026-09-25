"""Цены продажи игроков по правилам FPL: публичный API их не отдаёт, считаем по истории трансферов.

Правило: если цена выросла с момента покупки, при продаже получаешь покупку + половину прироста
(с округлением вниз до £0.1); если упала — текущую цену. Цены — в десятых долях миллиона.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from fplcopilot.data.schemas import TransferRow


def selling_price(purchase: int, now: int) -> int:
    """7.5 -> 7.8: продажа за 7.6; 7.5 -> 7.7: за 7.6; 7.5 -> 7.3: за 7.3."""
    return purchase + (now - purchase) // 2 if now > purchase else now


def purchase_prices(
    player_ids: Iterable[int],
    transfers: Sequence[TransferRow],
    start_cost: Mapping[int, int | None],
    *,
    freehit_events: Iterable[int] = (),
) -> dict[int, int]:
    """Цена покупки игрока в нынешнем составе: последний трансфер на него (вне Free Hit —
    после него состав возвращается), иначе цена на момент создания команды (start_cost).
    Игрок без известной цены в ответ не попадает."""
    skip = set(freehit_events)
    last_in: dict[int, int] = {}
    for t in sorted(transfers, key=lambda t: t.time):
        if t.event not in skip:
            last_in[t.element_in] = t.element_in_cost
    out: dict[int, int] = {}
    for pid in player_ids:
        cost = last_in.get(pid, start_cost.get(pid))
        if cost is not None:
            out[pid] = cost
    return out
