"""«Ценный актив с короткой травмой»: придержать своего, купить чужого на спаде цены.

Горизонт оптимизатора — 3 тура, и травмированный игрок в нём выглядит слабо, хотя после
возвращения он снова даёт топ-очки (и обычно дорожает, когда его покупают обратно). Правило:

- короткая травма: статус FPL d, либо i/s с туром возвращения не позже from_gw + RETURN_WITHIN
  (новость FPL «Expected back …» или сигнал из новостей); u/n — не актив;
- ценный: очки «если здоров» (xpts_fit, среднее по горизонту) не ниже P75 позиции среди
  регулярных стартеров, или очки за £1m не ниже P75 при очках не ниже медианы позиции;
- свой такой игрок — hold; чужой — buy_low, если травма совсем лёгкая (d с шансом от
  BUY_LOW_MIN_CHANCE % или возврат в пределах окна) и его сбрасывают: цена ниже стартовой при
  чистых продажах за тур или продажи больше BUY_LOW_NET_OUT.

Бонус в цель оптимизатора — за игрока в составе на конец горизонта: превышение его очков над
медианой позиции за TAIL_GWS туров после горизонта (с дисконтом стратегии), с весом WEIGHT за
неопределённость. Это «ценность сверх замены», которую 3-туровый горизонт не видит.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from pydantic import BaseModel

from fplcopilot.core.candidates import Candidate
from fplcopilot.core.strategy import POSITION_QUOTA, Strategy

TAIL_GWS = 2
WEIGHT = 0.5
RETURN_WITHIN = 2  # туров после from_gw
TOP_QUANTILE = 0.75
REGULAR_P_START = 0.6
MIN_REGULARS = 8  # меньше — уровни позиции не считаем
BUY_LOW_MIN_CHANCE = 50
BUY_LOW_NET_OUT = 20_000  # чистые продажи за тур, после которых цена обычно падает
MIN_BONUS = 0.2

AssetKind = Literal["hold", "buy_low"]


class AssetFlag(BaseModel):
    player_id: int
    name: str
    kind: AssetKind
    bonus: float  # очков-эквивалент в цель за игрока в составе на конец горизонта
    fit_xpts: float  # очки за тур, когда здоров (среднее по горизонту)
    position_median: float
    top_share: float  # доля регулярных стартеров позиции, у которых очков меньше
    status: str
    chance: int | None = None
    return_gw: int | None = None
    price: float
    price_change_start: float = 0.0
    transfers_net: int = 0


def fit_rate(c: Candidate, gws: Sequence[int]) -> float:
    return sum(c.xpts_fit(g) for g in gws) / len(gws) if gws else 0.0


def _quantile(values: Sequence[float], q: float) -> float:
    s = sorted(values)
    if not s:
        return 0.0
    i = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
    return s[i]


def position_levels(
    cands: Iterable[Candidate], gws: Sequence[int]
) -> dict[int, tuple[float, float, float, list[float]]]:
    """Позиция -> (медиана, P75 очков, P75 очков за £1m, отсортированные очки) среди здоровых
    регулярных стартеров (статус a, p_start >= 0.6 в ближайшем туре) — уровень «обычной замены»."""
    by_pos: dict[int, list[Candidate]] = {pos: [] for pos in POSITION_QUOTA}
    for c in cands:
        if c.status == "a" and c.p_start >= REGULAR_P_START and c.price > 0:
            by_pos[c.position].append(c)
    out = {}
    for pos, group in by_pos.items():
        if len(group) < MIN_REGULARS:
            continue
        rates = [fit_rate(c, gws) for c in group]
        per_m = [fit_rate(c, gws) / c.price for c in group]
        out[pos] = (
            statistics.median(rates),
            _quantile(rates, TOP_QUANTILE),
            _quantile(per_m, TOP_QUANTILE),
            sorted(rates),
        )
    return out


def is_short_absence(c: Candidate, from_gw: int) -> bool:
    if c.status == "d":
        return True
    if c.status in ("i", "s"):
        return c.return_gw is not None and c.return_gw <= from_gw + RETURN_WITHIN
    return False


def is_minor(c: Candidate, from_gw: int) -> bool:
    """Лёгкая травма для покупки: d с шансом от 50 % или известный скорый возврат."""
    if c.status == "d":
        return (c.chance if c.chance is not None else 50) >= BUY_LOW_MIN_CHANCE
    return is_short_absence(c, from_gw)


def is_falling(c: Candidate) -> bool:
    """Дешевеет на распродаже: цена ниже стартовой и его продают, либо продают массово."""
    return (c.price_change_start < 0 and c.transfers_net < 0) or c.transfers_net <= -BUY_LOW_NET_OUT


def tail_factor(strategy: Strategy, horizon: int) -> float:
    return sum(strategy.decay_base ** (horizon + k) for k in range(TAIL_GWS))


def asset_flags(
    cands: Mapping[int, Candidate],
    squad_ids: Iterable[int],
    gws: Sequence[int],
    strategy: Strategy,
) -> dict[int, AssetFlag]:
    if not gws:
        return {}
    from_gw = gws[0]
    squad = set(squad_ids)
    levels = position_levels(cands.values(), gws)
    tail = tail_factor(strategy, len(gws))
    out: dict[int, AssetFlag] = {}
    for pid, c in cands.items():
        lv = levels.get(c.position)
        if lv is None or not is_short_absence(c, from_gw) or c.price <= 0:
            continue
        median, top, top_per_m, rates = lv
        rate = fit_rate(c, gws)
        valuable = rate >= top or (rate / c.price >= top_per_m and rate >= median)
        if not valuable:
            continue
        in_squad = pid in squad
        if not in_squad and not (is_minor(c, from_gw) and is_falling(c)):
            continue
        bonus = WEIGHT * max(0.0, rate - median) * tail
        if bonus < MIN_BONUS:
            continue
        below = sum(1 for r in rates if r < rate)
        out[pid] = AssetFlag(
            player_id=pid,
            name=c.name,
            kind="hold" if in_squad else "buy_low",
            bonus=round(bonus, 3),
            fit_xpts=round(rate, 2),
            position_median=round(median, 2),
            top_share=round(below / len(rates), 2),
            status=c.status,
            chance=c.chance,
            return_gw=c.return_gw,
            price=c.price,
            price_change_start=round(c.price_change_start, 1),
            transfers_net=c.transfers_net,
        )
    return out
