"""Модель минут: P(старт), P(выход на замену), ожидаемые минуты, P(>= 60 минут).

Вход — история матчей игрока ДО дедлайна (leakage guard в core/history.rows_before), статус
FPL и (опционально) новостной сигнал из player_signals. Шаги:

1. p_start_raw = 0.7 · взвешенная доля стартов в последних 5 матчах (веса RECENT_WEIGHTS,
   свежие важнее) + 0.3 · доля стартов за сезон;
2. сжатие к приору: p_start = (n · p_start_raw + K · prior) / (n + K), K = START_PRIOR_K = 1,
   prior = доля стартов прошлого сезона (starts/38), иначе START_PRIOR_DEFAULT;
3. p_sub = P(выйти на замену | не в старте), exp_min_start, sub_minutes и P(>=60 | старт)
   оцениваются по тем же строкам со сжатием к приорам (константы ниже);
4. доступность FPL: a -> ×1 (или ×chance/100, если FPL указал), d -> ×chance/100 (нет chance -> 50%),
   i/s/u/n -> 0 — множитель к p_start и p_sub;
5. сигнал из новостей (если есть, не unknown; действует неделю, а injured/suspended/unavailable
   с return_gw — до этого тура, см. core/signals.signal_active; при status_gw >= return_gw
   сигнал считается отслужившим и не применяется):
   p_start = w · signal.start_probability + (1 - w) · p_start, w = confidence;
   injured/suspended/unavailable в сигнале дополнительно гасят p_sub на (1 - w).
   Исключение — `fit`: снизить p_start он может только с rotation_risk="high" (явное
   предупреждение о ротации); иначе «fit 0.9» лишь подтверждает (или поднимает игрока со
   статусом d) и никогда не режет p_start железного стартера;
6. горизонт (gw > status_gw — тур, к которому относится статус FPL): если у сигнала есть
   return_gw, то для gw < return_gw доступность = 0 (для статуса d — chance/100), для
   gw >= return_gw статус и сигнал игнорируются, первый тур после возвращения — ×POST_RETURN_DISCOUNT.

Выход: exp_minutes = p_start · exp_min_start + (1 - p_start) · p_sub · sub_minutes,
p_appear = p_start + (1 - p_start) · p_sub, p60 = p_start · p60_start + (1 - p_start) · p_sub · SUB_P60.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from fplcopilot.core.history import SeasonTotals
from fplcopilot.core.signals import SignalLite
from fplcopilot.core.stats import clamp, shrink
from fplcopilot.data import Position
from fplcopilot.data.schemas import PlayerGWHistory

RECENT_WEIGHTS = (0.35, 0.25, 0.18, 0.12, 0.10)  # последний матч первым
RECENT_SHARE = 0.7  # вес «последних 5» против доли за сезон
START_PRIOR_K = 1.0  # псевдо-матчи приора p_start
START_PRIOR_DEFAULT = 0.4  # игрок без истории и без прошлого сезона
PAST_SEASON_MATCHES = 38
SUB_APPEAR_PRIOR = 0.35  # P(выйти на замену | не в старте) для «среднего» игрока
SUB_K = 2.0
SUB_MINUTES_PRIOR = 20.0
START_MINUTES_PRIOR = {
    Position.GKP: 90.0,
    Position.DEF: 86.0,
    Position.MID: 80.0,
    Position.FWD: 78.0,
}
START_K = 2.0
P60_START_PRIOR = {Position.GKP: 0.99, Position.DEF: 0.95, Position.MID: 0.88, Position.FWD: 0.85}
SUB_P60 = 0.04  # замена на 30-й минуте после травмы партнёра — редкость
DOUBTFUL_DEFAULT_CHANCE = 50
UNAVAILABLE_STATUSES = frozenset({"i", "s", "u", "n"})
# Первый тур после возвращения из травмы: тренеры вводят через замену / снимают раньше.
POST_RETURN_DISCOUNT = 0.8
ROTATION_WARNING = "high"  # rotation_risk сигнала, при котором «fit» всё же может снизить p_start


@dataclass(frozen=True)
class MinutesEstimate:
    p_start: float
    p_sub: float  # P(появиться с замены | не в старте)
    p_appear: float
    p60: float
    exp_minutes_if_start: float
    sub_minutes: float
    exp_minutes: float
    availability_factor: float
    p_start_prior: float  # до статуса и сигнала (для --explain)
    signal_weight: float = 0.0
    notes: tuple[str, ...] = field(default=())


def availability_factor(status: str, chance: int | None) -> float:
    """Множитель доступности по статусу FPL и chance_of_playing_next_round."""
    if status in UNAVAILABLE_STATUSES:
        return 0.0
    if status == "d":
        return clamp((DOUBTFUL_DEFAULT_CHANCE if chance is None else chance) / 100)
    if chance is not None:  # 'a' с chance < 100 бывает сразу после возвращения
        return clamp(chance / 100)
    return 1.0


def start_prior_from_past(past: SeasonTotals | None) -> float | None:
    """Доля стартов прошлого сезона; для старых сезонов без starts — минуты/(38·90)."""
    if past is None or past.minutes <= 0:
        return None
    if past.starts > 0:
        return clamp(past.starts / PAST_SEASON_MATCHES)
    return clamp(past.minutes / (PAST_SEASON_MATCHES * 90))


def _weighted_recent_share(flags: Sequence[bool]) -> float:
    """flags — по времени; веса применяются с конца (последний матч — самый важный)."""
    recent = list(reversed(flags))[: len(RECENT_WEIGHTS)]
    weights = RECENT_WEIGHTS[: len(recent)]
    return sum(w * f for w, f in zip(weights, recent, strict=True)) / sum(weights)


def horizon_availability(
    signal: SignalLite | None,
    status: str,
    chance: int | None,
    *,
    gw: int | None,
    status_gw: int | None,
) -> tuple[float, bool, str] | None:
    """Правило горизонта по return_gw сигнала: (множитель доступности, смешивать ли сигнал, заметка).

    None — правило не применяется (прогноз на тур статуса FPL или у сигнала нет return_gw):
    тогда действует обычная логика статуса и смешивания. Статус FPL описывает ближайший тур,
    поэтому для gw <= status_gw он главнее любой даты возвращения из новостей.
    """
    if signal is None or signal.return_gw is None or gw is None or status_gw is None:
        return None
    if gw <= status_gw:
        return None
    if gw >= signal.return_gw:
        factor = POST_RETURN_DISCOUNT if gw == signal.return_gw else 1.0
        return (
            factor,
            False,
            f"signal return_gw={signal.return_gw}: back for GW{gw} -> ×{factor:.2f}",
        )
    factor = availability_factor(status, chance) if status == "d" else 0.0
    return factor, True, f"signal return_gw={signal.return_gw}: out for GW{gw} -> ×{factor:.2f}"


def fit_minutes(
    rows: Sequence[PlayerGWHistory],
    *,
    position: Position,
    past: SeasonTotals | None = None,
    gw: int | None = None,
    status_gw: int | None = None,
) -> MinutesEstimate:
    """Минуты «если здоров»: без статуса FPL и сигнала, а хвост матчей с 0 минут в конце истории
    отрезан — у травмированного это и есть пропуск, иначе он занизит долю стартов."""
    trimmed = list(rows)
    while trimmed and trimmed[-1].minutes == 0:
        trimmed.pop()
    return estimate_minutes(trimmed, position=position, past=past, gw=gw, status_gw=status_gw)


def estimate_minutes(
    rows: Sequence[PlayerGWHistory],
    *,
    position: Position,
    status: str = "a",
    chance: int | None = None,
    signal: SignalLite | None = None,
    past: SeasonTotals | None = None,
    gw: int | None = None,
    status_gw: int | None = None,
) -> MinutesEstimate:
    """gw — тур прогноза, status_gw — тур, к которому относится статус FPL (обычно ближайший).

    Оба нужны только для горизонта (gw > status_gw): см. horizon_availability.
    """
    notes: list[str] = []
    n = len(rows)
    prior = start_prior_from_past(past)
    if prior is None:
        prior = START_PRIOR_DEFAULT
        notes.append("prior p_start: default (no previous season)")
    else:
        notes.append(f"prior p_start: {prior:.2f} from previous season")

    if n:
        started = [r.starts > 0 for r in rows]
        raw = RECENT_SHARE * _weighted_recent_share(started) + (1 - RECENT_SHARE) * (
            sum(started) / n
        )
        p_start_model = shrink(raw, prior, n, START_PRIOR_K)
    else:
        p_start_model = prior
        notes.append("no matches this season")

    starts = [r for r in rows if r.starts > 0]
    non_starts = [r for r in rows if r.starts == 0]
    sub_apps = [r for r in non_starts if r.minutes > 0]

    p_sub = shrink(
        len(sub_apps) / len(non_starts) if non_starts else SUB_APPEAR_PRIOR,
        SUB_APPEAR_PRIOR,
        len(non_starts),
        SUB_K,
    )
    sub_minutes = shrink(
        sum(r.minutes for r in sub_apps) / len(sub_apps) if sub_apps else SUB_MINUTES_PRIOR,
        SUB_MINUTES_PRIOR,
        len(sub_apps),
        SUB_K,
    )
    exp_min_start = shrink(
        sum(r.minutes for r in starts) / len(starts) if starts else START_MINUTES_PRIOR[position],
        START_MINUTES_PRIOR[position],
        len(starts),
        START_K,
    )
    p60_start = shrink(
        sum(r.minutes >= 60 for r in starts) / len(starts) if starts else P60_START_PRIOR[position],
        P60_START_PRIOR[position],
        len(starts),
        START_K,
    )

    if signal is not None and signal.expired_for(status_gw):
        # тур возвращения наступил: жёсткий сигнал отслужил, ближайший тур описывает статус FPL
        notes.append(
            f"news signal {signal.availability} return_gw={signal.return_gw} expired "
            f"(status GW{status_gw}) -> FPL status only"
        )
        signal = None
    factor = availability_factor(status, chance)
    blend_signal = True
    horizon = horizon_availability(signal, status, chance, gw=gw, status_gw=status_gw)
    if horizon is not None:
        factor, blend_signal, note = horizon
        notes.append(note)
    elif factor < 1.0:
        notes.append(f"FPL status={status} chance={chance} -> ×{factor:.2f}")
    p_start = p_start_model * factor
    p_sub_eff = p_sub * factor

    weight = 0.0
    if signal is not None and signal.usable and blend_signal:
        weight = clamp(signal.confidence)
        blended = weight * signal.start_probability + (1 - weight) * p_start
        if signal.is_fit and signal.rotation_risk != ROTATION_WARNING and blended <= p_start:
            # «fit» без явного предупреждения о ротации подтверждает доступность: типовое 0.9
            # не должно резать p_start 0.98 железного стартера (снизу — поднимает, см. else)
            weight = 0.0
            notes.append(
                f"news signal fit p_start={signal.start_probability:.2f} confirms prior "
                f"{p_start:.2f} (not blended)"
            )
        else:
            p_start = weight * signal.start_probability + (1 - weight) * p_start
            if signal.rules_out:
                p_sub_eff *= 1 - weight
            notes.append(
                f"news signal {signal.availability} p_start={signal.start_probability:.2f} "
                f"conf={weight:.2f} as of {signal.as_of:%Y-%m-%d %H:%M}Z"
            )

    p_start = clamp(p_start)
    p_appear = clamp(p_start + (1 - p_start) * p_sub_eff)
    p60 = clamp(p_start * p60_start + (1 - p_start) * p_sub_eff * SUB_P60)
    exp_minutes = p_start * exp_min_start + (1 - p_start) * p_sub_eff * sub_minutes
    return MinutesEstimate(
        p_start=round(p_start, 4),
        p_sub=round(p_sub_eff, 4),
        p_appear=round(p_appear, 4),
        p60=round(p60, 4),
        exp_minutes_if_start=round(exp_min_start, 1),
        sub_minutes=round(sub_minutes, 1),
        exp_minutes=round(exp_minutes, 1),
        availability_factor=factor,
        p_start_prior=round(p_start_model, 4),
        signal_weight=weight,
        notes=tuple(notes),
    )
