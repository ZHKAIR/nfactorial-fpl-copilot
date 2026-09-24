"""Стратегия оптимизатора: набор чисел, которые меняют целевую функцию и ограничения MILP.

Стратегия никогда не меняет текст и не «интерпретирует» — только коэффициенты:
- variance_penalty (λ): штраф за дисперсию очков стартового состава (риск-аверсия);
- hit_threshold: минимальный чистый выигрыш (за вычетом −4) за горизонт, чтобы советовать хит;
- ownership_weight: ± очков-эквивалент за 1 % владения на игрока в составе за тур
  (плюс — тянет к шаблону, минус — к дифференциалам);
- ft_value: ценность одного бесплатного трансфера; начисляется в туре, когда FT появляется
  (ft[w+1] − ft[w]), с дисконтом — как gw_ft_gain в open-fpl-solver;
- itb_value: ценность £1 млн в банке за тур (в open-fpl-solver цены в миллионах и
  itb_value · in_the_bank[w] входит в сумму по турам);
- decay_base: дисконт очков будущих туров (decay^(w − from_gw));
- bench_weights: доля очков запасных, попадающая в целевую функцию, по слоту 12–15;
- wc_margin: на сколько очков план с Wildcard должен обыграть план трансферов.

Числа ft_value=1.5, itb_value=0.08, decay_base=0.84, bench_weights={0.03, 0.21, 0.06, 0.002}
— дефолты сообщества (sertalpbilal/FPL-Optimization-Tools, dev/solver.py, «regular» objective);
остальное — наши пресеты, см. docs/optimizer.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Literal

StrategyName = Literal["conservative", "balanced", "aggressive"]

HIT_COST = 4  # очков за платный трансфер
MAX_FREE_TRANSFERS = 5  # правила 2026/27: FT копятся до 5
SQUAD_SIZE = 15
STARTERS = 11
POSITION_QUOTA = {1: 2, 2: 5, 3: 5, 4: 3}  # GKP, DEF, MID, FWD в составе из 15
FORMATION_BOUNDS = {1: (1, 1), 2: (3, 5), 3: (2, 5), 4: (1, 3)}  # стартовые 11
MAX_PER_CLUB = 3
FIRST_HALF_LAST_GW = 19  # правила 2026/27: каждый чип дважды, первый набор сгорает после GW19

# Доля xPts запасного, идущая в цель: слот 12 — запасной вратарь, 13–15 — полевые по порядку.
DEFAULT_BENCH_WEIGHTS: dict[int, float] = {12: 0.03, 13: 0.21, 14: 0.06, 15: 0.002}


@dataclass(frozen=True)
class Strategy:
    name: str
    variance_penalty: float  # λ: очков штрафа за 1 очко² дисперсии стартера за тур
    hit_threshold: float  # мин. чистый выигрыш (уже минус 4) за горизонт, чтобы брать хит
    ownership_weight: float  # очков за 1 % владения на игрока состава за тур (± )
    wc_margin: float  # WC советуем, если план с WC лучше плана трансферов на столько очков
    ft_value: float = 1.5
    itb_value: float = 0.08  # очков за £1 млн в банке за тур
    decay_base: float = 0.84
    bench_weights: Mapping[int, float] = field(default_factory=lambda: dict(DEFAULT_BENCH_WEIGHTS))

    def decay(self, gw: int, from_gw: int) -> float:
        return self.decay_base ** max(0, gw - from_gw)

    def bench_weight(self, slot: int) -> float:
        return float(self.bench_weights.get(slot, 0.0))

    def with_overrides(self, **kwargs: float) -> Strategy:
        return replace(self, **kwargs)


PRESETS: dict[str, Strategy] = {
    # Риск-аверсия, шаблон, хиты только при явном выигрыше, WC лишь при большом отрыве.
    "conservative": Strategy(
        name="conservative",
        variance_penalty=0.03,
        hit_threshold=2.0,
        ownership_weight=0.005,
        wc_margin=3.0,
    ),
    "balanced": Strategy(
        name="balanced",
        variance_penalty=0.01,
        hit_threshold=1.0,
        ownership_weight=0.0,
        wc_margin=2.0,
    ),
    # Погоня за рангом: небольшая премия за дисперсию, дифференциалы, хит при любом плюсе.
    "aggressive": Strategy(
        name="aggressive",
        variance_penalty=-0.005,
        hit_threshold=0.25,
        ownership_weight=-0.005,
        wc_margin=1.0,
    ),
}


def get_strategy(name: str | Strategy = "balanced") -> Strategy:
    if isinstance(name, Strategy):
        return name
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"неизвестная стратегия {name!r}; есть {sorted(PRESETS)}") from None
