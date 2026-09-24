"""Страница «План»: фраза-итог, выбор чипов и их эффект — чистые функции над `PlanOut`.

«Ответ сначала»: одна фраза, что делать в ближайшем туре и сколько план даёт против варианта
«ничего не менять» (`baseline_total` — лучший состав из нынешних 15 на каждый тур, без
трансферов, с теми же фишками). Фишки: Bench Boost и Triple Captain (как в core/optimizer —
Free Hit не моделируется, Wildcard план сравнивает сам); пользователь выбирает тур для каждой
доступной фишки, в один тур — одна (правило FPL); эффект — разница двух вызовов
`build_gameweek_plan` (с фишками и без), оба кэшируются (app/plan_chips.py).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from fplcopilot.agent.tools import PlanOut
from fplcopilot.app import format as fmt
from fplcopilot.app.ui_kit import CHIP_LABELS
from fplcopilot.core.optimizer import PLANNABLE_CHIPS

CHIP_ORDER: tuple[str, ...] = tuple(PLANNABLE_CHIPS)  # bboost, 3xc
CHIP_HELP: dict[str, str] = {
    "bboost": (
        "Bench Boost: в выбранный тур очки приносят все 15 игроков — план заранее усиливает "
        "скамейку трансферами под матчи этого тура."
    ),
    "3xc": "Triple Captain: очки капитана в выбранный тур умножаются на 3, а не на 2.",
}


def chip_label(chip: str) -> str:
    return CHIP_LABELS.get(chip, chip)


def plannable_chips(available: Sequence[str]) -> list[str]:
    """Доступные менеджеру чипы в порядке показа (Bench Boost и Triple Captain — первыми)."""
    return [c for c in CHIP_ORDER if c in set(available)]


def chip_selection(
    choices: Mapping[str, int | None],
) -> tuple[tuple[tuple[int, str], ...], list[str]]:
    """Выбор «чип -> тур» из виджетов -> ((тур, чип), …) для `cached_plan` и предупреждения:
    в один тур только один чип (оставляем первый по порядку CHIP_ORDER)."""
    taken: dict[int, str] = {}
    notes: list[str] = []
    for chip in CHIP_ORDER:
        gw = choices.get(chip)
        if gw is None:
            continue
        if gw in taken:
            notes.append(
                f"В один тур можно сыграть только один чип: {chip_label(chip)} в GW{gw} не учтён "
                f"— там уже {chip_label(taken[gw])}."
            )
            continue
        taken[gw] = chip
    return tuple(sorted(taken.items())), notes


def chips_map(plan: PlanOut) -> dict[str, str]:
    """Фишки плана {«7»: «bboost»} — для бейджей на карточках туров."""
    return {str(c.gw): c.chip for c in plan.chips}


def chips_text(chips: Mapping[str, str] | Sequence[tuple[int, str]]) -> str:
    """«Bench Boost в GW7, Triple Captain в GW9» — для подписей."""
    items = chips.items() if isinstance(chips, Mapping) else chips
    return ", ".join(f"{chip_label(c)} в GW{g}" for g, c in sorted(items, key=lambda x: int(x[0])))


def _moves_word(n: int) -> str:
    return fmt.plural(n, "ход", "хода", "ходов")


def headline(plan: PlanOut) -> str:
    """Главная фраза: что сделать в первом туре и сколько даёт план против «ничего не менять»."""
    g = plan.from_gw
    tours = fmt.plural(plan.horizon, "тур", "тура", "туров")
    gain = plan.expected_total - plan.baseline_total
    total = fmt.points_text(plan.expected_total)
    if abs(gain) < 0.05:
        versus = "столько же, сколько если ничего не менять"
    else:
        more = "больше" if gain > 0 else "меньше"
        versus = f"на {fmt.points_text(abs(gain))} {more}, чем если ничего не менять"
    if plan.recommendation == "wildcard" and plan.wildcard:
        wc = plan.wildcard
        return (
            f"Выгоднее сыграть Wildcard в GW{wc.get('gw', g)}: "
            f"{fmt.points_text(float(wc.get('expected_total') or 0))} за {tours} — "
            f"на {fmt.points_text(abs(float(wc.get('delta_vs_plan') or 0)))} больше обычного плана."
        )
    moves = plan.moves_by_gw.get(str(g)) or []
    if moves:
        first = f"{moves[0].out} → {moves[0].in_}"
        rest = f" и ещё {_moves_word(len(moves) - 1)}" if len(moves) > 1 else ""
        paid = sum(1 for m in moves if m.paid)
        hit = f" (платных: {paid}, −{4 * paid} очка)" if paid else ""
        start = f"В GW{g} сделайте трансфер {first}{rest}{hit}"
    else:
        start = f"В GW{g} трансфер не нужен — бесплатный переносится на следующий тур"
    return f"{start}. План на {tours} даст {total} — {versus}."


def chip_effect(with_chips: PlanOut, without: PlanOut | None) -> str | None:
    """«С Bench Boost в GW7 план даёт на 8.3 очка больше, чем без чипов.»"""
    if not with_chips.chips or without is None:
        return None
    d = with_chips.expected_total - without.expected_total
    label = chips_text(chips_map(with_chips))
    if abs(d) < 0.05:
        return f"С чипами ({label}) план даёт столько же очков, сколько без них."
    more = "больше" if d > 0 else "меньше"
    return f"С {label} план даёт на {fmt.points_text(abs(d))} {more}, чем без чипов."


def answer_numbers(plan: PlanOut) -> list[tuple[str, str, str]]:
    """Три числа под фразой: по плану, если ничего не менять, разница (зелёная — в плюс)."""
    gain = plan.expected_total - plan.baseline_total
    tone = "good" if gain > 0.05 else "bad" if gain < -0.05 else "plain"
    return [
        (f"{plan.expected_total:.1f}", "очков по плану", "accent"),
        (f"{plan.baseline_total:.1f}", "если ничего не менять", "plain"),
        (f"{gain:+.1f}", "разница", tone),
    ]
