"""Фишки на странице «План»: выбор «фишка → тур» (Bench Boost / Triple Captain), вызов
build_gameweek_plan с `chips`, фишка и её вклад в таблице по турам.

Без выбранной фишки страница идёт через common.cached_plan / common.save_plan_snapshot как
раньше; с фишкой — через свои кэш и сохранение (ключ кэша включает фишки).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import streamlit as st

from fplcopilot.agent.tools import BuildPlanInput, PlanChipIn, PlanOut, SquadOverride
from fplcopilot.app import common
from fplcopilot.app import format as fmt
from fplcopilot.core.optimizer import BENCH_BOOST, PLANNABLE_CHIPS, TRIPLE_CAPTAIN, ChipPlanError

Chips = tuple[tuple[int, str], ...]  # ((тур, фишка), ...) — хэшируемый ключ кэша

NO_CHIP = ""
CHIP_KEY = "plan_chip"
CHIP_GW_KEY = "plan_chip_gw"
CHIP_LABELS: dict[str, str] = {BENCH_BOOST: "Bench Boost", TRIPLE_CAPTAIN: "Triple Captain"}
CHIP_HELP = (
    "Bench Boost: в выбранном туре считаются очки всех 15 — план заранее усиливает скамейку "
    "трансферами под матчи этого тура. Triple Captain: капитан в выбранном туре ×3 вместо ×2. "
    "В списке только фишки, которые у вас есть. Wildcard план сравнивает сам, Free Hit не "
    "учитывается."
)
NO_CHIPS_HELP = (
    "Bench Boost и Triple Captain в этой половине сезона уже сыграны. Wildcard план сравнивает "
    "сам, Free Hit не учитывается."
)
GW_HELP = "Тур горизонта, в котором играете фишку."
TOTALS_CAPTION = "Обе суммы — план и «без трансферов» — считаны с фишкой."
CHIP_ERROR_TITLE = "Фишку нельзя запланировать"
DIFF_KIND_CHIPS = "фишки"


def chip_label(chip: str) -> str:
    return CHIP_LABELS.get(chip, chip) if chip else "Без фишки"


def chip_options(chips_available: Iterable[str] | None) -> list[str]:
    """«Без фишки» + доступные менеджеру BB / TC (Wildcard и Free Hit здесь не выбираются)."""
    have = set(chips_available or ())
    return [NO_CHIP] + [c for c in PLANNABLE_CHIPS if c in have]


def horizon_gws(gw: int, horizon: int) -> list[int]:
    return list(range(gw, gw + horizon))


def selected(chip: str | None, chip_gw: int | None) -> Chips:
    return ((int(chip_gw), chip),) if chip and chip_gw is not None else ()


def chip_picker(chips_available: Iterable[str] | None, gw: int, horizon: int) -> Chips:
    """Две колонки как у горизонта/хитов: «Фишка» и «Тур» (туры только из горизонта)."""
    options = chip_options(chips_available)
    gws = horizon_gws(gw, horizon)
    if st.session_state.get(CHIP_KEY) not in options:
        st.session_state.pop(CHIP_KEY, None)
    if st.session_state.get(CHIP_GW_KEY) not in gws:
        st.session_state.pop(CHIP_GW_KEY, None)
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(
            fmt.label_with_tip("Фишка", CHIP_HELP if len(options) > 1 else NO_CHIPS_HELP),
            unsafe_allow_html=True,
        )
        chip = st.selectbox(
            "Фишка",
            options,
            format_func=chip_label,
            key=CHIP_KEY,
            disabled=len(options) == 1,
            label_visibility="collapsed",
        )
    with c2:
        st.markdown(fmt.label_with_tip("Тур фишки", GW_HELP), unsafe_allow_html=True)
        chip_gw = st.selectbox(
            "Тур фишки",
            gws,
            format_func=lambda g: f"GW{g}",
            key=CHIP_GW_KEY,
            disabled=not chip,
            label_visibility="collapsed",
        )
    return selected(chip, chip_gw)


# ---------- вызов инструмента ----------


def _plan_input(
    manager_id: int,
    gw: int,
    strategy: str,
    horizon: int,
    allow_hits: bool,
    chips: Chips,
    *,
    save: bool,
) -> BuildPlanInput:
    return BuildPlanInput(
        manager_id=manager_id,
        gw=gw,
        horizon=horizon,
        strategy=strategy,
        allow_hits=allow_hits,
        save=save,
        chips=[PlanChipIn(gw=g, chip=c) for g, c in chips],
    )


@st.cache_data(ttl=common.CACHE_TTL, show_spinner=False)
def _cached_chip_plan(
    manager_id: int,
    gw: int,
    strategy: str,
    horizon: int,
    allow_hits: bool,
    chips: Chips,
    fp: str | None,
    _override: SquadOverride | None,
) -> PlanOut:
    inp = _plan_input(manager_id, gw, strategy, horizon, allow_hits, chips, save=False)
    return common.squad_call(common.get_tools().build_gameweek_plan, inp, _override)


def cached_plan(ui: common.UI, gw: int, horizon: int, allow_hits: bool, chips: Chips) -> PlanOut:
    if not chips:
        return common.cached_plan(
            ui.squad_manager_id, gw, ui.strategy, horizon, allow_hits, ui.fp, ui.override
        )
    return _cached_chip_plan(
        ui.squad_manager_id, gw, ui.strategy, horizon, allow_hits, chips, ui.fp, ui.override
    )


def save_plan_snapshot(
    ui: common.UI, gw: int, horizon: int, allow_hits: bool, chips: Chips
) -> PlanOut:
    """Пересчёт с save=True — снимок с теми же фишками, что на экране (без кэша)."""
    if not chips:
        return common.save_plan_snapshot(ui.squad_manager_id, gw, ui.strategy, horizon, allow_hits)
    inp = _plan_input(ui.squad_manager_id, gw, ui.strategy, horizon, allow_hits, chips, save=True)
    return common.squad_call(common.get_tools().build_gameweek_plan, inp, common.current_override())


def guarded(fn: Any, *args: Any) -> Any:
    """common.guarded, но нарушение правил фишек — понятная ошибка, а не имя исключения."""
    try:
        return fn(*args)
    except ChipPlanError as exc:
        st.error(f"**{CHIP_ERROR_TITLE}**")
        st.caption(str(exc))
        return None
    except Exception as exc:  # noqa: BLE001 — как common.guarded
        common.show_error(exc)
        return None


# ---------- отображение ----------


def chip_lines(plan: PlanOut) -> list[str]:
    """Строка на фишку: тур, вклад в xPts и за счёт кого."""
    lines = []
    for c in plan.chips:
        who = ", ".join(c.players)
        if c.chip == BENCH_BOOST:
            lines.append(f"**Bench Boost в GW{c.gw}:** +{fmt.num(c.points)} xPts скамейки ({who})")
        elif c.chip == TRIPLE_CAPTAIN:
            lines.append(
                f"**Triple Captain в GW{c.gw}:** +{fmt.num(c.points)} xPts — капитан {who} ×3"
            )
    return lines


def gw_rows(plan: PlanOut, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Строки fmt.plan_gw_rows + «Фишка» и «Вклад фишки»; без фишек — как есть."""
    if not plan.chips:
        return [dict(r) for r in rows]
    by_gw = {c.gw: c for c in plan.chips}
    out = []
    for r in rows:
        c = by_gw.get(int(r["GW"]))
        out.append(
            {
                **r,
                "Фишка": chip_label(c.chip) if c else "",
                "Вклад фишки": f"+{fmt.num(c.points)}" if c else "",
            }
        )
    return out


def wildcard_value(plan: PlanOut, value: str) -> str:
    """Метрика «Wildcard»: альтернатива не строилась из-за фишки в первом туре — не «недоступен»."""
    if plan.wildcard is None and plan.notes and any(c.gw == plan.from_gw for c in plan.chips):
        return "не сравнивался"
    return value


def diff_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """fmt.diff_rows + русская подпись для смены фишек."""
    return [
        {**r, "Изменение": DIFF_KIND_CHIPS} if r.get("Изменение") == "chips" else dict(r)
        for r in rows
    ]
