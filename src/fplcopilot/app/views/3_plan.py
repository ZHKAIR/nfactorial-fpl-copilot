"""Страница «План» по принципу «ответ сначала» (Season Canvas референса).

1. Итог: одна фраза — что сделать в ближайшем туре и сколько план даёт против «ничего не
   менять», три числа, эффект чипов (если выбраны).
2. Настройки в одну строку: горизонт 3–6 туров, хиты, чипы (тур для каждого доступного чипа).
3. Таймлайн туров: ходы — главное; прогноз, капитан, FT, банк — мелко; бейдж чипа.
4. Прогноз по турам: «если ничего не менять» против «по плану».
5. Детали — в раскрывающихся блоках: целевой состав, Wildcard, учтённые проблемы, сравнение
   со снимком, таблицы; кнопка «Сохранить снимок».

Инструмент — build_gameweek_plan (с чипами и без; оба вызова кэшируются, app/plan_chips.py),
снимки — latest_plan / diff_plans / save_plan. Планируются Bench Boost и Triple Captain (как в
core/optimizer): Wildcard план сравнивает сам, Free Hit не моделируется.
"""

from __future__ import annotations

import streamlit as st

from fplcopilot.app import briefing, common, plan_chips, plan_view, ui_kit
from fplcopilot.app import format as fmt

HORIZONS = (3, 4, 5, 6)
NO_CHIP = "не играть"

ui = common.sidebar()
common.page_head(
    f"GW{ui.gw} · план трансферов" if ui.gw else "План трансферов",
    "План на несколько туров",
)
if not common.require_squad(ui):
    st.stop()
ctx = ui.ctx
assert ctx is not None
gw = ctx.gw

# ---------- настройки в одну строку ----------

c_h, c_hits, c_chips = st.columns([1.3, 1, 1.3], vertical_alignment="bottom")
with c_h:
    horizon = st.segmented_control(
        "Горизонт",
        HORIZONS,
        default=5,
        required=True,
        format_func=lambda h: f"{h} GW",
        key="plan_horizon",
    )
horizon = int(horizon or 5)
gws = [gw + i for i in range(horizon)]
with c_hits:
    allow_hits = st.toggle(
        "Разрешить платные трансферы (−4)",
        value=True,
        key="plan_allow_hits",
        help="Выключено — ни одного платного трансфера в горизонте, только бесплатные.",
    )
available = plan_view.plannable_chips(ctx.chips_available)
choices: dict[str, int | None] = {}
with c_chips:
    if not available:
        st.caption(
            "Чипов для планирования нет: Bench Boost и Triple Captain в этой половине сезона "
            "уже сыграны. Wildcard план сравнивает сам, Free Hit не учитывается."
        )
    else:
        chosen_now = {
            c: st.session_state.get(f"plan_chip_{c}")
            for c in available
            if st.session_state.get(f"plan_chip_{c}") not in (None, NO_CHIP)
        }
        label = (
            "Чипы: " + plan_view.chips_text([(int(g[2:]), c) for c, g in chosen_now.items()])
            if chosen_now
            else "Чипы: не выбраны"
        )
        with st.popover(label, width="stretch"):
            st.caption("Выберите тур для чипа — план пересчитается с ним.")
            for chip in available:
                pick = st.selectbox(
                    plan_view.chip_label(chip),
                    [NO_CHIP, *(f"GW{g}" for g in gws)],
                    key=f"plan_chip_{chip}",
                    help=plan_view.CHIP_HELP.get(chip),
                )
                choices[chip] = None if pick in (None, NO_CHIP) else int(str(pick)[2:])
chips, chip_notes = plan_view.chip_selection(
    {c: g for c, g in choices.items() if g is None or g in gws}
)
for note in chip_notes:
    st.warning(note)

# ---------- план (с чипами и без) ----------

with st.spinner(f"Строю план на {horizon} туров (до ~15 с)…"):
    plan = plan_chips.guarded(plan_chips.cached_plan, ui, gw, horizon, allow_hits, chips)
if plan is None:
    st.stop()
plain = plan
if chips:
    with st.spinner("Считаю тот же план без чипов для сравнения…"):
        plain = plan_chips.guarded(plan_chips.cached_plan, ui, gw, horizon, allow_hits, ())

# ---------- 1. итог ----------

with st.container(border=True, key="card_plan_answer"):
    numbers = plan_view.answer_numbers(plan)
    st.markdown(
        ui_kit.section_label("Итог плана", dot=True, tone="accent")
        + ui_kit.card_title(plan_view.headline(plan)),
        unsafe_allow_html=True,
    )
    cols = st.columns(len(numbers) + 1)
    for col, (value, note, tone) in zip(cols, numbers, strict=False):
        col.markdown(ui_kit.big_number(value, note, tone=tone), unsafe_allow_html=True)
    effect = plan_view.chip_effect(plan, plain)
    if effect:
        st.markdown(
            ui_kit.news_item(
                "Эффект чипов", effect, "сравнение с тем же планом без чипов", tone="good"
            ),
            unsafe_allow_html=True,
        )
        for line in plan_chips.chip_lines(plan):
            st.markdown(line)
    for note in plan.notes:
        st.info(note)
    st.caption(
        f"GW{gws[0]}–GW{gws[-1]} · «ничего не менять» — лучший состав из ваших 15 на каждый тур "
        "без трансферов (с теми же чипами) · солвер "
        f"{plan.solver} {plan.runtime_s:.1f} с"
        + (" (лимит времени)" if plan.time_limit_hit else "")
    )

# ---------- 2. таймлайн ----------

st.subheader("Ходы по турам")
bs = common.get_tools().bootstrap
deadlines = {e.id: e.deadline_time for e in getattr(bs, "events", [])}
st.markdown(
    ui_kit.plan_gw_cards(
        plan, deadlines=deadlines, reason_ru=fmt.ISSUE_KIND_RU, chips=plan_view.chips_map(plan)
    ),
    unsafe_allow_html=True,
)

# ---------- 3. прогноз по турам ----------

series = briefing.plan_vs_hold(plan)
if any(b is not None for _, b, _ in series):
    with st.container(border=True, key="card_plan_series"):
        st.markdown(
            ui_kit.section_label("Прогноз по турам")
            + ui_kit.paired_bars(series, ("если ничего не менять", "по плану"))
            + ui_kit.text(
                "Серый — лучший состав из ваших нынешних 15 в каждом туре без трансферов. "
                "Синий — состав по плану в этом туре с учётом чипов и за вычетом −4 за платные "
                "трансферы. Под парой — разница.",
                muted=True,
            ),
            unsafe_allow_html=True,
        )

# ---------- 4. детали ----------

st.subheader("Подробности")
with st.expander("Целевой состав к концу горизонта"):
    if plan.target_squad:
        st.markdown(ui_kit.name_chips(plan.target_squad), unsafe_allow_html=True)
    else:
        st.caption("Состав не меняется.")
if plan.wildcard:
    wc = plan.wildcard
    with st.expander(f"Альтернатива: Wildcard в GW{wc.get('gw', gw)}"):
        st.markdown(
            ui_kit.text(
                f"План с Wildcard даёт {fmt.num(wc.get('expected_total'))} xPts "
                f"({fmt.signed(wc.get('delta_vs_plan'))} к обычному плану). Wildcard "
                "рекомендуется, если выигрыш больше порога стратегии и в составе не меньше "
                "трёх проблемных игроков."
            ),
            unsafe_allow_html=True,
        )
        st.markdown(ui_kit.name_chips(wc.get("squad") or []), unsafe_allow_html=True)
if plan.issues:
    with st.expander("Проблемы состава, учтённые планом"):
        for i in plan.issues:
            st.write(
                f"- {i.get('name')}: {fmt.ISSUE_KIND_RU.get(i.get('kind'), i.get('kind'))} — {i.get('detail')}"
            )
with st.expander("Что изменилось со снимка"):
    if plan.diff_vs_previous is None:
        st.caption(
            "Сохранённого снимка для этого менеджера, тура и стратегии ещё нет. Сохраните "
            "снимок — в следующий раз здесь будет видно, что поменялось в плане."
        )
    elif not plan.diff_vs_previous:
        st.caption(f"План совпадает со снимком от {fmt.snapshot_time_text(plan.previous_plan_at)}.")
    else:
        n = len(plan.diff_vs_previous)
        st.caption(
            f"С момента снимка от {fmt.snapshot_time_text(plan.previous_plan_at)} — "
            + fmt.plural(n, "изменение", "изменения", "изменений")
            + "."
        )
        st.dataframe(
            plan_chips.diff_rows(fmt.diff_rows(plan.diff_vs_previous)),
            hide_index=True,
            width="stretch",
        )
with st.expander("Таблицы плана"):
    st.dataframe(fmt.plan_move_rows(plan), hide_index=True, width="stretch")
    st.dataframe(plan_chips.gw_rows(plan, fmt.plan_gw_rows(plan)), hide_index=True, width="stretch")

bcol, tcol = st.columns([8, 1])
save_clicked = bcol.button("Сохранить снимок плана", key="save_plan", type="primary")
tcol.markdown(
    fmt.TIP_WIDGET_CSS + fmt.tip("plan_snapshots (006_plans.sql)"),
    unsafe_allow_html=True,
)
if save_clicked:
    with st.spinner("Пересчитываю и сохраняю…"):
        saved = plan_chips.guarded(plan_chips.save_plan_snapshot, ui, gw, horizon, allow_hits, chips)
    if saved is not None:
        if saved.saved_id is not None:
            st.success(f"Снимок сохранён: plan_snapshots id={saved.saved_id}")
            st.cache_data.clear()
        else:
            st.warning("Снимок не сохранён (БД недоступна).")
