"""Страница «Мой состав» (`/squad`): шапка команды, строка серьёзных проблем, вкладки «Поле»
(футболки клубов по схеме, как «Tactics Room» референса) и «Таблица» — HTML-таблица 15 игроков
по образцу smartplay Players (Player в две строки со ссылкой на страницу «Игрок»
`/player?pid=<id>` · Price · TSB% · Form · Pts/Game · xMins · xPts · 3GW xPts · FRR# + цветной
календарь на 5 туров, разделитель «Bench»), проблемы состава с цитатами, скриншот в expander.

Инструменты: get_gameweek_context (+ diagnose_squad внутри), predict_player (горизонт 5),
analyze_player_risk(cached_only) — без LLM; TSB% / Form / Pts/Game — из bootstrap в кэше
LiveTools (без новых сетевых запросов); распознавание скриншота — vision.squad_from_image.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from fplcopilot.app import briefing, common, llm_budget, player_card, ui_kit
from fplcopilot.app import format as fmt
from fplcopilot.config import PROJECT_ROOT
from fplcopilot.data.schemas import Bootstrap

SAMPLES = {
    "Пример: валидный Pick Team": PROJECT_ROOT / "samples" / "pick_team_valid.png",
    "Пример: сломанный (14 игроков)": PROJECT_ROOT / "samples" / "pick_team_broken.png",
}
HORIZON = fmt.FRR_HORIZON  # туров в календаре (GW6..GW10) и в рейтинге календаря FRR#


def team_representatives(bs: Bootstrap) -> tuple[int, ...]:
    """По одному игроку на клуб (самый популярный) — его прогноз даёт календарь клуба для FRR#."""
    best: dict[int, Any] = {}
    for p in bs.elements:
        cur = best.get(p.team)
        if cur is None or float(p.selected_by_percent or 0) > float(cur.selected_by_percent or 0):
            best[p.team] = p
    return tuple(sorted(p.id for p in best.values()))


def fixture_run_ranks(bs: Bootstrap, gw: int, gws: list[int]) -> dict[str, int]:
    """FRR#: клуб (сокращение) -> ранг календаря 1–20 (app/player_card.team_fixture_run_rank)."""
    reps = team_representatives(bs)
    preds = common.guarded(common.cached_predictions, gw, HORIZON, reps) or {}
    team_fsi = player_card.team_fsi_by_gw(preds, bs, gws)
    out: dict[str, int] = {}
    for team_id in team_fsi:
        rr = player_card.team_fixture_run_rank(team_fsi, team_id)
        if rr is not None:
            out[bs.team(team_id).short_name] = rr.rank
    return out


def header(ui: common.UI, preds: dict[int, Any]) -> None:
    ctx = ui.ctx
    assert ctx is not None
    entry = common.cached_entry(ui.manager_id) if ui.manager_id else None
    st.markdown(fmt.stat_strip_html(fmt.squad_header(ctx, entry, preds)), unsafe_allow_html=True)


def problems_line(problems: list[dict[str, Any]]) -> None:
    """«Haaland — под вопросом (75 %), колено · Saka — риск ротации (…)»; без серьёзных
    проблем строки нет вовсе."""
    text = fmt.problems_line(problems)
    if text:
        st.warning(text, icon=":material/warning:")


def squad_views(ui: common.UI, preds: dict[int, Any], problems: list[dict[str, Any]]) -> None:
    """Вкладки «Поле» (футболки по схеме, C / V, точки проблем, матч и прогноз на тур) и
    «Таблица» (smartplay: все колонки и календарь на 5 туров) — из одних и тех же строк."""
    ctx = ui.ctx
    assert ctx is not None and ctx.squad
    bs = common.get_tools().bootstrap
    gws = [ctx.gw + i for i in range(HORIZON)]
    frr = fixture_run_ranks(bs, ctx.gw, gws)
    xi, bench = fmt.squad_table_rows(ctx, preds, bs, gws=gws, frr=frr, problems=problems)
    tab_pitch, tab_table = st.tabs(["Поле", "Таблица"])
    with tab_pitch:
        starters, subs = ui_kit.pitch_from_squad_rows(xi, bench, gw=ctx.gw)
        total = briefing.xi_outlook(ctx, preds, [ctx.gw])
        st.markdown(
            ui_kit.pitch_html(
                starters,
                subs,
                player_url=fmt.PLAYER_URL,
                toolbar_left=f"Прогноз на GW{ctx.gw}: под именем — соперник и ожидаемые очки",
                toolbar_right=(f"{total[0][1]:.1f}", "очков с учётом капитана") if total else None,
            ),
            unsafe_allow_html=True,
        )
    with tab_table:
        st.markdown(fmt.squad_table_html(xi, bench, gw=ctx.gw, gws=gws), unsafe_allow_html=True)
        st.markdown(fmt.fsi_legend_html(), unsafe_allow_html=True)


def problems_section(ctx: Any, problems: list[dict[str, Any]], signals: dict[int, Any]) -> None:
    """Expander на каждого игрока из строки проблем: все замечания diagnose_squad по нему и
    цитаты из новостей (источник, дата, ссылка)."""
    if not problems:
        return
    st.subheader("Проблемы состава")
    by_player = fmt.issues_by_player(ctx.issues)
    for pr in problems:
        pid = int(pr["id"])
        with st.expander(f"{fmt.FLAG_ICON[pr['flag']]} {pr['text']}"):
            for i in by_player.get(pid, []):
                kind = fmt.ISSUE_KIND_RU.get(i["kind"], i["kind"])
                st.write(f"- **{kind}** (важность {i['severity']}): {i['detail']}")
            risk = signals.get(pid)
            if risk is not None and risk.origin != "unavailable":
                st.write(
                    f"Новости об игроке: {fmt.availability_text(risk)} — "
                    f"{fmt.template_ru(risk.summary)}"
                )
                if risk.evidence:
                    st.dataframe(
                        fmt.evidence_rows(risk.evidence),
                        hide_index=True,
                        width="stretch",
                        column_config={"Ссылка": st.column_config.LinkColumn("Ссылка")},
                    )
            else:
                st.caption("Новостей об игроке нет — обновите на странице «Игрок».")


def demo_mode() -> bool:
    """Кнопки-примеры (samples/) — только для демо и AppTest: `?demo=1` в адресе."""
    return str(st.query_params.get("demo", "")) == "1"


def screenshot_section(ui: common.UI) -> None:
    """Загрузка скриншота: одна строка-подсказка, uploader, «Распознать», короткий итог и
    таблица карточек, «Использовать этот состав»; подробности (модель, токены, $) — в popover."""
    st.caption(
        "Загрузите скриншот экрана Pick Team или Transfers — состав будет распознан и применён "
        "на всех страницах."
    )
    if not common.openai_ready():
        st.error("Распознавание недоступно: не задан OPENAI_API_KEY.")
        return
    up = st.file_uploader(
        "PNG / JPG / WEBP", type=["png", "jpg", "jpeg", "webp"], key="shot_upload"
    )
    if demo_mode():
        cols = st.columns(len(SAMPLES))
        for col, (label, path) in zip(cols, SAMPLES.items(), strict=True):
            if col.button(label, key=f"sample_{path.stem}"):
                st.session_state["shot_bytes"] = path.read_bytes()
                st.session_state["shot_name"] = path.name
                st.session_state.pop("parsed_squad", None)
    if up is not None:
        st.session_state["shot_bytes"] = up.getvalue()
        st.session_state["shot_name"] = up.name
    data = st.session_state.get("shot_bytes")
    if data is None:
        return
    st.image(data, caption=st.session_state.get("shot_name"), width=320)
    if st.button("Распознать", key="parse_shot", type="primary"):
        from fplcopilot.vision import squad_from_image

        parsed = None
        if llm_budget.allow_llm():
            with st.spinner("Распознаю состав…"):
                parsed = common.guarded(squad_from_image, data, common.get_tools().bootstrap)
        if parsed is not None:
            st.session_state["parsed_squad"] = parsed
    parsed = st.session_state.get("parsed_squad")
    if parsed is None:
        return
    text, kind = fmt.parsed_result_text(parsed)
    (st.success if kind == "success" else st.error)(text)
    st.dataframe(fmt.parsed_squad_rows(parsed), hide_index=True, width="stretch")
    if parsed.issues:
        st.dataframe(fmt.parsed_issue_rows(parsed), hide_index=True, width="stretch")
    c_use, c_details = st.columns([3, 1])
    if parsed.is_valid and c_use.button("Использовать этот состав", key="use_shot", type="primary"):
        from fplcopilot.vision import to_squad

        tools = common.get_tools()
        squad = to_squad(parsed, tools.bootstrap, manager_id=ui.squad_manager_id)
        override = tools.squad_override_from(squad, manager_id=ui.manager_id)
        common.set_override(override, st.session_state.get("shot_name"))
        st.rerun()
    with c_details.popover("Подробности"):
        st.table([{"Поле": k, "Значение": v} for k, v in fmt.parsed_details(parsed).items()])


ui = common.sidebar()
common.page_head(f"GW{ui.gw} · состав" if ui.gw else "Состав", "Мой состав")
if ui.ctx is None:
    st.stop()
if ui.has_squad:
    ctx = ui.ctx
    pids = tuple(p.id for p in ctx.squad or [])
    with st.spinner("Считаю прогноз и календарь…"):
        preds = common.guarded(common.cached_predictions, ctx.gw, HORIZON, pids) or {}
    problems = fmt.serious_problems(ctx, preds)
    signals: dict[int, Any] = {}
    if problems:  # сохранённые вердикты из новостей — только для проблемных игроков
        problem_ids = tuple(int(p["id"]) for p in problems)
        signals = common.guarded(common.cached_signals, problem_ids, ui.as_of_minute) or {}
    header(ui, preds)
    problems_line(problems)
    squad_views(ui, preds, problems)
    problems_section(ctx, problems, signals)
elif ui.manager_id is None:
    st.info(
        "Укажите ID менеджера FPL в панели слева — или загрузите скриншот состава ниже, "
        "он будет применён на всех страницах."
    )
else:
    st.warning(f"Состав из FPL недоступен: {ui.ctx.squad_note}")
    st.info("Загрузите скриншот состава ниже — он будет применён на всех страницах.")
with st.expander("Скриншот состава", expanded=not ui.has_squad):
    screenshot_section(ui)
