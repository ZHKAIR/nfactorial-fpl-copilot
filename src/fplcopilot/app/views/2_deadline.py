"""Страница «К дедлайну»: лучший состав + капитан (optimize_team), top-3 маршрутов трансфера
(recommend_transfers) с вердиктами по платному трансферу, варианты капитана по владению.
"""

from __future__ import annotations

import threading
from datetime import timedelta

import streamlit as st

from fplcopilot.app import briefing, common, deadline_plan, ui_kit
from fplcopilot.app import format as fmt


def _why_hydrate(cache_key: str, payload: dict, use_llm: bool) -> None:
    from fplcopilot.core.why_narrate import narrate_why

    narrate_why(payload, disk_key=cache_key, use_llm=use_llm)


ui = common.sidebar()
common.page_head(
    f"GW{ui.gw} · решения до дедлайна" if ui.gw else "Решения до дедлайна",
    "К дедлайну",
    aside=common.deadline_aside(ui),
)
st.markdown(
    "<style>"
    '[data-testid="stHeading"] h3,[data-testid="stMetricLabel"]{'
    "letter-spacing:normal!important;word-spacing:normal!important;"
    "white-space:normal!important}"
    "</style>",
    unsafe_allow_html=True,
)
if not common.require_squad(ui):
    st.stop()
ctx = ui.ctx
assert ctx is not None
gw = ctx.gw

st.markdown(
    fmt.label_with_tip(
        "Разрешить платный трансфер (минус 4 очка)",
        "Вердикт по платному трансферу считается от лучшего бесплатного варианта: "
        "платный ход только если выигрыш выше порога выбранной стратегии.",
    ),
    unsafe_allow_html=True,
)
allow_hit = st.toggle(
    "Разрешить платный трансфер (минус 4 очка)",
    value=False,
    key="allow_hit",
    label_visibility="collapsed",
)
horizon = 3

with st.spinner("Оптимизирую состав…"):
    lu = common.guarded(
        common.cached_lineup, ui.squad_manager_id, gw, ui.strategy, ui.fp, ui.override
    )
with st.spinner("Считаю маршруты трансферов (top-3)…"):
    rt = common.guarded(
        common.cached_routes,
        ui.squad_manager_id,
        gw,
        ui.strategy,
        horizon,
        allow_hit,
        ui.fp,
        ui.override,
    )

sell: list[str] = []
buy: list[str] = []
if rt is not None and rt.recommended_rank is not None:
    rec = next((r for r in rt.routes if r.rank == rt.recommended_rank), None)
    if rec is not None:
        sell = list(rec.out)
        buy = list(rec.in_)

# Календари out/in + факты «Почему» (история / пенальти Understat). Текст — LLM или запасной.
why_preds: dict = {}
why_facts: dict = {}
why_data = None
if rt is not None and rt.routes:
    pids = tuple(sorted({pid for r in rt.routes for pid in (*r.out_ids, *r.in_ids)}))
    if pids:
        why_preds = common.guarded(common.cached_predictions, gw, horizon, pids) or {}
        from fplcopilot.core.why_facts import collect_facts_for_players, try_why_data

        try:
            why_data = try_why_data(common.get_tools(), gw, pids, ui.strategy)
            raw = collect_facts_for_players(pids, why_data)
            why_facts = {
                pid: [
                    {
                        "kind": f.kind,
                        "clause": f.clause,
                        "role": f.role,
                        "priority": f.priority,
                        "player_id": f.player_id,
                        "points": f.points,
                    }
                    for f in facts
                ]
                for pid, facts in raw.items()
            }
        except Exception:  # noqa: BLE001 — факты не роняют страницу
            why_facts = {}
            why_data = None

if lu is not None:
    sale = {}
    if ui.override is None and ui.manager_id and deadline_plan.recommended(rt) is not None:
        squad_ids = tuple(sorted(p.id for p in ctx.squad or []))
        sale = common.cached_sale_prices(ui.squad_manager_id, squad_ids)
    try:
        plan = deadline_plan.build_plan(
            lu, rt, ctx, common.get_tools().bootstrap, why_preds, sale, horizon=horizon
        )
    except Exception:  # noqa: BLE001 — объяснитель не роняет страницу
        common.log.warning("deadline plan failed", exc_info=True)
        plan = None
    if plan is not None:
        with st.container(border=True, key="card_plan"):
            st.markdown(
                ui_kit.section_label(f"План на GW{gw}", dot=True, tone="accent")
                + ui_kit.deadline_plan_html(plan, fmt.PLAYER_URL),
                unsafe_allow_html=True,
            )

st.subheader(f"Лучший состав на тур {gw}")
if lu is not None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Схема", lu.formation)
    c2.metric("Прогноз состава", fmt.num(lu.expected_points))
    c3.metric("Капитан / вице", f"{lu.captain} / {lu.vice}")
    if lu.current_xi_points is not None:
        # gap = лучший − ваш старт; подпись инвертирует знак: отставание — «−5.91 …».
        delta = lu.expected_points - lu.current_xi_points
        c4.metric(
            "Ваш старт",
            fmt.num(lu.current_xi_points),
            delta=briefing.vs_best_text(delta),
            delta_color="off",
            delta_arrow="down" if delta > 0.005 else "off",
        )
    if buy:
        st.markdown(fmt.buy_chips_html(buy), unsafe_allow_html=True)
    col_pitch, col_coach = st.columns([1.65, 1], gap="medium")
    with col_pitch:
        tab_pitch, tab_table = st.tabs(["Поле", "Таблица"])
        with tab_pitch:
            starters, subs = ui_kit.pitch_from_lineup(lu, sell=sell, buy=buy)
            st.markdown(
                ui_kit.pitch_html(
                    starters,
                    subs,
                    player_url=fmt.PLAYER_URL,
                    toolbar_left=f"Лучший состав на GW{gw}, схема {lu.formation}",
                    toolbar_right=(fmt.num(lu.expected_points, 1), "очков с учётом капитана"),
                ),
                unsafe_allow_html=True,
            )
        with tab_table:
            st.markdown(fmt.lineup_table_html(lu, sell=sell, buy=buy), unsafe_allow_html=True)
    with col_coach:
        with st.container(border=True, key="card_lineup_check"):
            start_names, bench_names = briefing.lineup_changes(lu, ctx)
            cap_note = briefing.captain_change(lu)
            if start_names or bench_names:
                body = ui_kit.news_item(
                    "Поменять стартовый состав",
                    (
                        f"Выпустить в старт: {', '.join(start_names) or '—'}. "
                        f"Убрать на скамейку: {', '.join(bench_names) or '—'}."
                    ),
                    "сравнение с лучшим составом на тур",
                )
            else:
                body = ui_kit.news_item(
                    "Стартовый состав менять не нужно",
                    "В старте те же игроки, что и в лучшем составе на тур.",
                    "сравнение с лучшим составом на тур",
                )
            if cap_note:
                body += ui_kit.news_item("Смените капитана", cap_note, "по прогнозу на тур", tone="warn")
            st.markdown(ui_kit.section_label("Проверка состава", tone="accent") + body, unsafe_allow_html=True)
        with st.container(border=True, key="card_captain_lab"):
            st.markdown(
                ui_kit.section_label("Варианты капитана")
                + ui_kit.rank_list(briefing.captain_ranks(lu))
                + f'<p class="fpl-reason" style="font-size:13px">{ui_kit.esc(fmt.CAPTAIN_TAG_HELP)}</p>',
                unsafe_allow_html=True,
            )
            if st.button("Спросить ассистента о капитане", key="ask_captain", type="tertiary"):
                common.ask_assistant("Кого поставить капитаном в этом туре?")

st.subheader("Трансферы")
if rt is not None:
    st.caption(fmt.routes_caption(rt, horizon=horizon))
    if not rt.routes:
        st.info("Маршрутов нет — держать трансфер.")
    jobs: list[tuple] = []
    pending_keys: list[str] = []
    narrated: list[tuple[str, str]] = []
    sig = ""
    if rt.routes:
        from fplcopilot.core.why_facts import enrich_route_penalty
        from fplcopilot.core.why_narrate import payload_hash

        bs = getattr(common.get_tools(), "bootstrap", None)
        for route in rt.routes:
            calendars, ownership = fmt.route_calendars_from_preds(
                route, why_preds, horizon=horizon  # type: ignore[arg-type]
            )
            fmap = (
                fmt.facts_by_route_name(route, why_facts)
                if hasattr(fmt, "facts_by_route_name")
                else {}
            )
            if why_data is not None and bs is not None:
                try:
                    outs = [bs.player(int(pid)) for pid in route.out_ids]
                    ins = [bs.player(int(pid)) for pid in route.in_ids]
                    extra = enrich_route_penalty(outs, ins, why_data)
                except Exception:  # noqa: BLE001
                    extra = None
                if extra is not None and route.in_:
                    taker = next(
                        (p.web_name for p in ins if p.penalties_order == 1),
                        route.in_[0],
                    )
                    fmap[taker] = list(fmap.get(taker, [])) + [extra]
            payload = fmt.compose_why_payload(
                route,
                horizon=horizon,
                calendars=calendars,
                ownership=ownership,
                facts=fmap,
            )
            cache_key = (
                f"v7-{gw}-{'-'.join(str(x) for x in route.out_ids)}-"
                f"{'-'.join(str(x) for x in route.in_ids)}-{ui.strategy}-"
                f"{payload_hash(payload)}"
            )
            jobs.append((route, calendars, ownership, fmap, payload, cache_key))

        from fplcopilot.core.why_narrate import disk_get, fallback_why

        use_llm = common.openai_ready()
        narrated: list[tuple[str, str]] = []
        pending_keys: list[str] = []
        for job in jobs:
            payload, cache_key = job[4], job[5]
            hit = disk_get(cache_key)
            if hit is not None:
                narrated.append((hit[0], hit[1] if hit[1] in ("llm", "fallback") else "fallback"))
            else:
                narrated.append((fallback_why(payload), "fallback"))
                pending_keys.append(cache_key)
        sig = "|".join(pending_keys)
        if pending_keys and st.session_state.get("why_bg_sig") != sig:
            st.session_state["why_bg_sig"] = sig
            for job in jobs:
                payload, cache_key = job[4], job[5]
                if disk_get(cache_key) is None:
                    threading.Thread(
                        target=_why_hydrate,
                        args=(cache_key, payload, use_llm),
                        daemon=True,
                    ).start()
        st.session_state["why_sources"] = [src for _, src in narrated]
        st.session_state["why_texts"] = [text for text, _ in narrated]

    cols = st.columns(max(1, len(rt.routes)))
    for i, (col, route) in enumerate(zip(cols, rt.routes, strict=False)):
        calendars, ownership, fmap = jobs[i][1], jobs[i][2], jobs[i][3]
        card_kw: dict = {
            "horizon": horizon,
            "calendars": calendars,
            "ownership": ownership,
            "facts": fmap,
        }
        card = fmt.route_card(route, **card_kw)
        card["why"] = narrated[i][0]
        with col.container(border=True, key=f"card_route_{route.rank}"):
            recommended = rt.recommended_rank == route.rank
            st.markdown(
                ui_kit.section_label(
                    f"Вариант {card['rank']}" + (" · рекомендуем" if recommended else ""),
                    dot=recommended,
                    tone="accent" if recommended else "",
                )
                + ui_kit.card_title(briefing.route_headline(route))
                + ui_kit.pills(briefing.route_card_pills(route, horizon)),
                unsafe_allow_html=True,
            )
            if card["verdict_note"]:
                st.markdown(
                    ui_kit.risk_note("Вердикт", card["verdict_note"]), unsafe_allow_html=True
                )
            with st.expander("Почему", expanded=False):
                st.write(card["why"])
                if pending_keys and jobs[i][5] in pending_keys:
                    st.caption("Уточняю формулировку…")
    if jobs and pending_keys:

        @st.fragment(run_every=timedelta(seconds=1))
        def _poll_why() -> None:
            from fplcopilot.core.why_narrate import disk_get as _dg

            if all(_dg(k) is not None for k in pending_keys) and st.session_state.get(
                "why_poll_done"
            ) != sig:
                st.session_state["why_poll_done"] = sig
                st.rerun()

        _poll_why()
    for note in rt.notes:
        st.caption(note)
