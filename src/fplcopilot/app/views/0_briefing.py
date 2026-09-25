"""Страница «Брифинг» (стартовая, `/`): что важно до дедлайна — по референсу «The Briefing».

Шапка: приветствие по имени менеджера (FPL entry), одна фраза о том, что нужно сделать,
карточка дедлайна. Карточки: главный ход (рекомендуемый маршрут recommend_transfers: продаём ->
покупаем, выигрыш за 3 тура, 2–3 аргумента, риски покупки из risk_note, «почему» — в
раскрывающемся блоке), капитан (два лучших варианта optimize_team), «Следить до дедлайна»
(серьёзные проблемы состава + сохранённый разбор новостей) и прогноз на тур (лучший состав
против вашего старта, общий ранг). Внизу — строка вопроса ассистенту. Действия — только
переходы на страницы и вопрос в чат.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from fplcopilot.app import briefing, common, ui_kit
from fplcopilot.app import format as fmt

HORIZON = fmt.FRR_HORIZON  # тот же ключ кэша прогнозов, что на «Мой состав»
ROUTE_HORIZON = 3  # тот же ключ кэша маршрутов, что на «К дедлайну» (без хита)
DEADLINE_PAGE = "views/2_deadline.py"
SQUAD_PAGE = "views/1_squad.py"
PLAN_PAGE = "views/3_plan.py"
COMPARE_PAGE = "views/7_compare.py"


def focus_card(route: Any, rt: Any, preds: dict[int, Any]) -> None:
    bs = common.get_tools().bootstrap
    tours = fmt.plural(ROUTE_HORIZON, "тур", "тура", "туров")
    with st.container(border=True, key="card_focus"):
        if route is None:
            st.markdown(
                ui_kit.section_label("Главный ход", dot=True, tone="accent")
                + ui_kit.card_title("Трансфер можно не делать", "Трансферы")
                + (
                    ui_kit.text(fmt.routes_caption(rt, horizon=ROUTE_HORIZON))
                    if rt is not None
                    else ""
                ),
                unsafe_allow_html=True,
            )
        else:
            outs, ins = briefing.route_sides(route, bs)
            gain = fmt.points_text(route.gain_horizon, signed=True)
            number, word = gain.split(" ", 1)
            risks = briefing.buy_risks(route)
            st.markdown(
                ui_kit.section_label("Главный ход", dot=True, tone="accent")
                + ui_kit.card_title(briefing.route_headline(route), "Рекомендуем трансфер")
                + ui_kit.transfer_row(outs, ins, number, f"{word} за {tours}")
                + ui_kit.pills(briefing.route_pills(route, preds))
                + (
                    ui_kit.risk_note("Что может пойти не так", "; ".join(risks) + ".")
                    if risks
                    else ""
                ),
                unsafe_allow_html=True,
            )
            calendars, ownership = fmt.route_calendars_from_preds(
                route, preds, horizon=ROUTE_HORIZON
            )
            why = fmt.route_card(
                route, horizon=ROUTE_HORIZON, calendars=calendars, ownership=ownership
            )["why"]
            with st.expander("Почему этот трансфер"):
                st.markdown(ui_kit.text(why), unsafe_allow_html=True)
        c1, c2, _ = st.columns([1.5, 1.4, 1.1])
        if c1.button(
            "Подробнее к дедлайну", key="brief_to_deadline", type="primary", width="stretch"
        ):
            common.go(DEADLINE_PAGE)
        if c2.button("Спросить ассистента", key="brief_ask_route", width="stretch"):
            q = (
                f"Стоит ли продать {', '.join(route.out)} ради {', '.join(route.in_)}?"
                if route is not None
                else "Стоит ли делать трансфер в этом туре?"
            )
            common.ask_assistant(q)


def captain_card(lu: Any) -> None:
    pair = briefing.captain_pair(lu)
    with st.container(border=True, key="card_captain"):
        if not pair:
            st.markdown(
                ui_kit.section_label("Капитан") + ui_kit.card_title(lu.captain),
                unsafe_allow_html=True,
            )
            return
        a, b = pair[0], pair[1] if len(pair) > 1 else None

        def side(o: Any) -> tuple[str, str, str, str]:
            return (o.name, o.team, ui_kit.fixture_short(o.fixture), f"{o.captain_points:.1f}")

        st.markdown(
            ui_kit.section_label("Капитан")
            + ui_kit.versus(side(a), side(b) if b else None)
            + ui_kit.text(briefing.captain_sentence(lu)),
            unsafe_allow_html=True,
        )
        if b is not None and st.button(
            "Сравнить двух игроков", key="brief_compare", type="tertiary"
        ):
            common.go(COMPARE_PAGE, a=str(a.id), b=str(b.id))


def watch_card(rows: list[dict[str, Any]]) -> None:
    """Одна строка на игрока старта: точка, имя (ссылка на карточку), короткий статус, источник."""
    with st.container(border=True, key="card_watch"):
        if not rows:
            body = ui_kit.news_item(
                "Серьёзных проблем нет",
                "У игроков старта нормальный статус FPL и нет риска ротации.",
                tone="good",
            )
            label = ui_kit.section_label("Следить до дедлайна")
        else:
            body = ui_kit.watch_list_html(rows, fmt.PLAYER_URL)
            label = ui_kit.section_label("Следить до дедлайна", tone="warn")
        st.markdown(label + body, unsafe_allow_html=True)


def news_card(items: list[dict[str, Any]]) -> None:
    """Лента: важные новости состава + свежие заголовки из корпуса RAG."""
    with st.container(border=True, key="card_news"):
        st.markdown(
            ui_kit.section_label(f"Новости за {briefing.NEWS_DAYS} дней")
            + (
                ui_kit.news_ticker_html(items, fmt.PLAYER_URL)
                if items
                else ui_kit.text(
                    "Свежих новостей пока нет — ни по составу, ни в корпусе за эти дни.",
                    muted=True,
                )
            ),
            unsafe_allow_html=True,
        )


def forecast_card(lu: Any, gw: int, share: str | None, entry: Any) -> None:
    """Прогноз на ближайший тур: лучший состав из ваших 15 против вашего нынешнего старта —
    оба числа из optimize_team; и место в общем зачёте. Прогноз по турам с трансферами и без —
    на странице «План»."""
    with st.container(border=True, key="card_forecast"):
        st.markdown(ui_kit.section_label(f"Прогноз на GW{gw}"), unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        if lu is not None:
            c1.markdown(
                ui_kit.big_number(f"{lu.expected_points:.1f}", "лучший состав из ваших 15"),
                unsafe_allow_html=True,
            )
            if lu.current_xi_points is not None:
                c2.markdown(
                    ui_kit.big_number(
                        f"{lu.current_xi_points:.1f}", "ваш нынешний старт", tone="plain"
                    ),
                    unsafe_allow_html=True,
                )
        rank = entry.get("rank") if entry else None
        if rank is not None:
            c3.markdown(
                ui_kit.big_number(
                    f"{int(rank):,}".replace(",", " "),
                    f"общий ранг · {share} менеджеров" if share else "общий ранг",
                    tone="plain",
                ),
                unsafe_allow_html=True,
            )
        if st.button("Прогноз по турам — на странице «План»", key="brief_to_plan", type="tertiary"):
            common.go(PLAN_PAGE)


ui = common.sidebar()
if ui.ctx is None:
    common.page_head("Брифинг", "Брифинг")
    st.stop()
ctx = ui.ctx
gw = ctx.gw
entry = common.cached_entry(ui.manager_id) if ui.manager_id else None
hour = ui.now.astimezone().hour
title = briefing.greeting((entry or {}).get("first_name"), hour)

if not ui.has_squad:
    common.page_head(f"GW{gw} · брифинг перед дедлайном", title, aside=common.deadline_aside(ui))
    common.require_squad(ui)
    if st.button("Открыть «Мой состав»", key="brief_to_squad", type="primary"):
        common.go(SQUAD_PAGE)
    st.stop()

pids = tuple(p.id for p in ctx.squad or [])
with st.spinner("Собираю брифинг: прогноз, лучший состав, трансферы…"):
    preds = common.guarded(common.cached_predictions, gw, HORIZON, pids) or {}
    lu = common.guarded(
        common.cached_lineup, ui.squad_manager_id, gw, ui.strategy, ui.fp, ui.override
    )
    rt = common.guarded(
        common.cached_routes,
        ui.squad_manager_id,
        gw,
        ui.strategy,
        ROUTE_HORIZON,
        False,
        ui.fp,
        ui.override,
    )
problems = briefing.watch_filter(fmt.serious_problems(ctx, preds), ctx, lu)
# сохранённые разборы новостей по всему составу (cached_only, без LLM) — для строки статуса и ленты
signals: dict[int, Any] = common.guarded(common.cached_signals, pids, ui.as_of_minute) or {}
route = briefing.recommended_route(rt)
route_preds: dict[int, Any] = {}
if route is not None:
    route_pids = tuple(sorted({*route.out_ids, *route.in_ids}))
    route_preds = common.guarded(common.cached_predictions, gw, ROUTE_HORIZON, route_pids) or {}

common.page_head(
    f"GW{gw} · брифинг перед дедлайном",
    title,
    briefing.lead_text(route, rt, lu, problems),
    aside=common.deadline_aside(ui),
)

left, right = st.columns([1.6, 1], gap="large")
with left:
    focus_card(route, rt, route_preds)
with right:
    if lu is not None:
        captain_card(lu)
    watch_card(briefing.watch_rows(problems, signals))

names = {p.id: p.name for p in ctx.squad or []}
articles = common.guarded(common.cached_recent_articles, briefing.NEWS_DAYS) or []
news_card(briefing.briefing_news(signals, names, ui.now, articles))

share = briefing.rank_share(
    (entry or {}).get("rank"), getattr(common.get_tools().bootstrap, "total_players", None)
)
forecast_card(lu, gw, share, entry)

# Строка вопроса ассистенту: компактная форма (не st.chat_input — тот в корне прокручивает
# страницу вниз), закреплена у нижнего края экрана CSS-правилом `.st-key-ask_bar` (sticky).
with st.container(key="ask_bar"), st.form("ask_form", clear_on_submit=True, border=False):
    c_icon, c_text, c_send = st.columns([0.04, 0.76, 0.2], vertical_alignment="center")
    c_icon.markdown(":blue[:material/smart_toy:]")
    question = c_text.text_input(
        "Вопрос ассистенту",
        placeholder="Спросите ассистента: трансфер, капитан, состав…",
        label_visibility="collapsed",
        key="brief_ask",
    )
    sent = c_send.form_submit_button(
        "Спросить", icon=":material/arrow_upward:", type="primary", width="stretch"
    )
if sent and question.strip():
    common.ask_assistant(question.strip())
