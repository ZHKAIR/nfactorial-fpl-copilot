"""Страница «Игрок» — карточка по образцу smartplayfpl.com/players/N (раскладка, порядок, плотность).

Поиск по bootstrap -> шапка «Player report» (цена · позиция · клуб, имя, клуб и статус, крупный
xPts, строка ключевых чисел xPts · Points · Form · 3GW xPts · Own% · Net Tx) -> секции парами
(Overview, Minutes, Attacking Output, Advanced Attacking, Defence & Reliability) строками «число +
процентиль среди игроков позиции», синяя точка — метрика входит в прогноз xPts -> Ownership &
Transfers полосой -> «Fixtures (next 6 GWs)» чипами -> похожие игроки (Competes for minutes,
Alternatives at the price, The differential bridge; «Compare» открывает «Сравнение»). Наше,
чего нет у smartplay, — ниже в раскрывающихся блоках: прогноз xPts по турам и компоненты, рейтинг
календаря, «Откуда очки», стандарты и Understat, новости об игроке (сохранённый разбор /
обновление RAG + LLM), новости клуба. Логика — в app/player_card.py (чистые функции).

Прямая ссылка: `/player?pid=<FPL element id>` открывает карточку сразу (поле поиска заполняется
именем); выбор через поиск или кнопку «Открыть» обновляет `?pid=` в адресе. Невалидный pid
игнорируется — показывается поиск.
"""

from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st

from fplcopilot.agent.llm import estimate_cost
from fplcopilot.agent.tools import PlayerRiskInput
from fplcopilot.app import common, form_notes_view, llm_budget, team_news_view, ui_kit
from fplcopilot.app import format as fmt
from fplcopilot.app import player_card as card
from fplcopilot.config import settings
from fplcopilot.core.ext_stats import load_ext_index, setpiece_phrase

HORIZON = card.FIXTURES_SHOWN  # 6 туров: календарь и прогноз
PICK_ID = "player_pick_id"  # выбранный игрок (кнопка «Открыть», ?pid=) -> id
PICK_QUERY = "player_pick_query"  # ... и текст поиска, при котором выбор действует
URL_PID = "player_url_pid"  # последний применённый ?pid= из адреса (чтобы не перебивать поиск)
PID_PARAM = "pid"  # /player?pid=<FPL element id> — ссылки со страницы «Мой состав»


@st.cache_data(ttl=common.CACHE_TTL, show_spinner=False)
def full_match_share() -> dict[int, float] | None:
    """player_id -> доля матчей с >= 60 мин по player_gw_history; None — БД недоступна
    (тогда Mins% считается из bootstrap)."""
    from fplcopilot.core.history import load_history

    try:
        rows = load_history()
    except Exception:  # noqa: BLE001 — без БД карточка работает по bootstrap
        return None
    return card.full_match_share(rows)


@st.cache_data(ttl=common.CACHE_TTL, show_spinner=False)
def cached_points_origin(pid: int, last_n: int, gw: int, pos: int) -> dict | None:
    """Раскладка очков за last_n туров. БД недоступна — None, страница жива."""
    from fplcopilot.core.history import load_history
    from fplcopilot.core.points_form import points_breakdown
    from fplcopilot.data.schemas import Position

    try:
        rows = load_history(before_gw=gw, player_ids=[pid])
    except Exception:  # noqa: BLE001
        return None
    window = points_breakdown(pid, last_n, history=rows, position=Position(pos))
    return None if window is None else window.as_dict()


def select_player(p) -> None:
    """Подставить игрока в поиск и зафиксировать выбор. Вызывается до создания поля поиска:
    из callback кнопки «Открыть» или при открытии по ссылке ?pid=."""
    st.session_state["player_query"] = p.web_name
    st.session_state[PICK_ID] = p.id
    st.session_state[PICK_QUERY] = p.web_name


def pid_from_url(bs) -> int | None:
    """?pid=<id> из адреса; невалидный (не число / нет в bootstrap) — молча игнорируется."""
    raw = st.query_params.get(PID_PARAM)
    if not raw or not str(raw).strip().isdigit():
        return None
    pid = int(str(raw).strip())
    try:
        bs.player(pid)
    except KeyError:
        return None
    return pid


ui = common.sidebar()
common.page_head("Карточка игрока", "Игрок")
COMPARE_PAGE = "views/7_compare.py"
if ui.ctx is None:
    st.stop()
gw = ui.ctx.gw
tools = common.get_tools()
bs = tools.bootstrap

# ---------- открытие по ссылке /player?pid=<id> ----------

url_pid = pid_from_url(bs)
if url_pid is not None and st.session_state.get(URL_PID) != url_pid:
    select_player(bs.player(url_pid))  # новый pid в адресе -> открыть карточку, минуя поиск
    st.session_state[URL_PID] = url_pid

# ---------- поиск ----------

query = st.text_input("Поиск игрока (фамилия или часть имени)", key="player_query")
player = None
if query.strip():
    found = bs.find_players(query.strip())
    pick_id = st.session_state.get(PICK_ID)
    picked = (
        next((p for p in found if p.id == pick_id), None)
        if pick_id is not None and st.session_state.get(PICK_QUERY) == query
        else None
    )
    if not found:
        st.info("Никого не нашёл — попробуйте фамилию как в FPL (например, «Palmer»).")
    elif picked is not None:
        player = picked
    elif len(found) == 1:
        player = found[0]
    else:
        found = sorted(found, key=lambda p: -float(p.selected_by_percent or 0))[:12]
        labels = {fmt.player_label(p, bs): p for p in found}
        choice = st.radio("Уточните игрока", list(labels), key="player_choice")
        player = labels[choice]
if player is None:
    if PID_PARAM in st.query_params:  # игрок не выбран — ссылка не должна на него указывать
        del st.query_params[PID_PARAM]
    st.session_state.pop(URL_PID, None)
    st.stop()
# адрес страницы всегда указывает на открытого игрока: /player?pid=<id> можно копировать
if st.query_params.get(PID_PARAM) != str(player.id):
    st.query_params[PID_PARAM] = str(player.id)
st.session_state[URL_PID] = player.id

# ---------- прогноз игрока и кандидатов той же позиции ----------

with st.spinner("Считаю xPts…"):
    preds = common.guarded(common.cached_predictions, gw, HORIZON, (player.id,)) or {}
pred = preds.get(player.id)
same_position = tuple(
    sorted(p.id for p in bs.elements if p.element_type == player.element_type and p.id != player.id)
)
with st.spinner("Сравниваю с игроками той же позиции…"):
    pool_preds = common.guarded(common.cached_predictions, gw, HORIZON, same_position) or {}
all_preds = {**pool_preds, **preds}
ext = common.guarded(load_ext_index, bs)


def ext_row(pid: int):
    return None if ext is None else ext.player_row(pid)


# ---------- шапка «Player report» (как у smartplay) ----------

head = card.header(player, bs)
keys = card.key_numbers(player, pred)
with st.container(border=True, key="card_player_report"):
    left, right = st.columns([3, 1], vertical_alignment="center")
    with left:
        st.markdown(card.report_tags_html(head, player), unsafe_allow_html=True)
        st.header(head.name)
        st.markdown(card.report_club_html(head), unsafe_allow_html=True)
    right.markdown(card.report_xpts_html(keys[0].value), unsafe_allow_html=True)
    st.markdown(card.key_numbers_html(keys), unsafe_allow_html=True)

st.markdown(
    '<div class="fpl-model-legend"><i></i>синяя точка у метрики — она входит в прогноз xPts '
    "(наведите, чтобы увидеть как); остальные — справка. Процент — место среди игроков той же "
    f"позиции с игровым временем (≥ {card.MIN_POOL_MINUTES} мин за сезон).</div>",
    unsafe_allow_html=True,
)

# ---------- секции статистики с перцентилями ----------

sections = {
    s.key: s for s in card.stat_sections(player, bs, mins_share=full_match_share(), ext_row=ext_row)
}
for pair in (("overview", "minutes"), ("attack", "advanced"), ("defence", None)):
    cols = st.columns(2, gap="medium")
    for col, key in zip(cols, pair, strict=True):
        if key and key in sections:
            col.markdown(card.section_card_html(sections[key]), unsafe_allow_html=True)
if "ownership" in sections:
    st.markdown(card.ownership_strip_html(sections["ownership"]), unsafe_allow_html=True)

# ---------- календарь ----------

st.markdown(
    fmt.label_with_tip(
        f"Fixtures (next {HORIZON} GWs)",
        "Сложность матча (индекс FSI 1–5) по xG-модели: 1 — очень лёгкий соперник, 5 — очень "
        "тяжёлый. Коэффициенты букмекеров не показываются.",
        heading=True,
    ),
    unsafe_allow_html=True,
)
cells = card.fixture_cells(pred, HORIZON)
if not cells:
    st.info("Календарь недоступен: нет прогноза.")
else:
    st.markdown(card.fixtures_strip_html(cells), unsafe_allow_html=True)
    st.markdown(card.fixture_legend_html(), unsafe_allow_html=True)

# ---------- похожие игроки ----------

st.subheader(f"Players related to {player.web_name}")


def render_similar(items: list[card.SimilarPlayer], key: str) -> None:
    if not items:
        st.caption("Подходящих игроков нет.")
        return
    cols = st.columns(3, gap="small")
    for col, s in zip(cols, items, strict=False):
        with col.container(border=True, key=f"card_sim_{key}_{s.id}"):
            st.markdown(card.similar_card_html(s), unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            if b1.button("Compare", key=f"cmp_{key}_{s.id}", width="stretch"):
                common.go(COMPARE_PAGE, a=str(player.id), b=str(s.id))
            b2.button(
                "Открыть",
                key=f"{key}_{s.id}",
                on_click=select_player,
                args=(bs.player(s.id),),
                type="tertiary",
                width="stretch",
            )


st.markdown("#### Competes for minutes")
st.caption("Тот же клуб и та же позиция — конкуренты за место в старте; статус показан как есть.")
render_similar(card.competitors_for_minutes(player, bs, all_preds), "open_comp")

st.markdown("#### Alternatives at the price")
st.caption(
    f"Та же позиция, цена в пределах ±£{card.PRICE_WINDOW:.1f}m, другой клуб, доступен сейчас."
)
render_similar(card.alternatives_at_price(player, bs, all_preds), "open_alt")

st.markdown("#### The differential bridge")
st.caption(
    f"Та же позиция, владение < {card.DIFFERENTIAL_OWNERSHIP:.0f} %, цена не выше и календарь "
    f"не тяжелее на ближайшие {card.SIMILAR_GWS} тура."
)
render_similar(card.differential_bridge(player, bs, all_preds), "open_diff")

# ---------- наше: модель, стандарты, новости — ниже и отдельно ----------

st.subheader("Модель FPL Copilot и новости")

with st.expander("Прогноз xPts по турам", expanded=True):
    if pred is None or not pred.by_gw:
        st.warning("Прогноза нет (нет матчей в горизонте или игрок вне модели).")
    else:
        series = fmt.xpts_series(pred)
        st.markdown(ui_kit.bars(list(series.items())), unsafe_allow_html=True)
        nxt = pred.by_gw[0]
        st.caption(
            f"Вероятность выхода в старте {fmt.num(nxt.p_start)} · ожидаемые минуты "
            f"{round(nxt.exp_minutes)} на GW{nxt.gw}."
        )
        st.dataframe(fmt.components_rows(pred), hide_index=True, width="stretch")
        if nxt.notes:
            st.caption("Заметки модели: " + "; ".join(nxt.notes))
    run_gws = [g.gw for g in pred.by_gw[: card.RUN_GWS]] if pred else []
    run = card.team_fixture_run_rank(card.team_fsi_by_gw(all_preds, bs, run_gws), player.team)
    if run is not None:
        st.markdown(
            fmt.label_with_tip(
                f"Рейтинг календаря команды: #{run.rank} из {run.n_teams}",
                f"Ранг {head.club} по средней сложности матчей (FSI) на ближайшие "
                f"{fmt.plural(len(run_gws), 'тур', 'тура', 'туров')}: 1 — самый лёгкий календарь "
                f"среди клубов лиги. Средняя сложность команды: {run.avg_fsi:.2f}; тур без "
                "матча считается как максимальная сложность.",
            ),
            unsafe_allow_html=True,
        )

origin = cached_points_origin(
    player.id, int(getattr(settings, "why_form_last_n", 3)), gw, int(player.position)
)
origin_rows = card.points_origin_rows(origin)
if origin_rows is not None:
    title, rows, label, total = origin_rows
    with st.expander(title):
        st.markdown(card.points_origin_html(rows, label, total), unsafe_allow_html=True)

with st.expander("Стандарты и Understat"):
    st.markdown(card.section_card_html(card.setpiece_section(player)), unsafe_allow_html=True)
    phrase = setpiece_phrase(player, bs.team(player.team))
    if phrase:
        st.caption(phrase)
    u_section = card.understat_section(player, ext_row(player.id))
    if u_section is not None:
        st.markdown(card.section_card_html(u_section), unsafe_allow_html=True)
    else:
        st.caption("Understat: игрок не сопоставился или снимок пуст — показываем только поля FPL.")

with st.expander("Новости об игроке", expanded=player.status != "a"):
    as_of = datetime.now(UTC)
    bcol, tcol = st.columns([8, 1])
    clicked_refresh = bcol.button(
        "Обновить из новостей",
        key="refresh_signal",
        disabled=not common.openai_ready(),
    )
    tcol.markdown(
        fmt.TIP_WIDGET_CSS + fmt.tip("Поиск свежих новостей и разбор LLM, ~$0.001"),
        unsafe_allow_html=True,
    )
    if clicked_refresh and llm_budget.allow_llm():
        with st.spinner("Ищу новости и разбираю…"):
            risk = common.guarded(
                tools.analyze_player_risk,
                PlayerRiskInput(player_id=player.id, as_of=as_of, force=True),
            )
        if risk is not None:
            st.session_state["fresh_risk"] = risk
            tools.invalidate()  # новый вердикт должен попасть в модель минут / xPts
            st.cache_data.clear()
    risk = st.session_state.get("fresh_risk")
    if risk is not None and risk.player_id != player.id:
        risk = None
    if risk is None:
        risk = (common.guarded(common.cached_signals, (player.id,), ui.as_of_minute) or {}).get(
            player.id
        )
    if risk is None:
        st.info("Разбора новостей нет.")
    else:
        if risk.origin == "extracted":
            cost = estimate_cost(
                risk.model or "gpt-4o-mini", risk.prompt_tokens, risk.completion_tokens
            )
            st.success("Разбор сделан сейчас: " + fmt.extraction_cost_text(risk, cost))
        elif risk.origin == "unavailable":
            note = fmt.template_ru(risk.note)
            st.info(
                "Сохранённого разбора нет"
                + (f" ({note})" if note else "")
                + f"; FPL: {fmt.status_text(risk.fpl_status, risk.fpl_chance)}"
            )
        else:
            note = fmt.template_ru(risk.note)
            st.caption(
                f"Вердикт по новостям, {fmt.age_text(risk.age_h)}" + (f" — {note}" if note else "")
            )
        st.table(card.news_verdict_rows(risk))
        if risk.summary:
            st.write(fmt.template_ru(risk.summary))
        if risk.evidence:
            st.dataframe(
                fmt.evidence_rows(risk.evidence),
                hide_index=True,
                width="stretch",
                column_config={"Ссылка": st.column_config.LinkColumn("Ссылка")},
            )
        form_notes_view.render(risk)

with st.expander("Новости клуба"):
    team_news_view.render(
        tools, player, datetime.now(UTC), openai_ready=common.openai_ready(), guarded=common.guarded
    )
