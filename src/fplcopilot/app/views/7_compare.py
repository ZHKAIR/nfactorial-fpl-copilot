"""Страница «Сравнение» — два игрока бок о бок + ИИ-объяснение по готовым фактам.

Выбор: поиск (как на «Игрок») или из состава; URL `?a=<pid>&b=<pid>`. Таблица метрик и
календарь на 6 туров — HTML; факты собирает app/compare.py; объяснение — gpt-4o-mini
(compare.system.md), кэш в session_state по (a, b, strategy).
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from fplcopilot.app import common, llm_budget
from fplcopilot.app import compare as cmp
from fplcopilot.app import format as fmt
from fplcopilot.app import player_card as card
from fplcopilot.core.ext_stats import load_ext_index, player_ext
from fplcopilot.data.schemas import Bootstrap, Player

HORIZON = cmp.FIXTURES_SHOWN
RUN_GWS = card.RUN_GWS
PARAM_A = "a"
PARAM_B = "b"


def _pid_param(name: str, bs: Bootstrap) -> int | None:
    raw = st.query_params.get(name)
    if not raw or not str(raw).strip().isdigit():
        return None
    pid = int(str(raw).strip())
    try:
        bs.player(pid)
    except KeyError:
        return None
    return pid


def _search_player(label: str, key_prefix: str, *, default_query: str = "") -> Player | None:
    """Текст + radio при неоднозначности (как на странице «Игрок»)."""
    q_key = f"{key_prefix}_query"
    pick_id = f"{key_prefix}_pick_id"
    pick_q = f"{key_prefix}_pick_query"
    if q_key not in st.session_state and default_query:
        st.session_state[q_key] = default_query
    query = st.text_input(label, key=q_key)
    if not query.strip():
        return None
    bs = common.get_tools().bootstrap
    found = bs.find_players(query.strip())
    chosen_id = st.session_state.get(pick_id)
    picked = (
        next((p for p in found if p.id == chosen_id), None)
        if chosen_id is not None and st.session_state.get(pick_q) == query
        else None
    )
    if not found:
        st.caption("Никого не нашёл — попробуйте фамилию как в FPL.")
        return None
    if picked is not None:
        return picked
    if len(found) == 1:
        return found[0]
    found = sorted(found, key=lambda p: -float(p.selected_by_percent or 0))[:12]
    labels = {fmt.player_label(p, bs): p for p in found}
    choice = st.radio("Уточните", list(labels), key=f"{key_prefix}_choice")
    return labels[choice]


def _squad_options(ui: common.UI) -> list[tuple[str, int]]:
    ctx = ui.ctx
    if ctx is None or not ctx.squad:
        return []
    bs = common.get_tools().bootstrap
    out = []
    for p in ctx.squad:
        try:
            pl = bs.player(p.id)
        except KeyError:
            continue
        out.append((fmt.player_label(pl, bs), pl.id))
    return out


def team_representatives(bs: Bootstrap) -> tuple[int, ...]:
    best: dict[int, Any] = {}
    for p in bs.elements:
        cur = best.get(p.team)
        if cur is None or float(p.selected_by_percent or 0) > float(cur.selected_by_percent or 0):
            best[p.team] = p
    return tuple(sorted(p.id for p in best.values()))


def fixture_run_rank_for(bs: Bootstrap, gw: int, team_id: int, preds: dict[int, Any]) -> int | None:
    gws = [gw + i for i in range(RUN_GWS)]
    team_fsi = card.team_fsi_by_gw(preds, bs, gws)
    run = card.team_fixture_run_rank(team_fsi, team_id)
    return run.rank if run else None


def render_explanation(expl: cmp.CompareExplanation, usage: Any) -> None:
    st.subheader("На ближайший тур")
    st.write(expl.verdict_next_gw)
    st.subheader("На 3 тура")
    st.write(expl.verdict_3gw)
    if expl.why:
        st.markdown("**Почему**")
        for item in expl.why:
            st.markdown(f"- {item}")
    st.markdown("**Из новостей**")
    if expl.news_points:
        for item in expl.news_points:
            st.markdown(f"- {item}")
    else:
        st.caption("Разбора новостей нет.")
    if expl.caveats:
        st.caption("Оговорки: " + " ".join(expl.caveats))
    with st.expander("Подробности", expanded=False):
        st.caption(
            f"Модель {usage.model} · {usage.prompt_tokens}+{usage.completion_tokens} токенов · "
            f"${usage.cost_usd:.4f} · {usage.latency_ms:.0f} мс"
        )


ui = common.sidebar()
common.page_head("Два игрока бок о бок", "Сравнение")
if ui.ctx is None:
    st.stop()

gw = ui.ctx.gw
tools = common.get_tools()
bs = tools.bootstrap

url_a, url_b = _pid_param(PARAM_A, bs), _pid_param(PARAM_B, bs)

# ---------- выбор игроков ----------

st.markdown(
    fmt.label_with_tip(
        "Как выбрать",
        "Два игрока для сравнения; ссылка с параметрами a и b сохраняет выбор",
    ),
    unsafe_allow_html=True,
)
mode = st.radio(
    "Как выбрать",
    ("Поиск", "Из состава"),
    horizontal=True,
    key="compare_mode",
    label_visibility="collapsed",
)

player_a: Player | None = None
player_b: Player | None = None

if mode == "Из состава":
    opts = _squad_options(ui)
    if len(opts) < 2:
        st.info("В составе меньше двух игроков — выберите через поиск.")
    else:
        labels = [o[0] for o in opts]
        id_by = {o[0]: o[1] for o in opts}
        default_a = next((i for i, o in enumerate(opts) if url_a and o[1] == url_a), 0)
        default_b = next(
            (i for i, o in enumerate(opts) if url_b and o[1] == url_b),
            1 if len(opts) > 1 else 0,
        )
        c1, c2 = st.columns(2)
        with c1:
            la = st.selectbox("Игрок A", labels, index=min(default_a, len(labels) - 1), key="sq_a")
        with c2:
            lb = st.selectbox("Игрок B", labels, index=min(default_b, len(labels) - 1), key="sq_b")
        player_a, player_b = bs.player(id_by[la]), bs.player(id_by[lb])
else:
    # Прямая ссылка ?a=&b= (в том числе «Compare» с карточки игрока) — подставить игроков по id,
    # если пара в адресе новая (поиск «Saka» неоднозначен)
    if (
        st.session_state.get("url_pair") != (url_a, url_b)
        and url_a is not None
        and url_b is not None
        and url_a != url_b
    ):
        pa, pb = bs.player(url_a), bs.player(url_b)
        st.session_state["cmp_a_query"] = pa.web_name
        st.session_state["cmp_a_pick_id"] = pa.id
        st.session_state["cmp_a_pick_query"] = pa.web_name
        st.session_state["cmp_b_query"] = pb.web_name
        st.session_state["cmp_b_pick_id"] = pb.id
        st.session_state["cmp_b_pick_query"] = pb.web_name
        st.session_state["compare_pair"] = (url_a, url_b)
        st.session_state["url_pair"] = (url_a, url_b)
    c1, c2 = st.columns(2)
    with c1:
        player_a = _search_player("Игрок A", "cmp_a")
    with c2:
        player_b = _search_player("Игрок B", "cmp_b")

compare_clicked = st.button(
    "Сравнить", type="primary", disabled=player_a is None or player_b is None
)

if compare_clicked and player_a is not None and player_b is not None:
    st.session_state["compare_pair"] = (player_a.id, player_b.id)
    st.query_params[PARAM_A] = str(player_a.id)
    st.query_params[PARAM_B] = str(player_b.id)

# Открытие по ссылке ?a=&b= (режим «Из состава» или уже выставленная пара)
if (
    st.session_state.get("compare_pair") is None
    and url_a is not None
    and url_b is not None
    and url_a != url_b
):
    st.session_state["compare_pair"] = (url_a, url_b)

pair = st.session_state.get("compare_pair")
ready = (
    player_a is not None
    and player_b is not None
    and pair == (player_a.id, player_b.id)
)
if not ready:
    st.caption("Выберите двух игроков и нажмите «Сравнить».")
    st.stop()

assert player_a is not None and player_b is not None
if player_a.id == player_b.id:
    st.warning("Выберите двух разных игроков.")
    st.stop()

# ---------- данные ----------

with st.spinner("Считаю прогнозы…"):
    preds = (
        common.guarded(common.cached_predictions, gw, HORIZON, (player_a.id, player_b.id)) or {}
    )
    reps = team_representatives(bs)
    # для FRR# — календарь всех клубов (как на «Мой состав»)
    league_preds = common.guarded(common.cached_predictions, gw, RUN_GWS, reps) or {}
    all_preds = {**league_preds, **preds}

pred_a, pred_b = preds.get(player_a.id), preds.get(player_b.id)
frr_a = fixture_run_rank_for(bs, gw, player_a.team, all_preds)
frr_b = fixture_run_rank_for(bs, gw, player_b.team, all_preds)

signals = (
    common.guarded(common.cached_signals, (player_a.id, player_b.id), ui.as_of_minute) or {}
)
risk_a, risk_b = signals.get(player_a.id), signals.get(player_b.id)

facts = cmp.build_compare_facts(
    player_a,
    player_b,
    pred_a,
    pred_b,
    risk_a,
    risk_b,
    bs=bs,
    frr_a=frr_a,
    frr_b=frr_b,
    strategy=ui.strategy,
)

# ---------- таблица ----------

st.subheader("Сравнение")
idx = common.guarded(load_ext_index, bs)
rows = cmp.metric_rows(
    player_a,
    player_b,
    pred_a,
    pred_b,
    frr_a=frr_a,
    frr_b=frr_b,
    gw=gw,
    ext_a=player_ext(player_a, idx),
    ext_b=player_ext(player_b, idx),
)
with st.container(border=True, key="card_compare_table"):
    st.markdown(
        cmp.compare_table_html(
            rows,
            player_a.web_name,
            player_b.web_name,
            team_a=bs.team(player_a.team).short_name,
            team_b=bs.team(player_b.team).short_name,
            pos_a=player_a.position.short,
            pos_b=player_b.position.short,
        ),
        unsafe_allow_html=True,
    )

st.subheader(f"Календарь (следующие {HORIZON} туров)")
fx_rows = cmp.fixture_compare_rows(pred_a, pred_b, player_a.web_name, player_b.web_name)
if not fx_rows:
    st.info("Календарь недоступен: нет прогноза.")
else:
    with st.container(border=True, key="card_compare_fixtures"):
        st.markdown(
            cmp.fixture_compare_html(fx_rows, player_a.web_name, player_b.web_name),
            unsafe_allow_html=True,
        )

# ---------- объяснение ИИ ----------

st.subheader("Объяснение")
cache_key = cmp.explanation_cache_key(player_a.id, player_b.id, ui.strategy)
cached = st.session_state.get(cache_key)

if cached is not None:
    render_explanation(cached["expl"], cached["usage"])
elif not common.openai_ready():
    st.info("Нужен OPENAI_API_KEY, чтобы получить объяснение. Таблица и факты уже готовы.")
    with st.expander("Факты (без ИИ)", expanded=False):
        for line in [
            facts["next_gw"].get("text"),
            facts["next_gw"].get("fixture_note"),
            facts["horizon_3"].get("text"),
            facts.get("calendar_summary"),
            *facts.get("news_lines", []),
        ]:
            if line:
                st.write(line)
elif llm_budget.allow_llm():
    with st.spinner("Готовлю объяснение…"):
        result = common.guarded(cmp.explain_compare, facts)
    if result is not None:
        expl, usage = result
        st.session_state[cache_key] = {"expl": expl, "usage": usage}
        render_explanation(expl, usage)
