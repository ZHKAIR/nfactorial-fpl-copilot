"""Страница «Чат»: LangGraph-агент со стримингом узлов и human-in-the-loop.

Прогон -> `agent.stream(...)` (в UI только спиннер, без лога узлов) -> `agent.snapshot(thread_id)`.
Два вида прерываний:
- перед `confirm_action` (`pending_action`: хит / Wildcard) — кнопки «Подтвердить» / «Отклонить»
  -> `agent.stream_resume(thread_id, decision)`;
- перед `resolve_clarification` (`pending_clarification`: неоднозначное имя) — кнопка на каждого
  кандидата -> `agent.stream_resume(thread_id, player_id=...)`, граф продолжает с выбранным игроком.
"""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from fplcopilot.agent.chat import history_from_messages
from fplcopilot.app import common, llm_budget
from fplcopilot.app import format as fmt

CHAT_KEY = "chat_messages"
MAX_CANDIDATE_BUTTONS = 6


def run_agent(
    agent: Any, prompt: str, ui: common.UI, history: list[dict[str, Any]] | None = None
) -> dict[str, Any] | None:
    if not llm_budget.allow_llm():
        return None
    started = time.perf_counter()
    thread_id: str | None = None
    kwargs: dict[str, Any] = {"manager_id": ui.manager_id, "strategy": ui.strategy}
    if history:
        kwargs["history"] = history  # роутер переписывает уточнение в самостоятельный вопрос
    if ui.override is not None:
        kwargs["squad_override"] = ui.override
    try:
        with st.spinner("Думаю…"):
            for node, line in agent.stream(prompt, **kwargs):
                if node == "start":
                    thread_id = line.split()[-1]
        assert thread_id is not None
        state = agent.snapshot(thread_id)
    except Exception as exc:  # noqa: BLE001
        common.show_error(exc)
        return None
    state["elapsed_s"] = round(time.perf_counter() - started, 1)
    return state


def resume_agent(
    agent: Any, thread_id: str, decision: str | None, *, player_id: int | None = None
) -> dict[str, Any] | None:
    if not llm_budget.allow_llm():
        return None
    started = time.perf_counter()
    label = "Продолжаю…"
    try:
        with st.spinner(label):
            for _node, _line in agent.stream_resume(thread_id, decision, player_id=player_id):
                pass
        state = agent.snapshot(thread_id)
    except Exception as exc:  # noqa: BLE001
        common.show_error(exc)
        return None
    state["elapsed_s"] = round(time.perf_counter() - started, 1)
    return state


def candidate_label(c: dict[str, Any]) -> str:
    price = f"£{float(c['price']):.1f}" if c.get("price") is not None else "—"
    flag = " · в составе" if c.get("in_squad") else ""
    return f"{c.get('full_name') or c.get('name')} ({c.get('team')}, {c.get('position')}, {price}){flag}"


def render_clarification(
    idx: int, msg: dict[str, Any], agent: Any, pending: dict[str, Any]
) -> None:
    """Кнопки-кандидаты для неоднозначного имени (HITL №2) + «Отменить»."""
    st.warning(f"Уточнение: какого **{pending.get('mention')}** вы имеете в виду?")
    candidates = list(pending.get("candidates") or [])[:MAX_CANDIDATE_BUTTONS]
    cols = st.columns(2)
    chosen: int | None = None
    for i, c in enumerate(candidates):
        if cols[i % 2].button(
            candidate_label(c), key=f"choose_{idx}_{c['player_id']}", width="stretch"
        ):
            chosen = int(c["player_id"])
    cancel = st.button("Отменить уточнение", key=f"cancel_clarify_{idx}")
    if chosen is not None or cancel:
        msg["choice"] = {"player_id": chosen, "decision": "choose" if chosen else "cancel"}
        final = resume_agent(
            agent,
            msg["thread_id"],
            None if chosen else "cancel",
            player_id=chosen,
        )
        if final is not None:
            msg["final_state"] = final
        st.rerun()


def render_assistant(idx: int, msg: dict[str, Any], agent: Any) -> None:
    state = msg.get("final_state") or msg["state"]
    interrupted = bool(state.get("interrupted"))
    pending = state.get("pending_action") if interrupted else None
    pending_clar = state.get("pending_clarification") if interrupted else None
    if pending and msg.get("decision") is None:
        st.warning(fmt.pending_action_text(pending))
        c1, c2 = st.columns(2)
        decision = None
        if c1.button("Подтвердить", key=f"confirm_{idx}", icon=":material/check:", type="primary"):
            decision = "confirm"
        if c2.button("Отклонить", key=f"reject_{idx}", icon=":material/close:"):
            decision = "reject"
        if decision:
            msg["decision"] = decision
            final = resume_agent(agent, msg["thread_id"], decision)
            if final is not None:
                msg["final_state"] = final
            st.rerun()
    elif pending_clar and msg.get("choice") is None:
        render_clarification(idx, msg, agent, pending_clar)
    if msg.get("decision"):
        st.caption(
            f"Решение: {'подтверждено' if msg['decision'] == 'confirm' else 'отклонено — пересчёт без хита/Wildcard'}"
        )
    if msg.get("choice"):
        ch = msg["choice"]
        st.caption(
            f"Уточнение: выбран игрок id {ch['player_id']}"
            if ch.get("player_id")
            else "Уточнение отменено"
        )
    if state.get("answer"):
        st.markdown(state["answer"])
    elif not pending and not pending_clar:
        st.info("Ответа нет.")


ui = common.sidebar()
common.page_head("Ассистент", "Чат с ассистентом")
if not common.openai_ready():
    st.error(
        "OPENAI_API_KEY не задан — агент недоступен. Заполните .env и перезапустите приложение."
    )
    st.stop()
agent = common.guarded(common.get_agent)
if agent is None:
    st.stop()

follow_gw = f"GW{ui.gw + 1}" if ui.gw and ui.gw < 38 else "следующий тур"
st.caption(
    f"Вопросы на русском или английском. Можно уточнять («а на {follow_gw}?», «а если без хита?») — "
    "чат помнит последние реплики. Платный трансфер и Wildcard требуют подтверждения."
)
with st.container(key="quick_prompts", horizontal=True, gap="small"):
    for i, example in enumerate(fmt.example_prompts(ui.gw)):
        if st.button(example, key=f"example_{i}"):
            st.session_state["queued_prompt"] = example

messages: list[dict[str, Any]] = st.session_state.setdefault(CHAT_KEY, [])
for idx, msg in enumerate(messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_assistant(idx, msg, agent)

prompt = st.chat_input("Спросите про состав, трансферы, капитана, план…")
prompt = prompt or st.session_state.pop("queued_prompt", None)
if prompt:
    history = history_from_messages(messages)  # до текущей реплики
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        state = run_agent(agent, prompt, ui, history)
        if state is not None:
            msg = {
                "role": "assistant",
                "state": state,
                "thread_id": state["thread_id"],
                "decision": None,
                "choice": None,
            }
            messages.append(msg)
            render_assistant(len(messages) - 1, msg, agent)
if messages and st.button("Очистить историю", key="clear_chat"):
    st.session_state[CHAT_KEY] = []
    st.rerun()
