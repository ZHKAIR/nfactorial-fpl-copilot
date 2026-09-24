"""Страница «О системе»: граф агента из скомпилированного LangGraph (оба узла-прерывания
подсвечены), схема потока данных, таблица «страница -> инструменты». Пояснения — в help (?)
подзаголовков; без метрик и команд запуска.
"""

from __future__ import annotations

import streamlit as st

from fplcopilot.agent.tools import TOOL_NAMES
from fplcopilot.app import common
from fplcopilot.app import format as fmt

N_PAGES = 8  # Home.py: Брифинг · Мой состав · К дедлайну · План · Игрок · Сравнение · Чат · О системе

common.sidebar()
common.page_head("Как устроен FPL Copilot", "О системе")
st.markdown(
    "Языковые модели **маршрутизируют, извлекают и объясняют**; считают детерминированные "
    "инструменты: прогноз очков (xPts), оптимизатор состава (MILP), вердикты по игрокам из "
    "новостей (RAG). Каждое число на экране и в ответе чата приходит из инструмента уже "
    "округлённым, а текст объяснения проверяется кодом."
)

st.markdown(
    fmt.label_with_tip(
        "Граф агента",
        "Построен из скомпилированного графа LangGraph. Пунктир — условные рёбра (ветвление); "
        "grade_signals -> rewrite_retry — ограниченный цикл (не больше 2 повторов). Жёлтые узлы "
        "— прерывания: перед confirm_action (платный трансфер / Wildcard) и resolve_clarification "
        "(уточнение имени игрока) граф останавливается и ждёт решения человека на странице «Чат».",
        heading=True,
    ),
    unsafe_allow_html=True,
)
agent = common.guarded(common.get_agent) if common.openai_ready() else None
if agent is not None:
    st.graphviz_chart(fmt.agent_graph_dot(agent.graph.get_graph()), width="stretch")
else:
    st.info("Граф агента недоступен: не задан OPENAI_API_KEY.")

st.markdown(
    fmt.label_with_tip(
        "Поток данных",
        "FPL API, новости и скриншот -> Postgres, RAG, прогноз очков, оптимизатор и распознавание "
        f"скриншота -> LiveTools ({len(TOOL_NAMES)} инструментов с pydantic-схемами) -> агент, "
        "интерфейс и MCP-сервер.",
        heading=True,
    ),
    unsafe_allow_html=True,
)
st.graphviz_chart(fmt.pipeline_dot(len(TOOL_NAMES), N_PAGES), width="stretch")

st.markdown(
    fmt.label_with_tip(
        "Страница → инструменты",
        "Какие инструменты LiveTools вызывает каждая страница; чат работает через агента.",
        heading=True,
    ),
    unsafe_allow_html=True,
)
st.table([{"Страница": p, "Инструменты": t} for p, t in fmt.PAGE_TOOLS])
