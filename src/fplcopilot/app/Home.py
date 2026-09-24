# ruff: noqa: N999 — Streamlit-конвенция: заглавная точка входа Home.py
"""Точка входа Streamlit: конфигурация страницы, эмблема, общий CSS и навигация по `views/`.

uv run streamlit run src/fplcopilot/app/Home.py

При заданном APP_PASSWORD сначала экран входа (app/auth.py). Навигация — вкладками в шапке; сайдбар — панель настроек (менеджер, стратегия, данные).
Стартовая страница (`/`) — «Брифинг»; «Мой состав» — `/squad`; последняя — «О системе».
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from fplcopilot.app import auth, theme
from fplcopilot.config import settings

HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="FPL Copilot", page_icon=str(theme.FAVICON), layout="wide")
st.logo(str(theme.LOGO), size="large", icon_image=str(theme.LOGO))
theme.inject()
# Прод: при APP_PASSWORD — экран входа; до верного пароля страницы и сайдбар не выполняются.
if not auth.gate(settings.app_password):
    st.stop()

# Страницы лежат в views/ (не pages/): каталог pages/ Streamlit подхватывает как «старую»
# многостраничность и до первого вызова st.navigation показывает сырые имена файлов.
VIEWS = HERE / "views"
pages = [
    st.Page(str(VIEWS / "0_briefing.py"), title="Брифинг", default=True),  # url /
    st.Page(str(VIEWS / "1_squad.py"), title="Мой состав", url_path="squad"),
    st.Page(str(VIEWS / "2_deadline.py"), title="К дедлайну", url_path="deadline"),
    st.Page(str(VIEWS / "3_plan.py"), title="План", url_path="plan"),
    st.Page(str(VIEWS / "4_player.py"), title="Игрок", url_path="player"),
    st.Page(str(VIEWS / "7_compare.py"), title="Сравнение", url_path="compare"),
    st.Page(str(VIEWS / "5_chat.py"), title="Чат", url_path="chat"),
    st.Page(str(VIEWS / "6_about.py"), title="О системе", url_path="about"),
]
st.navigation(pages, position="top").run()
