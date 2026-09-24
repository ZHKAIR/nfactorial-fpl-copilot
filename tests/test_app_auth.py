"""Вход по паролю (app/auth.py) и суточный лимит LLM (app/llm_budget.py): чистые функции и
AppTest точки входа Home.py — без пароля страница не рендерится, неверный — ошибка, верный —
Брифинг. Фейки — из tests/test_app_smoke.py (без сети/LLM)."""

from __future__ import annotations

from datetime import date

import pytest
from streamlit.testing.v1 import AppTest
from test_agent_fakes import FakeRouter, router_output
from test_app_smoke import APP, TIMEOUT, fakes, make_agent  # noqa: F401 — фикстура fakes

from fplcopilot.app import auth, common, llm_budget


def test_check_password_constant_time_and_optional():
    assert auth.check_password("s3cret", "s3cret")
    assert not auth.check_password("s3cre", "s3cret")
    assert not auth.check_password("", "s3cret")
    assert not auth.check_password(None, "s3cret")
    assert auth.check_password("", None) and auth.check_password("x", "  ")  # пароль не задан
    assert auth.check_password("пароль", "пароль")  # не-ASCII
    assert not auth.password_required("") and auth.password_required("x")


def test_failure_delay_grows_and_is_capped():
    assert auth.failure_delay(1) == 1.5
    assert auth.failure_delay(3) == 4.5
    assert auth.failure_delay(100) == auth.MAX_DELAY_S


def test_daily_budget_limits_and_resets_next_day():
    day = [date(2026, 9, 25)]
    b = llm_budget.DailyBudget(limit=2, today=lambda: day[0])
    assert b.spend() and b.spend() and not b.spend()
    assert b.used == 2 and b.remaining == 0
    day[0] = date(2026, 9, 26)  # новые сутки — счётчик с нуля
    assert b.remaining == 2 and b.spend()
    unlimited = llm_budget.DailyBudget(limit=None)
    assert all(unlimited.spend() for _ in range(100)) and unlimited.remaining is None
    assert "2 в сутки" in llm_budget.limit_message(2)


@pytest.fixture
def home(fakes, monkeypatch):  # noqa: F811
    agent = make_agent(fakes, FakeRouter(router_output("captain")))
    monkeypatch.setattr(common, "get_agent", lambda: agent)
    monkeypatch.setattr(common.settings, "app_password", "s3cret")
    monkeypatch.setattr(auth, "failure_delay", lambda attempts: 0.0)
    at = AppTest.from_file(str(APP / "Home.py"), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at, fakes


def login(at: AppTest, password: str) -> None:
    at.text_input(key="login_password").input(password)
    next(b for b in at.button if b.label == "Войти").click().run()
    assert not at.exception, [e.value for e in at.exception]


def test_home_without_password_shows_only_login(home):
    at, tools = home
    assert at.text_input(key="login_password").proto.type == 1  # PASSWORD
    assert not at.title and not at.sidebar.text_input  # ни страницы, ни сайдбара
    assert not tools.calls  # ни одного вызова инструментов до входа


def test_home_wrong_password_error_then_right_password_opens_briefing(home):
    at, tools = home
    login(at, "wrong")
    assert [e.value for e in at.error] == ["Неверный пароль"]
    assert not at.title and not tools.calls
    assert at.session_state[auth.ATTEMPTS_KEY] == 1
    login(at, "s3cret")
    assert not at.error
    assert at.title[0].value.endswith(", Test.")  # Брифинг
    assert at.session_state[auth.AUTH_KEY] is True
