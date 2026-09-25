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
    assert at.session_state[auth.COOKIE_WRITTEN_KEY] is True
    scripts = [h.proto.body for h in at.get("html") if auth.COOKIE_NAME in h.proto.body]
    assert scripts and f"{auth.COOKIE_NAME}=" in scripts[0] and "SameSite=Lax" in scripts[0]


def test_issue_token_roundtrip_and_rejects_tamper_or_expiry():
    token, age = auth.issue_token("s3cret", now=1_000_000, max_age=100)
    assert age == 100 and token.startswith("1000100.")
    assert auth.token_ok("s3cret", token, now=1_000_050)
    assert not auth.token_ok("s3cret", token, now=1_000_101)  # истек
    assert not auth.token_ok("other", token, now=1_000_050)
    exp, _, mac = token.partition(".")
    assert not auth.token_ok("s3cret", f"{exp}.{mac[:-1]}0", now=1_000_050)
    assert not auth.token_ok("s3cret", None) and not auth.token_ok("s3cret", "nope")
    assert not auth.token_ok("", token) and not auth.token_ok("s3cret", "abc.short")


def test_cookie_script_sets_signed_cookie():
    js = auth.cookie_script("1000100.abc", max_age=99)
    assert "fplc_auth=1000100.abc; Path=/; Max-Age=99; SameSite=Lax" in js
    assert "Secure" in js and "document.cookie=c" in js


def test_home_valid_cookie_skips_login(fakes, monkeypatch):  # noqa: F811
    token, _ = auth.issue_token("s3cret")  # срок от «сейчас», иначе token_ok в gate() отвергнет
    agent = make_agent(fakes, FakeRouter(router_output("captain")))
    monkeypatch.setattr(common, "get_agent", lambda: agent)
    monkeypatch.setattr(common.settings, "app_password", "s3cret")
    monkeypatch.setattr(auth, "session_cookie", lambda: token)
    monkeypatch.setattr(auth, "failure_delay", lambda attempts: 0.0)
    at = AppTest.from_file(str(APP / "Home.py"), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state[auth.AUTH_KEY] is True
    assert at.title[0].value.endswith(", Test.")
    assert not [t for t in at.text_input if t.key == "login_password"]
