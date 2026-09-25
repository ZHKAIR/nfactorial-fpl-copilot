"""Вход по паролю для прода: если задан `APP_PASSWORD`, Home.py до навигации показывает экран
входа, а скрипты страниц и сайдбар не выполняются (ни инструментов, ни OpenAI) до верного
пароля. Пустой `APP_PASSWORD` — вход не нужен (локальная разработка, тесты).

Сессия Streamlit живёт, пока открыт вебсокет. Ссылки на карточку игрока — обычные
`<a href="/player?pid=…">` (таблица состава, поле, лента новостей): браузер делает полный
переход, вебсокет новый, `session_state` пустой. Поэтому после входа пишем cookie
`fplc_auth` (как `fplc_manager` в manager_state): HMAC от пароля с сроком, читает новая
сессия из `st.context.cookies`. Смена `APP_PASSWORD` все cookie обесценивает.

Сравнение — `hmac.compare_digest` (время не зависит от совпавшего префикса); после неверного
пароля — задержка, растущая с числом попыток в сессии (перебор становится медленным).
"""

from __future__ import annotations

import hashlib
import hmac
import time

AUTH_KEY = "fplc_authenticated"
ATTEMPTS_KEY = "fplc_login_attempts"
SET_COOKIE_KEY = "fplc_set_auth_cookie"
COOKIE_WRITTEN_KEY = "fplc_auth_cookie_written"
COOKIE_NAME = "fplc_auth"
COOKIE_MAX_AGE_S = 14 * 24 * 3600
TOKEN_VERSION = "v1"
BASE_DELAY_S = 1.5
MAX_DELAY_S = 10.0
LOGO_URL = "/app/static/logo.svg"  # статика Streamlit (server.enableStaticServing)


def password_required(expected: str | None) -> bool:
    return bool(expected and expected.strip())


def check_password(entered: str | None, expected: str | None) -> bool:
    """Верный ли пароль; пустой ожидаемый — вход не требуется (True)."""
    if not password_required(expected):
        return True
    return hmac.compare_digest((entered or "").encode("utf-8"), str(expected).encode("utf-8"))


def failure_delay(attempts: int) -> float:
    """Пауза после неверного пароля: 1.5 с, 3 с, 4.5 с … не больше 10 с."""
    return min(MAX_DELAY_S, BASE_DELAY_S * max(1, attempts))


def issue_token(
    password: str, *, now: int | None = None, max_age: int = COOKIE_MAX_AGE_S
) -> tuple[str, int]:
    """Подпись входа: `{exp}.{hmac}`; ключ — сам пароль, в cookie пароля нет."""
    exp = int(now if now is not None else time.time()) + int(max_age)
    body = f"{TOKEN_VERSION}.{exp}".encode()
    mac = hmac.new(password.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"{exp}.{mac}", int(max_age)


def token_ok(password: str, token: str | None, *, now: int | None = None) -> bool:
    """Cookie ещё действует и подписан текущим паролем."""
    if not password or not token or "." not in token:
        return False
    exp_s, _, mac = token.partition(".")
    if not exp_s.isdigit() or not mac:
        return False
    exp = int(exp_s)
    if exp < int(now if now is not None else time.time()):
        return False
    body = f"{TOKEN_VERSION}.{exp}".encode()
    want = hmac.new(password.encode("utf-8"), body, hashlib.sha256).hexdigest()
    try:
        return hmac.compare_digest(mac, want)
    except Exception:  # noqa: BLE001 — разная длина подписи
        return False


def session_cookie() -> str | None:
    """Подпись входа с устройства: cookie запроса, открывшего сессию (вне браузера — None)."""
    try:
        import streamlit as st

        raw = st.context.cookies.get(COOKIE_NAME)
    except Exception:  # noqa: BLE001 — нет контекста клиента (bare-режим, AppTest)
        return None
    return raw or None


def cookie_script(token: str, max_age: int = COOKIE_MAX_AGE_S) -> str:
    """JS для документа приложения: запомнить вход (как manager_state.device_script)."""
    return (
        "<style>[data-testid='stElementContainer']:has(.fplc-auth){display:none}</style>"
        '<span class="fplc-auth"></span>'
        "<script>(function(){"
        f"var c='{COOKIE_NAME}={token}; Path=/; Max-Age={int(max_age)}; SameSite=Lax';"
        "if(location.protocol==='https:'){c+='; Secure';}"
        "document.cookie=c;"
        "})();</script>"
    )


def _remember_device(expected: str) -> None:
    """Один раз за сессию записать cookie, если браузер его ещё не прислал."""
    import streamlit as st

    force = bool(st.session_state.pop(SET_COOKIE_KEY, False))
    if st.session_state.get(COOKIE_WRITTEN_KEY) and not force:
        return
    if token_ok(expected, session_cookie()) and not force:
        st.session_state[COOKIE_WRITTEN_KEY] = True
        return
    token, age = issue_token(expected)
    st.html(cookie_script(token, age), unsafe_allow_javascript=True)
    st.session_state[COOKIE_WRITTEN_KEY] = True


LOGIN_CSS = """
<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"] { display: none; }
.st-key-login_card {
  max-width: 420px; margin: 6vh auto 0; background: var(--fpl-surface);
  border: 1px solid var(--fpl-line-strong); border-radius: 16px; padding: 28px 30px 22px;
  box-shadow: 0 12px 40px rgba(18, 32, 42, 0.08);
}
.fpl-login-crest { display: flex; justify-content: center; }
.fpl-login-crest img { width: 132px; height: 132px; }
.fpl-login-title { text-align: center; font-size: 24px; font-weight: 800; letter-spacing: -0.02em; margin: 6px 0 2px; }
.fpl-login-sub { text-align: center; font-size: 14px; color: var(--fpl-muted); margin: 0 0 14px; }
</style>
"""


def gate(expected: str | None, logo_url: str = LOGO_URL) -> bool:
    """True — можно показывать приложение; False — нарисован экран входа (вызвать st.stop())."""
    import streamlit as st

    if not password_required(expected):
        return True
    if st.session_state.get(AUTH_KEY):
        _remember_device(expected)
        return True
    if token_ok(expected, session_cookie()):
        st.session_state[AUTH_KEY] = True
        return True
    st.html(LOGIN_CSS)
    with st.container(key="login_card"):
        st.markdown(
            f'<div class="fpl-login-crest"><img src="{logo_url}" alt="FPL Copilot"></div>'
            '<div class="fpl-login-title">FPL Copilot</div>'
            '<p class="fpl-login-sub">Вход только для приглашённых. Введите пароль.</p>',
            unsafe_allow_html=True,
        )
        with st.form("login_form", border=False):
            entered = st.text_input(
                "Пароль", type="password", key="login_password", placeholder="Пароль"
            )
            submitted = st.form_submit_button("Войти", type="primary", width="stretch")
        if submitted:
            if check_password(entered, expected):
                st.session_state[AUTH_KEY] = True
                st.session_state[SET_COOKIE_KEY] = True
                st.session_state.pop(ATTEMPTS_KEY, None)
                st.rerun()
            attempts = int(st.session_state.get(ATTEMPTS_KEY, 0)) + 1
            st.session_state[ATTEMPTS_KEY] = attempts
            time.sleep(failure_delay(attempts))
            st.error("Неверный пароль")
    return False
