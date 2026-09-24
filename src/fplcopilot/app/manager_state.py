"""ID менеджера FPL в сайдбаре: адрес `?manager=`, сессия, cookie «запомнить на этом устройстве».

Источник применённого ID (первый подходящий): `?manager=<id>` в адресе → `session_state
["manager_id"]` → cookie `fplc_manager` этого браузера → `APP_DEFAULT_MANAGER_ID` (только
локальная разработка / демо, по умолчанию пусто) → пусто. `FPL_MANAGER_ID` — дефолт CLI и
скриптов, в UI не подставляется: каждый новый пользователь начинает с пустого поля.

- Адрес держится синхронным с применённым ID на каждом запуске страницы: меню `st.navigation`
  очищает query params, сайдбар возвращает `manager`, остальные параметры (`pid`, `a`/`b`,
  `demo`) не трогает. Мусор в `?manager=` молча убирается; ID, которого нет в FPL (404), —
  убирается с сообщением.
- Поле ввода — в форме: Enter и «Сохранить» — одно действие (проверить в FPL, применить,
  запомнить на устройстве); «Забыть» (или «Сохранить» с пустым полем) — стереть ID отовсюду.
- Cookie пишет `<script>` в `st.html(..., unsafe_allow_javascript=True)` (не iframe — документ
  приложения), читает новая сессия из заголовков открывшего её запроса (`st.context.cookies`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any, Literal

import httpx
import streamlit as st

from fplcopilot.app import format as fmt
from fplcopilot.config import settings

log = logging.getLogger(__name__)

URL_PARAM = "manager"
COOKIE_NAME = "fplc_manager"
COOKIE_MAX_AGE_S = 365 * 24 * 3600
MAX_MANAGER_ID = 2**31 - 1  # entry id FPL — int32; больше — заведомо 404
ENTRY_CHECK_TTL = 600

MANAGER_KEY = "manager_id"  # int в session_state: применённый ID, 0 — без менеджера (контракт)
MANAGER_TEXT_KEY = "manager_id_text"  # str: содержимое поля ввода
SAVE_KEY = "manager_save"
FORGET_KEY = "manager_forget"
_READY = "manager_sources_ready"  # cookie / дефолт уже разобраны в этой сессии
_USER_SET = "manager_user_set"  # применено кнопкой: в этом запуске адрес не перебивает сессию
_ERROR = "manager_error"  # str | None: сообщение под полем до следующего применения
_UNVERIFIED = "manager_unverified"  # ID, сохранённый без проверки (FPL API не ответил)
_DEVICE_OP = "manager_device_op"  # int: записать cookie (id) / стереть (0) в этом запуске
_TOAST = "manager_toast"

HELP = (
    "Число из адреса вашей команды: fantasy.premierleague.com/entry/ID/ (раздел Points). "
    "«Сохранить» или Enter — применить и запомнить на этом устройстве, «Забыть» — стереть. "
    "ID попадает и в адрес страницы — ссылкой можно поделиться."
)
WHERE_HINT = (
    "Где взять ID: [сайт FPL](https://fantasy.premierleague.com/) → Points — число в адресе "
    "`/entry/<ID>/`"
)
NOT_NUMBER = "Введите число"
UNVERIFIED_CAPTION = "FPL API не ответил — ID сохранён без проверки"

EntryCheck = Literal["ok", "not_found", "unverified"]
EntryLookup = Callable[[int], Mapping[str, Any] | None]
EntryFetch = Callable[[int], Any]


def parse_stored(raw: Any) -> int | None:
    """Значение из адреса / cookie -> ID; всё, кроме положительного числа из ASCII-цифр, — None."""
    s = "" if raw is None else str(raw).strip()
    if not (s.isascii() and s.isdigit()):
        return None
    mid = int(s)
    return mid if 0 < mid <= MAX_MANAGER_ID else None


def device_cookie() -> int | None:
    """ID, запомненный на этом устройстве: cookie запроса, открывшего сессию (вне браузера — None)."""
    try:
        return parse_stored(st.context.cookies.get(COOKIE_NAME))
    except Exception:  # noqa: BLE001 — нет контекста клиента (bare-режим, AppTest)
        return None


def app_default() -> int | None:
    """APP_DEFAULT_MANAGER_ID — предзаполнение для локальной разработки / демо."""
    return parse_stored(getattr(settings, "app_default_manager_id", None))


@st.cache_data(ttl=ENTRY_CHECK_TTL, show_spinner=False)
def _entry_exists(manager_id: int, _fetch: EntryFetch) -> bool:
    """entry/{id} есть в FPL? 404 -> False (кэшируется); прочие ошибки пробрасываются и не кэшируются."""
    try:
        _fetch(manager_id)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return False
        raise
    return True


def check_entry(manager_id: int, fetch: EntryFetch) -> EntryCheck:
    if manager_id > MAX_MANAGER_ID:
        return "not_found"
    try:
        return "ok" if _entry_exists(manager_id, fetch) else "not_found"
    except Exception:  # сеть / 5xx: сохраняем без проверки
        log.warning("entry(%s) check failed", manager_id, exc_info=True)
        return "unverified"


def not_found_text(manager_id: int, *, from_url: bool = False) -> str:
    where = " из ссылки" if from_url else ""
    return f"Команда с ID {manager_id}{where} не найдена в FPL"


def _apply(manager_id: int, status: EntryCheck = "ok") -> None:
    ss = st.session_state
    ss[MANAGER_KEY] = manager_id
    ss[MANAGER_TEXT_KEY] = str(manager_id) if manager_id else ""
    ss[_ERROR] = None
    ss[_UNVERIFIED] = manager_id if status == "unverified" else None


def _on_save(fetch: EntryFetch) -> None:
    """Enter / «Сохранить»: проверить ID в FPL, применить и запомнить на устройстве."""
    ss = st.session_state
    parsed = fmt.parse_manager_id(ss.get(MANAGER_TEXT_KEY))
    if parsed is None:
        ss[_ERROR] = NOT_NUMBER  # прежний ID остаётся применённым
        return
    if parsed == 0:
        _on_forget()
        return
    status = check_entry(parsed, fetch)
    if status == "not_found":
        ss[_ERROR] = not_found_text(parsed)
        return
    _apply(parsed, status)
    ss[_USER_SET] = True
    ss[_DEVICE_OP] = parsed
    ss[_TOAST] = (
        "ID сохранён на этом устройстве"
        if status == "ok"
        else "ID сохранён на этом устройстве, но не проверен: FPL API не ответил"
    )


def _on_forget() -> None:
    """«Забыть»: пусто в поле, в сессии, в адресе и в cookie."""
    ss = st.session_state
    _apply(0)
    ss[_USER_SET] = True
    ss[_DEVICE_OP] = 0
    ss[_TOAST] = "ID забыт на этом устройстве"


def _sync_url(manager_id: int) -> None:
    """`?manager=` = применённый ID (или нет параметра); остальные параметры не трогаем."""
    want = str(manager_id) if manager_id else None
    have = st.query_params.get_all(URL_PARAM)
    if want is None:
        if have:
            del st.query_params[URL_PARAM]
    elif have != [want]:
        st.query_params[URL_PARAM] = want


def resolve(fetch: EntryFetch) -> int:
    """Применённый ID на этот запуск (0 — без менеджера); адрес синхронизируется с ним."""
    ss = st.session_state
    user_set = ss.pop(_USER_SET, False)
    raw = st.query_params.get(URL_PARAM)
    url_error = None
    if raw is not None and not user_set:
        url_id = parse_stored(raw)  # мусор -> None, _sync_url уберёт / перепишет параметр
        if url_id is not None and url_id != ss.get(MANAGER_KEY):
            status = check_entry(url_id, fetch)
            if status == "not_found":
                url_error = not_found_text(url_id, from_url=True)
            else:
                _apply(url_id, status)
    if not ss.get(_READY):
        ss[_READY] = True
        if MANAGER_KEY not in ss:
            _apply(device_cookie() or app_default() or 0)
    if url_error:
        ss[_ERROR] = url_error
    manager_id = int(ss.get(MANAGER_KEY) or 0)
    ss[MANAGER_KEY] = manager_id
    # id виджета включает hash скрипта страницы: «трогаем» ключ до создания поля на каждой странице
    ss[MANAGER_TEXT_KEY] = ss.get(MANAGER_TEXT_KEY, str(manager_id) if manager_id else "")
    _sync_url(manager_id)
    return manager_id


def device_script(manager_id: int) -> str:
    """JS для документа приложения: запомнить ID в cookie (manager_id > 0) или стереть (0).
    Контейнер элемента скрывается CSS-правилом, чтобы в сайдбаре не было пустого отступа."""
    value, age = (str(int(manager_id)), COOKIE_MAX_AGE_S) if manager_id else ("", 0)
    return (
        "<style>[data-testid='stElementContainer']:has(.fplc-device){display:none}</style>"
        '<span class="fplc-device"></span>'
        "<script>(function(){"
        f"var c='{COOKIE_NAME}={value}; Path=/; Max-Age={age}; SameSite=Lax';"
        "if(location.protocol==='https:'){c+='; Secure';}"
        "document.cookie=c;"
        "})();</script>"
    )


def sidebar_input(*, entry: EntryLookup, fetch: EntryFetch) -> int | None:
    """Блок «ID менеджера FPL» в сайдбаре. entry — карточка для подписи «<команда> · очки · ранг»
    (None — 404 / сеть), fetch — прямой запрос entry/{id} для проверки при сохранении."""
    manager_id = resolve(fetch) or None
    st.sidebar.markdown(
        fmt.label_with_tip("ID менеджера FPL", HELP),
        unsafe_allow_html=True,
    )
    with st.sidebar.form("manager_form", border=False):
        st.text_input(
            "ID менеджера FPL",
            key=MANAGER_TEXT_KEY,
            placeholder="например, 1234567",
            label_visibility="collapsed",
        )
        # flex-строка, а не st.columns: колонки на узком экране складываются в столбик
        with st.container(horizontal=True, vertical_alignment="center"):
            # первая кнопка формы — её «нажимает» Enter в поле
            st.form_submit_button("Сохранить", key=SAVE_KEY, on_click=_on_save, args=(fetch,))
            st.form_submit_button("Забыть", key=FORGET_KEY, on_click=_on_forget, type="tertiary")
    error = st.session_state.get(_ERROR)
    if error:
        st.sidebar.caption(f":red[{error}]")
    card = entry(manager_id) if manager_id else None
    if manager_id and card is None and st.session_state.get(_UNVERIFIED) == manager_id:
        st.sidebar.caption(UNVERIFIED_CAPTION)
    else:
        st.sidebar.caption(fmt.manager_caption(manager_id, card))
    if not manager_id:
        st.sidebar.caption(WHERE_HINT)
    op = st.session_state.pop(_DEVICE_OP, None)
    if op is not None:
        st.sidebar.html(device_script(op), unsafe_allow_javascript=True)
    toast = st.session_state.pop(_TOAST, None)
    if toast:
        st.toast(toast)
    return manager_id
