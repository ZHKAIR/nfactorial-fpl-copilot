"""ID менеджера в сайдбаре (app/manager_state.py): `?manager=` в адресе, сессия, cookie устройства,
APP_DEFAULT_MANAGER_ID; форма «Сохранить» / «Забыть»; мусор и 404 в адресе; остальные query
params (pid, a/b, demo) не затираются. Фейки — из tests/test_app_smoke.py (без сети/LLM)."""

from __future__ import annotations

import httpx
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from test_agent_fakes import BS, NOW
from test_app_smoke import APP, TIMEOUT, AppTools, _FrozenDatetime

from fplcopilot.app import common, manager_state
from fplcopilot.data.schemas import Entry

PLAYER = str(APP / "views" / "4_player.py")
TEAMS = {1: ("Nfactorial_test", 312, 1234567), 7: ("Seven FC", 280, 50)}
OFFLINE_ID = 99  # entry/99 -> сеть недоступна


class _Client:
    """entry/{id}: 1 и 7 — известные команды, 99 — нет сети, остальные — 404 как у FPL API."""

    def __init__(self) -> None:
        self.entry_calls: list[int] = []

    def bootstrap(self, *, refresh: bool = False):
        return BS

    def entry(self, manager_id: int) -> Entry:
        self.entry_calls.append(manager_id)
        req = httpx.Request("GET", f"https://fantasy.premierleague.com/api/entry/{manager_id}/")
        if manager_id == OFFLINE_ID:
            raise httpx.ConnectError("offline", request=req)
        if manager_id not in TEAMS:
            raise httpx.HTTPStatusError(
                "404", request=req, response=httpx.Response(404, request=req)
            )
        name, points, rank = TEAMS[manager_id]
        return Entry(
            id=manager_id,
            player_first_name="Test",
            player_last_name="Manager",
            name=name,
            summary_overall_points=points,
            summary_overall_rank=rank,
        )

    def close(self):
        pass


class Tools(AppTools):
    def __init__(self) -> None:
        super().__init__()
        self.client = _Client()


@pytest.fixture
def fakes(monkeypatch):
    """Как владелец: FPL_MANAGER_ID задан, APP_DEFAULT_MANAGER_ID пуст, cookie нет."""
    st.cache_data.clear()
    tools = Tools()
    monkeypatch.setattr(common, "datetime", _FrozenDatetime)
    monkeypatch.setattr(common, "get_tools", lambda: tools)
    monkeypatch.setattr(common, "openai_ready", lambda: True)
    monkeypatch.setattr(
        common, "freshness", lambda gw: {"news_at": NOW, "signal_at": NOW, "xpts_rows": 700}
    )
    monkeypatch.setattr(manager_state.settings, "fpl_manager_id", 1)
    monkeypatch.setattr(manager_state.settings, "app_default_manager_id", None)
    monkeypatch.setattr(manager_state, "device_cookie", lambda: None)
    yield tools
    st.cache_data.clear()


def remembered(monkeypatch, manager_id: int) -> None:
    """Cookie `fplc_manager` пришла с запросом, открывшим сессию."""
    monkeypatch.setattr(manager_state, "device_cookie", lambda: manager_id)


def page(path: str = PLAYER, **params: str) -> AppTest:
    at = AppTest.from_file(path, default_timeout=TIMEOUT)
    for k, v in params.items():
        at.query_params[k] = v
    return at


def run(at: AppTest) -> AppTest:
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def qp(at: AppTest, name: str) -> str | None:
    """Параметр адреса: AppTest хранит значения query_params списками."""
    raw = at.query_params.get(name)
    if isinstance(raw, list):
        return raw[-1] if raw else None
    return raw


def captions(at: AppTest) -> list[str]:
    return [c.value for c in at.sidebar.caption]


def applied(at: AppTest) -> int:
    return at.session_state[common.MANAGER_KEY]


def device_scripts(at: AppTest) -> list[str]:
    return [h.proto.body for h in at.sidebar.get("html")]


def save(at: AppTest, text: str) -> AppTest:
    at.sidebar.text_input[0].input(text)
    at.button(key=manager_state.SAVE_KEY).click()
    return run(at)


def forget(at: AppTest) -> AppTest:
    at.button(key=manager_state.FORGET_KEY).click()
    return run(at)


# ---------- чистые функции ----------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("7", 7),
        (" 007 ", 7),
        (10835228, 10835228),
        (None, None),
        ("", None),
        ("0", None),
        ("-5", None),
        ("12a", None),
        ("1e3", None),
        ("١٢", None),  # не-ASCII цифры
        (str(2**31), None),
    ],
)
def test_parse_stored(raw, expected):
    assert manager_state.parse_stored(raw) == expected


def test_device_script_sets_and_deletes_cookie():
    js = manager_state.device_script(7)
    assert "fplc_manager=7; Path=/; Max-Age=31536000; SameSite=Lax" in js
    assert "Secure" in js and "document.cookie=c" in js
    assert "fplc_manager=; Path=/; Max-Age=0; SameSite=Lax" in manager_state.device_script(0)


def test_app_default_parses_setting(monkeypatch):
    for raw, expected in (("123", 123), ("", None), ("abc", None), (None, None)):
        monkeypatch.setattr(manager_state.settings, "app_default_manager_id", raw)
        assert manager_state.app_default() == expected


# ---------- источники и приоритет ----------


def test_new_user_gets_empty_field_despite_fpl_manager_id(fakes):
    at = run(page())
    assert at.sidebar.text_input[0].value == "" and applied(at) == 0
    caps = captions(at)
    assert caps[0] == "Без менеджера"
    assert any("/entry/<ID>/" in c and "Points" in c for c in caps)  # где взять ID
    assert "manager" not in at.query_params
    assert fakes.client.entry_calls == []
    assert not device_scripts(at)


def test_url_manager_applied_and_other_params_kept(fakes):
    at = run(page(pid="4", manager="7"))
    assert applied(at) == 7 and at.sidebar.text_input[0].value == "7"
    assert captions(at)[0] == "Seven FC · 280 очков · ранг 50"
    assert qp(at, "manager") == "7" and qp(at, "pid") == "4"
    assert [h.value for h in at.header] == ["Gabriel dos Santos Magalhães"]
    assert not device_scripts(at)  # ID из ссылки не запоминается на устройстве


@pytest.mark.parametrize("bad", ["abc", "0", "-5", "12a", " ", "99999999999"])
def test_garbage_manager_param_removed_silently(fakes, bad):
    at = run(page(pid="4", manager=bad))
    assert "manager" not in at.query_params and qp(at, "pid") == "4"
    assert applied(at) == 0
    assert not any(c.startswith(":red[") for c in captions(at))
    assert fakes.client.entry_calls == []


def test_garbage_manager_param_replaced_by_remembered_id(fakes, monkeypatch):
    remembered(monkeypatch, 7)
    at = run(page(pid="4", manager="abc"))
    assert applied(at) == 7 and qp(at, "manager") == "7" and qp(at, "pid") == "4"


def test_url_manager_missing_in_fpl_dropped_with_message(fakes, monkeypatch):
    remembered(monkeypatch, 7)
    at = run(page(manager="42"))
    assert applied(at) == 7 and qp(at, "manager") == "7"  # дальше — cookie устройства
    assert ":red[Команда с ID 42 из ссылки не найдена в FPL]" in captions(at)


@pytest.mark.parametrize(
    ("url", "session", "cookie", "default", "expected"),
    [
        ("7", 1, 1, "1", 7),  # адрес > сессия
        (None, 7, 1, "1", 7),  # сессия > устройство
        (None, None, 7, "1", 7),  # устройство > APP_DEFAULT_MANAGER_ID
        (None, None, None, "1", 1),  # APP_DEFAULT_MANAGER_ID > пусто
        (None, None, None, None, 0),  # FPL_MANAGER_ID (=1) в UI не идёт
    ],
)
def test_source_priority(fakes, monkeypatch, url, session, cookie, default, expected):
    monkeypatch.setattr(manager_state.settings, "app_default_manager_id", default)
    if cookie is not None:
        remembered(monkeypatch, cookie)
    at = page(**({"manager": url} if url else {}))
    if session is not None:
        at.session_state[common.MANAGER_KEY] = session
    run(at)
    assert applied(at) == expected
    assert qp(at, "manager") == (str(expected) if expected else None)
    assert at.sidebar.text_input[0].value == (str(expected) if expected else "")


def test_url_restored_after_navigation_and_followed_on_change(fakes):
    """Меню st.navigation очищает query params — сайдбар возвращает `manager`; новый ID в адресе
    (назад в истории, ручная правка) применяется."""
    at = run(page(manager="7"))
    del at.query_params["manager"]
    run(at)
    assert applied(at) == 7 and qp(at, "manager") == "7"
    at.query_params["manager"] = "1"
    run(at)
    assert applied(at) == 1 and captions(at)[0] == "Nfactorial_test · 312 очков · ранг 1 234 567"


def test_compare_and_demo_params_kept(fakes, monkeypatch):
    remembered(monkeypatch, 7)
    at = run(page(str(APP / "views" / "7_compare.py"), a="4", b="4"))
    assert (qp(at, "a"), qp(at, "b"), qp(at, "manager")) == ("4", "4", "7")
    at = run(page(str(APP / "views" / "1_squad.py"), demo="1"))
    assert qp(at, "demo") == "1" and qp(at, "manager") == "7"


# ---------- «Сохранить» / «Забыть» ----------


def test_save_applies_remembers_and_updates_url(fakes):
    at = run(page(pid="4"))
    save(at, " 7 ")
    assert applied(at) == 7 and at.sidebar.text_input[0].value == "7"
    assert qp(at, "manager") == "7" and qp(at, "pid") == "4"
    assert captions(at)[0] == "Seven FC · 280 очков · ранг 50"
    scripts = device_scripts(at)
    assert len(scripts) == 1 and "fplc_manager=7; Path=/" in scripts[0]
    assert [t.value for t in at.toast] == ["ID сохранён на этом устройстве"]
    run(at)  # cookie пишется один раз — на следующем запуске скрипта нет
    assert not device_scripts(at) and not at.toast and applied(at) == 7


def test_save_rejects_garbage_and_unknown_id(fakes):
    at = run(page(manager="7"))
    save(at, "abc")
    assert applied(at) == 7 and qp(at, "manager") == "7"
    assert ":red[Введите число]" in captions(at)
    assert at.sidebar.text_input[0].value == "abc"  # введённое не теряется — можно исправить
    save(at, "42")
    assert applied(at) == 7 and qp(at, "manager") == "7"
    assert ":red[Команда с ID 42 не найдена в FPL]" in captions(at)
    assert "Seven FC · 280 очков · ранг 50" in captions(at)
    save(at, str(2**31))  # больше int32 — без запроса к FPL
    assert applied(at) == 7 and fakes.client.entry_calls.count(2**31) == 0
    assert not device_scripts(at) and not at.toast


def test_forget_clears_session_url_and_cookie(fakes, monkeypatch):
    remembered(monkeypatch, 7)
    at = run(page(pid="4"))
    assert applied(at) == 7
    forget(at)
    assert applied(at) == 0 and at.sidebar.text_input[0].value == ""
    assert "manager" not in at.query_params and qp(at, "pid") == "4"
    scripts = device_scripts(at)
    assert len(scripts) == 1 and "fplc_manager=; Path=/; Max-Age=0" in scripts[0]
    assert [t.value for t in at.toast] == ["ID забыт на этом устройстве"]
    assert captions(at)[0] == "Без менеджера"
    run(at)  # в этой сессии cookie больше не возвращает ID
    assert applied(at) == 0 and "manager" not in at.query_params


def test_save_with_empty_field_forgets(fakes):
    at = run(page(manager="7"))
    save(at, "  ")
    assert applied(at) == 0 and "manager" not in at.query_params
    assert "fplc_manager=; Path=/; Max-Age=0" in device_scripts(at)[0]


def test_save_when_fpl_does_not_answer_is_unverified(fakes):
    at = run(page())
    save(at, str(OFFLINE_ID))
    assert applied(at) == OFFLINE_ID and qp(at, "manager") == str(OFFLINE_ID)
    assert f"fplc_manager={OFFLINE_ID};" in device_scripts(at)[0]
    assert "не проверен" in at.toast[0].value
    assert manager_state.UNVERIFIED_CAPTION in captions(at)
