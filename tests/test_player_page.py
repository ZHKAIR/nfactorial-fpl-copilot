"""AppTest страницы «Игрок»: открытие по прямой ссылке `/player?pid=<id>`, синхронизация
`?pid=` с выбором через поиск и кнопку «Открыть», невалидный pid игнорируется. Фейки — из
tests/test_app_smoke.py (без сети/LLM)."""

from __future__ import annotations

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from test_agent_fakes import NOW
from test_app_smoke import APP, TIMEOUT, AppTools, _FrozenDatetime

from fplcopilot.app import common

PAGE = str(APP / "views" / "4_player.py")


@pytest.fixture
def fakes(monkeypatch):
    st.cache_data.clear()
    tools = AppTools()
    monkeypatch.setattr(common, "datetime", _FrozenDatetime)
    monkeypatch.setattr(common, "get_tools", lambda: tools)
    monkeypatch.setattr(common, "openai_ready", lambda: True)
    monkeypatch.setattr(
        common, "freshness", lambda gw: {"news_at": NOW, "signal_at": NOW, "xpts_rows": 700}
    )
    monkeypatch.setattr(common.settings, "app_default_manager_id", "1")
    yield tools
    st.cache_data.clear()


def headers(at: AppTest) -> list[str]:
    return [h.value for h in at.header]


def pid(at: AppTest) -> str | None:
    """?pid= из адреса: AppTest хранит значения query_params списками."""
    raw = at.query_params.get("pid")
    if isinstance(raw, list):
        return raw[0] if raw else None
    return raw


def test_open_by_pid_skips_search_and_keeps_query_filled(fakes):
    at = AppTest.from_file(PAGE, default_timeout=TIMEOUT)
    at.query_params["pid"] = "4"  # Gabriel dos Santos Magalhães — один из четырёх «Gabriel»
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert headers(at) == ["Gabriel dos Santos Magalhães"]
    assert at.text_input(key="player_query").value == "Gabriel"  # поиск заполнен именем
    assert not at.radio  # карточка открыта сразу, без уточнения
    assert pid(at) == "4"
    report = " ".join(m.value for m in at.markdown if "fpl-report-top" in m.value)
    assert "Player report" in report and all(x in report for x in ("£8.0", "DEF", "ARS"))
    # похожие игроки: три группы, у каждой карточки — «Compare» и «Открыть»
    cmp_keys = {b.key for b in at.button if b.key and b.key.startswith("cmp_")}
    assert cmp_keys and all(b.label == "Compare" for b in at.button if b.key in cmp_keys)
    md = " ".join(m.value for m in at.markdown)
    assert "Alternatives at the price" in md and "The differential bridge" in md
    # поиск дальше работает: pid из адреса не перебивает новый запрос
    at.text_input(key="player_query").input("Haaland").run()
    assert not at.exception, [e.value for e in at.exception]
    assert headers(at) == ["Erling Haaland"]
    assert pid(at) == "411"  # адрес обновился под выбранного игрока


def test_invalid_pid_is_ignored_and_search_shown(fakes):
    for bad in ("999999", "abc", ""):
        at = AppTest.from_file(PAGE, default_timeout=TIMEOUT)
        at.query_params["pid"] = bad
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        assert headers(at) == [] and not at.error
        assert at.text_input(key="player_query").value == ""
        assert "pid" not in at.query_params  # мусорный параметр убран из адреса


def test_search_and_open_button_update_pid(fakes):
    at = AppTest.from_file(PAGE, default_timeout=TIMEOUT)
    at.run()
    assert "pid" not in at.query_params
    at.text_input(key="player_query").input("Haaland").run()
    assert not at.exception, [e.value for e in at.exception]
    assert headers(at) == ["Erling Haaland"] and pid(at) == "411"
    # «Открыть» у похожего игрока -> карточка и адрес переключаются на него
    opens = [b for b in at.button if b.key and b.key.startswith("open_")]
    assert opens, "нет карточек похожих игроков"
    target = int(opens[0].key.rsplit("_", 1)[1])
    opens[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert pid(at) == str(target)
    assert headers(at) == [fakes.bootstrap.player(target).full_name]
    assert at.text_input(key="player_query").value == fakes.bootstrap.player(target).web_name
    # очистили поиск -> карточки нет, pid из адреса убран
    at.text_input(key="player_query").input("").run()
    assert not at.exception, [e.value for e in at.exception]
    assert headers(at) == [] and "pid" not in at.query_params
