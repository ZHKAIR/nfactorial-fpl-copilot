"""Streamlit AppTest: страницы рендерятся без исключений на фейках (без сети/LLM/БД);
чат показывает кнопки HITL при прерывании и продолжает обе ветки; squad_override доходит до
инструментов уровня состава."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pytest
import streamlit as st
from langgraph.checkpoint.memory import MemorySaver
from streamlit.testing.v1 import AppTest
from test_agent_fakes import BS, NOW, FakeExplain, FakeRouter, route, router_output
from test_app_override import OverrideAwareTools, make_squad

from fplcopilot.agent.graph import Agent, Deps
from fplcopilot.agent.resolve import PlayerResolver
from fplcopilot.agent.tools import SquadOverride
from fplcopilot.app import common
from fplcopilot.data.schemas import Entry

APP = Path(__file__).resolve().parents[1] / "src" / "fplcopilot" / "app"
TIMEOUT = 60


class _Client:
    def bootstrap(self, *, refresh: bool = False):
        return BS

    def entry(self, manager_id: int) -> Entry:
        """entry/{id}: id 1 — известная команда, остальные — 404 как у FPL API."""
        if manager_id != 1:
            req = httpx.Request("GET", f"https://fantasy.premierleague.com/api/entry/{manager_id}/")
            raise httpx.HTTPStatusError(
                "404", request=req, response=httpx.Response(404, request=req)
            )
        return Entry(
            id=1,
            player_first_name="Test",
            player_last_name="Manager",
            name="Nfactorial_test",
            summary_overall_points=312,
            summary_overall_rank=1234567,
        )

    def close(self):
        pass


class AppTools(OverrideAwareTools):
    """FakeTools + то, что страницы берут у LiveTools помимо 9 инструментов."""

    client = _Client()

    @property
    def bootstrap(self):
        return BS

    def now(self):
        return NOW

    def next_gw(self):
        return 5

    def squad_override_from(self, squad, *, manager_id=None, source="screenshot"):
        return SquadOverride.from_squad(squad, source=source)


def make_agent(tools, router, explain=None):
    deps = Deps(
        tools=tools,
        router_llm=router,
        explain_llm=explain or FakeExplain(),
        resolver=lambda squad: PlayerResolver(BS, squad_ids=squad),
        clock=lambda: NOW,
        extra_known_names=["Arsenal", "Chelsea"],
    )
    return Agent(deps, MemorySaver())


class _FrozenDatetime(datetime):
    """datetime.now() сайдбара = NOW фейков, чтобы «N мин назад» в тестах было детерминированным."""

    @classmethod
    def now(cls, tz=None):
        return NOW if tz is None else NOW.astimezone(tz)


@pytest.fixture
def fakes(monkeypatch):
    """Подменяем ресурсы UI фейками; кэши Streamlit чистим между тестами."""
    st.cache_data.clear()
    tools = AppTools()
    monkeypatch.setattr(common, "datetime", _FrozenDatetime)
    monkeypatch.setattr(common, "get_tools", lambda: tools)
    monkeypatch.setattr(common, "openai_ready", lambda: True)
    monkeypatch.setattr(
        common,
        "freshness",
        lambda gw: {"news_at": NOW, "signal_at": NOW, "xpts_rows": 700},
    )
    # UI не берёт FPL_MANAGER_ID; менеджер 1 — через предзаполнение для разработки / демо
    monkeypatch.setattr(common.settings, "app_default_manager_id", "1")
    yield tools
    st.cache_data.clear()


def run_page(name: str) -> AppTest:
    at = AppTest.from_file(str(APP / "views" / name), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_home_renders_without_exceptions(fakes, monkeypatch):
    """Точка входа Home.py: стартовая страница (/) — «Брифинг» (приветствие по имени из FPL
    entry), «Мой состав» переехал на /squad, «Обзора» больше нет."""
    agent = make_agent(fakes, FakeRouter(router_output("captain")))
    monkeypatch.setattr(common, "get_agent", lambda: agent)
    at = AppTest.from_file(str(APP / "Home.py"), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value.endswith(", Test.")  # имя менеджера из entry/{id}
    md = " ".join(m.value for m in at.markdown)
    assert "Главный ход" in md and "Капитан" in md and "Следить до дедлайна" in md
    assert not any("Обзор" in h.value or "Как запустить" in h.value for h in at.subheader)
    # ID менеджера — текстовое поле без степпера; контракт session_state["manager_id"] = int
    assert at.sidebar.text_input[0].value == "1"
    assert at.session_state[common.MANAGER_KEY] == 1
    assert at.sidebar.selectbox[0].value == "balanced"  # внутреннее значение — как у агента
    captions = [c.value for c in at.sidebar.caption]
    assert captions[0] == "Nfactorial_test · 312 очков · ранг 1 234 567"  # подтверждение ID
    assert captions[1] == "Применено: сбалансированная"
    assert "Новости: 0 мин назад" in captions
    assert captions[-1] in ("LangSmith: включён", "LangSmith: выключен")  # зависит от .env
    md = [m.value for m in at.sidebar.markdown]
    assert any("Состав: из FPL за GW4" in m for m in md)
    assert any("Разбор новостей об игроках: 0 мин назад" in m for m in md)  # CSS-tip вместо help=
    assert any(m.startswith("**GW5** · дедлайн ") and "· через **1 д 7 ч**" in m for m in md)
    # лишнего в сайдбаре нет: ни expander'а, ни длинных подписей, ни жаргона
    assert not at.sidebar.expander and not at.sidebar.info and not at.sidebar.error
    everything = "\n".join(captions + md)
    for banned in ("сигнал", "БД", "строк", "фикстур", "picks", "Данные FPL", "xPts"):
        assert banned not in everything, banned


def test_briefing_page_cards_from_tools(fakes):
    """«Брифинг»: главный ход — рекомендуемый маршрут, капитан — два лучших варианта,
    «Следить до дедлайна» — серьёзные проблемы; действия — переходы и вопрос в чат."""
    fakes.risk = doubtful_risk
    at = run_page("0_briefing.py")
    md = " ".join(m.value for m in at.markdown)
    route_ = fakes.routes[0]
    assert f"{', '.join(route_.out)} → {', '.join(route_.in_)}" in md
    assert "за 3 тура" in md and "fpl-transfer" in md
    assert "fpl-vs" in md and "с учётом удвоения" in md and "Вице-капитан" in md
    # «Следить до дедлайна» — строка на игрока старта со ссылкой на карточку
    assert '<a href="/player?pid=165" target="_self">João Pedro</a>' in md
    assert "под вопросом (75 %)" in md and "fpl-watch-row" in md
    # лента: важные новости состава + заголовки из корпуса
    assert "Новости за 10 дней" in md and "fpl-ticker" in md
    assert "Knee problem, 75% chance" in md
    assert all(inp.cached_only for inp in fakes.called("analyze_player_risk"))
    assert "Прогноз на GW5" in md and "лучший состав из ваших 15" in md
    assert {"brief_to_deadline", "brief_ask_route", "brief_compare", "brief_to_plan"} <= keys(at)
    assert at.button(key="brief_ask_route").label == "Спросить ассистента"
    assert not any("Copilot" in b.label for b in at.button)
    assert at.text_input(key="brief_ask").placeholder.startswith("Спросите ассистента")
    assert not at.chat_input  # не st.chat_input: тот прокручивает страницу вниз
    for banned in ("confidence", "уверенност", "Save team", "Apply"):
        assert banned not in md, banned
    rt = fakes.called("recommend_transfers")
    assert rt and rt[0].horizon == 3 and rt[0].allow_hit is False


def doubtful_risk(inp):
    """João Pedro под вопросом (свежая цитата) — важная новость; остальные в строю."""
    from test_agent_fakes import fit_risk

    risk = fit_risk(inp)
    if inp.player_id != 165:
        return risk
    ev = risk.evidence[0].model_copy(update={"quote": "Knee problem, 75% chance to feature."})
    return risk.model_copy(update={"availability": "doubtful", "evidence": [ev]})


def test_briefing_without_picks_points_to_squad_page(fakes):
    fakes.squad = False
    at = run_page("0_briefing.py")
    assert any(w.value.startswith("Состав недоступен") for w in at.warning)
    assert "brief_to_squad" in keys(at)
    assert not fakes.called("recommend_transfers")


def test_home_sidebar_unknown_manager_stale_news_and_missing_xpts(fakes, monkeypatch):
    agent = make_agent(fakes, FakeRouter(router_output("captain")))
    monkeypatch.setattr(common, "get_agent", lambda: agent)
    monkeypatch.setattr(common.settings, "app_default_manager_id", "42")  # entry/42 -> 404
    monkeypatch.setattr(
        common,
        "freshness",
        lambda gw: {"news_at": NOW, "signal_at": NOW - timedelta(hours=72), "xpts_rows": 0},
    )
    at = AppTest.from_file(str(APP / "Home.py"), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    captions = [c.value for c in at.sidebar.caption]
    assert captions[0] == "Менеджер не найден в FPL"
    assert any(
        "Разбор новостей об игроках: 3 дн назад · устарел" in m.value for m in at.sidebar.markdown
    )
    assert any(e.value == "Прогноз на GW5 не рассчитан" for e in at.sidebar.error)


def test_sidebar_manager_text_input_validation(fakes):
    """Поле ID — в форме: применяется по Enter / «Сохранить» (в AppTest — клик по кнопке)."""
    at = run_page("4_player.py")

    def save(text: str) -> None:
        at.sidebar.text_input[0].input(text)
        at.button(key="manager_save").click().run()
        assert not at.exception, [e.value for e in at.exception]

    # не число -> ошибка, прежний id (1) остаётся применённым
    save("abc")
    captions = [c.value for c in at.sidebar.caption]
    assert ":red[Введите число]" in captions and "Nfactorial_test · 312 очков · ранг 1 234 567" in (
        captions
    )
    assert at.session_state[common.MANAGER_KEY] == 1
    # пробелы обрезаются; ID, которого нет в FPL (404), не применяется и не сохраняется
    save(" 42 ")
    assert at.session_state[common.MANAGER_KEY] == 1
    assert ":red[Команда с ID 42 не найдена в FPL]" in [c.value for c in at.sidebar.caption]
    # пусто -> без менеджера (entry не вызывается)
    save("")
    assert at.session_state[common.MANAGER_KEY] == 0
    assert "Без менеджера" in [c.value for c in at.sidebar.caption]
    assert any(w.value.startswith("Состав: нет — укажите ID") for w in at.sidebar.warning)


def test_squad_page_header_table_and_problems(fakes):
    at = run_page("1_squad.py")
    assert at.title[0].value == "Мой состав"
    # шапка команды: 6 метрик через HTML (CSS-tip вместо st.metric help=)
    header_md = " ".join(m.value for m in at.markdown if 'class="fpl-metric"' in m.value)
    for label in (
        "Очки сезона",
        "Общий ранг",
        "Банк",
        "Бесплатных трансферов",
        "Чипы",
        "Прогноз очков на GW5",
    ):
        assert label in header_md, label
    assert "312" in header_md and "1 234 567" in header_md
    assert "£0.1m" in header_md and "32.8" in header_md
    assert 'class="tip-text"' in header_md
    # строка серьёзных проблем — именная (João Pedro: FPL doubtful 75 %), без «тяжёлого календаря»
    assert [w.value for w in at.warning] == ["João Pedro — под вопросом (75 %)"]
    assert not at.success
    # таблица — HTML через st.markdown (не st.dataframe): колонки smartplay + календарь GW5..GW9,
    # старт по позициям, имя — ссылка на страницу «Игрок» с pid = FPL element id
    tables = [m.value for m in at.markdown if '<table class="fpl-table fpl-squad">' in m.value]
    assert len(tables) == 1
    assert len(at.dataframe) == 1  # единственный st.dataframe — цитаты в expander João Pedro
    html = tables[0]
    for label in ("Player", "Price", "TSB%", "Form", "Pts/Game", "xMins GW5", "xPts GW5"):
        assert f">{label}<span" in html, label
    for label in ("3GW xPts", "FRR#", "GW5", "GW9"):
        assert f">{label}<span" in html, label
    names = ["Gabriel", "Guéhi", "Palmer", "João Pedro", "Haaland"]
    links = [
        f'<a href="/player?pid={pid}" target="_self">{n}</a>'
        for pid, n in zip((4, 388, 154, 165, 411), names, strict=True)
    ]
    assert all(link in html for link in links), links
    assert [html.index(link) for link in links] == sorted(html.index(link) for link in links)
    assert '<div class="sub">MCI · FWD · 72.6%</div>' in html  # Haaland: клуб · поз · TSB
    assert html.count('<span class="tip flag"') == 1  # точка только у João Pedro
    assert (
        '<span class="tip flag"><span class="fpl-dot warn"></span>'
        '<span class="tip-text">Под вопросом (75 %)</span></span>' in html
    )
    assert (
        html.count("BRE (A)") == 25 and '<span class="fpl-fx fpl-fsi-3">BRE (A)</span>' in html
    )  # 5 игроков × 5 туров
    assert html.count('<td class="num">#1</td>') == 5  # FRR#: у всех клубов один календарь (BRE)
    assert "Bench" not in html  # скамейки в фейке нет -> без разделителя
    # легенда сложности — как на smartplay
    legend = " ".join(m.value for m in at.markdown)
    assert "Fixture difficulty:" in legend and "Very Tough" in legend
    # проблемы состава: expander только по игрокам из строки проблем
    assert any(h.value == "Проблемы состава" for h in at.subheader)
    # цветная иконка Material в подписи, не эмодзи
    assert [e.label for e in at.expander if "João Pedro" in e.label] == [
        ":orange[:material/warning:] João Pedro — под вопросом (75 %)"
    ]
    page_text = " ".join(
        x.value for x in [*at.subheader, *at.caption, *at.markdown, *at.warning, *at.info]
    ).lower()
    for banned in ("диагностик", "сигнал", "фикстур", "picks"):
        assert banned not in page_text, banned
    assert ">Статус<" not in html  # колонки «Статус» больше нет
    # скриншот — в свёрнутом expander «Скриншот состава»: одна подсказка, без кнопок-примеров,
    # без упоминаний модели / стоимости / docs / эталонов
    shot = [e for e in at.expander if e.label == "Скриншот состава"]
    assert len(shot) == 1 and shot[0].proto.expanded is False
    captions = [c.value for c in at.caption]
    assert (
        "Загрузите скриншот экрана Pick Team или Transfers — состав будет распознан и применён "
        "на всех страницах."
    ) in captions
    assert not {"sample_pick_team_valid", "sample_pick_team_broken", "parse_shot"} & keys(at)
    everything = " ".join(captions + [m.value for m in at.markdown if "<table" not in m.value])
    for banned in ("gpt-4o", "$", "docs/", "эталон", "vision", "токен"):
        assert banned not in everything.lower(), banned


def test_squad_page_demo_mode_shows_sample_buttons(fakes):
    """?demo=1 — кнопки-примеры для защиты и тестов; «Распознать» появляется после выбора картинки."""
    at = AppTest.from_file(str(APP / "views" / "1_squad.py"), default_timeout=TIMEOUT)
    at.query_params["demo"] = "1"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert {"sample_pick_team_valid", "sample_pick_team_broken"} <= keys(at)
    assert "parse_shot" not in keys(at)
    at.button(key="sample_pick_team_valid").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.button(key="parse_shot").label == "Распознать" and len(at.image) == 1


def keys(at: AppTest) -> set[str]:
    return {b.key for b in at.button if b.key}


def test_squad_page_without_picks_offers_screenshot(fakes, monkeypatch):
    fakes.squad = False
    at = run_page("1_squad.py")
    assert any(w.value.startswith("Состав из FPL недоступен") for w in at.warning)
    assert any("Состав: нет — загрузите скриншот" == w.value for w in at.sidebar.warning)
    assert not at.metric and not at.dataframe  # без состава нет шапки и таблицы
    shot = [e for e in at.expander if e.label == "Скриншот состава"]
    assert len(shot) == 1 and shot[0].proto.expanded is True  # раскрыт, когда состава нет
    assert at.get("file_uploader")  # загрузка доступна и без демо-кнопок


def test_squad_page_without_manager_asks_for_id_in_russian(fakes, monkeypatch):
    monkeypatch.setattr(common.settings, "app_default_manager_id", None)
    fakes.squad = False
    at = run_page("1_squad.py")
    assert not at.exception
    assert not any(w.value.startswith("Состав из FPL недоступен") for w in at.warning)
    assert any(i.value.startswith("Укажите ID менеджера FPL") for i in at.info)
    assert at.get("file_uploader")


def test_about_page_graphs_and_tools_table(fakes, monkeypatch):
    agent = make_agent(fakes, FakeRouter(router_output("captain")))
    monkeypatch.setattr(common, "get_agent", lambda: agent)
    at = run_page("6_about.py")
    assert at.title[0].value == "О системе"
    md = " ".join(m.value for m in at.markdown)
    assert "Граф агента" in md and "Поток данных" in md
    assert "Страница → инструменты" in md
    assert "Как запустить" not in md and not at.metric
    # пояснения — CSS-tip у заголовков, на экране без docs / команд запуска
    assert 'class="tip-text"' in md
    screen = " ".join(x.value for x in [*at.caption, *at.markdown, *at.info])
    assert "docs/" not in screen and "uv run" not in screen
    charts = at.get("graphviz_chart")
    assert len(charts) == 2
    agent_dot, pipeline = charts[0].proto.spec, charts[1].proto.spec
    assert "confirm_action\\n(interrupt before" in agent_dot
    assert "resolve_clarification\\n(interrupt before" in agent_dot
    assert "LiveTools: 11 инструментов" in pipeline and "9 инструментов" not in pipeline
    assert any("Мой состав" in t.value.to_string() for t in at.table)


def test_deadline_page_uses_squad_override_from_session(fakes):
    at = AppTest.from_file(str(APP / "views" / "2_deadline.py"), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    assert any(m.value == "56.14" for m in at.metric)  # picks
    assert any(h.value == "Лучший состав на тур 5" for h in at.subheader)
    assert {m.label for m in at.metric} >= {"Прогноз состава", "Ваш старт", "Капитан / вице"}
    labels = " ".join([*(h.value for h in at.subheader), *(m.label for m in at.metric)])
    assert "\u200b" not in labels and "Лучшийсостав" not in labels
    screen = " ".join(
        [
            *(x.value for x in [*at.subheader, *at.caption, *at.markdown]),
            *(m.label for m in at.metric),
            *((m.delta or "") for m in at.metric),
        ]
    )
    for banned in ("Лучшие 11", "Текущие 11", "лучших 11", "к оптимуму", "XI", "11-ка"):
        assert banned not in screen, banned
    assert fakes.overrides.get("optimize_team") is None

    at.session_state[common.SESSION_OVERRIDE] = SquadOverride.from_squad(make_squad())
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert isinstance(fakes.overrides["optimize_team"], SquadOverride)
    assert isinstance(fakes.overrides["recommend_transfers"], SquadOverride)
    assert any(m.value == "52.44" for m in at.metric)  # числа изменились
    assert any("Состав: со скриншота (GW5)" == m.value for m in at.sidebar.markdown)
    assert "reset_override" in keys(at)
    assert at.button(key="reset_override").label == "Сбросить"


def test_sidebar_without_manager_caption(fakes, monkeypatch):
    """Без APP_DEFAULT_MANAGER_ID -> пустое поле и «Без менеджера» (FPL_MANAGER_ID в UI не идёт)."""
    monkeypatch.setattr(common.settings, "app_default_manager_id", None)
    monkeypatch.setattr(common.settings, "fpl_manager_id", 1)
    at = run_page("4_player.py")
    assert at.sidebar.text_input[0].value == ""
    assert at.sidebar.caption[0].value == "Без менеджера"


def test_plan_and_player_pages_render(fakes):
    at = run_page("3_plan.py")
    md = " ".join(m.value for m in at.markdown)
    assert "Итог плана" in md and "План на 5 туров даст" in md  # ответ сначала
    assert "fpl-gw-grid" in md and "если ничего не менять" in md
    assert any(e.label == "Что изменилось со снимка" for e in at.expander)
    assert any("Чипов для планирования нет" in c.value for c in at.caption)  # у фейка чипов нет
    at = run_page("4_player.py")
    at.text_input(key="player_query").input("Palmer").run()
    assert not at.exception
    assert at.radio(key="player_choice") is not None  # неоднозначно -> выбор
    assert any("Cole Palmer" in h.value for h in at.header)
    assert any("4.76" in m.value for m in at.markdown if 'class="fpl-keynums"' in m.value)
    assert any(
        'class="fpl-sp-card"' in m.value and "Attacking Output" in m.value for m in at.markdown
    )
    # календарь вместо «Фикстуры»; слова «фикстур» на странице нет
    assert any("Fixtures (next 6 GWs)" in m.value for m in at.markdown)
    page_text = " ".join(x.value for x in [*at.subheader, *at.caption, *at.markdown])
    assert "фикстур" not in page_text.lower()


def hit_agent(tools):
    tools.routes = [route(1, ["Palmer", "João Pedro"], ["Saka", "Guéhi"], hit=4)]
    router = FakeRouter(router_output("transfer", ["Palmer"], sell=["Palmer"], allow_hit=True))
    return make_agent(tools, router, FakeExplain(["**Verdict:** ok after decision."]))


def test_chat_page_hitl_buttons_and_reject_branch(fakes, monkeypatch):
    agent = hit_agent(fakes)
    monkeypatch.setattr(common, "get_agent", lambda: agent)
    at = AppTest.from_file(str(APP / "views" / "5_chat.py"), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    assert len([b for b in at.button if b.key and b.key.startswith("example_")]) == 6
    at.chat_input[0].set_value("Should I sell Palmer even for a -4?").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Требует подтверждения" in w.value for w in at.warning)
    assert {"confirm_1", "reject_1"} <= keys(at)
    assert len(at.chat_message) == 2
    at.button(key="reject_1").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("отклонено" in c.value for c in at.caption)
    assert any("ok after decision" in m.value for m in at.markdown)
    calls = fakes.called("recommend_transfers")
    assert [c.allow_hit for c in calls] == [True, False]  # пересчёт без хита
    assert not {"confirm_1", "reject_1"} & keys(at)  # кнопки исчезли после решения


def test_chat_page_confirm_branch_and_example_button(fakes, monkeypatch):
    agent = hit_agent(fakes)
    monkeypatch.setattr(common, "get_agent", lambda: agent)
    at = AppTest.from_file(str(APP / "views" / "5_chat.py"), default_timeout=TIMEOUT)
    at.run()
    at.button(key="example_1").click().run()  # кнопка-пример; фейковый роутер отвечает хитом
    assert not at.exception, [e.value for e in at.exception]
    assert "confirm_1" in keys(at)
    at.button(key="confirm_1").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("подтверждено" in c.value for c in at.caption)
    assert len(fakes.called("recommend_transfers")) == 1  # без пересчёта
    assert any("ok after decision" in m.value for m in at.markdown)


def test_chat_page_clarification_buttons_resume_with_chosen_player(fakes, monkeypatch):
    """HITL №2: неоднозначное имя -> кнопки-кандидаты -> клик -> граф продолжает с выбранным."""
    router = FakeRouter(router_output("transfer", ["Gabriel"], sell=["Gabriel"]))
    agent = make_agent(fakes, router, FakeExplain(["**Verdict:** sell Gabriel for Saka."]))
    monkeypatch.setattr(common, "get_agent", lambda: agent)
    at = AppTest.from_file(str(APP / "views" / "5_chat.py"), default_timeout=TIMEOUT)
    at.run()
    at.chat_input[0].set_value("Should I sell Gabriel?").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Уточнение" in w.value and "Gabriel" in w.value for w in at.warning)
    choose_keys = {k for k in keys(at) if k.startswith("choose_1_")}
    assert choose_keys == {"choose_1_4", "choose_1_18", "choose_1_27", "choose_1_331"}
    assert "cancel_clarify_1" in keys(at)
    assert not {"confirm_1", "reject_1"} & keys(at)  # это не подтверждение хита
    labels = {b.label for b in at.button if b.key and b.key.startswith("choose_1_")}
    assert any("Gabriel dos Santos Magalhães" in lbl and "в составе" in lbl for lbl in labels)
    assert not fakes.called("recommend_transfers")  # до выбора инструменты состава не звались

    at.button(key="choose_1_4").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("выбран игрок id 4" in c.value for c in at.caption)
    assert any("sell Gabriel for Saka" in m.value for m in at.markdown)
    calls = fakes.called("recommend_transfers")
    assert len(calls) == 1 and calls[0].sell == [4]  # Gabriel Magalhães, тот, что в составе
    assert not {k for k in keys(at) if k.startswith("choose_1_")}  # кнопки исчезли после выбора


def test_chat_page_clarification_cancel_button(fakes, monkeypatch):
    router = FakeRouter(router_output("transfer", ["Gabriel"], sell=["Gabriel"]))
    agent = make_agent(fakes, router, FakeExplain())
    monkeypatch.setattr(common, "get_agent", lambda: agent)
    at = AppTest.from_file(str(APP / "views" / "5_chat.py"), default_timeout=TIMEOUT)
    at.run()
    at.chat_input[0].set_value("Should I sell Gabriel?").run()
    at.button(key="cancel_clarify_1").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Уточнение отменено" in c.value for c in at.caption)
    assert any("Clarification cancelled" in m.value for m in at.markdown)
    assert not fakes.called("recommend_transfers")


def test_chat_page_without_openai_key_shows_hint(fakes, monkeypatch):
    monkeypatch.setattr(common, "openai_ready", lambda: False)
    at = AppTest.from_file(str(APP / "views" / "5_chat.py"), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    assert any("OPENAI_API_KEY" in e.value for e in at.error)
