"""Фишки плана через инструмент build_gameweek_plan (LiveTools на игрушечных входах, без сети и
БД) и страница «План» (AppTest на фейках): выбор «фишка → тур», вклад фишки, понятная ошибка."""

from __future__ import annotations

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from test_agent_fakes import NOW
from test_app_smoke import APP, TIMEOUT, AppTools, _FrozenDatetime
from test_core_optimizer import as_pool, base_squad, cand, ids

from fplcopilot.agent import tools as tools_mod
from fplcopilot.agent.tools import BuildPlanInput, LiveTools, PlanChipIn, PlanChipOut
from fplcopilot.app import common, plan_chips
from fplcopilot.core.optimizer import ChipPlanError, ManagerInputs

ALL_CHIPS = ["wildcard", "freehit", "bboost", "3xc"]


def toy_inputs(chips: list[str]) -> ManagerInputs:
    squad = base_squad()
    for c in squad:
        c.in_squad = True
    pool = as_pool(squad, [cand(61, 4, 61, 5.0, 4.8)])
    return ManagerInputs(
        manager_id=1,
        from_gw=5,
        gws=[5, 6, 7],
        squad=ids(squad),
        bank=0.0,
        free_transfers=1,
        chips_available=list(chips),
        cands=dict(pool),
        pool=pool,
        issues=[],
        store=None,  # type: ignore[arg-type]
        squad_gw=4,
        chips_by_gw={g: list(chips) for g in (5, 6, 7)},
    )


@pytest.fixture
def live(monkeypatch):
    """LiveTools с подменёнными входами (состав/пул/чипы) и без снимков в БД."""
    tools = LiveTools(client=object(), now=NOW)  # type: ignore[arg-type]
    tools.chips = list(ALL_CHIPS)  # type: ignore[attr-defined]
    tools.input_calls = []  # type: ignore[attr-defined]

    def fake_inputs(*args, **kwargs):
        tools.input_calls.append((args, kwargs))  # type: ignore[attr-defined]
        return toy_inputs(tools.chips)  # type: ignore[attr-defined]

    monkeypatch.setattr(tools, "inputs", fake_inputs)
    monkeypatch.setattr(tools_mod, "latest_plan", lambda *a, **k: None)
    return tools


def plan_input(**kw) -> BuildPlanInput:
    return BuildPlanInput(manager_id=1, gw=5, horizon=3, save=False, **kw)


# ---------- инструмент ----------


def test_build_plan_input_chips_optional_and_default_empty():
    assert BuildPlanInput(manager_id=1, gw=5).chips == []
    schema = BuildPlanInput.model_json_schema()
    assert "chips" in schema["properties"] and "chips" not in schema.get("required", [])


def test_tool_without_chips_unchanged_and_with_triple_captain(live):
    plain = live.build_gameweek_plan(plan_input())
    assert plain.chips == [] and plain.notes == [] and plain.wildcard is not None
    tc = live.build_gameweek_plan(plan_input(chips=[PlanChipIn(gw=6, chip="3xc")]))
    assert tc.chips == [
        PlanChipOut(gw=6, chip="3xc", name="Triple Captain", points=7.5, players=["P13"])
    ]
    assert tc.xi_points_by_gw["6"] == pytest.approx(plain.xi_points_by_gw["6"] + 7.5)
    assert tc.xi_points_by_gw["5"] == plain.xi_points_by_gw["5"]
    assert tc.expected_total == pytest.approx(plain.expected_total + 7.5)
    assert tc.wildcard is not None  # фишка не в первом туре — WC-сравнение остаётся


def test_tool_bench_boost_in_first_gw_reports_bench_and_skips_wildcard(live):
    out = live.build_gameweek_plan(plan_input(chips=[PlanChipIn(gw=5, chip="bboost")]))
    (chip,) = out.chips
    assert chip.chip == "bboost" and chip.name == "Bench Boost" and len(chip.players) == 4
    assert out.wildcard is None
    assert out.notes and "Wildcard в GW5" in out.notes[0]


def test_tool_rejects_invalid_chip_conditions(live):
    with pytest.raises(ChipPlanError, match="Wildcard и Bench Boost"):
        live.build_gameweek_plan(
            plan_input(use_wildcard=True, chips=[PlanChipIn(gw=5, chip="bboost")])
        )
    assert live.input_calls == []  # до загрузки входов
    with pytest.raises(ChipPlanError, match="две фишки в одном туре"):
        live.build_gameweek_plan(
            plan_input(chips=[PlanChipIn(gw=6, chip="bboost"), PlanChipIn(gw=6, chip="3xc")])
        )
    with pytest.raises(ChipPlanError, match="вне горизонта"):
        live.build_gameweek_plan(plan_input(chips=[PlanChipIn(gw=9, chip="bboost")]))
    live.chips = ["wildcard", "freehit"]  # BB и TC уже сыграны
    with pytest.raises(ChipPlanError, match="Bench Boost недоступен"):
        live.build_gameweek_plan(plan_input(chips=[PlanChipIn(gw=6, chip="bboost")]))


# ---------- страница «План» ----------


class ChipTools(AppTools):
    """Фейк страницы: чипы менеджера в контексте тура; план с фишкой — с её вкладом."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.chips_available = ["wildcard", "bboost", "3xc"]
        self.chip_error: str | None = None

    def get_gameweek_context(self, inp, *, squad_override=None):
        ctx = super().get_gameweek_context(inp, squad_override=squad_override)
        ctx.chips_available = list(self.chips_available)
        return ctx

    def build_gameweek_plan(self, inp):
        plan = super().build_gameweek_plan(inp)
        if not inp.chips:
            return plan
        if self.chip_error:
            raise ChipPlanError(self.chip_error)
        c = inp.chips[0]
        chip = PlanChipOut(
            gw=c.gw,
            chip=c.chip,
            name=plan_chips.CHIP_LABELS[c.chip],
            points=8.4,
            players=["Forster", "Slater", "Konsa", "O'Shea"],
        )
        return plan.model_copy(update={"chips": [chip], "expected_total": 315.9})


@pytest.fixture
def chip_fakes(monkeypatch):
    st.cache_data.clear()
    tools = ChipTools()
    monkeypatch.setattr(common, "datetime", _FrozenDatetime)
    monkeypatch.setattr(common, "get_tools", lambda: tools)
    monkeypatch.setattr(common, "openai_ready", lambda: True)
    monkeypatch.setattr(
        common, "freshness", lambda gw: {"news_at": NOW, "signal_at": NOW, "xpts_rows": 700}
    )
    monkeypatch.setattr(common.settings, "app_default_manager_id", "1")
    yield tools
    st.cache_data.clear()


def plan_page() -> AppTest:
    at = AppTest.from_file(str(APP / "views" / "3_plan.py"), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_plan_page_bench_boost_in_chosen_gw(chip_fakes):
    """Страница «План» (ответ сначала, выбор тура для каждой фишки во всплывающем окне):
    только BB / TC из доступных, план с фишкой, эффект против того же плана без фишки, бейдж на
    карточке тура, снимок с той же фишкой."""
    at = plan_page()
    keys = {s.key for s in at.selectbox if s.key and s.key.startswith("plan_chip_")}
    assert keys == {"plan_chip_bboost", "plan_chip_3xc"}  # без WC / FH
    box = at.selectbox(key="plan_chip_bboost")
    assert box.options == ["не играть"] + [f"GW{g}" for g in range(5, 10)]
    assert chip_fakes.called("build_gameweek_plan")[-1].chips == []

    box.set_value("GW6").run()
    assert not at.exception, [e.value for e in at.exception]
    calls = chip_fakes.called("build_gameweek_plan")
    # план с фишкой; тот же план без неё (для эффекта) — из кэша первого прогона
    assert [c.chips for c in calls] == [[], [PlanChipIn(gw=6, chip="bboost")]]
    md = " ".join(m.value for m in at.markdown)
    assert "**Bench Boost в GW6:** +8.40 xPts скамейки (Forster, Slater, Konsa, O'Shea)" in md
    assert "С Bench Boost в GW6 план даёт на" in md  # эффект против плана без фишки
    assert '<b class="chip">Bench Boost</b>' in md  # бейдж на карточке GW6

    at.button(key="save_plan").click().run()
    assert not at.exception, [e.value for e in at.exception]
    saved = chip_fakes.called("build_gameweek_plan")[-1]
    assert saved.save is True and saved.chips == [PlanChipIn(gw=6, chip="bboost")]


def test_plan_page_shows_chip_rule_error_instead_of_traceback(chip_fakes):
    chip_fakes.chip_error = "Triple Captain недоступен менеджеру в GW7"
    at = plan_page()
    at.selectbox(key="plan_chip_3xc").set_value("GW7").run()
    assert not at.exception, [e.value for e in at.exception]
    assert [e.value for e in at.error] == [f"**{plan_chips.CHIP_ERROR_TITLE}**"]
    assert "Triple Captain недоступен менеджеру в GW7" in [c.value for c in at.caption]
    assert not any("Итог плана" in m.value for m in at.markdown)  # план не показан


def test_plan_page_without_bb_tc_disables_chip_choice(chip_fakes):
    chip_fakes.chips_available = ["wildcard", "freehit"]
    at = plan_page()
    assert not [s for s in at.selectbox if s.key and s.key.startswith("plan_chip_")]
    assert any("Чипов для планирования нет" in c.value for c in at.caption)
    assert chip_fakes.called("build_gameweek_plan")[-1].chips == []
    assert any("Итог плана" in m.value for m in at.markdown)
