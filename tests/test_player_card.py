"""Чистые помощники карточки игрока (app/player_card.py): перцентили по позиции, секции
статистики, Mins% по истории, календарь с цветами сложности, FRR, похожие игроки, вердикт по
новостям. Без Streamlit/сети/БД."""

from __future__ import annotations

from datetime import UTC, datetime

from fplcopilot.agent.tools import (
    FixtureBrief,
    GWPrediction,
    PlayerPrediction,
    PlayerRef,
    PlayerRisk,
)
from fplcopilot.app import player_card as card
from fplcopilot.data.schemas import Bootstrap, Event, Player, PlayerGWHistory, Position, Team

NOW = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


def pl(pid, web, team, pos, cost, own, **stats) -> Player:
    base = {
        "id": pid,
        "web_name": web,
        "first_name": web,
        "second_name": "Test",
        "team": team,
        "element_type": pos,
        "now_cost": cost,
        "selected_by_percent": own,
        "minutes": 450,
        "starts": 5,
        "total_points": 10,
    }
    base.update(stats)
    return Player(**base)


def make_bs() -> Bootstrap:
    teams = [
        Team(id=1, name="Arsenal", short_name="ARS"),
        Team(id=2, name="Chelsea", short_name="CHE"),
        Team(id=3, name="Leeds", short_name="LEE"),
    ]
    players = [
        # защитники: пул перцентилей (>= 90 мин) — 4, 5, 6, 7; 8 — без минут, вне пула
        pl(
            4,
            "Gabriel",
            1,
            2,
            80,
            22.8,
            total_points=25,
            bonus=2,
            points_per_game=5.0,
            form=5.0,
            clean_sheets=3,
            expected_goals_conceded=4.0,
            influence=94.8,
            goals_scored=0,
            assists=0,
            expected_goals=0.5,
            expected_assists=0.2,
            creativity=22.0,
            threat=84.0,
            ict_index=20.1,
            transfers_in_event=15419,
            transfers_out_event=47919,
        ),
        pl(
            5,
            "Saliba",
            1,
            2,
            60,
            30.0,
            total_points=20,
            clean_sheets=3,
            expected_goals_conceded=4.5,
        ),
        pl(
            6,
            "Timber",
            1,
            2,
            58,
            12.0,
            total_points=15,
            clean_sheets=2,
            expected_goals_conceded=5.0,
        ),
        pl(
            7,
            "Cucurella",
            2,
            2,
            62,
            5.0,
            total_points=12,
            clean_sheets=1,
            expected_goals_conceded=6.0,
        ),
        pl(8, "Rodon", 3, 2, 40, 1.0, minutes=0, starts=0, total_points=0, clean_sheets=0),
        pl(9, "Struijk", 3, 2, 45, 2.2, total_points=9, goals_scored=1),
        # полузащитник — другая позиция, в пул защитников не входит
        pl(10, "Saka", 1, 3, 100, 40.0, total_points=40, goals_scored=3),
    ]
    events = [Event(id=6, name="Gameweek 6", deadline_time=NOW, is_next=True)]
    return Bootstrap(events=events, teams=teams, elements=players)


BS = make_bs()


def fx(opp: str, home: bool, fsi: int) -> FixtureBrief:
    return FixtureBrief(
        opponent=opp, is_home=home, fsi=fsi, xg_for=1.5, xg_against=1.2, clean_sheet_prob=0.3
    )


def pred(pid: int, gws: dict[int, tuple[float, list[FixtureBrief]]]) -> PlayerPrediction:
    p = BS.player(pid)
    by_gw = [
        GWPrediction(
            gw=g,
            xpts=x,
            sd=2.0,
            p_start=0.9,
            exp_minutes=85,
            components={"appearance": 1.9},
            fixtures=fixtures,
        )
        for g, (x, fixtures) in sorted(gws.items())
    ]
    return PlayerPrediction(
        player=PlayerRef(
            id=p.id,
            name=p.web_name,
            full_name=p.full_name,
            team=BS.team(p.team).short_name,
            position=p.position.short,
            price=p.price,
            status=p.status,
            chance=None,
            ownership=float(p.selected_by_percent or 0),
        ),
        by_gw=by_gw,
        total_xpts=round(sum(g.xpts for g in by_gw), 2),
    )


def ars(x: float) -> dict[int, tuple[float, list[FixtureBrief]]]:
    """Календарь Arsenal: LEE(H) 2, NFO(A) 3, EVE(H) 2, LIV(A) 5, HUL(H) 1, blank."""
    return {
        6: (x, [fx("LEE", True, 2)]),
        7: (x, [fx("NFO", False, 3)]),
        8: (x, [fx("EVE", True, 2)]),
        9: (x, [fx("LIV", False, 5)]),
        10: (x, [fx("HUL", True, 1)]),
        11: (0.0, []),
    }


def che(x: float) -> dict[int, tuple[float, list[FixtureBrief]]]:
    """Chelsea: тяжёлый календарь (4, 4, 5, 3, 4) + double в GW11."""
    return {
        6: (x, [fx("MCI", False, 4)]),
        7: (x, [fx("LIV", True, 4)]),
        8: (x, [fx("ARS", False, 5)]),
        9: (x, [fx("TOT", True, 3)]),
        10: (x, [fx("NEW", False, 4)]),
        11: (x, [fx("BUR", True, 1), fx("SUN", False, 3)]),
    }


def lee(x: float) -> dict[int, tuple[float, list[FixtureBrief]]]:
    """Leeds: лёгкий календарь (1, 1, 2, 1, 2)."""
    return {
        6: (x, [fx("ARS", False, 1)]),
        7: (x, [fx("BUR", True, 1)]),
        8: (x, [fx("SUN", False, 2)]),
        9: (x, [fx("WOL", True, 1)]),
        10: (x, [fx("BHA", False, 2)]),
        11: (x, [fx("CRY", True, 3)]),
    }


PREDS = {
    4: pred(4, ars(4.3)),  # 3GW 12.9, FSI 7
    5: pred(5, ars(4.0)),  # 12.0
    6: pred(6, ars(2.5)),  # 7.5
    7: pred(7, che(3.5)),  # 10.5, FSI 13
    8: pred(8, lee(1.0)),  # 3.0, FSI 4
    9: pred(9, lee(3.0)),  # 9.0, FSI 4
    10: pred(10, ars(6.0)),
}


# ---------- перцентили и пул ----------


def test_percentile_strict_ties_and_direction():
    pool = [25, 20, 15, 12]
    assert card.percentile(pool, 25) == 75  # лучше трёх из четырёх (сам входит в пул)
    assert card.percentile(pool, 12) == 0
    assert card.percentile(pool, 18) == 50
    assert card.percentile([0, 0, 0, 0], 0) == 0  # Goals 0 -> 0 %, как у smartplay
    # меньше — лучше (xGC): 4.0 крепче, чем 4.5 / 5.0 / 6.0
    assert card.percentile([4.0, 4.5, 5.0, 6.0], 4.0, higher_is_better=False) == 75
    assert card.percentile([], 5) is None and card.percentile([1, 2], None) is None
    assert card.percentile([None, 3, None], 5) == 100


def test_position_pool_filters_position_and_minutes():
    pool = card.position_pool(BS, Position.DEF)
    assert [p.id for p in pool] == [4, 5, 6, 7, 9]  # Rodon (0 мин) и Saka (MID) вне пула
    assert [p.id for p in card.position_pool(BS, Position.DEF, min_minutes=1000)] == []


# ---------- шапка и ключевые числа ----------


def test_header_and_status_badge():
    h = card.header(BS.player(4), BS)
    assert h.meta == "Player report · £8.0 · DEF · ARS"
    assert h.name == "Gabriel Test" and h.club == "Arsenal"
    assert (h.status_tone, h.status_text) == ("green", "Играет")
    assert card.status_badge("d", 75) == ("orange", "Под вопросом 75 %")
    assert card.status_badge("d") == ("orange", "Под вопросом")
    assert card.status_badge("i") == ("red", "Травма")
    assert card.status_badge("s")[1] == "Дисквалификация"
    assert card.status_badge("u")[1] == "Недоступен"
    assert card.status_badge("n") == ("gray", "Не в заявке")
    assert card.badge_md("orange", "Под вопросом 75 %") == ":orange-badge[Под вопросом 75 %]"


def test_key_numbers_labels_and_values():
    ks = card.key_numbers(BS.player(4), PREDS[4])
    assert [k.label for k in ks] == ["xPts", "Points", "Form", "3GW xPts", "Own%", "Net Tx"]
    assert [k.value for k in ks] == ["4.30", "25", "5.0", "12.90", "22.8%", "−32 500"]
    assert ks[0].delta == "±2.0 sd" and "GW6" in ks[0].help
    assert all(k.help for k in ks)
    # без прогноза — прочерки, остальное из bootstrap
    ks0 = card.key_numbers(BS.player(4), None)
    assert ks0[0].value == "—" and ks0[3].value == "—" and ks0[1].value == "25"


def test_signed_int_and_money_delta():
    assert card.signed_int(15419) == "+15 419"
    assert card.signed_int(-47919) == "−47 919"
    assert card.signed_int(0) == "0" and card.signed_int(None) == "—"
    assert card.money_delta(-2.2) == "£2.2m дешевле"
    assert card.money_delta(0.5) == "£0.5m дороже"
    assert card.money_delta(0.0) == "та же цена"


# ---------- секции ----------


def test_stat_sections_titles_labels_and_percentiles():
    sections = card.stat_sections(BS.player(4), BS)
    titles = [s.title for s in sections]
    assert titles == [  # порядок и названия — как у smartplay
        "Overview",
        "Minutes",
        "Attacking Output",
        "Advanced Attacking",
        "Defence & Reliability",
        "Ownership & Transfers",
    ]
    labels = {s.key: [m.label for m in s.metrics] for s in sections}
    assert labels["overview"] == ["Pts", "Bonus", "Pts/Game", "Form"]
    assert labels["minutes"] == ["Mins", "Matches", "Avg Mins", "Mins%"]
    assert labels["attack"] == ["Goals", "Assists", "xG", "xA", "xGI"]
    assert labels["advanced"] == ["Creativity", "Threat", "xG/90"]  # без Understat
    assert labels["defence"] == ["Clean S.", "xGC", "Def. contr."]  # защитник: без сейвов
    assert labels["ownership"] == ["TSB%", "Transf. In", "Transf. Out", "Net Transf."]
    by_key = {m.key: m for s in sections for m in s.metrics}
    # Pts 25 — лучше 4 из 5 защитников пула -> 80 %
    assert by_key["pts"].text == "25" and by_key["pts"].pct == 80
    assert "лучше 80 % защитников" in by_key["pts"].help
    # Goals 0 при пуле [0, 0, 0, 0, 1] -> 0 %
    assert by_key["goals"].text == "0" and by_key["goals"].pct == 0
    # xGC: меньше лучше; 4.0 крепче 4.5 / 5.0 / 6.0, у Struijk None -> пул из 4 -> 75 %
    assert by_key["xgc"].text == "4.0" and by_key["xgc"].pct == 75
    assert "меньше, чем у 75 % защитников" in by_key["xgc"].help
    assert by_key["threat"].text == "84.0"
    # «в прогнозе»: минуты, xG / xA / xGI / xG/90, защитные действия; фактические голы, очки,
    # Creativity / Threat, сухие матчи, владение — только справка
    in_model = {k for k, m in by_key.items() if m.model}
    assert in_model == {
        "mins",
        "matches",
        "avg_mins",
        "mins_share",
        "xg",
        "xa",
        "xgi",
        "xg90",
        "defcon",
    }
    assert by_key["avg_mins"].text == "90" and by_key["mins_share"].text == "100%"
    # владение и трансферы — без перцентилей, со знаком
    own = sections[-1]
    assert all(m.pct is None for m in own.metrics)
    assert [m.text for m in own.metrics] == ["22.8%", "+15 419", "−47 919", "−32 500"]
    assert not any(s.collapsed for s in sections)  # DEF: оборона раскрыта
    # help на русском у каждой метрики
    assert all(m.help and any("а" <= ch <= "я" for ch in m.help.lower()) for m in by_key.values())


def test_stat_sections_collapse_defence_for_attackers_and_use_history_share():
    share = {4: 0.6, 5: 1.0, 6: 1.0, 7: 0.8, 9: 0.4}
    sections = card.stat_sections(BS.player(10), BS, mins_share=share)
    defence = next(s for s in sections if s.key == "defence")
    assert defence.collapsed
    # MID один в пуле -> 0 % везде, но значения есть
    pts = next(m for s in sections for m in s.metrics if m.key == "pts")
    assert pts.text == "40" and pts.pct == 0
    mins = next(m for s in sections for m in s.metrics if m.key == "mins_share")
    assert mins.text == "—" and "истории туров" in mins.help  # Saka нет в share
    g = card.stat_sections(BS.player(4), BS, mins_share=share)
    m = next(m for s in g for m in s.metrics if m.key == "mins_share")
    assert m.text == "60%" and m.pct == 20  # хуже только Struijk 0.4


def test_full_match_share_from_history():
    def row(pid, rnd, mins):
        return PlayerGWHistory(element=pid, fixture=rnd, round=rnd, was_home=True, minutes=mins)

    rows = [row(4, 1, 90), row(4, 2, 90), row(4, 3, 45), row(4, 4, 0), row(5, 1, 60), row(8, 1, 0)]
    share = card.full_match_share(rows)
    assert share[4] == 2 / 3  # 0 минут — не сыгран
    assert share[5] == 1.0 and 8 not in share


def test_percentile_bar_html():
    html = card.percentile_bar_html(88)
    assert "width:88%" in html and "88 %" in html and card.percentile_color(88) in html
    assert card.percentile_color(88).startswith("var(--fpl-good")
    assert card.percentile_color(50).startswith("var(--fpl-warn")
    assert card.percentile_color(10).startswith("var(--fpl-bad")
    assert card.percentile_bar_html(None) == ""


# ---------- календарь ----------


def test_fixture_cells_labels_colors_blank_and_double():
    cells = card.fixture_cells(PREDS[4])
    assert [c.gw for c in cells] == [6, 7, 8, 9, 10, 11]
    assert [c.label for c in cells] == ["LEE (H)", "NFO (A)", "EVE (H)", "LIV (A)", "HUL (H)", "—"]
    assert cells[0].css == "fpl-fsi-2" and cells[3].css == "fpl-fsi-5"
    assert cells[4].css == "fpl-fsi-1"
    assert cells[5].fsi is None and cells[5].css == card.BLANK_CLASS == "fpl-fsi-0"
    assert "тяжёлый" in cells[3].title and "Нет матча" in cells[5].title
    double = card.fixture_cells(PREDS[7])[-1]
    assert double.label == "BUR (H) · SUN (A)" and double.fsi == 2.0
    assert card.fixture_cells(None) == []
    html = card.fixture_cell_html(cells[0])
    assert "GW6" in html and "LEE (H)" in html and cells[0].css in html
    assert 'class="tip' in html and cells[0].title in html and 'title="' not in html
    legend = card.fixture_legend_html()
    assert "(H) — дома" in legend and all(f"fpl-fsi-{i}" in legend for i in range(1, 6))


def test_fsi_and_xpts_over_horizon():
    assert card.xpts_over(PREDS[4]) == 12.9
    assert card.fsi_over(PREDS[4]) == 7.0  # 2 + 3 + 2
    assert card.fsi_over(PREDS[7]) == 13.0  # 4 + 4 + 5
    assert card.fsi_over(None) == 3 * card.BLANK_FSI
    short = pred(4, {6: (4.0, [fx("LEE", True, 2)])})  # прогноз короче горизонта
    assert card.fsi_over(short) == 2 + 2 * card.BLANK_FSI
    assert card.gw_difficulty([]) == card.BLANK_FSI and card.gw_difficulty([1, 3]) == 2.0


# ---------- FRR ----------


def test_team_fsi_by_gw_and_run_rank():
    gws = [6, 7, 8, 9, 10]
    by_team = card.team_fsi_by_gw(PREDS, BS, gws)
    assert by_team[1] == [2, 3, 2, 5, 1]  # Arsenal, один представитель на клуб
    assert by_team[2] == [4, 4, 5, 3, 4]
    assert by_team[3] == [1, 1, 2, 1, 2]
    assert card.team_fixture_run_rank(by_team, 3) == card.RunRank(1, 3, 1.4)
    assert card.team_fixture_run_rank(by_team, 1) == card.RunRank(2, 3, 2.6)
    assert card.team_fixture_run_rank(by_team, 2) == card.RunRank(3, 3, 4.0)
    assert card.team_fixture_run_rank(by_team, 99) is None
    # blank в горизонте считается как максимальная сложность
    with_blank = card.team_fsi_by_gw(PREDS, BS, [10, 11])
    assert with_blank[1] == [1, card.BLANK_FSI]
    # равные средние делят ранг
    tied = {1: [2.0, 2.0], 2: [2.0, 2.0], 3: [3.0, 3.0]}
    assert card.team_fixture_run_rank(tied, 1).rank == 1
    assert card.team_fixture_run_rank(tied, 2).rank == 1
    assert card.team_fixture_run_rank(tied, 3).rank == 3
    # чужой id в прогнозах игнорируется
    assert card.team_fsi_by_gw({999: PREDS[4]}, BS, gws) == {}


# ---------- похожие игроки ----------


def test_competitors_for_minutes_same_club_position_top3():
    comps = card.competitors_for_minutes(BS.player(4), BS, PREDS)
    assert [c.name for c in comps] == ["Saliba", "Timber"]  # Saka — MID, Cucurella — CHE
    saliba = comps[0]
    assert saliba.club == "ARS" and saliba.position == "DEF" and saliba.price == 6.0
    assert saliba.xpts == 12.0 and saliba.status_tone == "green"
    assert saliba.compare == "£2.0m дешевле · -0.9 xPts за 3 тура"
    assert comps[1].compare == "£2.2m дешевле · -5.4 xPts за 3 тура"
    # top-3: добавим третьего защитника ARS дороже
    bs2 = Bootstrap(
        events=BS.events, teams=BS.teams, elements=[*BS.elements, pl(11, "White", 1, 2, 90, 3.0)]
    )
    preds2 = {**PREDS, 11: pred(4, ars(5.0))}  # ключ словаря — id кандидата
    comps2 = card.competitors_for_minutes(BS.player(4), bs2, preds2)
    assert [c.name for c in comps2] == ["White", "Saliba", "Timber"]
    assert comps2[0].compare == "£1.0m дороже · +2.1 xPts за 3 тура"


def test_differential_bridge_filters_ownership_price_and_run():
    # Gabriel: £8.0, FSI 7 за 3 тура. Кандидаты DEF: Saliba (30 % — много), Timber (12 % — много),
    # Cucurella (5 %, £6.2, FSI 13 — тяжелее), Rodon (1 %, £4.0, FSI 4), Struijk (2.2 %, FSI 4)
    bridge = card.differential_bridge(BS.player(4), BS, PREDS)
    assert [b.name for b in bridge] == ["Struijk", "Rodon"]
    assert bridge[0].compare == "2.2 % владение · календарь легче: 4.0 против 7.0"
    assert bridge[0].xpts == 9.0 and bridge[0].ownership == 2.2
    # одинаковый календарь — другая формулировка
    same = card.differential_bridge(BS.player(5), BS, {**PREDS, 9: pred(9, ars(3.0))})
    struijk = next(b for b in same if b.name == "Struijk")
    assert struijk.compare == "2.2 % владение · календарь такой же: 7.0"
    # цена выше текущего — отсекается
    assert card.differential_bridge(BS.player(8), BS, PREDS) == []


# ---------- новости ----------


def test_news_verdict_rows_without_signal_word():
    risk = PlayerRisk(
        player_id=4,
        player="Gabriel",
        fpl_status="a",
        fpl_chance=None,
        availability="fit",
        start_probability=0.9,
        expected_minutes=85,
        rotation_risk="low",
        confidence=0.8,
        origin="cached",
        age_h=1.5,
    )
    rows = card.news_verdict_rows(risk)
    table = {r["Поле"]: r["Значение"] for r in rows}
    assert table["Доступность"] == "в строю"
    assert table["Источник"] == "сохранённый разбор (1.5 ч назад)"
    assert table["Статус FPL"] == "доступен" and table["Возвращение"] == "—"
    assert table["Риск ротации"] == "низкий"
    text = " ".join(f"{r['Поле']} {r['Значение']}" for r in rows).lower()
    assert "сигнал" not in text and "фикстур" not in text
    now = card.news_verdict_rows(risk.model_copy(update={"origin": "extracted"}))
    assert {r["Поле"]: r["Значение"] for r in now}["Источник"] == "разобрано сейчас"


# ---------- smartplay-раскладка: альтернативы по цене, Understat, HTML ----------


def test_alternatives_at_price_same_position_other_club_available_within_window():
    # Saliba: £6.0, ARS. DEF других клубов в пределах ±0.5: Cucurella (£6.2, CHE, FSI 13).
    alts = card.alternatives_at_price(BS.player(5), BS, PREDS)
    assert [a.name for a in alts] == ["Cucurella"]
    assert alts[0].compare == "£0.2m дороже · календарь тяжелее: 13.0 против 7.0"
    # под вопросом (d) и тот же клуб — не альтернатива; дешевле и легче — «дешевле · легче»
    bs2 = Bootstrap(
        events=BS.events,
        teams=BS.teams,
        elements=[
            *BS.elements,
            pl(12, "Doubt", 2, 2, 59, 3.0, status="d", chance_of_playing_next_round=50),
            pl(13, "Leeds", 3, 2, 57, 3.0),
        ],
    )
    preds2 = {**PREDS, 12: pred(4, che(9.0)), 13: pred(9, lee(2.0))}
    alts2 = card.alternatives_at_price(BS.player(5), bs2, preds2)
    assert [a.name for a in alts2] == ["Cucurella", "Leeds"]
    assert alts2[1].compare == "£0.3m дешевле · календарь легче: 4.0 против 7.0"
    assert card.run_word(7.0, 7.0) == "календарь такой же: 7.0"


def test_stat_sections_understat_only_when_matched_and_gk_saves():
    class Row:
        def __init__(self, shots, npxg):
            self.shots, self.npxg = shots, npxg

    rows = {4: Row(10, 0.4), 5: Row(20, 1.0), 6: Row(5, 0.1)}
    sections = {s.key: s for s in card.stat_sections(BS.player(4), BS, ext_row=rows.get)}
    adv = [m.label for m in sections["advanced"].metrics]
    assert adv == ["Creativity", "Threat", "xG/90", "Shots", "npxG"]
    shots = next(m for m in sections["advanced"].metrics if m.key == "u_shots")
    assert shots.text == "10" and shots.pct is not None and shots.model is None
    # игрок без строки Understat — метрик Understat нет
    no_row = {s.key: s for s in card.stat_sections(BS.player(7), BS, ext_row=rows.get)}
    assert [m.label for m in no_row["advanced"].metrics] == ["Creativity", "Threat", "xG/90"]
    # вратарь: в «Defence & Reliability» — сейвы в прогнозе
    gk_bs = Bootstrap(
        events=BS.events, teams=BS.teams, elements=[*BS.elements, pl(20, "Raya", 1, 1, 60, 30.0)]
    )
    gk = {s.key: s for s in card.stat_sections(gk_bs.player(20), gk_bs)}
    assert [m.label for m in gk["defence"].metrics] == ["Clean S.", "xGC", "Saves"]
    assert gk["defence"].metrics[-1].model == card.MODEL_SAVES


def test_section_card_html_rows_pct_bar_and_model_dot():
    sections = {s.key: s for s in card.stat_sections(BS.player(4), BS)}
    html = card.section_card_html(sections["overview"])
    assert '<div class="fpl-sp-title">Overview</div>' in html
    assert html.count('class="fpl-sp-row"') == 4
    assert '<b class="val">25</b>' in html and "width:80%" in html and ">80%<" in html
    assert 'class="tip model"' not in html  # в обзоре нет метрик прогноза
    minutes = card.section_card_html(sections["minutes"])
    assert minutes.count('class="tip model"') == 4 and "Влияет на прогноз: модель минут" in minutes
    # без перцентилей (стандарты) — полоска скрыта
    sp = card.section_card_html(card.setpiece_section(BS.player(4)))
    assert 'class="bar none"' in sp and "width:" not in sp
    own = card.ownership_strip_html(sections["ownership"])
    assert '<b class="good">+15 419</b>' in own and '<b class="bad">−47 919</b>' in own


def test_key_numbers_and_similar_card_html():
    keys = card.key_numbers(BS.player(4), PREDS[4])
    html = card.key_numbers_html(keys)
    assert html.count("<div") == 7 and '<div class="accent"><b>4.30</b>' in html
    s = card.competitors_for_minutes(BS.player(4), BS, PREDS)[0]
    sim = card.similar_card_html(s)
    assert "<strong>Saliba</strong>" in sim and "ARS · DEF · £6.0" in sim and "<b>12.0</b>" in sim
    assert "fpl-status" not in sim  # статус «играет» не показываем
    assert s.compare in sim


def test_points_origin_rows_list_without_zero_categories():
    data = {
        "n_rounds": 3,
        "last_n": 3,
        "total_points": 25,
        "categories": {"appearance": 6, "assists": 3, "clean_sheet": 8, "goals": 0, "cards": -1},
        "label": "repeatable",
        "label_ru": "повторяемые",
    }
    title, rows, label, total = card.points_origin_rows(data)
    assert title == "Откуда очки за 3 тура"
    assert rows == [("выход", 6), ("ассисты", 3), ("сухие", 8), ("карточки", -1)]
    assert label == "повторяемые" and total == 25
    html = card.points_origin_html(rows, label, total)
    assert html.count('class="fpl-sp-row origin"') == 4 and "всего 25 очков" in html
    assert "var(--fpl-bad)" in html  # минус — красным
    assert card.points_origin_rows(None) is None
