"""Чистые помощники форматирования UI (app/format.py): без streamlit, сети и БД."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, timezone

from test_agent_fakes import BS, DEADLINE, NOW, FakeTools, fit_risk, route

from fplcopilot.agent.tools import (
    TOOL_NAMES,
    FixtureBrief,
    GameweekContextInput,
    OptimizeTeamInput,
    PlanMoveOut,
    PlanOut,
    PlayerRiskInput,
    PredictPlayerInput,
)
from fplcopilot.app import format as fmt


def test_deadline_countdown_and_ages():
    assert fmt.deadline_countdown(DEADLINE, NOW) == "1 д 7 ч 30 мин"
    assert fmt.deadline_countdown(NOW - timedelta(minutes=1), NOW) == "дедлайн прошёл"
    assert fmt.deadline_countdown(NOW + timedelta(minutes=5), NOW) == "5 мин"
    assert fmt.deadline_countdown(None, NOW) == "—"
    assert fmt.age_text(0.5) == "30 мин назад"
    assert fmt.age_text(1.4) == "1.4 ч назад"
    assert fmt.age_text(72) == "3 дн назад" and fmt.age_text(60) == "2.5 дн назад"
    assert fmt.deadline_countdown(DEADLINE, NOW, short=True) == "1 д 7 ч"
    assert (
        fmt.deadline_countdown(NOW + timedelta(hours=2, minutes=5), NOW, short=True) == "2 ч 5 мин"
    )
    assert fmt.deadline_text(DEADLINE, UTC) == "18 сен, 17:30"
    assert fmt.deadline_text(DEADLINE, timezone(timedelta(hours=5))) == "18 сен, 22:30"
    assert fmt.since_text(NOW - timedelta(hours=2), NOW) == "2.0 ч назад"
    assert fmt.since_text(None, NOW) == "нет данных"


def test_primitives_match_cli_rounding():
    assert fmt.money(7.8) == "£7.8m" and fmt.money(None) == "—"
    assert fmt.signed(4.03) == "+4.03" and fmt.signed(-0.19) == "-0.19"
    assert fmt.vs_best_label(5.91) == "−5.91 к лучшему составу"
    assert fmt.vs_best_label(-1.2) == "+1.20 к лучшему составу"
    assert fmt.vs_best_label(0) == "0.00 к лучшему составу"
    assert fmt._best_label(5.91) == fmt.vs_best_label(5.91)
    assert " " in fmt._best_label(5.91) and "\u200b" not in fmt._best_label(5.91)
    assert "оптимум" not in fmt._best_label(5.91)
    assert fmt.num(56.14) == "56.14" and fmt.num(3.0, 1) == "3.0"
    assert fmt.pct(72.6, 1) == "72.6 %"
    assert fmt.fsi_label(1) == "FSI 1 · очень лёгкий" and fmt.fsi_label(None) == "—"
    assert fmt.status_text("d", 75) == "под вопросом (75 %)"
    assert fmt.status_text("a") == "доступен"


def test_squad_table_rows_smartplay_columns_and_calendar():
    """Таблица «Мой состав» по образцу smartplay Players: старт GKP->DEF->MID->FWD с # и ролями,
    скамейка Зап1…, числа числами, календарь «BRE (A)» со сложностью для раскраски, FRR#,
    значок проблемы только у проблемных игроков."""
    tools = FakeTools()
    ctx = tools.get_gameweek_context(GameweekContextInput(manager_id=1))
    ctx.squad[-1].is_starting = False  # Guéhi -> скамейка
    ctx.squad[0].is_vice = True  # Palmer -> VC
    preds = {
        p.id: tools.predict_player(PredictPlayerInput(player_id=p.id, gw=5, horizon=5))
        for p in ctx.squad
    }
    gws = [5, 6, 7, 8, 9]
    problems = fmt.serious_problems(ctx, preds)
    xi, bench = fmt.squad_table_rows(ctx, preds, BS, gws=gws, frr={"CHE": 3}, problems=problems)
    # старт отсортирован по позиции: Gabriel (DEF), Palmer (MID), João Pedro / Haaland (FWD)
    assert [r["Player"] for r in xi] == ["Gabriel", "Palmer", "João Pedro", "Haaland"]
    assert [(r["_team"], r["_pos"]) for r in xi] == [
        ("ARS", "DEF"),
        ("CHE", "MID"),
        ("CHE", "FWD"),
        ("MCI", "FWD"),
    ]
    assert [r["#"] for r in xi] == [1, 2, 3, 4] and [r["#"] for r in bench] == [5]
    assert [r["Роль"] for r in xi] == ["—", "VC", "—", "C"] and bench[0]["Роль"] == "Зап1"
    assert list(xi[0]) == [
        *fmt.SQUAD_COLUMNS,
        "GW5",
        "GW6",
        "GW7",
        "GW8",
        "GW9",
        "_id",
        "_team",
        "_pos",
        "_flag",
        "_fsi",
    ]
    palmer = xi[1]
    assert palmer["Price"] == 9.7 and palmer["TSB%"] == 26.8  # из bootstrap
    assert palmer["Form"] is None and palmer["Pts/Game"] is None  # в мини-bootstrap нет
    assert palmer["xMins"] == 80 and palmer["xPts"] == 4.76 and palmer["3GW xPts"] == 14.28
    assert palmer["FRR#"] == 3 and xi[0]["FRR#"] is None  # ARS не в рейтинге
    assert palmer["GW5"] == "BRE (A)" and palmer["_fsi"] == dict.fromkeys(
        ("GW5", "GW6", "GW7", "GW8", "GW9"), 3
    )
    assert palmer["_flag"] is None and palmer["_id"] == 154
    assert xi[2]["_flag"] == ("warn", "Под вопросом (75 %)")  # João Pedro, FPL doubtful 75 %
    # без прогноза: ячейки календаря «—», сложности нет, xPts — из контекста тура
    xi2, _ = fmt.squad_table_rows(ctx, {}, None, gws=gws)
    assert xi2[1]["GW5"] == "—" and xi2[1]["_fsi"]["GW5"] is None
    assert xi2[1]["xPts"] == 4.76 and xi2[1]["xMins"] is None and xi2[1]["TSB%"] == 26.8


def test_squad_table_html_links_badges_calendar_and_bench():
    tools = FakeTools()
    ctx = tools.get_gameweek_context(GameweekContextInput(manager_id=1))
    ctx.squad[-1].is_starting = False  # Guéhi -> скамейка
    preds = {
        p.id: tools.predict_player(PredictPlayerInput(player_id=p.id, gw=5, horizon=5))
        for p in ctx.squad
    }
    gws = [5, 6, 7, 8, 9]
    problems = fmt.serious_problems(ctx, preds)
    xi, bench = fmt.squad_table_rows(ctx, preds, BS, gws=gws, frr={"CHE": 3}, problems=problems)
    html = fmt.squad_table_html(xi, bench, gw=5, gws=gws)
    assert html.lstrip().startswith("<style>") and '<table class="fpl-table fpl-squad">' in html
    # заголовки smartplay с CSS-tip и «?»; xMins / xPts — с туром
    for label in (
        "Player",
        "Price",
        "TSB%",
        "Form",
        "Pts/Game",
        "xMins GW5",
        "xPts GW5",
        "3GW xPts",
        "FRR#",
        "GW9",
    ):
        assert f'>{label}<span class="tip' in html, label
    assert "Team Selected By — доля менеджеров FPL" in html
    assert 'class="tip-text"' in html
    assert 'title="' not in html
    assert "Статус" not in html and "Next Opp" not in html
    # имя — ссылка на страницу «Игрок» с pid = FPL element id; вторая строка «клуб · поз · TSB»
    assert '<a href="/player?pid=154" target="_self">Palmer</a>' in html
    assert '<div class="sub">CHE · MID · 26.8%</div>' in html
    assert '<a href="/player?pid=411" target="_self">Haaland</a>' in html
    # цветная точка только у проблемного (João Pedro), с подсказкой
    assert html.count('<span class="tip flag"') == 1
    assert (
        '<span class="tip flag"><span class="fpl-dot warn"></span>'
        '<span class="tip-text">Под вопросом (75 %)</span></span>'
        in html
    )
    # числа и «—» для отсутствующих; FRR только у CHE
    assert '<td class="num">£9.7</td>' in html and '<td class="num">#3</td>' in html
    assert '<td class="num">80</td>' in html and '<td class="num xpts">4.8</td>' in html
    assert '<td class="num">—</td>' in html
    # календарь: чип по сложности 3 (серый), формат smartplay «BRE (A)»
    assert '<td class="cal"><span class="fpl-fx fpl-fsi-3">BRE (A)</span></td>' in html
    # разделитель скамейки внутри той же таблицы
    assert html.count('<tr class="sep">') == 1 and html.count('<tr class="bench">') == 1
    assert '<td colspan="16">Bench</td>' in html  # 11 колонок + 5 туров
    assert html.index("Bench") > html.index("Haaland") and html.index("Bench") < html.index("Guéhi")
    # без скамейки — без разделителя; без проблем — без значков
    assert "Bench" not in fmt.squad_table_html(xi, [], gw=5, gws=gws)
    xi_plain, _ = fmt.squad_table_rows(ctx, preds, BS, gws=gws)
    assert '<span class="fpl-dot' not in fmt.squad_table_html(xi_plain, [], gw=5, gws=gws)


def test_cell_text_calendar_cell_and_fsi_css():
    def fx(opp: str, home: bool, fsi: int) -> FixtureBrief:
        return FixtureBrief(
            opponent=opp, is_home=home, fsi=fsi, xg_for=1, xg_against=1, clean_sheet_prob=0.3
        )

    assert fmt.calendar_cell([fx("TOT", True, 2)]) == ("TOT (H)", 2)
    assert fmt.calendar_cell([fx("MUN", True, 1), fx("LIV", False, 4)]) == (
        "MUN (H) · LIV (A)",
        2,
    )
    assert fmt.calendar_cell([]) == ("—", None) and fmt.calendar_cell(None) == ("—", None)
    assert fmt.fsi_css(1) == "fpl-fsi-1" and fmt.fsi_css(5) == "fpl-fsi-5"
    assert fmt.fsi_css(None) == "" and fmt.fsi_css(9) == ""
    legend = fmt.fsi_legend_html()
    assert '<div class="fpl-legend">' in legend and "Fixture difficulty:" in legend
    assert all(f"fpl-fsi-{k}" in legend for k in range(1, 6))
    for label in ("Very Easy", "Easy", "Medium", "Tough", "Very Tough"):
        assert label in legend
    assert fmt.cell_text("Price", 4.6) == "£4.6" and fmt.cell_text("TSB%", 42.25) == "42.2%"
    assert fmt.cell_text("xMins", 89.6) == "90" and fmt.cell_text("FRR#", 1) == "#1"
    assert fmt.cell_text("xPts", 4.76) == "4.8" and fmt.cell_text("Form", None) == "—"
    assert fmt.cell_text("Роль", "VC") == "VC"


def test_serious_problems_and_problems_line():
    tools = FakeTools()
    ctx = tools.get_gameweek_context(GameweekContextInput(manager_id=1))
    preds = {
        p.id: tools.predict_player(PredictPlayerInput(player_id=p.id, gw=5, horizon=3))
        for p in ctx.squad
    }
    # фейк: João Pedro — FPL doubtful 75 % (статус из SquadPlayerRow, без detail diagnose_squad)
    problems = fmt.serious_problems(ctx, preds)
    assert [p["name"] for p in problems] == ["João Pedro"]
    jp = problems[0]
    assert jp["id"] == 165 and jp["flag"] == "warn" and jp["label"] == "под вопросом (75 %)"
    assert jp["text"] == "João Pedro — под вопросом (75 %)" and jp["title"] == "Под вопросом (75 %)"
    assert fmt.problems_line(problems) == "João Pedro — под вопросом (75 %)"
    # травма с новостью FPL, ротация с p(start) из прогноза, календарь — не серьёзная проблема
    ctx.squad[0].status, ctx.squad[0].news = "i", "Knee injury - expected back 20 Oct"
    ctx.squad[1].status, ctx.squad[1].news = "d", "Knock"  # João Pedro c текстом новости
    issues = [
        *ctx.issues,
        {"player_id": 411, "name": "Haaland", "kind": "not_playing", "severity": 2, "detail": ""},
        {"player_id": 4, "name": "Gabriel", "kind": "fixtures", "severity": 1, "detail": ""},
    ]
    problems = fmt.serious_problems(ctx, preds, issues)
    assert [p["name"] for p in problems] == ["Palmer", "João Pedro", "Haaland"]  # порядок состава
    assert problems[0]["flag"] == "bad"
    assert problems[0]["text"] == "Palmer — травма, Knee injury - expected back 20 Oct"
    assert problems[0]["title"] == "Травма — Knee injury - expected back 20 Oct"
    assert problems[1]["text"] == "João Pedro — под вопросом (75 %), Knock"
    assert problems[2] == {
        "id": 411,
        "name": "Haaland",
        "flag": "warn",
        "label": "риск ротации (выйдет в старте 93 %)",
        "news": "",
        "text": "Haaland — риск ротации (выйдет в старте 93 %)",
        "title": "Риск ротации (выйдет в старте 93 %)",
    }
    assert fmt.problems_line(problems) == (
        "Palmer — травма, Knee injury - expected back 20 Oct · João Pedro — под вопросом (75 %), "
        "Knock · Haaland — риск ротации (выйдет в старте 93 %)"
    )
    # серьёзных проблем нет -> строки нет (None), «проблем нет» не показываем
    for p in ctx.squad:
        p.status, p.news = "a", ""
    assert fmt.serious_problems(ctx, preds, []) == [] and fmt.problems_line([]) is None
    # без прогноза p(start) неизвестен — только «риск ротации»
    only_rot = fmt.serious_problems(ctx, {}, [issues[-2]])
    assert only_rot[0]["label"] == "риск ротации"


def test_squad_column_help_is_russian_and_parametrised():
    assert fmt.SQUAD_COLUMNS == (
        "#",
        "Роль",
        "Player",
        "Price",
        "TSB%",
        "Form",
        "Pts/Game",
        "xMins",
        "xPts",
        "3GW xPts",
        "FRR#",
    )
    assert fmt.column_help("xMins", 6).startswith("Ожидаемые минуты в GW6")
    assert fmt.column_label("xMins", 6) == "xMins GW6" and fmt.column_label("Price", 6) == "Price"
    assert fmt.column_help("xPts", 6).startswith("Прогноз очков на GW6")
    assert fmt.column_help("GW", 7).startswith("Соперник в туре GW7")
    assert fmt.column_help("TSB%", 6).startswith("Team Selected By — доля менеджеров FPL")
    assert fmt.column_help("FRR#", 6).startswith("Рейтинг календаря команды среди 20 клубов")
    assert (
        "/player" not in fmt.SQUAD_COLUMN_HELP["Player"] and fmt.PLAYER_URL == "/player?pid={pid}"
    )
    joined = " ".join(fmt.SQUAD_COLUMN_HELP.values())
    for banned in ("фикстур", "сигнал", "picks", "хит", "FSI", "диагностик"):
        assert banned not in joined, banned


def test_squad_header():
    tools = FakeTools()
    ctx = tools.get_gameweek_context(GameweekContextInput(manager_id=1))
    preds = {
        p.id: tools.predict_player(PredictPlayerInput(player_id=p.id, gw=5, horizon=3))
        for p in ctx.squad
    }
    entry = {"team": "Nfactorial_test", "points": 312, "rank": 1234567}
    cells = fmt.squad_header(ctx, entry, preds)
    assert [c[0] for c in cells] == [
        "Очки сезона",
        "Общий ранг",
        "Банк",
        "Бесплатных трансферов",
        "Чипы",
        "Прогноз очков на GW5",
    ]
    values = {c[0]: c[1] for c in cells}
    assert values["Очки сезона"] == "312" and values["Общий ранг"] == "1 234 567"
    assert values["Банк"] == "£0.1m" and values["Бесплатных трансферов"] == "2"
    assert values["Чипы"] == "—"
    # 4.76 + 3.3 + 4.58 + 7.13 × 2 (капитан Haaland) + 5.94 = 32.84
    assert values["Прогноз очков на GW5"] == "32.8"
    ctx.chips_available = ["wildcard"]
    no_entry = {c[0]: c[1] for c in fmt.squad_header(ctx, None, {})}
    assert no_entry["Очки сезона"] == "—" and no_entry["Общий ранг"] == "—"
    assert no_entry["Чипы"] == "WC" and no_entry["Прогноз очков на GW5"] == "32.8"
    ctx.chips_available = ["wildcard", "freehit", "3xc"]
    three = fmt.squad_header(ctx, None, {})
    chips_cell = next(c for c in three if c[0] == "Чипы")
    assert chips_cell[1] == "WC · FH · TC" and chips_cell[3] == "wrap"
    assert "Wildcard" in chips_cell[2] and "Triple Captain" in chips_cell[2]
    html = fmt.stat_strip_html(three)
    assert 'class="fpl-metric-value wrap"' in html and "WC · FH · TC" in html
    assert all(c[2] for c in cells)  # у каждой метрики есть help
    assert fmt.availability_text(None) == "нет разбора новостей"
    assert fmt.availability_text(fit_risk(PlayerRiskInput(player_id=154, as_of=NOW))) == (
        "в строю · уверенность 0.80"
    )


def test_route_card_and_why_prose():
    r = route(1, ["Ajayi"], ["Calafiori"], hit=0, verdict="go")
    r = r.model_copy(
        update={
            "risk_note": "in Calafiori: template (51% owned)",
            "xi_points_after": 55.66,
            "new_bank": 1.4,
            "gain_next_gw": 1.91,
            "gain_horizon": 6.06,
        }
    )
    calendars = {
        "Ajayi": [
            fmt.WhyMatch("Лидс", True, win_prob=0.35, xg_for=1.2, xg_against=1.5, fsi=4),
            fmt.WhyMatch("Брайтон", False, win_prob=0.40, xg_for=1.1, xg_against=1.4, fsi=4),
            fmt.WhyMatch("Вест Хэм", True, win_prob=0.55, xg_for=1.5, xg_against=1.2, fsi=3),
        ],
        "Calafiori": [
            fmt.WhyMatch("Борнмут", True, win_prob=0.60, xg_for=1.8, xg_against=0.9, fsi=2),
            fmt.WhyMatch("Ипсвич", False, win_prob=0.50, xg_for=1.4, xg_against=1.1, fsi=2),
            fmt.WhyMatch("Вулверхэмптон", True, win_prob=0.55, xg_for=1.6, xg_against=1.0, fsi=2),
        ],
    }
    card = fmt.route_card(r, horizon=3, calendars=calendars, ownership={"Calafiori": 51})
    assert card["title"] == "Продаём Ajayi, покупаем Calafiori."
    assert card["delta_next"] == "+1.9" and card["delta_horizon"] == "+6.1"
    assert card["hit"] == "нет" and card["horizon_label"] == "За 3 тура"
    assert card["verdict_note"] is None
    why = card["why"]
    assert why == (
        "Впереди Лидс дома, затем Брайтон и Вест Хэм — поэтому Ajayi уходит. "
        "Calafiori сыграет с Борнмутом дома, Ипсвичем и Вулверхэмптоном. "
        "После замены ваш состав набирает 55.7 очков за 3 тура, в банке останется 1.4 миллиона."
    )
    assert "это повторяется" not in why
    assert "спокойный выбор для осторожной стратегии" not in why
    for banned in (
        "→",
        "Δ",
        "×",
        "±",
        "sd",
        "xPts",
        "FT",
        "route",
        "template",
        "status d",
        "fixtures",
        "плюс 1",
        "команда забивает",
        "пропускает",
        "шансы команды",
        "Вердикт",
    ):
        assert banned not in why, banned
    # без win_prob — тоже без троек xG и без процентов победы
    no_win = {
        "Ajayi": [
            fmt.WhyMatch("Лидс", True, xg_for=1.3, xg_against=1.6, fsi=5),
            fmt.WhyMatch("Брайтон", True, xg_for=1.1, xg_against=1.5, fsi=3),
            fmt.WhyMatch("Вест Хэм", True, xg_for=1.4, xg_against=1.3, fsi=3),
        ],
        "Calafiori": [
            fmt.WhyMatch("Борнмут", True, xg_for=1.8, xg_against=0.9, fsi=2),
        ],
    }
    why2 = fmt.route_why(r, horizon=3, calendars=no_win, ownership={"Calafiori": 51})
    assert why2 == (
        "Впереди Лидс дома и два домашних матча — поэтому Ajayi уходит. "
        "Calafiori сыграет с Борнмутом дома. "
        "После замены ваш состав набирает 55.7 очков за 3 тура, в банке останется 1.4 миллиона."
    )
    assert fmt.club_name_ru("MCI") == "Сити" and fmt.club_name_ru("MUN") == "Юнайтед"
    assert fmt.club_name_ru("NFO") == "Ноттингем"
    assert fmt.sell_ahead_phrase(
        [
            fmt.WhyMatch("Сити", True, fsi=5),
            fmt.WhyMatch("Брентфорд", False, fsi=3),
            fmt.WhyMatch("Брайтон", False, fsi=3),
        ]
    ) == "впереди Сити дома и два выездных матча"
    assert "команда забивает" not in why2 and "победу" not in why2
    free = fmt.route_card(route(2, ["Konsa"], ["Guéhi"]))
    assert free["hit"] == "нет" and free["verdict_note"] is None
    assert free["title"] == "Продаём Konsa, покупаем Guéhi."
    assert free["why"] == (
        "Konsa уходит из состава. Вместо него берём Guéhi. "
        "После замены ваш состав набирает 57.6 очков за 3 тура, в банке останется 0.3 миллиона."
    )
    hold = fmt.route_card(route(3, ["Gakpo"], ["Rogers"], verdict="hold"))
    assert hold["verdict_note"] == "Лучше подождать, выигрыш слишком маленький."
    paid = fmt.route_card(route(4, ["Gakpo"], ["Palmer"], hit=4, verdict="hit_not_worth"))
    assert paid["hit"] == "−4"
    assert paid["verdict_note"] == "За минус 4 очка этот трансфер не окупается."
    assert "Это платный трансфер за минус 4 очка." in paid["why"]
    html = fmt.route_metrics_html(
        next_label="Следующий тур",
        next_value="+1.9",
        horizon_label="За 3 тура",
        horizon_value="+6.1",
        hit_label="Платный трансфер",
        hit_value="нет",
    )
    assert "Следующий тур" in html and "Платный трансфер" in html
    assert "+1.9" in html and "нет" in html
    assert "ellipsis" not in html and "плюс" not in html


def test_route_why_with_stats_not_only_calendar():
    from fplcopilot.core.why_facts import WhyFact

    r = route(1, ["Ajayi"], ["Calafiori"], hit=0, verdict="go")
    r = r.model_copy(update={"xi_points_after": 55.66, "new_bank": 1.4, "gain_next_gw": 1.91})
    calendars = {
        "Ajayi": [fmt.WhyMatch("Сити", True, fsi=5), fmt.WhyMatch("Брайтон", False, fsi=3)],
        "Calafiori": [fmt.WhyMatch("Борнмут", True, fsi=2)],
    }
    facts = {
        "Ajayi": [WhyFact("few_chances", "в последних 3 турах почти не создавал моменты", "sell", 20)],
        "Calafiori": [
            WhyFact(
                "moments_no_goals",
                "в последних 3 турах создавал моменты (1.8 ожидаемых голов), но не забивал",
                "buy",
                10,
            ),
            WhyFact("leaky_opp", "дальше Борнмут — уже пропустили 12 мячей", "buy", 18),
        ],
    }
    why = fmt.route_why(r, horizon=3, calendars=calendars, facts=facts)
    assert "моменты" in why and "не забивал" in why
    assert "пропустили 12" in why
    assert "xG" not in why and "selected_by" not in why
    assert "против слабых" not in why
    assert why.count(".") >= 2
    card = fmt.route_card(r, horizon=3, calendars=calendars, facts=facts)
    assert card["delta_next"] == "+1.9" and card["hit"] == "нет"


def test_route_why_penalty_only_as_contrast():
    from fplcopilot.core.why_facts import WhyFact

    calendars = {
        "Gakpo": [fmt.WhyMatch("Сити", True, fsi=4)],
        "Palmer": [fmt.WhyMatch("Борнмут", True, fsi=2)],
    }
    contrast_route = route(1, ["Gakpo"], ["Palmer"], hit=0, verdict="go")
    contrast_route = contrast_route.model_copy(
        update={"xi_points_after": 55.0, "new_bank": 1.0, "gain_next_gw": 2.0}
    )
    outlook = WhyFact(
        "penalty_outlook",
        "примерно 35%, что у Челси будет пенальти за эти 3 тура, бить будет Palmer",
        "buy",
        26,
    )
    why = fmt.route_why(
        contrast_route,
        horizon=3,
        calendars=calendars,
        facts={
            "Gakpo": [],
            "Palmer": [WhyFact("penalty", "бьёт пенальти у Челси", "buy", 25), outlook],
        },
    )
    assert "35%" in why and "Palmer" in why
    assert "бьёт пенальти у Челси" not in why

    both = fmt.route_why(
        contrast_route,
        horizon=3,
        calendars=calendars,
        facts={
            "Gakpo": [WhyFact("penalty", "бьёт пенальти у Ливерпуля", "buy", 25)],
            "Palmer": [WhyFact("penalty", "бьёт пенальти у Челси", "buy", 25), outlook],
        },
    )
    assert "пенальти" not in both

    standalone = route(1, ["Ajayi"], ["Palmer"], hit=0, verdict="go")
    standalone = standalone.model_copy(
        update={"xi_points_after": 55.0, "new_bank": 1.0, "gain_next_gw": 2.0}
    )
    why_solo = fmt.route_why(
        standalone,
        horizon=3,
        calendars={"Ajayi": calendars["Gakpo"], "Palmer": calendars["Palmer"]},
        facts={"Ajayi": [], "Palmer": [outlook]},
    )
    assert "35%" in why_solo and "Palmer" in why_solo

    bare = fmt.route_why(
        contrast_route,
        horizon=3,
        calendars=calendars,
        facts={"Gakpo": [], "Palmer": [WhyFact("penalty", "бьёт пенальти у Челси", "buy", 25)]},
    )
    assert "пенальти" not in bare


def test_route_why_form_phrase_when_meaningful():
    from fplcopilot.core.why_facts import WhyFact

    r = route(1, ["Gakpo"], ["Rogers"], hit=0, verdict="go")
    r = r.model_copy(update={"xi_points_after": 55.0, "new_bank": 1.4, "gain_next_gw": 2.0})
    calendars = {
        "Gakpo": [fmt.WhyMatch("Сити", True, fsi=4)],
        "Rogers": [fmt.WhyMatch("Борнмут", True, fsi=2)],
    }
    facts = {
        "Gakpo": [
            WhyFact(
                "form_unlucky",
                "моменты создавал, а очков мало (6 за 3 тура) — продавать из-за очков рано",
                "sell",
                8,
            )
        ],
        "Rogers": [
            WhyFact(
                "form_lucky",
                "16 очков за 3 тура во многом везение — такого снова не ждите",
                "buy",
                8,
            )
        ],
    }
    why = fmt.route_why(r, horizon=3, calendars=calendars, facts=facts)
    assert "продавать из-за очков рано" in why
    assert "везение" in why and "не ждите" in why
    assert "на удаче" not in why
    assert "это повторяется" not in why
    assert "xG" not in why
    card = fmt.route_card(r, horizon=3, calendars=calendars, facts=facts)
    assert card["delta_next"] == "+2.0" and card["hit"] == "нет"


def test_routes_caption_russian():
    from fplcopilot.agent.tools import RoutesOut

    rt = RoutesOut(
        gw=6,
        horizon=3,
        strategy="balanced",
        free_transfers=1,
        bank=3.0,
        routes=[],
        recommended_rank=1,
        recommendation="route 1",
        baseline_xi_points=53.76,
    )
    assert fmt.routes_caption(rt) == (
        "У вас один бесплатный трансфер и 3.0 миллиона в банке; считаем на 3 тура. "
        "Без трансферов ваш лучший состав наберёт 53.76 очка — рекомендуем вариант 1."
    )
    hold = rt.model_copy(update={"recommendation": "hold", "recommended_rank": None})
    assert fmt.routes_caption(hold).endswith("рекомендуем держать трансфер.")


def test_lineup_and_captain_rows():
    tools = FakeTools()
    lu = tools.optimize_team(OptimizeTeamInput(manager_id=1, gw=5))
    rows = fmt.lineup_rows(lu, sell=["Saka"], buy=[])
    assert rows[0]["Роль"] == "—" and {r["Роль"] for r in rows} >= {"C", "VC"}
    assert "Статус" not in rows[0]
    assert "±sd" not in rows[0] and "p(start)" not in rows[0] and "xPts" not in rows[0]
    assert rows[0]["разброс"] and "%" in rows[0]["шанс старта"]
    assert "(д)" in rows[0]["Матч"] or "(в)" in rows[0]["Матч"] or rows[0]["Матч"] == "—"
    html = fmt.lineup_table_html(lu, sell=["Saka"], buy=["Guéhi"])
    assert "продать" in html and "купить" in html and "tag-sell" in html
    assert "XI" not in html and "лучшие 11" not in html
    assert "XI" not in fmt.LINEUP_COLUMN_HELP["Роль"]
    assert "Guéhi" in fmt.buy_chips_html(["Guéhi"])
    caps = fmt.captain_rows(lu.captain_options)
    assert caps[0]["Игрок"] == "Haaland" and caps[0]["Как капитан"] == "14.26"
    assert caps[0]["Тип"] == "надёжный"
    assert fmt.CAPTAIN_TAG_RU == {
        "safe": "надёжный",
        "balanced": "обычный",
        "differential": "редкий",
    }


def test_fixture_match_label():
    assert fmt.fixture_match_label("vTOT (FSI 4)") == "TOT (д)"
    assert fmt.fixture_match_label("@LIV (FSI 2)") == "LIV (в)"
    assert fmt.fixture_match_label("vSUN (FSI 1) · @CHE (FSI 3)") == "SUN (д) · CHE (в)"
    assert fmt.fixture_match_label("TOT (H)") == "TOT (д)"
    assert fmt.fixture_match_label(None) == "—"


def test_plan_rows_summary_and_diff():
    plan = PlanOut(
        from_gw=5,
        horizon=2,
        strategy="balanced",
        moves_by_gw={
            "5": [
                PlanMoveOut(
                    gw=5,
                    out="Konsa",
                    in_="Guéhi",
                    price_out=4.5,
                    price_in=6.0,
                    delta_xpts_horizon=13.45,
                    out_problem=None,
                ),
                PlanMoveOut(
                    gw=5,
                    out="João Pedro",
                    in_="Barry",
                    price_out=7.8,
                    price_in=5.6,
                    delta_xpts_horizon=2.49,
                    out_problem="doubtful",
                    paid=True,
                ),
            ],
            "6": [],
        },
        hits_by_gw={"5": 4, "6": 0},
        ft_by_gw={"5": 2, "6": 1},
        bank_by_gw={"5": 0.8, "6": 0.8},
        xi_points_by_gw={"5": 60.17, "6": 59.66},
        captain_by_gw={"5": "Haaland", "6": "Saka"},
        expected_total=307.54,
        baseline_total=283.13,
        recommendation="transfers",
        wildcard={"gw": 5, "expected_total": 321.9, "delta_vs_plan": 14.36, "squad": ["A"]},
        solver="HiGHS",
        runtime_s=5.2,
        diff_vs_previous=[
            {
                "gw": 5,
                "kind": "changed_target",
                "player_ids": [1, 2, 3],
                "detail": "Konsa: X -> Guéhi",
            }
        ],
    )
    moves = fmt.plan_move_rows(plan)
    assert moves[0]["Out"] == "Konsa (£4.5)" and moves[0]["Δ xPts горизонт"] == "+13.45"
    assert moves[1]["Хит"] == "−4" and moves[1]["Причина"] == "под вопросом"
    assert moves[2]["Причина"] == "roll (без трансфера)"
    gws = fmt.plan_gw_rows(plan)
    assert gws[0]["Прогноз состава"] == "60.17" and gws[1]["Капитан"] == "Saka"
    summary = fmt.plan_summary(plan)
    assert summary["План"] == "307.54 xPts" and summary["Выигрыш"] == "+24.41"
    assert summary["Wildcard"] == "321.90 xPts (+14.36 к плану)"
    assert summary["Рекомендация"] == "трансферы"
    assert fmt.diff_rows(plan.diff_vs_previous)[0]["Изменение"] == "другая цель"
    assert fmt.diff_rows(None) == []


def test_player_helpers():
    tools = FakeTools()
    pred = tools.predict_player(PredictPlayerInput(player_id=154, gw=5, horizon=3))
    assert fmt.xpts_series(pred) == {"GW5": 4.76, "GW6": 4.76, "GW7": 4.76}
    comp = fmt.components_rows(pred)
    assert comp[0]["GW"] == 5 and comp[0]["голы"] == "1.10" and comp[0]["мин"] == 80
    fx = fmt.fixture_rows(pred)
    assert fx[0] == {"GW": 5, "Соперник": "BRE", "Где": "в гостях", "Сложность": "FSI 3 · средний"}
    risk = fit_risk(PlayerRiskInput(player_id=154, as_of=NOW))
    summary = fmt.risk_summary(risk)
    assert summary["Доступность"] == "в строю" and summary["Источник"].startswith("сохранённый")
    assert fmt.extraction_cost_text(risk.model_copy(update={"llm_calls": 1}), 0.0008).startswith(
        "1 LLM-вызов(ов)"
    )
    assert fmt.evidence_rows(risk.evidence)[0]["Источник"] == "bbc"
    assert "Cole Palmer" in fmt.player_label(BS.player(154), BS)


def test_agent_run_summary_and_pending_action():
    state = {
        "intent": "what_if",
        "strategy": "balanced",
        "gw": 5,
        "thread_id": "abc",
        "answer": "**Verdict:** ok",
        "retries": 0,
        "cost_estimate": 0.0011,
        "validation": {"passed": True, "checked_names": 12, "checked_numbers": 4, "attempts": 1},
        "tool_log": [
            {"node": "compute", "tool": "simulate_scenario", "latency_ms": 1162, "ok": True},
            {"node": "compute", "tool": "-", "latency_ms": 0, "ok": True},
        ],
        "llm_calls": [
            {
                "node": "router",
                "purpose": "route",
                "model": "gpt-4o-mini",
                "prompt_tokens": 1397,
                "completion_tokens": 73,
                "latency_ms": 1073,
                "cost_usd": 0.000253,
            }
        ],
    }
    summary = fmt.run_summary(state, elapsed_s=2.3)
    assert summary["Интент"] == "what_if" and summary["Стоимость"] == "≈ $0.0011"
    assert summary["LLM-вызовов"] == "1 (токены 1397/73)" and summary["Время"] == "2.3 с"
    assert summary["Валидация"].startswith("пройдена: имён 12")
    assert len(fmt.tool_log_rows(state)) == 1  # «-» отфильтрован
    assert fmt.llm_call_rows(state)[0]["$"] == "0.0003"
    assert fmt.validation_text({"answer": None}) == "n/a (ответа ещё нет)"
    assert fmt.validation_text({"answer": "x"}) == "n/a (детерминированный ответ без LLM)"
    text = fmt.pending_action_text({"kind": "hit", "detail": "A -> B (GW5)", "cost": 4})
    assert text == "Требует подтверждения: платный трансфер (хит) — A -> B (GW5), цена 4 очков."
    assert fmt.pending_action_text({"kind": "wildcard", "detail": "WC", "cost": 0}).endswith(
        "Wildcard — WC."
    )


def test_plural():
    assert fmt.plural(1, "очко", "очка", "очков") == "1 очко"
    assert fmt.plural(2, "очко", "очка", "очков") == "2 очка"
    assert fmt.plural(5, "очко", "очка", "очков") == "5 очков"
    assert fmt.plural(11, "очко", "очка", "очков") == "11 очков"
    assert fmt.plural(21, "очко", "очка", "очков") == "21 очко"
    assert fmt.plural(312, "очко", "очка", "очков") == "312 очков"
    assert fmt.plural(None, "очко", "очка", "очков") == "0 очков"


def test_strategy_labels_and_help_are_plain_russian():
    assert [fmt.strategy_label(s) for s in fmt.STRATEGIES] == [
        "Осторожная",
        "Сбалансированная",
        "Агрессивная",
    ]
    assert fmt.strategy_label("custom") == "custom"
    for jargon in ("хит", "шаблон", "дифференциал", "conservative", "balanced", "aggressive"):
        assert jargon not in fmt.STRATEGY_HELP
    assert fmt.STRATEGY_HELP.startswith("Осторожная — надёжные игроки")
    assert fmt.MANAGER_HELP == "Число из адреса вашей команды: fantasy.premierleague.com/entry/ID/"


def test_parse_manager_id():
    assert fmt.parse_manager_id("10835228") == 10835228
    assert fmt.parse_manager_id(" 42 ") == 42
    assert fmt.parse_manager_id("") == 0 and fmt.parse_manager_id(None) == 0
    assert fmt.parse_manager_id("0") == 0 and fmt.parse_manager_id("  ") == 0
    for bad in ("abc", "12a", "-5", "1.5", "1 2"):
        assert fmt.parse_manager_id(bad) is None, bad


def test_manager_caption():
    entry = {"team": "Nfactorial_test", "points": 312, "rank": 1234567}
    assert fmt.manager_caption(10835228, entry) == "Nfactorial_test · 312 очков · ранг 1 234 567"
    assert fmt.manager_caption(7, {"team": "X", "points": None, "rank": None}) == "X"
    assert fmt.manager_caption(7, {"team": "", "points": 1, "rank": None}) == "команда 7 · 1 очко"
    assert fmt.manager_caption(42, None) == "Менеджер не найден в FPL"
    assert fmt.manager_caption(None, None) == "Без менеджера"
    assert fmt.manager_caption(0, entry) == "Без менеджера"


def test_squad_source_text_sidebar():
    tools = FakeTools()
    ctx = tools.get_gameweek_context(GameweekContextInput(manager_id=1))
    text, kind, help_text = fmt.squad_source_text(ctx, None)
    assert (text, kind) == ("Состав: из FPL за GW4", "markdown")
    assert help_text == (
        "FPL отдаёт состав только за завершённый тур. Сделали трансферы на GW5? "
        "Загрузите скриншот на странице «Мой состав»."
    )
    assert fmt.squad_source_text(ctx, 5) == ("Состав: со скриншота (GW5)", "markdown", None)
    no_squad = FakeTools(squad=False).get_gameweek_context(GameweekContextInput(manager_id=1))
    assert fmt.squad_source_text(no_squad, None) == (
        "Состав: нет — загрузите скриншот",
        "warning",
        None,
    )
    text, kind, _ = fmt.squad_source_text(no_squad, None, has_manager=False)
    assert kind == "warning" and text == "Состав: нет — укажите ID менеджера или загрузите скриншот"


def test_freshness_lines_and_stale_flag():
    fr = {
        "news_at": NOW - timedelta(minutes=11),
        "signal_at": NOW - timedelta(hours=72),
        "xpts_rows": 659,
    }
    lines = fmt.freshness_lines(fr, NOW)
    assert lines["news"] == "Новости: 11 мин назад"
    assert lines["signals"] == "Разбор новостей об игроках: 3 дн назад · устарел"
    assert lines["signals_stale"] and not lines["xpts_missing"]
    fresh = fmt.freshness_lines({**fr, "signal_at": NOW - timedelta(hours=2), "xpts_rows": 0}, NOW)
    assert fresh["signals"] == "Разбор новостей об игроках: 2.0 ч назад"
    assert not fresh["signals_stale"] and fresh["xpts_missing"]
    empty = fmt.freshness_lines({}, NOW)
    assert empty["news"] == "Новости: нет данных"
    assert empty["signals"] == "Разбор новостей об игроках: нет данных" and empty["xpts_missing"]
    joined = " ".join(str(v) for v in [*lines.values(), fmt.SIGNALS_HELP])
    for banned in ("сигнал", "БД", "строк", "устарели", "игроков"):
        assert banned not in joined, banned


def test_agent_graph_dot_highlights_both_interrupts_and_pipeline_dot():
    class Edge:
        def __init__(self, s, t, c):
            self.source, self.target, self.conditional = s, t, c

    class Graph:
        def __init__(self):
            self.nodes = dict.fromkeys(
                ("__start__", "router", "confirm_action", "resolve_clarification", "explain")
            )
            self.edges = [
                Edge("__start__", "router", False),
                Edge("router", "confirm_action", True),
                Edge("router", "resolve_clarification", True),
            ]

    dot = fmt.agent_graph_dot(Graph())
    assert dot.startswith("digraph agent") and '"router" -> "confirm_action" [style=dashed]' in dot
    assert (
        '"confirm_action" [label="confirm_action\\n(interrupt before: платный трансфер / Wildcard)", fillcolor="#ffe8b3"]'
        in dot
    )
    assert (
        '"resolve_clarification" [label="resolve_clarification\\n(interrupt before: уточнение имени игрока)", fillcolor="#ffe8b3"]'
        in dot
    )
    assert '"explain" [label="explain"];' in dot  # обычный узел без подсветки
    assert set(fmt.INTERRUPT_NODES) == {"confirm_action", "resolve_clarification"}

    pipe = fmt.pipeline_dot(len(TOOL_NAMES), 6)
    assert pipe.lstrip().startswith("digraph pipeline")
    assert f"LiveTools: {len(TOOL_NAMES)} инструментов" in pipe and "6 страниц" in pipe
    assert "9 инструментов" not in pipe and "сигнал" not in pipe and "FSI" not in pipe
    assert "лучший состав" in pipe and "лучшие 11" not in pipe
    assert "инструмента" in fmt.pipeline_dot(3, 1) and "1 страница" in fmt.pipeline_dot(3, 1)
    assert datetime.now(UTC)  # sanity: чистые функции без побочных эффектов


def test_app_ui_sources_avoid_xi_jargon():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "fplcopilot" / "app"
    files = [
        root / "Home.py",
        root / "common.py",
        root / "format.py",
        *sorted((root / "views").glob("*.py")),
    ]
    blob = "\n".join(p.read_text(encoding="utf-8") for p in files)
    for phrase in (
        "Лучшие 11",
        "текущие 11",
        "Текущие 11",
        "лучших 11",
        "best 11",
        "Best XI",
        "current XI",
        "11-ка",
        "к оптимуму",
        "XI xPts",
        "Прогноз лучших",
        '"XI"',
        "'XI'",
    ):
        assert phrase not in blob, phrase


def test_double_transfer_facts_stay_per_player():
    from fplcopilot.core.why_facts import WhyFact
    from fplcopilot.core.why_narrate import fallback_why, validate_why_text

    r = route(1, ["Raya", "Calafiori"], ["Groß", "Saka"], hit=0, verdict="go")
    r = r.model_copy(
        update={
            "out_ids": [1, 2],
            "in_ids": [3, 4],
            "xi_points_after": 56.0,
            "new_bank": 1.2,
        }
    )
    calendars = {
        "Raya": [fmt.WhyMatch("Лидс", True)],
        "Calafiori": [fmt.WhyMatch("Лидс", True), fmt.WhyMatch("Ноттингем", False)],
        "Groß": [fmt.WhyMatch("Сандерленд", True)],
        "Saka": [fmt.WhyMatch("Вест Хэм", True)],
    }
    facts = {
        "Raya": [WhyFact("form_repeatable", "6 из 12 — сейвы", "sell", 30, 1, 12)],
        "Calafiori": [
            WhyFact("form_repeatable", "4 из 9 — сухие матчи", "sell", 30, 2, 9)
        ],
        "Groß": [
            WhyFact(
                "form_lucky",
                "20 очков за 3 тура во многом везение — такого снова не ждите",
                "buy",
                8,
                3,
                20,
            ),
            WhyFact(
                "penalty_outlook",
                "Сандерленд отдал пенальти в прошлом туре, а у Брайтона их бьёт Groß",
                "buy",
                26,
            ),
        ],
        "Saka": [
            WhyFact(
                "form_unlucky",
                "моменты были, очков мало (12 за 3 тура)",
                "buy",
                9,
                4,
                12,
            )
        ],
    }
    payload = fmt.compose_why_payload(r, horizon=3, calendars=calendars, facts=facts)
    assert len(payload["pairs"]) == 2
    p0, p1 = payload["pairs"]
    assert p0["sell"]["name"] == "Raya" and p0["buy"]["name"] == "Groß"
    assert p1["sell"]["name"] == "Calafiori" and p1["buy"]["name"] == "Saka"
    assert p0["buy"]["lucky_points"] == 20 and p0["buy"]["points"] == 20
    assert p1["buy"]["lucky_points"] is None and p1["buy"]["points"] == 12
    assert p0["buy"]["opponents"] == ["Сандерленд"]
    assert p1["buy"]["opponents"] == ["Вест Хэм"]
    assert "Сандерленд" in (p0["buy"].get("penalty") or "")
    assert not p1["buy"].get("penalty")
    text = fallback_why(payload)
    assert "32" not in text
    assert "20" in text and "Groß" in text
    assert "Saka" in text
    glued = (
        "Calafiori и Groß набрали 32 очка на везении, но впереди Лидс. "
        "Состав — 56.0 за 3 тура, в банке 1.2."
    )
    ok, reason = validate_why_text(glued, payload)
    assert ok is False
    assert reason.startswith(("number:32", "points:", "summed")) or "summed" in reason
    saka_wrong = (
        "Saka набрал 20 очков во многом на везении. "
        "Состав — 56.0 за 3 тура, в банке 1.2."
    )
    ok, reason = validate_why_text(saka_wrong, payload)
    assert ok is False
    assert reason == "points:Saka:20" or "summed" in reason or reason.startswith("points:")
    ok, reason = validate_why_text(text, payload)
    assert ok, reason


def test_buyer_interesting_stat_not_calendar_only():
    from fplcopilot.core.why_facts import WhyFact

    r = route(1, ["Ajayi"], ["Calafiori"], hit=0, verdict="go")
    r = r.model_copy(update={"xi_points_after": 55.66, "new_bank": 1.4})
    calendars = {
        "Ajayi": [fmt.WhyMatch("Сити", True)],
        "Calafiori": [
            fmt.WhyMatch("Лидс", True),
            fmt.WhyMatch("Ноттингем", False),
            fmt.WhyMatch("Эвертон", True),
        ],
    }
    facts = {
        "Ajayi": [],
        "Calafiori": [
            WhyFact("form_repeatable", "4 из 9 — сухие матчи", "buy", 11, 10, 9)
        ],
    }
    payload = fmt.compose_why_payload(r, horizon=3, calendars=calendars, facts=facts)
    assert payload["buy"]["stance"] == "repeatable"
    assert payload["buy"]["interesting"]
    assert "4" in payload["buy"]["interesting"]
    why = fmt.route_why(r, horizon=3, calendars=calendars, facts=facts)
    assert "4" in why and "сух" in why
    assert "Лидс" in why or "календар" in why.lower() or "Дальше" in why
    from fplcopilot.core.why_narrate import validate_why_text

    ok, reason = validate_why_text(
        "Calafiori сыграет с Лидсом, Ноттингемом и Эвертоном. "
        "Состав — 55.7 за 3 тура, в банке 1.4.",
        payload,
    )
    assert ok is False and reason.startswith("interesting:")


def test_buy_status_must_appear_in_why_text():
    from fplcopilot.core.why_facts import WhyFact
    from fplcopilot.core.why_narrate import fallback_why, validate_why_text

    r = route(1, ["Gakpo"], ["Palmer"], hit=0, verdict="go")
    r = r.model_copy(update={"xi_points_after": 57.7, "new_bank": 0.5})
    calendars = {
        "Gakpo": [fmt.WhyMatch("Сити", True)],
        "Palmer": [fmt.WhyMatch("Борнмут", True)],
    }
    facts = {
        "Gakpo": [],
        "Palmer": [WhyFact("status", "под вопросом — шанс сыграть 75%", "any", 5)],
    }
    payload = fmt.compose_why_payload(r, horizon=3, calendars=calendars, facts=facts)
    assert payload["buy"]["status"]
    assert "под вопросом" in payload["buy"]["status"]
    assert "75%" in payload["buy"]["status"]
    silent = (
        "Palmer сыграет с Борнмутом дома. "
        "После замены ваш состав набирает 57.7 очков за 3 тура, в банке останется 0.5 миллиона."
    )
    ok, reason = validate_why_text(silent, payload)
    assert ok is False and reason.startswith("buy_status")
    text = fallback_why(payload)
    assert "под вопросом" in text and "75%" in text
    ok, reason = validate_why_text(text, payload)
    assert ok, reason


def test_ownership_does_not_replace_interesting_stat():
    from fplcopilot.core.why_narrate import validate_why_text

    payload = {
        "sell": {"name": "Gakpo", "names": ["Gakpo"], "stance": "weak_form", "form_ok": False},
        "buy": {
            "name": "Palmer",
            "names": ["Palmer"],
            "stance": "neutral",
            "calendar": "с Борнмутом дома",
            "opponents": ["Борнмут"],
            "interesting": None,
            "ownership": 26,
        },
        "after": {"xi_points": "57.7", "bank": "0.5", "horizon": 3, "hit": 0},
    }
    ok, reason = validate_why_text(
        "Palmer сыграет с Борнмутом дома, его выбрали 26%. "
        "Состав — 57.7 за 3 тура, в банке 0.5.",
        payload,
    )
    assert ok is False and reason == "ownership_as_stat"


def test_template_ru_translates_code_templates_and_keeps_model_text():
    from fplcopilot.rag.extract import ABSTAIN_SUMMARY
    from fplcopilot.rag.team_news import NO_NEWS_SUMMARY

    assert fmt.template_ru(NO_NEWS_SUMMARY) == "Новостей о клубе не найдено."
    assert fmt.template_ru(ABSTAIN_SUMMARY) == "Новостей именно об игроке не найдено."
    assert (
        fmt.template_ru(f"{ABSTAIN_SUMMARY} FPL API status d (doubtful) is the only signal.")
        == "Новостей именно об игроке не найдено; есть только статус FPL: под вопросом."
    )
    assert fmt.template_ru(
        "No evidence about Martin Ødegaard in the retrieved documents; "
        "FPL API status a (available) is the only signal."
    ) == ("В найденных статьях нет сведений о Martin Ødegaard; есть только статус FPL: доступен.")
    assert fmt.template_ru("FPL status i is the only signal.") == (
        "Есть только статус FPL: травма."
    )
    assert fmt.template_ru("no saved news signal (cached_only)") == ""
    assert fmt.template_ru("stale digest (30.1 h > 12 h), not refreshed") == (
        "дайджест старше срока (30.1 ч > 12 ч), не обновлялся"
    )
    assert fmt.template_ru(
        "signal extraction failed: APIConnectionError: boom (stale cached signal used)"
    ) == ("обновить разбор не удалось (APIConnectionError: boom); показан прежний сохранённый")
    model_text = "Arteta confirmed Saka trained fully on Thursday."
    assert fmt.template_ru(model_text) == model_text
    assert fmt.template_ru(None) == ""


def test_example_prompts_are_russian_and_follow_the_next_gameweek():
    ex = fmt.example_prompts(6)
    assert len(ex) == 6
    assert "Кого поставить капитаном на GW6?" in ex
    assert any("перед GW6" in e for e in ex) and any("Bench Boost в GW7" in e for e in ex)
    assert not any("GW5" in e for e in ex)
    assert all(re.search("[А-Яа-я]", e) for e in ex)
    no_ctx = fmt.example_prompts(None)
    assert len(no_ctx) == 6 and not any(re.search(r"GW\d", e) for e in no_ctx)
    assert not any("GW39" in e for e in fmt.example_prompts(38))
