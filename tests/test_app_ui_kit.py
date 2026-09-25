"""Чистые помощники нового дизайна: app/briefing.py (данные «Брифинга») и app/ui_kit.py
(HTML-компоненты: поле, карточки туров, аватары) — без Streamlit, сети и LLM."""

from __future__ import annotations

from test_agent_fakes import BS, DEADLINE, NOW, FakeTools, route

from fplcopilot.agent.tools import (
    CaptainOption,
    EvidenceItem,
    GameweekContext,
    LineupOut,
    LineupPlayer,
    PlanMoveOut,
    PlanOut,
    PlayerRisk,
    PredictPlayerInput,
    RoutesOut,
    SquadPlayerRow,
)
from fplcopilot.app import briefing, theme, ui_kit
from fplcopilot.app import format as fmt


def lp(pid, name, team, pos, xpts=5.0, fixture="vTOT (FSI 2)"):
    return LineupPlayer(
        id=pid,
        name=name,
        team=team,
        position=pos,
        price=6.0,
        xpts=xpts,
        sd=2.0,
        p_start=0.9,
        ownership=10.0,
        fixture=fixture,
    )


def cap(pid, name, team, xpts, tag="safe", fixture="vTOT (FSI 2)"):
    return CaptainOption(
        id=pid,
        name=name,
        team=team,
        xpts=xpts,
        captain_points=2 * xpts,
        sd=2.0,
        ownership=40.0,
        tag=tag,
        fixture=fixture,
        p_start=0.93,
    )


def lineup() -> LineupOut:
    return LineupOut(
        gw=5,
        formation="3-4-3",
        starters=[
            lp(1, "Raya", "ARS", "GKP"),
            lp(4, "Gabriel", "ARS", "DEF"),
            lp(388, "Guéhi", "MCI", "DEF"),
            lp(154, "Palmer", "CHE", "MID", 6.2),
            lp(411, "Haaland", "MCI", "FWD", 7.1, "@LIV (FSI 5) vTOT (FSI 2)"),
        ],
        bench=[lp(165, "João Pedro", "CHE", "FWD", 3.3)],
        captain="Haaland",
        vice="Palmer",
        expected_points=56.14,
        captain_options=[
            cap(154, "Palmer", "CHE", 6.2, "balanced"),
            cap(411, "Haaland", "MCI", 7.1),
            cap(12, "Saka", "ARS", 5.0, "differential"),
        ],
        current_xi_points=52.0,
    )


def squad_row(pid, name, team, pos, *, start=True, captain=False):
    return SquadPlayerRow(
        id=pid,
        name=name,
        team=team,
        position=pos,
        price=6.0,
        is_starting=start,
        is_captain=captain,
    )


def context() -> GameweekContext:
    return GameweekContext(
        gw=5,
        current_gw=4,
        deadline=DEADLINE,
        as_of=NOW,
        squad=[
            squad_row(1, "Raya", "ARS", "GKP"),
            squad_row(4, "Gabriel", "ARS", "DEF"),
            squad_row(154, "Palmer", "CHE", "MID"),
            squad_row(411, "Haaland", "MCI", "FWD", captain=True),
            squad_row(165, "João Pedro", "CHE", "FWD"),
            squad_row(388, "Guéhi", "MCI", "DEF", start=False),
        ],
    )


# ---------- briefing ----------


def test_greeting_by_hour_and_name():
    assert briefing.greeting("Jasper", 8) == "Доброе утро, Jasper."
    assert briefing.greeting("Jasper", 14) == "Добрый день, Jasper."
    assert briefing.greeting(" ", 20) == "Добрый вечер."
    assert briefing.greeting(None, 2) == "Доброй ночи."


def test_rank_share_uses_total_players_and_skips_missing():
    assert briefing.rank_share(1_974_000, 10_970_599) == "топ 18 %"
    assert briefing.rank_share(250_000, 10_000_000) == "топ 2.5 %"
    assert briefing.rank_share(1, 10_970_599) == "топ 0.01 %"
    assert briefing.rank_share(None, 10_000_000) is None
    assert briefing.rank_share(5, None) is None


def test_recommended_route_sides_and_pills_are_real_data():
    r = route(1, ["Palmer"], ["Saka"])
    rt = RoutesOut(
        gw=5,
        horizon=3,
        strategy="balanced",
        free_transfers=1,
        bank=0.5,
        routes=[r],
        recommended_rank=1,
        recommendation="route 1",
        baseline_xi_points=50.0,
    )
    assert briefing.recommended_route(rt) is r
    assert briefing.recommended_route(rt.model_copy(update={"recommended_rank": None})) is None
    outs, ins = briefing.route_sides(r, BS)  # фейк route(): out_ids=[154], in_ids=[301]
    assert outs == [("Palmer", "CHE", 9.7)] and ins == [("Saka", "IPS", 4.5)]
    preds = {301: FakeTools().predict_player(PredictPlayerInput(player_id=301, gw=5, horizon=3))}
    pills_ = briefing.route_pills(r, preds)
    assert pills_[0] == ("бесплатный трансфер", "good")
    assert pills_[1] == ("+1.5 очка уже в ближайшем туре", "good")
    assert pills_[2] == ("Saka выходит в старте с шансом 93 %", "")
    hit = briefing.route_pills(r.model_copy(update={"hit_cost": 4}), {})
    assert hit[0] == ("платный: −4 очка", "warn")
    assert briefing.route_headline(r) == "Palmer → Saka"


def test_buy_risks_only_buy_side_without_template():
    r = route(1, ["Palmer"], ["Saka"]).model_copy(
        update={
            "risk_note": "out Palmer: doubtful (FPL doubtful 75%); in Saka: high variance "
            "(sd 3.9); in Guéhi: template (45% owned); in Neto: FPL status d"
        }
    )
    risks = briefing.buy_risks(r)
    assert risks == [
        "Saka: нестабильные очки, разброс около 3.9 очков",
        "Neto: под вопросом по статусу FPL",
    ]


def test_captain_pair_sentence_and_ranks():
    lu = lineup()
    pair = briefing.captain_pair(lu)
    assert [o.name for o in pair] == ["Haaland", "Palmer"]
    assert briefing.captain_sentence(lu) == (
        "Капитан — Haaland: в среднем 14.2 очка с учётом удвоения, на 1.8 больше, чем у "
        "Palmer. Вице-капитан — Palmer."
    )
    ranks = briefing.captain_ranks(lu)
    assert [r["name"] for r in ranks] == ["Haaland", "Palmer", "Saka"]
    assert ranks[0]["sub"] == "TOT (H) · шанс старта 93 %" and ranks[0]["tag"] == "надёжный"
    assert ranks[2]["tag"] == "редкий"


def test_captain_change_only_when_armband_differs():
    lu = lineup()
    assert briefing.captain_change(lu) is None  # current_captain не известен
    assert briefing.captain_change(lu.model_copy(update={"current_captain": "Haaland"})) is None
    assert briefing.captain_change(lu.model_copy(update={"current_captain": "Palmer"})) == (
        "Капитаном лучше поставить Haaland, а не Palmer: на 1.8 очка больше с учётом удвоения."
    )


def test_snapshot_time_text_local_and_fallback():
    from datetime import UTC

    assert fmt.snapshot_time_text("2026-09-23T20:49:16.227827+00:00", UTC) == "23.09 в 20:49"
    assert fmt.snapshot_time_text("not a date") == "not a date"
    assert fmt.snapshot_time_text(None) == "—"


def test_lineup_changes_and_xi_outlook():
    lu = lineup()
    ctx = context()
    start, bench = briefing.lineup_changes(lu, ctx)
    assert start == ["Guéhi"] and bench == ["João Pedro"]
    tools = FakeTools()
    preds = {
        pid: tools.predict_player(PredictPlayerInput(player_id=pid, gw=5, horizon=2))
        for pid in (4, 154, 411, 165, 388)  # у Raya (id 1) прогноза нет — пропускается
    }
    out = briefing.xi_outlook(ctx, preds, [5, 6, 99])
    assert [g for g, _ in out] == ["GW5", "GW6"]  # тур без прогноза пропущен
    starters = [p for p in ctx.squad if p.is_starting]
    exp = sum(
        preds[p.id].by_gw[0].xpts * (2 if p.is_captain else 1) for p in starters if p.id in preds
    )
    assert out[0][1] == round(exp, 1)


def test_watch_filter_rows_and_injury_ru():
    ctx = context()  # старт: Raya, Gabriel, Palmer, Haaland, João Pedro; Guéhi — запасной
    lu = lineup()  # лучший состав: Raya, Gabriel, Guéhi, Palmer, Haaland
    problems = [
        {
            "id": 165,
            "name": "João Pedro",
            "flag": "warn",
            "label": "под вопросом (75 %)",
            "news": "Knee injury - 75% chance of playing",
        },
        {
            "id": 388,
            "name": "Guéhi",
            "flag": "warn",
            "label": "риск ротации (выйдет в старте 0 %)",
            "news": "",
        },
        {"id": 999, "name": "Bench", "flag": "warn", "label": "риск ротации", "news": ""},
    ]
    kept = briefing.watch_filter(problems, ctx, lu)
    # João Pedro — в нашем старте, Guéhi — в лучшем составе; запасной вне обоих — не проблема
    assert [p["id"] for p in kept] == [165, 388]
    assert briefing.watch_filter(problems, ctx, None) == [problems[0]]
    ev = EvidenceItem(
        player_id=165,
        player="João Pedro",
        source="ffscout",
        url="https://x",
        published_at="2026-09-17T10:00:00Z",
        date="17.09",
        quote="no one is ruled out",
    )
    signals = {
        165: PlayerRisk(
            player_id=165,
            player="João Pedro",
            fpl_status="d",
            fpl_chance=75,
            evidence=[ev],
            origin="cached",
        )
    }
    rows = briefing.watch_rows(kept, signals)
    assert rows[0] == {
        "id": 165,
        "name": "João Pedro",
        "tone": "warn",
        "status": "под вопросом (75 %), колено",
        "meta": "ffscout · 17.09",
    }
    assert rows[1]["meta"] == "по статусу FPL"
    assert (
        briefing.injury_ru("Hamstring injury - Expected back 26 Sep") == "задняя поверхность бедра"
    )
    assert briefing.injury_ru("Expected back 26 Sep") is None  # «back» — не спина
    assert briefing.injury_ru("") is None


def risk_with(
    pid, availability="doubtful", rotation="unknown", days_ago=1, quote="Knock", return_gw=None
):
    from datetime import timedelta

    published = (NOW - timedelta(days=days_ago)).isoformat()
    ev = EvidenceItem(
        player_id=pid,
        player=f"P{pid}",
        source="bbc",
        url=f"https://n/{pid}",
        published_at=published,
        date="01.01",
        quote=quote,
    )
    return PlayerRisk(
        player_id=pid,
        player=f"P{pid}",
        fpl_status="a",
        fpl_chance=None,
        availability=availability,
        rotation_risk=rotation,
        evidence=[ev],
        origin="cached",
        return_gw=return_gw,
    )


def test_is_important_and_news_feed():
    assert briefing.is_important(risk_with(1))
    assert briefing.is_important(risk_with(1, "fit", "medium"))
    assert briefing.is_important(risk_with(1, "fit", "low", return_gw=8))
    assert not briefing.is_important(risk_with(1, "fit", "low"))
    assert not briefing.is_important(risk_with(1, "unknown"))
    no_ev = risk_with(1).model_copy(update={"evidence": []})
    assert not briefing.is_important(no_ev)
    signals = {
        1: risk_with(
            1, "injured", days_ago=2, quote="Ruled out for a month with a hamstring tear " * 6
        ),
        2: risk_with(2, "fit", "medium", days_ago=1, quote="May be rested for the cup"),
        3: risk_with(3, "doubtful", days_ago=30, quote="Old news"),  # старше 10 дней
        4: risk_with(4, "fit", "low", quote="Trained fully"),  # не важно
    }
    feed = briefing.news_feed(signals, {1: "Saka", 2: "Rice"}, NOW)
    assert [i["player"] for i in feed] == ["Rice", "Saka"]  # свежие сверху
    assert feed[0]["status"] == "риск ротации" and feed[0]["tone"] == "warn"
    assert feed[1]["status"] == "травма" and feed[1]["tone"] == "bad"
    assert len(feed[1]["quote"]) <= 150 and feed[1]["quote"].endswith("…")
    assert feed[1]["meta"] == "bbc · 01.01" and feed[1]["url"] == "https://n/1"


def test_watch_list_and_ticker_html():
    rows = [
        {
            "id": 165,
            "name": "João Pedro",
            "tone": "warn",
            "status": "под вопросом",
            "meta": "bbc · 17.09",
        }
    ]
    html = ui_kit.watch_list_html(rows, fmt.PLAYER_URL)
    assert '<a href="/player?pid=165" target="_self">João Pedro</a>' in html
    assert 'class="dot warn"' in html and "bbc · 17.09" in html
    item = {
        "id": 1,
        "player": "Saka",
        "status": "травма",
        "tone": "bad",
        "quote": "Out",
        "meta": "bbc · 01.01",
        "url": "https://n/1",
    }
    static = ui_kit.news_ticker_html([item, item], fmt.PLAYER_URL)
    assert 'class="fpl-ticker static"' in static and "dup" not in static
    moving = ui_kit.news_ticker_html([item] * 4, fmt.PLAYER_URL)
    assert "animation-duration:28s" in moving and 'class="dup" aria-hidden="true"' in moving
    assert moving.count("«Out»") == 8  # вторая копия для бесшовного цикла
    css = theme.global_css()
    assert "prefers-reduced-motion" in css and "animation-play-state: paused" in css


def test_lead_text_counts_decisions():
    lu = lineup()
    r = route(1, ["Palmer"], ["Saka"])
    assert briefing.lead_text(r, None, lu, [{"id": 1}]) == (
        "До дедлайна нужно сделать трансфер, выбрать капитана и проверить 1 игрока с проблемами."
    )
    assert briefing.lead_text(None, None, lu, [{"id": 1}, {"id": 2}]) == (
        "До дедлайна нужно выбрать капитана и проверить 2 игроков с проблемами."
    )
    assert briefing.lead_text(None, None, None, []) == "Данных для брифинга пока нет."


# ---------- ui_kit ----------


def test_initials_club_colors_and_fixture_short():
    assert ui_kit.initials("B.Fernandes") == "BF"
    assert ui_kit.initials("O'Shea") == "OS"
    assert ui_kit.initials("João Pedro") == "JP"
    assert ui_kit.initials("Haaland") == "HA"
    assert ui_kit.club_colors("mci") == ui_kit.CLUB_COLORS["MCI"]
    assert ui_kit.club_colors("XXX") == ui_kit.NEUTRAL_CLUB
    assert ui_kit.fixture_short("vTOT (FSI 2)") == "TOT (H)"
    assert ui_kit.fixture_short("@LIV (FSI 5) vTOT (FSI 2)") == "LIV (A) · TOT (H)"
    assert ui_kit.fixture_short("blank") == "—"


def test_pitch_from_lineup_rows_marks_and_links():
    starters, bench = ui_kit.pitch_from_lineup(lineup(), sell=["Palmer"], buy=["Guéhi"])
    by = {p.name: p for p in starters}
    assert by["Haaland"].role == "C" and by["Palmer"].role == "VC"
    assert by["Palmer"].tag == "sell" and by["Guéhi"].tag == "buy"
    assert by["Haaland"].sub == "LIV (A) · TOT (H)" and by["Haaland"].value == "7.1"
    html = ui_kit.pitch_html(starters, bench, player_url=fmt.PLAYER_URL)
    lines = [seg.split('"')[0] for seg in html.split('class="line ')[1:]]
    assert lines == ["fwd", "mid", "def", "gkp"]  # нападение вверху, вратарь внизу
    assert '<a class="fpl-token sell" href="/player?pid=154" target="_self"' in html
    assert '<span class="mark buy">купить</span>' in html
    assert html.count('<b class="role">') == 2 and "Скамейка" in html
    assert "XI" not in html  # страница «К дедлайну» не говорит «XI»


def test_pitch_from_squad_rows_uses_table_rows():
    xi = [
        {
            "#": 1,
            "Роль": "C",
            "Player": "Haaland",
            "xPts": 7.13,
            "_id": 411,
            "_team": "MCI",
            "_pos": "FWD",
            "GW5": "BRE (A)",
            "_flag": None,
        },
        {
            "#": 2,
            "Роль": "—",
            "Player": "João Pedro",
            "xPts": None,
            "_id": 165,
            "_team": "CHE",
            "_pos": "FWD",
            "GW5": "BRE (A)",
            "_flag": ("warn", "Под вопросом (75 %)"),
        },
    ]
    starters, bench = ui_kit.pitch_from_squad_rows(xi, [], gw=5)
    assert starters[0] == ui_kit.PitchPlayer(
        id=411, name="Haaland", club="MCI", pos="FWD", sub="BRE (A)", value="7.1", role="C"
    )
    assert starters[1].value == "" and starters[1].flag == ("warn", "Под вопросом (75 %)")
    html = ui_kit.pitch_html(starters, bench, player_url=fmt.PLAYER_URL)
    assert 'title="Под вопросом (75 %)"' in html and '<span class="flag warn">' in html
    assert "fpl-bench" not in html  # скамейки нет -> без полосы


def test_plan_gw_cards_from_plan_only():
    plan = PlanOut(
        from_gw=5,
        horizon=3,
        strategy="balanced",
        moves_by_gw={
            "5": [
                PlanMoveOut(
                    gw=5,
                    out="Konsa",
                    **{"in": "Guéhi"},
                    price_out=4.6,
                    price_in=5.1,
                    delta_xpts_horizon=13.45,
                    out_problem="doubtful",
                    paid=False,
                ),
                PlanMoveOut(
                    gw=5,
                    out="João Pedro",
                    **{"in": "Barry"},
                    price_out=7.8,
                    price_in=5.6,
                    delta_xpts_horizon=2.49,
                    paid=True,
                ),
            ],
            "6": [],
        },
        hits_by_gw={"5": 4, "6": 0, "7": 0},
        ft_by_gw={"5": 1, "6": 1, "7": 2},
        bank_by_gw={"5": 0.3, "6": 0.3, "7": 0.3},
        xi_points_by_gw={"5": 60.1, "6": 58.2, "7": 61.0},
        captain_by_gw={"5": "Haaland", "6": "Saka", "7": "Haaland"},
        expected_total=179.3,
        baseline_total=170.0,
        recommendation="transfers",
    )
    html = ui_kit.plan_gw_cards(
        plan, deadlines={5: DEADLINE}, reason_ru=fmt.ISSUE_KIND_RU, chips={"6": "bboost"}
    )
    assert html.count("<article") == 3
    assert html.index("GW5") < html.index("GW6") < html.index("GW7")
    assert '<article class="fpl-gw current">' in html and html.count(">сейчас<") == 1
    assert DEADLINE.strftime("%d.%m") in html
    assert "Konsa → Guéhi" in html and "+13.4 очка</span> за 3 тура" in html
    assert fmt.ISSUE_KIND_RU["doubtful"] in html and "сильнее по прогнозу" in html
    assert html.count('<b class="chip">Bench Boost</b>') == 1  # бейдж чипа только в GW6
    assert html.index("Bench Boost") > html.index("GW6") > html.index("GW5")
    assert html.count('<span class="hit">−4</span>') == 1 and "хит −4" in html
    assert html.count("Без трансфера") == 2  # GW6 (пусто) и GW7 (нет ключа)
    assert "Haaland" in html and "£0.3m" in html and "FT 2" in html
    for banned in ("On track", "Flexibility", "8 GW"):
        assert banned not in html


def test_components_escape_and_css_is_injected_once():
    html = ui_kit.news_item("<b>x</b>", "a & b", "src · 01.01", tone="warn")
    assert "&lt;b&gt;x&lt;/b&gt;" in html and "a &amp; b" in html and 'class="ico warn"' in html
    assert ui_kit.card("x", cls="accent") == '<div class="fpl-card accent">x</div>'
    assert ui_kit.card("x") == '<div class="fpl-card">x</div>'
    strip = ui_kit.summary_strip([("План", "307.5", "+24.4")])
    assert "<small>План</small><strong>307.5</strong><span>+24.4</span>" in strip
    assert ui_kit.bars([]) == ""
    b = ui_kit.bars([("GW5", 60.0), ("GW6", 30.0)])
    assert 'style="height:100%"' in b and 'class="bar" style="height:50%"' in b
    pb = ui_kit.paired_bars([("GW5", 50.0, 55.0), ("GW6", 60.0, None)], ("без", "план"))
    assert 'class="bar base" style="height:83%"' in pb and '<b class="good">+5.0</b>' in pb
    assert 'class="bar plan none"' in pb and "без" in pb and "план" in pb
    assert ui_kit.paired_bars([], ("a", "b")) == ""
    css = theme.global_css()
    assert css.count(".fpl-pitch {") == 1 and "--fpl-mono" in css and "JetBrains Mono" in css
    assert "Manrope" in css and "Space Mono" not in css and "DM Sans" not in css
    assert "Copilot" not in css
    assert "stStatusWidgetRunningIcon" in css and "fpl-ball" in css
