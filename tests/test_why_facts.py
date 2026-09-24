"""Сборщик фактов «Почему»: только то, что есть в данных; без выдумок."""

from __future__ import annotations

import pytest

from fplcopilot.core.points_form import form_label, points_breakdown
from fplcopilot.core.why_facts import (
    UpcomingOpp,
    WhyData,
    WhyFact,
    collect_player_facts,
    pick_facts,
)
from fplcopilot.data.schemas import Bootstrap, Fixture, Player, PlayerGWHistory, Team


def _player(**kw) -> Player:
    base = {
        "id": 1,
        "web_name": "Rogers",
        "first_name": "Morgan",
        "second_name": "Rogers",
        "team": 2,
        "element_type": 3,
        "now_cost": 70,
    }
    base.update(kw)
    return Player.model_validate(base)


def _hist(pid: int, rnd: int, *, xg: float, goals: int, minutes: int, fixture: int = 0) -> PlayerGWHistory:
    return PlayerGWHistory(
        element=pid,
        fixture=fixture or (pid * 10 + rnd),
        round=rnd,
        was_home=True,
        minutes=minutes,
        goals_scored=goals,
        expected_goals=xg,
    )


def _fx(fid: int, home: int, away: int, hs: int, a_s: int) -> Fixture:
    return Fixture(
        id=fid,
        event=1,
        team_h=home,
        team_a=away,
        finished=True,
        team_h_score=hs,
        team_a_score=a_s,
    )


def _bs(*players: Player) -> Bootstrap:
    teams = [Team(id=i, name=f"T{i}", short_name=f"T{i}") for i in range(1, 21)]
    teams[0] = Team(id=1, name="Arsenal", short_name="ARS")
    teams[1] = Team(id=2, name="Aston Villa", short_name="AVL")
    teams[19] = Team(id=20, name="Bournemouth", short_name="BOU")
    return Bootstrap(
        events=[{"id": 6, "name": "GW6", "deadline_time": "2026-10-10T10:00:00Z", "is_next": True}],
        teams=teams,
        elements=list(players),
    )


def test_penalty_taker_gets_phrase():
    p = _player(penalties_order=1)
    facts = collect_player_facts(p, WhyData(bs=_bs(p)))
    kinds = [f.kind for f in facts]
    assert "penalty" in kinds
    clause = next(f.clause for f in facts if f.kind == "penalty")
    assert "пенальти" in clause
    assert "xG" not in clause


def test_last3_xg_over_goals_is_moments_not_xg_jargon():
    p = _player()
    rows = [
        _hist(1, 3, xg=0.7, goals=0, minutes=90),
        _hist(1, 4, xg=0.6, goals=0, minutes=85),
        _hist(1, 5, xg=0.5, goals=0, minutes=90),
    ]
    facts = collect_player_facts(p, WhyData(bs=_bs(p), history=rows))
    hit = next(f for f in facts if f.kind == "moments_no_goals")
    assert "моменты" in hit.clause
    assert "не забивал" in hit.clause
    assert "1.8" in hit.clause
    assert "xG" not in hit.clause
    assert "последних 3" in hit.clause


def _league_with_standout(*, leaky_id: int = 20, leaky_against: int = 3, other: int = 1, n: int = 5):
    """n матчей на команду: leaky_id пропускает leaky_against за игру, остальные — other."""
    fixtures: list[Fixture] = []
    fid = 1
    for _rnd in range(n):
        for h, a in ((1, 2), (3, 4), (5, 6), (7, 8)):
            fixtures.append(_fx(fid, h, a, other, other))
            fid += 1
        fixtures.append(_fx(fid, 9, leaky_id, leaky_against, 0))
        fid += 1
    return fixtures


def test_high_conceded_opponent_says_leaks_with_number():
    p = _player()
    fixtures = _league_with_standout()
    facts = collect_player_facts(
        p,
        WhyData(bs=_bs(p), fixtures=fixtures, gw=6),
        upcoming=[UpcomingOpp(20, "BOU", True)],
    )
    leaky = next(f for f in facts if f.kind == "leaky_opp")
    assert "Борнмут" in leaky.clause
    assert "пропускает по" in leaky.clause
    assert "3" in leaky.clause
    assert "21" not in leaky.clause
    assert "слабых" not in leaky.clause


def test_average_opponents_stay_silent():
    p = _player()
    fixtures = _league_with_standout(leaky_against=1, other=1)
    facts = collect_player_facts(
        p,
        WhyData(bs=_bs(p), fixtures=fixtures, gw=6),
        upcoming=[
            UpcomingOpp(20, "SUN", True),
            UpcomingOpp(2, "CRY", False),
            UpcomingOpp(4, "LIV", True),
        ],
    )
    assert not any(f.kind == "leaky_opp" for f in facts)


def test_tight_opp_requires_below_league_average():
    p = _player(element_type=2, web_name="Calafiori")
    fixtures = _league_with_standout(leaky_id=20, leaky_against=2, other=2)
    # team 20 (NFO) забивает 0 в этих матчах — ниже среднего
    facts = collect_player_facts(
        p,
        WhyData(bs=_bs(p), fixtures=fixtures, gw=6),
        upcoming=[UpcomingOpp(20, "NFO", True)],
    )
    tight = next(f for f in facts if f.kind == "tight_opp")
    assert "Ноттингем" in tight.clause
    assert "забил" in tight.clause
    quiet = collect_player_facts(
        p,
        WhyData(bs=_bs(p), fixtures=fixtures, gw=6),
        upcoming=[UpcomingOpp(2, "AVL", True)],
    )
    assert not any(f.kind == "tight_opp" for f in quiet)


def test_no_data_stays_silent():
    p = _player()
    facts = collect_player_facts(p, WhyData())
    assert facts == []
    empty = collect_player_facts(
        p,
        WhyData(bs=_bs(p), fixtures=[], history=[], ext=None),
        upcoming=[],
    )
    assert not any(f.kind in ("moments_no_goals", "leaky_opp", "penalty") for f in empty)


def test_does_not_invent_vs_weak_teams():
    p = _player()
    text = " ".join(f.clause for f in collect_player_facts(p, WhyData(bs=_bs(p))))
    assert "против слабых" not in text
    assert "слабых команд" not in text
    assert "vs weak" not in text.lower()


def test_pick_facts_caps_interesting():
    facts = [
        WhyFact("moments_no_goals", "создавал моменты", "buy", 10),
        WhyFact("leaky_opp", "уже пропустили 12 мячей", "buy", 18),
        WhyFact("penalty", "бьёт пенальти", "buy", 25),
        WhyFact("strategy", "его держат 10 процентов", "buy", 40),
    ]
    picked = pick_facts(facts, role="buy", interesting_limit=2)
    assert [f.kind for f in picked] == ["moments_no_goals", "leaky_opp"]
    assert "penalty" not in [f.kind for f in picked]
    assert "strategy" not in [f.kind for f in picked]


def _gw(pid: int, rnd: int, **kw) -> PlayerGWHistory:
    base = {
        "element": pid,
        "fixture": pid * 10 + rnd,
        "round": rnd,
        "was_home": True,
        "minutes": 90,
        "total_points": 2,
    }
    base.update(kw)
    return PlayerGWHistory.model_validate(base)


def test_points_breakdown_sum_equals_total_points():
    rows = [
        _gw(1, 3, goals_scored=1, bonus=2, total_points=9),  # app 2 + goal 5 + bonus 2
        _gw(1, 4, assists=1, yellow_cards=1, total_points=4),  # app 2 + assist 3 - 1
        _gw(1, 5, total_points=2),
    ]
    p = _player()
    window = points_breakdown(1, 3, history=rows, player=p)
    assert window is not None
    assert window.n_rounds == 3
    assert window.total_points == 15
    assert sum(window.categories.values()) == window.total_points
    assert window.sums_to_total
    assert window.categories["goals"] == 5
    assert window.categories["assists"] == 3
    assert window.categories["bonus"] == 2
    assert window.categories["cards"] == -1
    assert window.categories["other"] == 0


def test_points_breakdown_residual_other_keeps_sum():
    rows = [_gw(1, 5, minutes=90, total_points=10)]  # app 2, остаток 8 — нет колонки в БД
    window = points_breakdown(1, 3, history=rows, player=_player())
    assert window is not None
    assert window.categories["appearance"] == 2
    assert window.categories["other"] == 8
    assert sum(window.categories.values()) == 10


def test_form_label_luck_and_unlucky_thresholds():
    cats = {"appearance": 6, "goals": 10, "assists": 0, "clean_sheet": 0, "saves": 0, "defcon": 0}
    assert form_label(total_points=16, ga_points=10, moments_points=4.5, categories=cats) == "lucky"
    assert form_label(total_points=16, ga_points=10, moments_points=5.0, categories=cats) == "lucky"
    assert form_label(total_points=16, ga_points=10, moments_points=5.1, categories=cats) is None
    assert form_label(total_points=6, ga_points=0, moments_points=4.0, categories=cats) == "unlucky"
    assert form_label(total_points=6, ga_points=0, moments_points=3.9, categories=cats) is None


def test_form_label_repeatable_share():
    cats = {"appearance": 6, "clean_sheet": 12, "defcon": 4, "goals": 0, "saves": 0}
    assert (
        form_label(total_points=22, ga_points=0, moments_points=0.2, categories=cats)
        == "repeatable"
    )
    thin = {"appearance": 2, "clean_sheet": 0, "defcon": 0, "saves": 0, "goals": 0}
    assert form_label(total_points=2, ga_points=0, moments_points=0.0, categories=thin) is None


def test_lucky_form_fact_from_history():
    # 2 гола MID = 10, xG 0.4 → моменты 2, дельта +8 ≥ 5
    rows = [
        _gw(1, 3, goals_scored=1, expected_goals=0.2, total_points=7),
        _gw(1, 4, goals_scored=1, expected_goals=0.1, total_points=7),
        _gw(1, 5, expected_goals=0.1, total_points=2),
    ]
    p = _player()
    window = points_breakdown(1, 3, history=rows, player=p)
    assert window is not None
    assert window.label == "lucky"
    facts = collect_player_facts(p, WhyData(bs=_bs(p), history=rows))
    hit = next(f for f in facts if f.kind == "form_lucky")
    assert "везение" in hit.clause
    assert "на удаче" not in hit.clause
    assert "19" in hit.clause or "16" in hit.clause or str(window.total_points) in hit.clause
    assert "xG" not in hit.clause


def test_unlucky_form_is_argument_against_sale():
    rows = [
        _gw(1, 3, expected_goals=0.8, expected_assists=0.2, total_points=2),
        _gw(1, 4, expected_goals=0.7, expected_assists=0.2, total_points=2),
        _gw(1, 5, expected_goals=0.6, expected_assists=0.1, total_points=2),
    ]
    p = _player()
    window = points_breakdown(1, 3, history=rows, player=p)
    assert window is not None
    assert window.label == "unlucky"
    facts = collect_player_facts(p, WhyData(bs=_bs(p), history=rows))
    sell = next(f for f in facts if f.kind == "form_unlucky" and f.role == "sell")
    assert "продавать из-за очков рано" in sell.clause
    assert "моменты" in sell.clause


def test_repeatable_form_from_clean_sheets_and_defcon():
    rows = [
        _gw(
            1,
            3,
            clean_sheets=1,
            clearances_blocks_interceptions=8,
            tackles=3,
            total_points=8,
        ),
        _gw(
            1,
            4,
            clean_sheets=1,
            clearances_blocks_interceptions=7,
            tackles=3,
            total_points=8,
        ),
        _gw(
            1,
            5,
            clean_sheets=1,
            clearances_blocks_interceptions=6,
            tackles=4,
            total_points=8,
        ),
    ]
    p = _player(element_type=2, web_name="Gabriel")  # DEF: CS 4 + app 2 + defcon 2 = 8
    window = points_breakdown(1, 3, history=rows, player=p)
    assert window is not None
    assert window.categories["clean_sheet"] == 12
    assert window.categories["defcon"] == 6
    assert window.label == "repeatable"
    facts = collect_player_facts(p, WhyData(bs=_bs(p), history=rows))
    hit = next(f for f in facts if f.kind == "form_repeatable" and f.role == "buy")
    assert "сух" in hit.clause
    assert str(window.categories["clean_sheet"]) in hit.clause
    assert str(window.total_points) in hit.clause
    assert "повторяется" not in hit.clause
    assert "CS" not in hit.clause
    sell = next(f for f in facts if f.kind == "form_repeatable" and f.role == "sell")
    assert "форма нормальная" in sell.clause


def test_decent_unlabeled_form_is_ok_to_sell_for_calendar():
    rows = [
        _gw(1, 3, minutes=90, total_points=7),
        _gw(1, 4, minutes=90, total_points=7),
        _gw(1, 5, minutes=90, total_points=7),
    ]
    p = _player()
    window = points_breakdown(1, 3, history=rows, player=p)
    assert window is not None
    assert window.label is None
    assert window.total_points == 21
    facts = collect_player_facts(p, WhyData(bs=_bs(p), history=rows))
    hit = next(f for f in facts if f.kind == "form_ok_sell")
    assert "форма нормальная" in hit.clause
    assert hit.role == "sell"


def test_no_history_no_form_label():
    p = _player()
    assert points_breakdown(1, 3, history=[], player=p) is None
    facts = collect_player_facts(p, WhyData(bs=_bs(p), history=[]))
    assert not any(f.kind.startswith("form_") for f in facts)


def test_appearance_heavy_not_repeatable():
    """Palmer-кейс: 6 из 8 за выход — метка «повторяемые» не ставится."""
    cats = {
        "appearance": 6,
        "goals": 2,
        "assists": 0,
        "clean_sheet": 0,
        "saves": 0,
        "defcon": 0,
        "bonus": 0,
    }
    assert form_label(total_points=8, ga_points=2, moments_points=2.0, categories=cats) is None
    cs = {
        "appearance": 6,
        "clean_sheet": 12,
        "defcon": 4,
        "goals": 0,
        "saves": 0,
    }
    assert form_label(total_points=22, ga_points=0, moments_points=0.2, categories=cs) == "repeatable"


def test_validator_rejects_invented_number():
    from fplcopilot.core.why_narrate import fallback_why, narrate_why, validate_why_text

    payload = {
        "sell": {"names": ["Gakpo"], "calendar": "впереди Сити дома", "reason": "", "form_ok": False},
        "buy": {
            "names": ["Palmer"],
            "calendar": "с Борнмутом дома",
            "opponents": ["Борнмут"],
            "lucky_points": None,
            "repeatable": None,
            "moments": None,
            "leaky": None,
            "penalty": "примерно 35%, что у Челси будет пенальти за эти 3 тура, бить будет Palmer",
            "status": None,
            "ownership": 26,
        },
        "after": {"xi_points": "57.3", "bank": "0.5", "horizon": 3, "hit": 0},
        "variant": 1,
    }
    ok, reason = validate_why_text(
        "Palmer берём, потому что наберёт 99 очков, а банк 0.5.", payload
    )
    assert ok is False
    assert reason.startswith("number:99")
    text, src = narrate_why(
        payload,
        llm=lambda _p: "Palmer берём, потому что наберёт 99 очков.",
        use_disk=False,
    )
    assert src == "fallback"
    assert "99" not in text
    text2, src2 = narrate_why(
        payload,
        llm=lambda _p: (
            "Palmer берём под календарь с Борнмутом дома. "
            "Есть вероятность пенальти у Челси, бить будет Palmer. "
            "После замены ваш состав набирает 57.3 очков за 3 тура, в банке останется 0.5 миллиона."
        ),
        use_disk=False,
    )
    assert src2 == "fallback"
    assert "15%" in text2 or "35%" in text2
    assert fallback_why(payload)
    for stamp in (
        "это повторяется",
        "на удаче",
        "спокойный выбор для осторожной стратегии",
    ):
        assert stamp not in fallback_why(payload)


def test_fallback_why_has_no_banned_stamps():
    from fplcopilot.core.why_narrate import STAMPS, fallback_why, has_stamp

    payload = {
        "sell": {
            "names": ["Maitland-Niles"],
            "calendar": "впереди Арсенал в гостях",
            "reason": "",
            "form_ok": True,
        },
        "buy": {
            "names": ["Rogers"],
            "calendar": "с Борнмутом дома, Эвертоном и Тоттенхэмом",
            "opponents": ["Борнмут", "Эвертон", "Тоттенхэм"],
            "lucky_points": 16,
            "repeatable": None,
            "moments": None,
            "leaky": None,
            "penalty": None,
            "status": None,
            "ownership": 12,
        },
        "after": {"xi_points": "57.3", "bank": "2.5", "horizon": 3, "hit": 0},
        "variant": 0,
    }
    text = fallback_why(payload)
    assert has_stamp(text) is False
    for s in STAMPS:
        if s != "набрал ":
            assert s not in text.lower()
    assert "16 очков" in text and "везение" in text
    assert "форма нормальная" in text
    assert "набрал " not in text


def test_fallback_why_names_weak_form_seller_in_every_variant():
    from fplcopilot.core.why_narrate import fallback_why

    base = {
        "sell": {
            "names": ["Palmer"],
            "calendar": "впереди Борнмут дома, затем Эвертон и Тоттенхэм",
            "stance": "weak_form",
            "detail": "под вопросом — шанс сыграть 75%",
            "form_ok": False,
        },
        "buy": {
            "names": ["Groß"],
            "calendar": "Сандерленд, Кристал Пэлас и Ливерпуль",
            "opponents": ["Сандерленд", "Кристал Пэлас", "Ливерпуль"],
            "lucky_points": 32,
            "stance": "lucky_but_fixtures",
        },
        "after": {"xi_points": "62.7", "bank": "0.0", "horizon": 3, "hit": 0},
    }
    for variant in range(3):
        text = fallback_why({**base, "variant": variant})
        assert "Palmer" in text.split("под вопросом")[0], (variant, text)
        assert "32 очка Groß" in text and "32 очков" not in text, (variant, text)


def test_penalty_outlook_shrink_and_grows_with_matches():
    from fplcopilot.core.penalty_outlook import (
        book_from_teams,
        match_lambda,
        penalty_outlook,
        shrink_rate,
    )
    from fplcopilot.core.understat import UnderstatTeam
    from fplcopilot.core.why_facts import UpcomingOpp, enrich_route_penalty

    def ut(title: str, matches: int, won: int, conc: int, last: tuple[tuple[str, int, int], ...] = ()):
        return UnderstatTeam(
            understat_id=title,
            title=title,
            matches=matches,
            xg=1.0,
            xga=1.0,
            pens_won=won,
            pens_conceded=conc,
            match_pens=last,
        )

    by_team = {
        1: ut("ARS", 5, 2, 0),
        2: ut("AVL", 5, 1, 1),
        3: ut("BOU", 5, 0, 1),
        4: ut("BRE", 5, 1, 0),
        5: ut("BHA", 5, 0, 1),
        6: ut("CHE", 5, 0, 0),
        7: ut("CRY", 5, 1, 1),
        8: ut("EVE", 5, 0, 2, (("2026-09-01", 0, 1), ("2026-09-14", 0, 1))),
        9: ut("FUL", 5, 1, 0),
        10: ut("LIV", 5, 1, 1),
        11: ut("MCI", 5, 2, 0),
        12: ut("MUN", 5, 0, 0),
        13: ut("NEW", 5, 0, 1),
        14: ut("NFO", 5, 0, 0),
        15: ut("SUN", 5, 0, 0),
        16: ut("TOT", 5, 0, 1),
        17: ut("WHU", 5, 0, 0),
        18: ut("WOL", 5, 0, 0),
        19: ut("LEE", 5, 0, 0),
        20: ut("BUR", 5, 0, 0),
    }
    book = book_from_teams(by_team)
    assert book.league_won == book.league_conceded
    assert book.league_won == 9
    mu = book.league_avg
    assert mu == pytest.approx(9 / 100)
    # сжатие к среднему: 0 из 5 ближе к μ, чем сырой 0
    raw = 0 / 5
    shrunk = shrink_rate(0, 5, mu, 10)
    assert raw < shrunk < mu
    one = [UpcomingOpp(3, "BOU", True)]
    three = [
        UpcomingOpp(3, "BOU", True),
        UpcomingOpp(8, "EVE", False),
        UpcomingOpp(16, "TOT", True),
    ]
    p1 = penalty_outlook(6, one, book, taker_name="Palmer")
    p3 = penalty_outlook(6, three, book, taker_name="Palmer")
    assert p1 is not None and p3 is not None
    assert p3.p_any > p1.p_any
    # λ растёт с числом матчей
    assert sum(p3.lambdas) > sum(p1.lambdas)
    assert match_lambda(mu, mu, mu) == pytest.approx(mu)

    buyer = _player(id=154, web_name="Palmer", team=6, penalties_order=1)
    seller = _player(id=80, web_name="Gakpo", team=10, penalties_order=3)
    both = _player(id=81, web_name="Salah", team=10, penalties_order=1)
    data = WhyData(bs=_bs(buyer, seller, both), ext=type("E", (), {"by_team": by_team})(), gw=6)
    data.upcoming = {154: three}
    hit = enrich_route_penalty([seller], [buyer], data)
    assert hit is not None and hit.kind == "penalty_outlook"
    assert "Palmer" in hit.clause
    none = enrich_route_penalty([both], [buyer], data)
    assert none is None
    none2 = enrich_route_penalty([seller], [seller], data)
    assert none2 is None


def test_validator_bans_water_long_text_and_cyrillic_names():
    from fplcopilot.core.why_narrate import narrate_why, sentence_count, validate_why_text

    payload = {
        "sell": {
            "name": "Gakpo",
            "names": ["Gakpo"],
            "stance": "weak_form",
            "calendar": "впереди Сити дома",
            "detail": "почти не создавал моменты",
            "form_ok": False,
            "reason": "почти не создавал моменты",
        },
        "buy": {
            "name": "Rogers",
            "names": ["Rogers"],
            "stance": "lucky_but_fixtures",
            "calendar": "с Борнмутом дома",
            "opponents": ["Борнмут"],
            "lucky_points": 16,
            "penalty": None,
        },
        "after": {"xi_points": "57.6", "bank": "2.5", "horizon": 3, "hit": 0},
        "variant": 0,
    }
    water = (
        "Gakpo выглядит интересным вариантом. Rogers может быть интересным вариантом. "
        "Состав — 57.6 за 3 тура, в банке 2.5."
    )
    ok, reason = validate_why_text(water, payload)
    assert ok is False and reason == "stamp"
    long = (
        "Gakpo почти не создавал моменты. Впереди Сити дома. "
        "Rogers сыграет с Борнмутом дома. Ещё одна лишняя фраза. "
        "После замены ваш состав набирает 57.6 очков за 3 тура, в банке останется 2.5 миллиона."
    )
    assert sentence_count(long) == 5
    ok, reason = validate_why_text(long, payload)
    assert ok is False and reason.startswith("sentences")
    cyr = (
        "Гакпо почти не создавал моменты, впереди Сити дома. "
        "Rogers — 16 очков везение, берём за Борнмут. "
        "Состав — 57.6 за 3 тура, в банке 2.5."
    )
    ok, reason = validate_why_text(cyr, payload)
    assert ok is False and reason.startswith("cyr_name")
    calls = {"n": 0}

    def llm(_payload, reject=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return water
        return (
            "Gakpo почти не создавал моменты, впереди Сити дома. "
            "16 очков Rogers — во многом везение, берём за Борнмут. "
            "После замены ваш состав набирает 57.6 очков за 3 тура, в банке останется 2.5 миллиона."
        )

    text, src = narrate_why(payload, llm=llm, use_disk=False)
    assert src == "llm" and calls["n"] == 2
    assert "Gakpo" in text and "Гакпо" not in text
    assert "интересным" not in text


def test_compose_stances_are_exclusive():
    from test_agent_fakes import route

    from fplcopilot.app import format as fmt
    from fplcopilot.core.why_facts import WhyFact

    r = route(1, ["Gakpo"], ["Rogers"])
    facts = {
        "Gakpo": [
            WhyFact("few_chances", "почти не создавал моменты", "sell", 20),
            WhyFact("form_ok_sell", "форма нормальная", "sell", 28),
        ],
        "Rogers": [
            WhyFact("form_lucky", "16 очков за 3 тура во многом везение", "buy", 8),
            WhyFact("moments_no_goals", "создавал моменты, но не забивал", "buy", 10),
        ],
    }
    payload = fmt.compose_why_payload(r, horizon=3, facts=facts)
    assert payload["sell"]["stance"] == "weak_form"
    assert payload["sell"]["form_ok"] is False
    assert "форма нормальная" not in (payload["sell"].get("detail") or "")
    assert payload["buy"]["stance"] == "lucky_but_fixtures"
    assert payload["buy"]["moments"] is None
    assert payload["buy"]["lucky_points"] == 16


def test_penalty_clause_respects_min_p(monkeypatch):
    import fplcopilot.core.penalty_outlook as po
    from fplcopilot.core.penalty_outlook import book_from_teams, penalty_outlook
    from fplcopilot.core.understat import UnderstatTeam
    from fplcopilot.core.why_facts import UpcomingOpp

    monkeypatch.setattr(po.config.settings, "why_pen_min_p", 0.30)

    def ut(title: str, matches: int, won: int, conc: int):
        return UnderstatTeam(
            understat_id=title, title=title, matches=matches, xg=1.0, xga=1.0,
            pens_won=won, pens_conceded=conc,
        )

    book = book_from_teams({
        6: ut("CHE", 5, 0, 0),
        3: ut("BOU", 5, 0, 0),
        8: ut("EVE", 5, 0, 2),
    })
    low = penalty_outlook(6, [UpcomingOpp(3, "BOU", True)], book, taker_name="Taker")
    assert low is not None and low.kind == "below_threshold" and low.clause == ""
    stand = penalty_outlook(6, [UpcomingOpp(8, "EVE", True)], book, taker_name="Taker")
    assert stand is not None and stand.kind == "standout" and "Taker" in stand.clause
    monkeypatch.setattr(po.config.settings, "why_pen_min_p", 0.01)
    hi = penalty_outlook(6, [UpcomingOpp(3, "BOU", True)] * 3, book, taker_name="Taker")
    assert hi is not None and hi.kind == "probability" and hi.clause


def test_gk_form_label_skips_goal_luck():
    from fplcopilot.data.schemas import Position

    cats = {
        "appearance": 6,
        "saves": 4,
        "goals": 10,
        "clean_sheet": 4,
        "defcon": 0,
    }
    assert (
        form_label(
            total_points=20,
            ga_points=10,
            moments_points=0.2,
            categories=cats,
            position=Position.GKP,
        )
        != "lucky"
    )
    assert (
        form_label(
            total_points=20,
            ga_points=0,
            moments_points=8.0,
            categories=cats,
            position=Position.GKP,
        )
        != "unlucky"
    )
    cs = {"appearance": 6, "saves": 8, "clean_sheet": 8, "goals": 0, "defcon": 0}
    assert (
        form_label(
            total_points=22,
            ga_points=0,
            moments_points=0.1,
            categories=cs,
            position=Position.GKP,
        )
        == "repeatable"
    )
    rows = [
        _gw(1, 3, goals_scored=1, expected_goals=0.1, total_points=10),
        _gw(1, 4, goals_scored=1, expected_goals=0.1, total_points=10),
        _gw(1, 5, total_points=2),
    ]
    p = _player(element_type=1, web_name="Raya")
    window = points_breakdown(1, 3, history=rows, player=p)
    assert window is not None
    assert window.label != "lucky"
    facts = collect_player_facts(p, WhyData(bs=_bs(p), history=rows))
    assert not any(f.kind == "form_lucky" for f in facts)


def test_club_ru_genitive_all_twenty_plus_aliases():
    from fplcopilot.core.why_facts import TEAM_RU, TEAM_RU_GEN, club_ru, club_ru_gen

    assert set(TEAM_RU) == set(TEAM_RU_GEN)
    assert club_ru("BHA") == "Брайтон"
    assert club_ru_gen("BHA") == "Брайтона"
    assert club_ru_gen("SUN") == "Сандерленда"
    assert club_ru_gen("ARS") == "Арсенала"
    assert club_ru_gen("CHE") == "Челси"
    assert f"у {club_ru_gen('BHA')}" == "у Брайтона"


def test_penalty_last_tour_wording_and_genitive():
    from fplcopilot.core.penalty_outlook import book_from_teams, penalty_outlook
    from fplcopilot.core.understat import UnderstatTeam

    def ut(title: str, matches: int, won: int, conc: int, match_pens=()):
        return UnderstatTeam(
            understat_id=title,
            title=title,
            matches=matches,
            xg=1.0,
            xga=1.0,
            pens_won=won,
            pens_conceded=conc,
            match_pens=tuple(match_pens),
        )

    book = book_from_teams(
        {
            36: ut("BHA", 5, 0, 0),
            20: ut(
                "SUN",
                5,
                0,
                1,
                match_pens=(("2026-09-01", 0, 0), ("2026-09-14", 0, 1)),
            ),
        }
    )
    out = penalty_outlook(
        36, [UpcomingOpp(20, "SUN", True)], book, taker_name="Groß"
    )
    assert out is not None and out.clause
    assert "в прошлом туре" in out.clause
    assert "у Брайтона" in out.clause
    assert "Groß" in out.clause
    assert "отдал 1 пенальти за" not in out.clause
    assert out.opponents[0].name == "Сандерленд"


def test_buyer_status_is_human_and_not_sell_only():
    p = _player(web_name="Palmer", status="d", chance_of_playing_next_round=75)
    facts = collect_player_facts(p, WhyData(bs=_bs(p)))
    hit = next(f for f in facts if f.kind == "status")
    assert hit.role == "any"
    assert "под вопросом" in hit.clause
    assert "75%" in hit.clause
    assert "по статусу FPL" not in hit.clause
    available = _player(web_name="Saka", status="a")
    assert not any(
        f.kind == "status"
        for f in collect_player_facts(available, WhyData(bs=_bs(available)))
    )
