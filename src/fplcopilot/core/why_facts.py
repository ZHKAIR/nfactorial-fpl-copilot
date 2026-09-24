"""Детерминированные факты для текста «Почему» на карточках трансфера.

Никакого LLM: только FPL (bootstrap, фикстуры, element-summary / player_gw_history)
и кэш Understat. Нет числа в данных — нет фразы. Нет сплита «против слабых» —
его не выдумываем.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from fplcopilot.core.points_form import points_breakdown
from fplcopilot.data.schemas import Bootstrap, Fixture, Player, PlayerGWHistory, Position

log = logging.getLogger(__name__)

LAST_N = 3
MOMENTS_XG_MIN = 0.70
MOMENTS_GAP = 0.50
SEASON_XG_MIN = 1.50
SEASON_GAP = 1.00
LOW_MINUTES = 90
FEW_CHANCES_XG = 0.35
OWN_DIFF = 15.0
OWN_SAFE = 40.0
MIN_TEAMS_FOR_LEAGUE = 8

INTERESTING = frozenset(
    {
        "form_lucky",
        "form_unlucky",
        "form_repeatable",
        "form_ok_sell",
        "moments_no_goals",
        "leaky_opp",
        "penalty_outlook",
        "penalty_contrast",
        "few_chances",
        "low_minutes",
        "tight_opp",
    }
)

TEAM_RU: dict[str, str] = {
    "ARS": "Арсенал",
    "AVL": "Астон Вилла",
    "BOU": "Борнмут",
    "BRE": "Брентфорд",
    "BHA": "Брайтон",
    "BUR": "Бернли",
    "CHE": "Челси",
    "COV": "Ковентри",
    "CRY": "Кристал Пэлас",
    "EVE": "Эвертон",
    "FUL": "Фулхэм",
    "HUL": "Халл",
    "IPS": "Ипсвич",
    "LEE": "Лидс",
    "LEI": "Лестер",
    "LIV": "Ливерпуль",
    "LUT": "Лутон",
    "MCI": "Сити",
    "MUN": "Юнайтед",
    "NEW": "Ньюкасл",
    "NFO": "Ноттингем",
    "SHU": "Шеффилд Юнайтед",
    "SOU": "Саутгемптон",
    "SUN": "Сандерленд",
    "TOT": "Тоттенхэм",
    "WHU": "Вест Хэм",
    "WOL": "Вулверхэмптон",
}

# Родительный падеж: «у Брайтона», «у Сандерленда». Несклоняемые — как именительный.
TEAM_RU_GEN: dict[str, str] = {
    "ARS": "Арсенала",
    "AVL": "Астон Виллы",
    "BOU": "Борнмута",
    "BRE": "Брентфорда",
    "BHA": "Брайтона",
    "BUR": "Бернли",
    "CHE": "Челси",
    "COV": "Ковентри",
    "CRY": "Кристал Пэласа",
    "EVE": "Эвертона",
    "FUL": "Фулхэма",
    "HUL": "Халла",
    "IPS": "Ипсвича",
    "LEE": "Лидса",
    "LEI": "Лестера",
    "LIV": "Ливерпуля",
    "LUT": "Лутона",
    "MCI": "Сити",
    "MUN": "Юнайтед",
    "NEW": "Ньюкасла",
    "NFO": "Ноттингема",
    "SHU": "Шеффилд Юнайтед",
    "SOU": "Саутгемптона",
    "SUN": "Сандерленда",
    "TOT": "Тоттенхэма",
    "WHU": "Вест Хэма",
    "WOL": "Вулверхэмптона",
}

STATUS_CLAUSE: dict[str, str] = {
    "d": "под вопросом",
    "i": "травма",
    "s": "дисквалификация",
    "u": "недоступен",
    "n": "не в заявке",
}


@dataclass(frozen=True)
class WhyFact:
    """Одна опора для склейки «Почему»: готовая клауза без имени игрока и без точки."""

    kind: str
    clause: str
    role: str  # buy | sell | any
    priority: int = 50
    player_id: int | None = None
    points: int | None = None


@dataclass(frozen=True)
class UpcomingOpp:
    team_id: int
    short_name: str
    is_home: bool = True


@dataclass
class WhyData:
    """Всё, что можно честно показать. Пустые коллекции = молчим, не врём."""

    bs: Bootstrap | None = None
    fixtures: Sequence[Fixture] = field(default_factory=tuple)
    history: Sequence[PlayerGWHistory] = field(default_factory=tuple)
    ext: Any = None
    gw: int = 1
    strategy: str = "balanced"
    horizon: int = 3
    upcoming: Mapping[int, Sequence[UpcomingOpp]] | None = None


def club_ru(short_name: str | None) -> str:
    if not short_name:
        return "соперник"
    key = short_name.strip()
    return TEAM_RU.get(key.upper(), key)


def club_ru_gen(short_name: str | None) -> str:
    """Родительный: «Брайтон» → «Брайтона». Неизвестный клуб — как именительный."""
    if not short_name:
        return "соперника"
    key = short_name.strip()
    return TEAM_RU_GEN.get(key.upper(), club_ru(key))


def _plural(n: int, one: str, few: str, many: str) -> str:
    tail, tail2 = abs(n) % 10, abs(n) % 100
    if tail == 1 and tail2 != 11:
        word = one
    elif 2 <= tail <= 4 and not 12 <= tail2 <= 14:
        word = few
    else:
        word = many
    return f"{n} {word}"


def _xg_num(x: float) -> str:
    if abs(x - round(x)) < 0.05:
        return str(round(x))
    return f"{x:.1f}"


def _pct(n: float) -> str:
    return _plural(round(float(n)), "процент", "процента", "процентов")


def _goals_word(n: int) -> str:
    return _plural(n, "мяч", "мяча", "мячей")


def last_n_round_rows(
    rows: Sequence[PlayerGWHistory], n: int = LAST_N
) -> list[PlayerGWHistory]:
    if not rows:
        return []
    rounds = sorted({int(r.round) for r in rows})
    keep = set(rounds[-n:])
    return [r for r in rows if int(r.round) in keep]


def team_goals_conceded(fixtures: Sequence[Fixture]) -> dict[int, int]:
    """Сумма пропущенных по сыгранным матчам. Нет счёта — команда не попадает в словарь."""
    out: dict[int, int] = {}
    for f in fixtures:
        if not f.finished or f.team_h_score is None or f.team_a_score is None:
            continue
        out[f.team_h] = out.get(f.team_h, 0) + int(f.team_a_score)
        out[f.team_a] = out.get(f.team_a, 0) + int(f.team_h_score)
    return out


def leaky_cutoff(conceded: Mapping[int, int]) -> int | None:
    """Нижняя граница «много пропускает»: не ниже p75 и не ниже медианы+2. Иначе None."""
    vals = sorted(int(v) for v in conceded.values())
    if len(vals) < MIN_TEAMS_FOR_LEAGUE:
        return None
    p75 = vals[int(len(vals) * 0.75)]
    median = vals[len(vals) // 2]
    return max(p75, median + 2)


def upcoming_opponents(
    player: Player,
    fixtures: Sequence[Fixture],
    *,
    gw: int,
    horizon: int,
    bs: Bootstrap | None,
) -> list[UpcomingOpp]:
    rows: list[tuple[int, Any, UpcomingOpp]] = []
    for f in fixtures:
        if f.event is None or f.finished:
            continue
        if f.event < gw:
            continue
        if f.team_h != player.team and f.team_a != player.team:
            continue
        home = f.team_h == player.team
        opp_id = f.team_a if home else f.team_h
        short = "?"
        if bs is not None:
            try:
                short = bs.team(opp_id).short_name
            except (KeyError, AttributeError):
                short = "?"
        rows.append((int(f.event), f.kickoff_time, UpcomingOpp(opp_id, short, home)))
    rows.sort(key=lambda r: (r[0], r[1] is None, r[1] or 0))
    return [r[2] for r in rows[: max(1, horizon)]]


def _player_history(player_id: int, rows: Sequence[PlayerGWHistory]) -> list[PlayerGWHistory]:
    return [r for r in rows if int(r.element) == int(player_id)]


def _xga_per_game(ext: Any, team_id: int) -> float | None:
    if ext is None:
        return None
    row = ext.team_row(team_id) if hasattr(ext, "team_row") else None
    if row is None:
        return None
    return row.xga_per_game


def _league_pg(ext: Any, attr: str) -> list[float]:
    if ext is None or not hasattr(ext, "by_team"):
        return []
    return [
        float(v)
        for t in ext.by_team.values()
        if (v := getattr(t, attr, None)) is not None
    ]


def _xga_leaky(ext: Any, team_id: int) -> bool:
    vals = _league_pg(ext, "xga_per_game")
    if len(vals) < MIN_TEAMS_FOR_LEAGUE:
        return False
    vals_s = sorted(vals)
    p75 = vals_s[int(len(vals_s) * 0.75)]
    median = vals_s[len(vals_s) // 2]
    mine = _xga_per_game(ext, team_id)
    if mine is None:
        return False
    return mine >= p75 and mine > median * 1.10


def _xg_per_game(ext: Any, team_id: int) -> float | None:
    if ext is None:
        return None
    row = ext.team_row(team_id) if hasattr(ext, "team_row") else None
    if row is None:
        return None
    return getattr(row, "xg_per_game", None)


def _xg_tight(ext: Any, team_id: int) -> bool:
    vals = _league_pg(ext, "xg_per_game")
    if len(vals) < MIN_TEAMS_FOR_LEAGUE:
        return False
    vals_s = sorted(vals)
    p25 = vals_s[int(len(vals_s) * 0.25)]
    median = vals_s[len(vals_s) // 2]
    mine = _xg_per_game(ext, team_id)
    if mine is None:
        return False
    return mine <= p25 and mine < median * 0.90


def _moments_from_window(rows: Sequence[PlayerGWHistory], player: Player) -> WhyFact | None:
    window = last_n_round_rows(rows, LAST_N)
    if not window:
        return None
    xg = sum(float(r.expected_goals or 0.0) for r in window)
    goals = sum(int(r.goals_scored) for r in window)
    mins = sum(int(r.minutes) for r in window)
    n_rounds = len({int(r.round) for r in window})
    tours = _plural(n_rounds, "туре", "турах", "турах").split(" ", 1)[1]
    if mins >= 60 and xg >= MOMENTS_XG_MIN and goals <= xg - MOMENTS_GAP:
        xg_s = _xg_num(xg)
        if goals == 0:
            clause = (
                f"в последних {n_rounds} {tours} создавал моменты "
                f"({xg_s} ожидаемых голов), но не забивал"
            )
        else:
            scored = _plural(goals, "гол", "гола", "голов")
            clause = (
                f"в последних {n_rounds} {tours} {xg_s} ожидаемых голов "
                f"и только {scored}"
            )
        return WhyFact("moments_no_goals", clause, "buy", 10)
    if (
        mins >= 120
        and player.position in (Position.MID, Position.FWD)
        and xg < FEW_CHANCES_XG
    ):
        return WhyFact(
            "few_chances",
            f"в последних {n_rounds} {tours} почти не создавал моменты",
            "sell",
            20,
        )
    if n_rounds >= 2 and mins < LOW_MINUTES:
        return WhyFact(
            "low_minutes",
            f"в последних {n_rounds} {tours} мало минут ({mins})",
            "sell",
            15,
        )
    return None


def _season_moments(player: Player, ext: Any) -> WhyFact | None:
    xg = player.expected_goals
    goals = int(player.goals_scored)
    if (
        xg is not None
        and player.minutes >= 180
        and xg >= SEASON_XG_MIN
        and goals <= xg - SEASON_GAP
    ):
        return WhyFact(
            "moments_no_goals",
            f"за сезон {_xg_num(xg)} ожидаемых голов при {goals} забитых",
            "buy",
            16,
        )
    row = None
    if ext is not None and hasattr(ext, "player_row"):
        row = ext.player_row(player.id)
    if (
        row is not None
        and row.minutes >= 180
        and row.xg >= SEASON_XG_MIN
        and row.goals <= row.xg - SEASON_GAP
    ):
        return WhyFact(
            "moments_no_goals",
            f"за сезон {_xg_num(row.xg)} ожидаемых голов при {row.goals} забитых",
            "buy",
            17,
        )
    return None


def team_matches(fixtures: Sequence[Fixture]) -> dict[int, int]:
    """Сыгранные матчи с известным счётом."""
    out: dict[int, int] = {}
    for f in fixtures:
        if not f.finished or f.team_h_score is None or f.team_a_score is None:
            continue
        out[f.team_h] = out.get(f.team_h, 0) + 1
        out[f.team_a] = out.get(f.team_a, 0) + 1
    return out


def _opp_thresholds() -> tuple[float, int]:
    from fplcopilot.config import settings

    ratio = float(getattr(settings, "why_opp_ratio", 1.3))
    min_m = int(getattr(settings, "why_opp_min_matches", 5))
    return ratio, min_m


def _rates_from_totals(
    totals: Mapping[int, int], matches: Mapping[int, int]
) -> dict[int, tuple[float, int]]:
    out: dict[int, tuple[float, int]] = {}
    for tid, n in totals.items():
        m = int(matches.get(tid, 0) or 0)
        if m > 0:
            out[int(tid)] = (float(n) / m, m)
    return out


def _rates_from_ext(ext: Any, attr: str) -> dict[int, tuple[float, int]]:
    out: dict[int, tuple[float, int]] = {}
    if ext is None or not hasattr(ext, "by_team"):
        return out
    for tid, row in ext.by_team.items():
        val = getattr(row, attr, None)
        m = int(getattr(row, "matches", 0) or 0)
        if val is not None and m > 0:
            out[int(tid)] = (float(val), m)
    return out


def league_rate(
    rates: Mapping[int, tuple[float, int]], *, min_matches: int
) -> float | None:
    vals = [r for r, m in rates.values() if m >= min_matches]
    if len(vals) < MIN_TEAMS_FOR_LEAGUE:
        return None
    return sum(vals) / len(vals)


def _best_standout(
    upcoming: Sequence[UpcomingOpp],
    *,
    rates: Mapping[int, tuple[float, int]],
    avg: float | None,
    min_m: int,
    ratio: float,
    leaky: bool,
) -> tuple[UpcomingOpp, float, int] | None:
    if avg is None or avg <= 0:
        return None
    best: tuple[UpcomingOpp, float, int] | None = None
    best_score = 0.0
    for opp in upcoming[:LAST_N]:
        pair = rates.get(opp.team_id)
        if pair is None:
            continue
        rate, m = pair
        if m < min_m:
            continue
        if leaky:
            if rate + 1e-12 < ratio * avg:
                continue
            score = rate / avg
        else:
            if rate - 1e-12 > avg / ratio:
                continue
            score = (avg / rate) if rate > 0 else ratio + 1
        if best is None or score > best_score:
            best = (opp, rate, m)
            best_score = score
    return best


def _leaky_fact(
    upcoming: Sequence[UpcomingOpp],
    fixtures: Sequence[Fixture],
    ext: Any,
) -> WhyFact | None:
    """Один соперник заметно дырявее среднего по лиге. Сумму по трём не пишем."""
    ratio, min_m = _opp_thresholds()
    matches = team_matches(fixtures)
    goal_rates = _rates_from_totals(team_goals_conceded(fixtures), matches)
    avg_goals = league_rate(goal_rates, min_matches=min_m)
    hit = _best_standout(
        upcoming, rates=goal_rates, avg=avg_goals, min_m=min_m, ratio=ratio, leaky=True
    )
    if hit is not None:
        opp, rate, _m = hit
        return WhyFact(
            "leaky_opp",
            f"{club_ru(opp.short_name)} пропускает по {_xg_num(rate)} за матч",
            "buy",
            18,
        )
    xga_rates = _rates_from_ext(ext, "xga_per_game")
    avg_xga = league_rate(xga_rates, min_matches=min_m)
    hit = _best_standout(
        upcoming, rates=xga_rates, avg=avg_xga, min_m=min_m, ratio=ratio, leaky=True
    )
    if hit is not None:
        opp, rate, _m = hit
        return WhyFact(
            "leaky_opp",
            f"{club_ru(opp.short_name)} пропускает по {_xg_num(rate)} ожидаемых за матч",
            "buy",
            18,
        )
    return None


def _tight_fact(
    upcoming: Sequence[UpcomingOpp],
    fixtures: Sequence[Fixture],
    ext: Any,
) -> WhyFact | None:
    """Соперник заметно слабее среднего в атаке. Только с числом и порогом."""
    ratio, min_m = _opp_thresholds()
    matches = team_matches(fixtures)
    scored = team_goals_scored(fixtures)
    goal_rates = _rates_from_totals(scored, matches)
    avg_goals = league_rate(goal_rates, min_matches=min_m)
    hit = _best_standout(
        upcoming, rates=goal_rates, avg=avg_goals, min_m=min_m, ratio=ratio, leaky=False
    )
    if hit is not None:
        opp, _rate, m = hit
        n = int(scored.get(opp.team_id, 0) or 0)
        return WhyFact(
            "tight_opp",
            f"{club_ru(opp.short_name)} забил всего {_goals_word(n)} за {m} матчей",
            "buy",
            18,
        )
    xg_rates = _rates_from_ext(ext, "xg_per_game")
    avg_xg = league_rate(xg_rates, min_matches=min_m)
    hit = _best_standout(
        upcoming, rates=xg_rates, avg=avg_xg, min_m=min_m, ratio=ratio, leaky=False
    )
    if hit is not None:
        opp, rate, _m = hit
        return WhyFact(
            "tight_opp",
            f"{club_ru(opp.short_name)} забивает по {_xg_num(rate)} ожидаемых за матч",
            "buy",
            18,
        )
    return None


def team_goals_scored(fixtures: Sequence[Fixture]) -> dict[int, int]:
    """Сумма забитых по сыгранным матчам. Нет счёта — команда не попадает в словарь."""
    out: dict[int, int] = {}
    for f in fixtures:
        if not f.finished or f.team_h_score is None or f.team_a_score is None:
            continue
        out[f.team_h] = out.get(f.team_h, 0) + int(f.team_h_score)
        out[f.team_a] = out.get(f.team_a, 0) + int(f.team_a_score)
    return out


def stingy_cutoff(scored: Mapping[int, int]) -> int | None:
    """Верхняя граница «мало забивает»: не выше p25. Иначе None."""
    vals = sorted(int(v) for v in scored.values())
    if len(vals) < MIN_TEAMS_FOR_LEAGUE:
        return None
    p25 = vals[int(len(vals) * 0.25)]
    median = vals[len(vals) // 2]
    cut = min(p25, max(0, median - 2))
    return cut if cut < median else None


def _status_fact(player: Player) -> WhyFact | None:
    """Статус покупаемого — всегда; у продаваемого — если это причина."""
    chance = player.chance_of_playing_next_round
    st = player.status
    if st == "a":
        if chance is None or chance >= 100:
            return None
        clause = f"шанс сыграть {round(float(chance))}%"
    else:
        head = STATUS_CLAUSE.get(st)
        if not head:
            return None
        if chance is not None and st in ("d", "i"):
            clause = f"{head} — шанс сыграть {round(float(chance))}%"
        else:
            clause = head
    return WhyFact("status", clause, "any", 5)


def _form_facts(player: Player, rows: Sequence[PlayerGWHistory]) -> list[WhyFact]:
    """Метка удачи / недобора / повторяемых очков. Нет метки — пусто."""
    from fplcopilot.config import settings

    last_n = int(getattr(settings, "why_form_last_n", 3))
    window = points_breakdown(player.id, last_n, history=rows, player=player)
    if window is None:
        return []
    n = window.total_points
    if window.label is None:
        if n >= 8:
            return [WhyFact("form_ok_sell", "форма нормальная", "sell", 28, player.id, n)]
        return []
    tours = _plural(window.n_rounds, "тур", "тура", "туров")
    if window.label == "lucky":
        return [
            WhyFact(
                "form_lucky",
                f"{n} очков за {tours} во многом везение — такого снова не ждите",
                "buy",
                8,
                player.id,
                n,
            ),
            WhyFact(
                "form_lucky",
                f"недавние {n} очков больше похожи на везение, чем на устойчивую форму",
                "sell",
                14,
                player.id,
                n,
            ),
        ]
    if window.label == "unlucky":
        return [
            WhyFact(
                "form_unlucky",
                f"моменты создавал, а очков мало ({n} за {tours}) — продавать из-за очков рано",
                "sell",
                8,
                player.id,
                n,
            ),
            WhyFact(
                "form_unlucky",
                f"моменты были, очков мало ({n} за {tours})",
                "buy",
                9,
                player.id,
                n,
            ),
        ]
    bits: list[str] = []
    cats = window.categories
    cs = int(cats.get("clean_sheet") or 0)
    dc = int(cats.get("defcon") or 0)
    sv = int(cats.get("saves") or 0)
    if cs:
        bits.append(f"{cs} из {n} — сухие матчи")
    if dc:
        bits.append(f"{dc} — защитные действия")
    if sv:
        bits.append(f"{sv} — сейвы")
    extra = ", ".join(bits[:2]) or "стабильных источников"
    return [
        WhyFact("form_repeatable", extra, "buy", 11, player.id, n),
        WhyFact("form_repeatable", "форма нормальная", "sell", 30, player.id, n),
    ]


def _penalty_fact(player: Player, bs: Bootstrap | None) -> WhyFact | None:
    if player.penalties_order != 1:
        return None
    short = ""
    if bs is not None:
        try:
            short = bs.team(player.team).short_name
        except (KeyError, AttributeError):
            short = ""
    club = club_ru_gen(short) if short else "команды"
    return WhyFact("penalty", f"бьёт пенальти у {club}", "buy", 25)


def _strategy_fact(player: Player, strategy: str) -> WhyFact | None:
    own = player.selected_by_percent
    if own is None:
        return None
    if strategy == "aggressive" and own < OWN_DIFF:
        return WhyFact(
            "strategy",
            f"его держат {_pct(own)} менеджеров — подходит под агрессивную стратегию",
            "buy",
            40,
        )
    if strategy == "conservative" and own >= OWN_SAFE:
        return WhyFact(
            "strategy",
            f"его держат {_pct(own)} менеджеров — спокойный выбор для осторожной стратегии",
            "buy",
            45,
        )
    return None


def collect_player_facts(
    player: Player,
    data: WhyData,
    *,
    upcoming: Sequence[UpcomingOpp] | None = None,
) -> list[WhyFact]:
    """Факты по одному игроку. Пустой список, если сказать нечего."""
    facts: list[WhyFact] = []
    rows = _player_history(player.id, data.history)
    try:
        facts.extend(_form_facts(player, rows))
    except Exception:
        log.debug("why_facts: form window unavailable", exc_info=True)
    window = _moments_from_window(rows, player)
    if window is not None:
        facts.append(window)
    elif _season_moments(player, data.ext) is not None:
        facts.append(_season_moments(player, data.ext))  # type: ignore[arg-type]

    pen = _penalty_fact(player, data.bs)
    if pen is not None:
        facts.append(pen)

    status = _status_fact(player)
    if status is not None:
        facts.append(status)

    opps = list(upcoming) if upcoming is not None else []
    if not opps and data.upcoming is not None:
        opps = list(data.upcoming.get(player.id, ()))
    if not opps and data.bs is not None and data.fixtures:
        opps = upcoming_opponents(
            player, data.fixtures, gw=data.gw, horizon=data.horizon, bs=data.bs
        )
    pos = player.position
    if pos in (Position.FWD, Position.MID):
        leaky = _leaky_fact(opps, data.fixtures, data.ext)
        if leaky is not None:
            facts.append(leaky)
    elif pos in (Position.DEF, Position.GKP):
        tight = _tight_fact(opps, data.fixtures, data.ext)
        if tight is not None:
            facts.append(tight)

    strat = _strategy_fact(player, data.strategy)
    if strat is not None:
        facts.append(strat)

    facts.sort(key=lambda f: f.priority)
    return facts


def collect_facts_for_players(
    player_ids: Iterable[int], data: WhyData
) -> dict[int, list[WhyFact]]:
    out: dict[int, list[WhyFact]] = {}
    if data.bs is None:
        return out
    for pid in player_ids:
        try:
            player = data.bs.player(int(pid))
        except (KeyError, AttributeError):
            continue
        out[int(pid)] = collect_player_facts(player, data)
    return out


def try_why_data(
    tools: Any,
    gw: int,
    player_ids: Sequence[int],
    strategy: str,
    *,
    horizon: int = 3,
) -> WhyData:
    """Живые источники. Сеть/БД не бросают наружу: нет данных — пустой WhyData.bs или пустые списки."""
    bs = getattr(tools, "bootstrap", None)
    if bs is None:
        return WhyData(gw=gw, strategy=strategy, horizon=horizon)
    client = getattr(tools, "client", None)
    fixtures: list[Fixture] = []
    fn = getattr(client, "fixtures", None)
    if callable(fn):
        try:
            fixtures = list(fn())
        except Exception:
            log.debug("why_facts: fixtures unavailable", exc_info=True)
    history: list[PlayerGWHistory] = []
    ext = None
    if fixtures:
        try:
            from fplcopilot.core.history import load_history

            history = load_history(before_gw=gw, player_ids=list(player_ids))
        except Exception:
            log.debug("why_facts: history unavailable", exc_info=True)
        try:
            from fplcopilot.core.ext_stats import load_ext_index

            ext = load_ext_index(bs)
        except Exception:
            log.debug("why_facts: understat unavailable", exc_info=True)
    return WhyData(
        bs=bs,
        fixtures=fixtures,
        history=history,
        ext=ext,
        gw=gw,
        strategy=strategy,
        horizon=horizon,
    )


def enrich_route_penalty(
    out_players: Sequence[Player],
    in_players: Sequence[Player],
    data: WhyData,
) -> WhyFact | None:
    """Пенальти только при контрасте + посчитанной статистике. Иначе молчим."""
    buyers = [p for p in in_players if p.penalties_order == 1]
    if not buyers:
        return None
    if any(p.penalties_order == 1 for p in out_players):
        return None
    from fplcopilot.core.penalty_outlook import book_from_ext, penalty_outlook

    book = book_from_ext(data.ext)
    if book is None:
        return None
    buyer = buyers[0]
    opps = list((data.upcoming or {}).get(buyer.id, ()))
    if not opps and data.bs is not None and data.fixtures:
        opps = upcoming_opponents(
            buyer, data.fixtures, gw=data.gw, horizon=data.horizon, bs=data.bs
        )
    outlook = penalty_outlook(
        buyer.team, opps, book, taker_name=buyer.web_name, bs=data.bs
    )
    if outlook is None or not outlook.clause:
        return None
    return WhyFact("penalty_outlook", outlook.clause, "buy", 26)


def pick_facts(
    facts: Sequence[WhyFact],
    *,
    role: str,
    interesting_limit: int = 2,
) -> list[WhyFact]:
    """Не больше 2 интересных. Штамп про стратегию/владение больше не клеим."""
    ok = [f for f in facts if f.role in (role, "any")]
    interesting = [f for f in ok if f.kind in INTERESTING]
    interesting.sort(key=lambda f: f.priority)
    return interesting[:interesting_limit]
