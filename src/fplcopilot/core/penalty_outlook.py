"""Вероятность пенальти на ближайшие туры по Understat (xG − npxG).

Фолов нет ни в FPL, ни в Understat — не упоминаем. Один пенальти ≈ 0.76 xG.
За 5 туров выборка крошечная: rate сжимаем к среднему лиги (k≈10).
В xPts не подмешивается: бонус пенальтиста остаётся плоским XPTS_SETPIECE_WEIGHT.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fplcopilot import config
from fplcopilot.core.understat import PENALTY_XG, UnderstatTeam
from fplcopilot.data.schemas import Bootstrap

DEFAULT_LEAGUE_AVG = 0.14
SHRINK_K = 10.0
STANDOUT_CONCEDED = 2


@dataclass(frozen=True)
class TeamPens:
    team_id: int
    title: str
    matches: int
    pens_won: int
    pens_conceded: int
    match_pens: tuple[tuple[str, int, int], ...] = ()


@dataclass(frozen=True)
class PenaltyBook:
    by_team: dict[int, TeamPens]
    league_avg: float
    league_won: int
    league_conceded: int
    team_matches: int

    @property
    def balanced(self) -> bool:
        return self.league_won == self.league_conceded


@dataclass(frozen=True)
class OppPen:
    team_id: int
    name: str
    pens_conceded: int
    last2_conceded: int
    standout: bool
    last1_conceded: int = 0


@dataclass(frozen=True)
class PenaltyOutlook:
    team_id: int
    taker: str
    club: str
    league_avg: float
    won_rate: float
    lambdas: tuple[float, ...]
    p_any: float
    p_pct: int
    n_fixtures: int
    opponents: tuple[OppPen, ...]
    clause: str
    kind: str  # standout | probability

    def as_dict(self) -> dict[str, Any]:
        return {
            "team_id": self.team_id,
            "taker": self.taker,
            "club": self.club,
            "league_avg": round(self.league_avg, 4),
            "won_rate": round(self.won_rate, 4),
            "lambda_sum": round(sum(self.lambdas), 4),
            "p_any": round(self.p_any, 4),
            "p_pct": self.p_pct,
            "n_fixtures": self.n_fixtures,
            "opponents": [
                {
                    "team_id": o.team_id,
                    "name": o.name,
                    "pens_conceded": o.pens_conceded,
                    "last2_conceded": o.last2_conceded,
                    "last1_conceded": o.last1_conceded,
                    "standout": o.standout,
                }
                for o in self.opponents
            ],
            "clause": self.clause,
            "kind": self.kind,
        }


def shrink_rate(pens: int, matches: int, league_avg: float, k: float = SHRINK_K) -> float:
    """(pens + k·μ) / (matches + k). Пустые матчи → среднее лиги."""
    m = max(0, int(matches))
    p = max(0, int(pens))
    mu = max(0.0, float(league_avg))
    kk = max(0.0, float(k))
    return (p + kk * mu) / (m + kk) if (m + kk) > 0 else mu


def match_lambda(won_rate: float, opp_conceded_rate: float, league_avg: float) -> float:
    if league_avg <= 0:
        return 0.0
    return league_avg * (won_rate / league_avg) * (opp_conceded_rate / league_avg)


def p_at_least_one(lambdas: Sequence[float]) -> float:
    return 1.0 - math.exp(-sum(max(0.0, float(x)) for x in lambdas))


def last_n_conceded(row: TeamPens, n: int = 2) -> int:
    if n <= 0:
        return 0
    return sum(int(c) for _, _, c in row.match_pens[-n:])


def book_from_teams(by_team: Mapping[int, UnderstatTeam]) -> PenaltyBook:
    rows: dict[int, TeamPens] = {}
    won = conc = matches = 0
    for tid, t in by_team.items():
        row = TeamPens(
            team_id=int(tid),
            title=t.title,
            matches=int(t.matches),
            pens_won=int(getattr(t, "pens_won", 0) or 0),
            pens_conceded=int(getattr(t, "pens_conceded", 0) or 0),
            match_pens=tuple(getattr(t, "match_pens", ()) or ()),
        )
        rows[int(tid)] = row
        won += row.pens_won
        conc += row.pens_conceded
        matches += row.matches
    fallback = float(getattr(config.settings, "why_pen_league_avg_fallback", DEFAULT_LEAGUE_AVG))
    avg = (won / matches) if matches >= 8 else fallback
    if avg <= 0:
        avg = fallback
    return PenaltyBook(
        by_team=rows,
        league_avg=avg,
        league_won=won,
        league_conceded=conc,
        team_matches=matches,
    )


def book_from_ext(ext: Any) -> PenaltyBook | None:
    if ext is None or not getattr(ext, "by_team", None):
        return None
    return book_from_teams(ext.by_team)


def _club_ru(short_name: str | None) -> str:
    from fplcopilot.core.why_facts import club_ru

    return club_ru(short_name)


def _club_ru_gen(short_name: str | None) -> str:
    from fplcopilot.core.why_facts import club_ru_gen

    return club_ru_gen(short_name)


def _opp_row(
    opp: Any,
    book: PenaltyBook,
    *,
    standout_n: int,
) -> OppPen:
    tid = int(getattr(opp, "team_id", 0) or 0)
    short = getattr(opp, "short_name", "") or ""
    row = book.by_team.get(tid)
    name = _club_ru(short)
    if row is None:
        return OppPen(tid, name, 0, 0, False)
    last2 = last_n_conceded(row, 2)
    last1 = last_n_conceded(row, 1)
    standout = row.pens_conceded >= standout_n or last2 >= 1
    return OppPen(row.team_id, name, row.pens_conceded, last2, standout, last1)


def _tours_word(n: int) -> str:
    n = abs(n)
    tail, tail2 = n % 10, n % 100
    if tail == 1 and tail2 != 11:
        return f"{n} тур"
    if 2 <= tail <= 4 and not 12 <= tail2 <= 14:
        return f"{n} тура"
    return f"{n} туров"


def penalty_outlook(
    team_id: int,
    next_fixtures: Sequence[Any],
    book: PenaltyBook,
    *,
    taker_name: str,
    bs: Bootstrap | None = None,
    k: float | None = None,
) -> PenaltyOutlook | None:
    """P(хотя бы один пенальти у команды team_id за ближайшие фикстуры) + фраза."""
    if not next_fixtures or not taker_name:
        return None
    settings = config.settings
    kk = float(getattr(settings, "why_pen_shrink_k", SHRINK_K) if k is None else k)
    standout_n = int(getattr(settings, "why_pen_standout_conceded", STANDOUT_CONCEDED))
    mine = book.by_team.get(int(team_id))
    matches = mine.matches if mine else 0
    won = mine.pens_won if mine else 0
    won_rate = shrink_rate(won, matches, book.league_avg, kk)
    club = "команды"
    club_short = ""
    if bs is not None:
        try:
            club_short = bs.team(int(team_id)).short_name
            club = _club_ru(club_short)
        except (KeyError, AttributeError):
            club_short = mine.title if mine else ""
            club = _club_ru(club_short) if club_short else (mine.title if mine else "команды")
    elif mine is not None:
        club_short = mine.title
        club = _club_ru(mine.title)
    club_gen = _club_ru_gen(club_short) if club_short else club

    opps: list[OppPen] = []
    lambdas: list[float] = []
    for opp in next_fixtures:
        row = _opp_row(opp, book, standout_n=standout_n)
        opps.append(row)
        opp_team = book.by_team.get(int(opp.team_id))
        conc = opp_team.pens_conceded if opp_team else 0
        m = opp_team.matches if opp_team else 0
        conc_rate = shrink_rate(conc, m, book.league_avg, kk)
        lambdas.append(match_lambda(won_rate, conc_rate, book.league_avg))

    p = p_at_least_one(lambdas)
    p_pct = round(p * 100)
    n = len(next_fixtures)
    tours = _tours_word(n)
    min_p = float(getattr(settings, "why_pen_min_p", 0.30))
    highlighted = next((o for o in opps if o.standout), None)
    if highlighted is not None:
        if highlighted.last1_conceded >= 1:
            clause = (
                f"{highlighted.name} отдал пенальти в прошлом туре, "
                f"а у {club_gen} их бьёт {taker_name}"
            )
        elif highlighted.last2_conceded >= 1:
            clause = (
                f"{highlighted.name} отдал пенальти в одном из последних двух туров, "
                f"а у {club_gen} их бьёт {taker_name}"
            )
        else:
            n_conc = highlighted.pens_conceded
            n_games = (
                book.by_team[highlighted.team_id].matches
                if highlighted.team_id in book.by_team
                else n
            )
            clause = (
                f"{highlighted.name} уже отдал {n_conc} "
                f"пенальти за {n_games} "
                f"{'тур' if n_games == 1 else 'тура' if n_games in (2, 3, 4) else 'туров'}, "
                f"а у {club_gen} их бьёт {taker_name}"
            )
        kind = "standout"
    elif p + 1e-12 >= min_p:
        clause = (
            f"примерно {p_pct}%, что у {club_gen} будет пенальти за эти {tours}, "
            f"бить будет {taker_name}"
        )
        kind = "probability"
    else:
        clause = ""
        kind = "below_threshold"
    return PenaltyOutlook(
        team_id=int(team_id),
        taker=taker_name,
        club=club,
        league_avg=book.league_avg,
        won_rate=won_rate,
        lambdas=tuple(lambdas),
        p_any=p,
        p_pct=p_pct,
        n_fixtures=n,
        opponents=tuple(opps),
        clause=clause.strip(),
        kind=kind,
    )


def flat_setpiece_bonus_horizon(horizon: int = 3) -> float:
    """Плоский бонус модели xPts за горизонт (не меняем xPts в этой итерации)."""
    w = float(getattr(config.settings, "xpts_setpiece_weight", 0.12))
    return w * max(1, int(horizon))


def lambda_xpts_mid(lambda_sum: float) -> float:
    """Грубая оценка очков MID с пенальти: Σλ × 0.76 × 5. Для сравнения с плоским бонусом."""
    return float(lambda_sum) * PENALTY_XG * 5.0
