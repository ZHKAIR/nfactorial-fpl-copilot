"""Сила фикстур: ожидаемые голы каждой стороны матча (odds-ready интерфейс).

Протокол FixtureStrengthProvider.expected_goals(fixture) -> (home_xg, away_xg). Реализации:
- TeamRatingProvider (по умолчанию): рейтинги FPL (strength_*) + xG/xGC команд этого сезона,
  сжатые к рейтингу с весом n/(n+k);
- OddsProvider (core/odds.py): по рынку 1X2 The Odds API для 1–2 ближайших туров, остальные
  матчи прозрачно считает TeamRatingProvider (гибрид; без ключа/сети — целиком fallback).

Из (home_xg, away_xg) для каждой команды строится FixtureView: xg_for, xg_against,
clean_sheet_prob = exp(-xg_against), fixture_strength_index 1–5 и fsi_source
('odds' | 'team_rating') для подсказки в UI. Пользователю показываются индекс, xG и источник —
сами коэффициенты и вероятности исходов наружу не выходят.

Константы (см. docs/xpts.md, раздел «Сила фикстур»):
- LEAGUE_AVG_XG = 1.45 — среднее xG команды за матч в АПЛ последних сезонов (≈1.40–1.50);
- HOME_FACTOR/AWAY_FACTOR = 1.10/0.90 — домашнее преимущество ≈ +10% / −10% xG;
- SHRINK_MATCHES = 6 — после 6 матчей данные сезона весят столько же, сколько рейтинг;
- RATING_EXPONENT = 0.75 — множитель силы = (strength / средняя strength) ** exponent.
  В 2026/27 FPL отдаёт strength_attack_*/strength_defence_* = 0, заполнены только
  strength_overall_home/away (2–5); тогда атака и защита берутся из overall. Показатель
  0.75 даёт ARS(д) vs COV ≈ 2.8–0.7 xG — как у букмекеров; по 40 матчам GW1–4 MAE к факту
  почти не зависит от показателя (0.67–0.73), см. docs/xpts.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from fplcopilot.config import settings
from fplcopilot.core.history import TeamMatchStat
from fplcopilot.core.odds import SOURCE_TEAM_RATING, OddsProvider, clean_sheet_prob
from fplcopilot.core.stats import shrink
from fplcopilot.data import Fixture, Team

LEAGUE_AVG_XG = 1.45
HOME_FACTOR = 1.10
AWAY_FACTOR = 0.90
SHRINK_MATCHES = 6
RATING_EXPONENT = 0.75
# fixture_strength_index по разности xg_for - xg_against: >= +0.9 -> 1 (очень лёгкая) ... < -0.9 -> 5
FSI_EDGES = (0.9, 0.3, -0.3, -0.9)


def fixture_strength_index(xg_for: float, xg_against: float) -> int:
    diff = xg_for - xg_against
    for idx, edge in enumerate(FSI_EDGES, start=1):
        if diff >= edge:
            return idx
    return 5


@dataclass(frozen=True)
class FixtureView:
    """Матч глазами одной команды."""

    fixture_id: int
    gw: int | None
    team_id: int
    opponent_id: int
    is_home: bool
    kickoff_time: datetime | None
    xg_for: float
    xg_against: float
    fsi_source: str = SOURCE_TEAM_RATING  # 'odds' | 'team_rating' — чем посчитана сложность

    @property
    def clean_sheet_prob(self) -> float:
        return clean_sheet_prob(self.xg_against)

    @property
    def fixture_strength_index(self) -> int:
        return fixture_strength_index(self.xg_for, self.xg_against)


class FixtureStrengthProvider(Protocol):
    name: str

    def expected_goals(self, fixture: Fixture) -> tuple[float, float]:
        """(ожидаемые голы хозяев, ожидаемые голы гостей)."""
        ...


def provider_source(provider: FixtureStrengthProvider, fixture: Fixture) -> str:
    """Источник сложности конкретного матча: provider.source(fixture), иначе имя провайдера.

    Гибридный OddsProvider отвечает 'odds' только для матчей, покрытых рынком; провайдеры без
    метода source (в т.ч. фейки в тестах) считаются однородными.
    """
    source = getattr(provider, "source", None)
    return str(source(fixture)) if callable(source) else provider.name


@dataclass(frozen=True)
class TeamRating:
    team_id: int
    attack: float  # множитель к среднему xG за (1.0 = средняя команда)
    defence: float  # множитель к среднему xG против (меньше = крепче)
    matches: int  # сколько матчей сезона учтено
    rating_attack: float  # чистый рейтинг FPL (до сжатия)
    rating_defence: float
    data_attack: float | None  # чистые данные сезона (до сжатия)
    data_defence: float | None


class TeamRatingProvider:
    name = SOURCE_TEAM_RATING

    def __init__(
        self,
        teams: list[Team],
        match_stats: dict[tuple[int, int], TeamMatchStat] | None = None,
        *,
        league_avg: float = LEAGUE_AVG_XG,
        shrink_matches: float = SHRINK_MATCHES,
        exponent: float = RATING_EXPONENT,
        home_factor: float = HOME_FACTOR,
        away_factor: float = AWAY_FACTOR,
        calibration_fixtures: list[Fixture] | None = None,
    ) -> None:
        self.league_avg = league_avg
        self.home_factor = home_factor
        self.away_factor = away_factor
        self.ratings = self._build_ratings(teams, match_stats or {}, shrink_matches, exponent)
        # Калибровка: среднее λ по всем фикстурам сезона должно равняться league_avg.
        self.scale = 1.0
        if calibration_fixtures:
            lams = [x for f in calibration_fixtures for x in self._raw(f)]
            if lams:
                self.scale = league_avg / (sum(lams) / len(lams))

    # ---------- рейтинги ----------

    @staticmethod
    def _fpl_strengths(teams: list[Team]) -> dict[int, tuple[float, float]]:
        """team_id -> (attack, defence) в сырых единицах FPL; fallback на strength_overall."""
        use_split = any((t.strength_attack_home or 0) > 0 for t in teams)
        out: dict[int, tuple[float, float]] = {}
        for t in teams:
            if use_split:
                att = ((t.strength_attack_home or 0) + (t.strength_attack_away or 0)) / 2
                dfn = ((t.strength_defence_home or 0) + (t.strength_defence_away or 0)) / 2
            else:
                overall = ((t.strength_overall_home or 3) + (t.strength_overall_away or 3)) / 2
                att = dfn = overall
            out[t.id] = (att or 1.0, dfn or 1.0)
        return out

    def _build_ratings(
        self,
        teams: list[Team],
        match_stats: dict[tuple[int, int], TeamMatchStat],
        k: float,
        exponent: float,
    ) -> dict[int, TeamRating]:
        raw = self._fpl_strengths(teams)
        mean_att = sum(a for a, _ in raw.values()) / len(raw)
        mean_def = sum(d for _, d in raw.values()) / len(raw)

        per_team: dict[int, list[TeamMatchStat]] = {t.id: [] for t in teams}
        for (_, team_id), st in match_stats.items():
            per_team.setdefault(team_id, []).append(st)
        all_xg = [st.xg_for for st in match_stats.values()]
        league_xg = (sum(all_xg) / len(all_xg)) if all_xg else self.league_avg

        ratings: dict[int, TeamRating] = {}
        for t in teams:
            att_raw, def_raw = raw[t.id]
            r_att = (att_raw / mean_att) ** exponent
            r_def = (mean_def / def_raw) ** exponent
            stats = per_team.get(t.id, [])
            n = len(stats)
            if n:
                d_att = (sum(s.xg_for for s in stats) / n) / league_xg
                d_def = (sum(s.xg_against for s in stats) / n) / league_xg
            else:
                d_att = d_def = None
            ratings[t.id] = TeamRating(
                team_id=t.id,
                attack=shrink(d_att, r_att, n, k) if d_att is not None else r_att,
                defence=shrink(d_def, r_def, n, k) if d_def is not None else r_def,
                matches=n,
                rating_attack=r_att,
                rating_defence=r_def,
                data_attack=d_att,
                data_defence=d_def,
            )
        return ratings

    # ---------- ожидаемые голы ----------

    def _raw(self, fixture: Fixture) -> tuple[float, float]:
        h = self.ratings[fixture.team_h]
        a = self.ratings[fixture.team_a]
        home = self.league_avg * h.attack * a.defence * self.home_factor
        away = self.league_avg * a.attack * h.defence * self.away_factor
        return home, away

    def expected_goals(self, fixture: Fixture) -> tuple[float, float]:
        home, away = self._raw(fixture)
        return round(home * self.scale, 3), round(away * self.scale, 3)

    def source(self, fixture: Fixture) -> str:
        return self.name


# ---------- фабрика и представления ----------


def get_provider(
    name: str | None = None,
    *,
    teams: list[Team] | None = None,
    match_stats: dict[tuple[int, int], TeamMatchStat] | None = None,
    calibration_fixtures: list[Fixture] | None = None,
) -> FixtureStrengthProvider:
    """Провайдер по имени (по умолчанию settings.xpts_fixture_provider).

    Для odds фикстуры сезона (calibration_fixtures) нужны ещё и для сопоставления матчей API с
    FPL, а TeamRatingProvider на тех же данных становится fallback для непокрытых туров.
    """
    name = name or settings.xpts_fixture_provider
    if name == "team_rating":
        if teams is None:
            raise ValueError("TeamRatingProvider требует список команд")
        return TeamRatingProvider(teams, match_stats, calibration_fixtures=calibration_fixtures)
    if name == "odds":
        fallback = (
            TeamRatingProvider(teams, match_stats, calibration_fixtures=calibration_fixtures)
            if teams is not None
            else None
        )
        return OddsProvider(
            settings.odds_api_key,
            teams=teams or (),
            fixtures=calibration_fixtures or (),
            fallback=fallback,
            league_avg=LEAGUE_AVG_XG,
        )
    raise ValueError(f"неизвестный провайдер силы фикстур: {name!r}")


def fixture_views(
    provider: FixtureStrengthProvider, fixtures: list[Fixture], gw: int
) -> dict[int, list[FixtureView]]:
    """team_id -> матчи тура gw глазами команды (пусто = blank GW, 2+ = double GW)."""
    out: dict[int, list[FixtureView]] = {}
    for f in fixtures:
        if f.event != gw:
            continue
        home_xg, away_xg = provider.expected_goals(f)
        src = provider_source(provider, f)
        out.setdefault(f.team_h, []).append(
            FixtureView(
                f.id, f.event, f.team_h, f.team_a, True, f.kickoff_time, home_xg, away_xg, src
            )
        )
        out.setdefault(f.team_a, []).append(
            FixtureView(
                f.id, f.event, f.team_a, f.team_h, False, f.kickoff_time, away_xg, home_xg, src
            )
        )
    return out
