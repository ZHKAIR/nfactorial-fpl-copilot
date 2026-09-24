"""История матчей игроков (element-summary) в Postgres + защита от утечки из будущего.

    uv run python -m fplcopilot.core.history --sync [--refresh] [--players 1,2,3]
    uv run python -m fplcopilot.core.history --stats

Таблицы (005_core.sql): player_gw_history — строка на игрока и матч, player_season_history —
итоги прошлых сезонов (приор для per-90 ставок). Синхронизация идемпотентна (upsert), 659
element-summary тянутся через дисковый кэш FPLClient с паузой между сетевыми запросами.

Правило без исключений: прогноз на тур g строится по `rows_before(rows, g, deadline)` —
только раунды < g и только матчи, сыгранные до дедлайна g.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict
from sqlalchemy import text
from sqlalchemy.orm import Session

from fplcopilot.data import Fixture, FPLClient
from fplcopilot.data.schemas import PlayerGWHistory
from fplcopilot.db import session_scope

log = logging.getLogger(__name__)

SEASON = "2026/27"
PREVIOUS_SEASON = "2025/26"
NETWORK_SLEEP_S = 0.07  # пауза после реального HTTP-запроса (кэш-хиты не тормозим)
NETWORK_HIT_THRESHOLD_S = 0.02  # быстрее — значит ответ пришёл из дискового кэша

HISTORY_COLUMNS = (
    "opponent_team",
    "round",
    "kickoff_time",
    "was_home",
    "minutes",
    "starts",
    "total_points",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "bonus",
    "bps",
    "expected_goals",
    "expected_assists",
    "expected_goals_conceded",
    "tackles",
    "recoveries",
    "clearances_blocks_interceptions",
    "defensive_contribution",
    "yellow_cards",
    "red_cards",
    "value",
    "selected",
    "team_h_score",
    "team_a_score",
)

_UPSERT_HISTORY = text(
    "INSERT INTO player_gw_history (player_id, season, fixture, "
    + ", ".join(HISTORY_COLUMNS)
    + ") VALUES (:player_id, :season, :fixture, "
    + ", ".join(f":{c}" for c in HISTORY_COLUMNS)
    + ") ON CONFLICT (season, player_id, fixture) DO UPDATE SET "
    + ", ".join(f"{c} = EXCLUDED.{c}" for c in HISTORY_COLUMNS)
    + ", synced_at = now()"
)

SEASON_COLUMNS = (
    "element_code",
    "start_cost",
    "end_cost",
    "total_points",
    "minutes",
    "starts",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "bonus",
    "bps",
    "expected_goals",
    "expected_assists",
    "expected_goals_conceded",
    "tackles",
    "recoveries",
    "clearances_blocks_interceptions",
    "defensive_contribution",
    "yellow_cards",
    "red_cards",
)

_UPSERT_SEASON = text(
    "INSERT INTO player_season_history (player_id, season_name, "
    + ", ".join(SEASON_COLUMNS)
    + ") VALUES (:player_id, :season_name, "
    + ", ".join(f":{c}" for c in SEASON_COLUMNS)
    + ") ON CONFLICT (player_id, season_name) DO UPDATE SET "
    + ", ".join(f"{c} = EXCLUDED.{c}" for c in SEASON_COLUMNS)
    + ", synced_at = now()"
)

_SELECT_HISTORY = (
    "SELECT player_id AS element, fixture, "
    + ", ".join(HISTORY_COLUMNS)
    + " FROM player_gw_history"
)


class SeasonTotals(BaseModel):
    """Итоги одного прошлого сезона игрока (history_past)."""

    model_config = ConfigDict(extra="ignore")

    season_name: str
    element_code: int | None = None
    start_cost: int | None = None
    end_cost: int | None = None
    total_points: int = 0
    minutes: int = 0
    starts: int = 0
    goals_scored: int = 0
    assists: int = 0
    clean_sheets: int = 0
    goals_conceded: int = 0
    saves: int = 0
    bonus: int = 0
    bps: int = 0
    expected_goals: float | None = None
    expected_assists: float | None = None
    expected_goals_conceded: float | None = None
    tackles: int = 0
    recoveries: int = 0
    clearances_blocks_interceptions: int = 0
    defensive_contribution: int = 0
    yellow_cards: int = 0
    red_cards: int = 0


@dataclass
class SyncStats:
    players: int = 0
    rows: int = 0
    past_rows: int = 0
    network_fetches: int = 0
    elapsed_s: float = 0.0

    def __str__(self) -> str:
        return (
            f"players={self.players} rows={self.rows} past_rows={self.past_rows} "
            f"network={self.network_fetches} elapsed={self.elapsed_s:.1f}s"
        )


# ---------- запись ----------


def history_params(player_id: int, row: PlayerGWHistory, season: str = SEASON) -> dict[str, Any]:
    data = row.model_dump()
    params = {c: data.get(c) for c in HISTORY_COLUMNS}
    params.update(player_id=player_id, season=season, fixture=row.fixture)
    return params


def upsert_history(
    session: Session, player_id: int, rows: list[PlayerGWHistory], season: str = SEASON
) -> int:
    if not rows:
        return 0
    session.execute(_UPSERT_HISTORY, [history_params(player_id, r, season) for r in rows])
    return len(rows)


def upsert_past_seasons(session: Session, player_id: int, past: list[dict[str, Any]]) -> int:
    params = []
    for raw in past:
        totals = SeasonTotals.model_validate(raw)
        p = totals.model_dump()
        p["player_id"] = player_id
        params.append(p)
    if params:
        session.execute(_UPSERT_SEASON, params)
    return len(params)


def sync_history(
    client: FPLClient,
    player_ids: list[int] | None = None,
    *,
    sleep_s: float = NETWORK_SLEEP_S,
    progress_every: int = 100,
) -> SyncStats:
    """Тянет element-summary всех игроков (или player_ids) и upsert-ит историю + прошлые сезоны.

    Кэш FPLClient переиспользуется (TTL из настроек; `FPLClient(cache_ttl=0)` — принудительно
    свежие данные). После каждого сетевого запроса — пауза sleep_s.
    """
    started = time.perf_counter()
    bs = client.bootstrap()
    ids = player_ids or [p.id for p in bs.elements]
    stats = SyncStats(players=len(ids))
    with session_scope() as s:
        for i, pid in enumerate(ids, start=1):
            t = time.perf_counter()
            summary = client.element_summary(pid)
            if time.perf_counter() - t > NETWORK_HIT_THRESHOLD_S:
                stats.network_fetches += 1
                time.sleep(sleep_s)
            stats.rows += upsert_history(s, pid, summary.history)
            stats.past_rows += upsert_past_seasons(s, pid, summary.history_past)
            if progress_every and i % progress_every == 0:
                log.info("synced %d/%d players (%d rows)", i, len(ids), stats.rows)
    stats.elapsed_s = time.perf_counter() - started
    log.info("history sync done: %s", stats)
    return stats


# ---------- чтение ----------


def _row_to_history(mapping: Any) -> PlayerGWHistory:
    return PlayerGWHistory.model_validate(dict(mapping))


def load_history(
    *,
    before_gw: int | None = None,
    deadline: datetime | None = None,
    player_ids: list[int] | None = None,
    season: str = SEASON,
) -> list[PlayerGWHistory]:
    """Строки истории из БД. before_gw/deadline — защита от утечки (см. rows_before)."""
    sql = _SELECT_HISTORY + " WHERE season = :season"
    params: dict[str, Any] = {"season": season}
    if player_ids:
        sql += " AND player_id = ANY(:ids)"
        params["ids"] = list(player_ids)
    sql += " ORDER BY player_id, round, kickoff_time"
    with session_scope() as s:
        rows = [_row_to_history(m) for m in s.execute(text(sql), params).mappings()]
    if before_gw is not None:
        rows = rows_before(rows, before_gw, deadline)
    return rows


def rows_before(
    rows: list[PlayerGWHistory], gw: int, deadline: datetime | None = None
) -> list[PlayerGWHistory]:
    """Leakage guard: только раунды < gw и (если известен дедлайн) матчи с kickoff < deadline.

    Второе условие защищает от перенесённых матчей ранних туров, сыгранных уже после дедлайна.
    """
    out = []
    for r in rows:
        if r.round >= gw:
            continue
        if deadline is not None and r.kickoff_time is not None:
            ko = r.kickoff_time if r.kickoff_time.tzinfo else r.kickoff_time.replace(tzinfo=UTC)
            if ko >= deadline:
                continue
        out.append(r)
    return out


def group_by_player(rows: list[PlayerGWHistory]) -> dict[int, list[PlayerGWHistory]]:
    """player_id -> строки по возрастанию (round, kickoff)."""
    grouped: dict[int, list[PlayerGWHistory]] = defaultdict(list)
    for r in rows:
        grouped[r.element].append(r)
    far_future = datetime.max.replace(tzinfo=UTC)
    for lst in grouped.values():
        lst.sort(key=lambda r: (r.round, r.kickoff_time or far_future, r.fixture))
    return dict(grouped)


def load_past_seasons(season_name: str = PREVIOUS_SEASON) -> dict[int, SeasonTotals]:
    sql = (
        "SELECT player_id, season_name, "
        + ", ".join(SEASON_COLUMNS)
        + " FROM player_season_history WHERE season_name = :season"
    )
    with session_scope() as s:
        mappings = s.execute(text(sql), {"season": season_name}).mappings().all()
    return {m["player_id"]: SeasonTotals.model_validate(dict(m)) for m in mappings}


def actual_points(gw: int, season: str = SEASON) -> dict[int, tuple[int, int]]:
    """player_id -> (сумма очков, сумма минут) за раунд gw (для бэктеста и обзора тура)."""
    sql = text(
        "SELECT player_id, sum(total_points), sum(minutes) FROM player_gw_history "
        "WHERE season = :season AND round = :gw GROUP BY player_id"
    )
    with session_scope() as s:
        return {
            int(r[0]): (int(r[1]), int(r[2])) for r in s.execute(sql, {"season": season, "gw": gw})
        }


def history_stats(season: str = SEASON) -> dict[str, Any]:
    with session_scope() as s:
        n, players, rounds, last = s.execute(
            text(
                "SELECT count(*), count(distinct player_id), max(round), max(synced_at) "
                "FROM player_gw_history WHERE season = :s"
            ),
            {"s": season},
        ).one()
        past = s.execute(text("SELECT count(*) FROM player_season_history")).scalar_one()
    return {
        "rows": n,
        "players": players,
        "max_round": rounds,
        "last_sync": last,
        "past_rows": past,
    }


# ---------- командная статистика ----------


@dataclass(frozen=True)
class TeamMatchStat:
    fixture: int
    round: int
    team: int
    opponent: int
    was_home: bool
    xg_for: float
    xg_against: float
    goals_for: int | None
    goals_against: int | None


def team_match_stats(
    gw_max: int | None = None,
    *,
    rows: list[PlayerGWHistory] | None = None,
    fixtures: list[Fixture] | None = None,
    deadline: datetime | None = None,
) -> dict[tuple[int, int], TeamMatchStat]:
    """(fixture, team) -> xG за / против по матчу: xG команды = сумма expected_goals её игроков.

    gw_max — включительно верхняя граница раунда (для прогноза на тур g передаём g-1).
    rows/fixtures можно передать явно (тесты); иначе берутся из БД и FPLClient.
    """
    if rows is None:
        rows = load_history(before_gw=gw_max + 1 if gw_max is not None else None, deadline=deadline)
    elif gw_max is not None:
        rows = rows_before(rows, gw_max + 1, deadline)
    if fixtures is None:
        fixtures = FPLClient().fixtures()
    by_id = {f.id: f for f in fixtures}

    xg: dict[tuple[int, int], float] = defaultdict(float)
    meta: dict[tuple[int, int], tuple[int, int, bool, int | None, int | None]] = {}
    for r in rows:
        fx = by_id.get(r.fixture)
        if fx is None:
            continue
        team = fx.team_h if r.was_home else fx.team_a
        opp = fx.team_a if r.was_home else fx.team_h
        key = (r.fixture, team)
        xg[key] += float(r.expected_goals or 0.0)
        if key not in meta:
            gf, ga = (
                (r.team_h_score, r.team_a_score) if r.was_home else (r.team_a_score, r.team_h_score)
            )
            meta[key] = (r.round, opp, r.was_home, gf, ga)

    out: dict[tuple[int, int], TeamMatchStat] = {}
    for (fixture, team), (rnd, opp, home, gf, ga) in meta.items():
        out[(fixture, team)] = TeamMatchStat(
            fixture=fixture,
            round=rnd,
            team=team,
            opponent=opp,
            was_home=home,
            xg_for=round(xg[(fixture, team)], 3),
            xg_against=round(xg.get((fixture, opp), 0.0), 3),
            goals_for=gf,
            goals_against=ga,
        )
    return out


# ---------- CLI ----------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FPL Copilot: синхронизация истории матчей игроков")
    ap.add_argument("--sync", action="store_true", help="загрузить element-summary и записать в БД")
    ap.add_argument("--refresh", action="store_true", help="игнорировать дисковый кэш FPL API")
    ap.add_argument("--players", help="id игроков через запятую (по умолчанию все)")
    ap.add_argument("--stats", action="store_true", help="показать, что есть в БД")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.sync:
        client = FPLClient(cache_ttl=0) if args.refresh else FPLClient()
        ids = [int(x) for x in args.players.split(",")] if args.players else None
        stats = sync_history(client, ids)
        print(f"OK: {stats}")
    if args.stats or not args.sync:
        print(history_stats())
    return 0


if __name__ == "__main__":
    sys.exit(main())
