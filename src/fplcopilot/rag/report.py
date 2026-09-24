"""Сводка по новостному корпусу: источники, даты, самые упоминаемые игроки и клубы."""

from __future__ import annotations

import logging

import httpx
from sqlalchemy import text

from fplcopilot.data import Bootstrap, FPLClient
from fplcopilot.db import session_scope

log = logging.getLogger(__name__)


def _names(bs: Bootstrap | None) -> tuple[dict[int, str], dict[int, str]]:
    if bs is None:
        return {}, {}
    return (
        {p.id: f"{p.web_name} ({bs.team(p.team).short_name})" for p in bs.elements},
        {t.id: t.name for t in bs.teams},
    )


def build_report(bs: Bootstrap | None = None) -> str:
    if bs is None:
        try:
            bs = FPLClient().bootstrap()
        except (httpx.HTTPError, ValueError) as exc:  # отчёт должен работать и без сети
            log.warning("bootstrap unavailable, showing ids only: %s", exc)
    players, teams = _names(bs)
    lines: list[str] = []

    with session_scope() as s:
        total, lo, hi, with_text = s.execute(
            text(
                "SELECT count(*), min(published_at), max(published_at), "
                "count(*) FILTER (WHERE raw->>'fulltext' = 'true') FROM news_articles"
            )
        ).one()
        lines.append(f"Статей всего: {total}, с полным текстом: {with_text}")
        if lo:
            lines.append(f"Даты публикации: {lo:%Y-%m-%d %H:%M} .. {hi:%Y-%m-%d %H:%M} UTC")

        lines.append("\nПо источникам:")
        rows = s.execute(
            text(
                "SELECT source, count(*), min(published_at), max(published_at) "
                "FROM news_articles GROUP BY source ORDER BY 2 DESC"
            )
        ).all()
        for source, n, lo, hi in rows:
            lines.append(f"  {source:<18}{n:>6}   {lo:%Y-%m-%d} .. {hi:%Y-%m-%d}")

        lines.append("\nТоп-15 игроков по упоминаниям (всего / без fpl_api):")
        rows = s.execute(
            text(
                "SELECT pid, count(*), count(*) FILTER (WHERE source <> 'fpl_api') "
                "FROM news_articles, unnest(players) AS pid "
                "GROUP BY pid ORDER BY 2 DESC, 3 DESC LIMIT 15"
            )
        ).all()
        for pid, n, media in rows:
            lines.append(f"  {players.get(pid, f'player#{pid}'):<28}{n:>5} / {media}")

        lines.append("\nТоп-10 клубов по упоминаниям:")
        rows = s.execute(
            text(
                "SELECT tid, count(*) FROM news_articles, unnest(teams) AS tid "
                "GROUP BY tid ORDER BY 2 DESC LIMIT 10"
            )
        ).all()
        for tid, n in rows:
            lines.append(f"  {teams.get(tid, f'team#{tid}'):<28}{n:>5}")

        n_snap, n_players, n_recovered = s.execute(
            text(
                "SELECT count(*), count(DISTINCT player_id), "
                "count(*) FILTER (WHERE status = 'a' AND news = '') FROM player_status_snapshots"
            )
        ).one()
        lines.append(
            f"\nСнимков статусов (player_status_snapshots): {n_snap}, игроков: {n_players}, "
            f"из них возвратов в строй (status=a, news=''): {n_recovered}"
        )

        lines.append("\n5 последних статей:")
        rows = s.execute(
            text(
                "SELECT source, published_at, title FROM news_articles "
                "ORDER BY published_at DESC LIMIT 5"
            )
        ).all()
        for source, published, title in rows:
            lines.append(f"  [{source}] {published:%Y-%m-%d %H:%M}  {title[:90]}")

    return "\n".join(lines)
