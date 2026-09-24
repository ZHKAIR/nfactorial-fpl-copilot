"""Реестр источников новостей. Результаты проверки от 17.09.2026 — в docs/sources.md."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal
from urllib.parse import quote_plus

from fplcopilot.data.schemas import Team

SourceKind = Literal["rss", "google_news", "fpl_api"]


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    kind: SourceKind
    enabled: bool = True
    fetch_fulltext: bool = True  # тянуть ли полный текст статьи через trafilatura
    note: str = ""


GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={q}&hl=en-GB&gl=GB&ceid=GB:en"

SOURCES: tuple[Source, ...] = (
    Source("bbc_football", "https://feeds.bbci.co.uk/sport/football/rss.xml", "rss"),
    Source("sky_football", "https://www.skysports.com/rss/11095", "rss"),
    Source(
        "sky_news_all",
        "https://www.skysports.com/rss/12040",
        "rss",
        enabled=False,
        note="общая лента Sky Sports News (MMA, крикет, F1...) — футбола мало, дублирует 11095",
    ),
    Source("guardian_football", "https://www.theguardian.com/football/rss", "rss"),
    Source("ffscout", "https://www.fantasyfootballscout.co.uk/feed/", "rss"),
    Source(
        "premierinjuries",
        "https://www.premierinjuries.com/feed/",
        "rss",
        enabled=False,
        note="403: Cloudflare JS-challenge на всём сайте, RSS и таблица травм недоступны",
    ),
    Source(
        "premierleague_com",
        "https://www.premierleague.com/news",
        "rss",
        enabled=False,
        note="RSS нет (/rss -> 404), страница новостей рендерится JS — нужен headless-браузер",
    ),
    Source(
        "google_news",
        GOOGLE_NEWS_RSS.format(q=quote_plus('"Premier League" injury')),
        "google_news",
        fetch_fulltext=False,
        note="только --backfill; ссылки — JS-редиректы news.google.com, полный текст недоступен",
    ),
    Source("fpl_api", "https://fantasy.premierleague.com/api/bootstrap-static/", "fpl_api"),
)


def enabled_sources(kind: SourceKind | None = None) -> list[Source]:
    return [s for s in SOURCES if s.enabled and (kind is None or s.kind == kind)]


# Полные названия для поисковых запросов (в FPL API имена укороченные: "Man City", "Spurs").
SEARCH_NAMES: dict[str, str] = {
    "MCI": "Manchester City",
    "MUN": "Manchester United",
    "TOT": "Tottenham",
    "NFO": "Nottingham Forest",
    "WOL": "Wolves",
    "WHU": "West Ham",
    "BHA": "Brighton",
    "NEW": "Newcastle",
    "LEE": "Leeds United",
    "IPS": "Ipswich Town",
    "SHU": "Sheffield United",
}

BACKFILL_KEYWORDS = "injury OR injured OR doubt OR fit OR press conference"


def google_news_url(query: str) -> str:
    return GOOGLE_NEWS_RSS.format(q=quote_plus(query))


def backfill_queries(teams: Iterable[Team]) -> list[tuple[str, str]]:
    """(метка, поисковый запрос) для Google News: общий запрос + по одному на клуб."""
    out = [("general", "Premier League injury news")]
    for t in teams:
        name = SEARCH_NAMES.get(t.short_name, t.name)
        out.append((t.short_name, f'"{name}" {BACKFILL_KEYWORDS}'))
    return out
