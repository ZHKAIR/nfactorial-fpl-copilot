"""Сбор новостного корпуса в Postgres: RSS-ленты, Google News (backfill) и статусы из FPL API.

Запуск:
    uv run python -m fplcopilot.rag.ingest --once [--backfill] [--limit N] [--index]
    uv run python -m fplcopilot.rag.ingest --loop --every 30 --index   # минуты; --index = эмбеддить новое
    uv run python -m fplcopilot.rag.ingest --report                    # только отчёт

Принципы: идемпотентность (url UNIQUE + ON CONFLICT DO NOTHING), вежливость к сайтам
(User-Agent, таймаут, пауза между загрузками полного текста, лимит на источник),
даты только timezone-aware UTC, записи без даты пропускаются.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import re
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, Self
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import feedparser
import httpx
import trafilatura
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from fplcopilot.config import settings
from fplcopilot.data import Bootstrap, FPLClient, Player
from fplcopilot.db import session_scope
from fplcopilot.rag.entity_matcher import EntityMatcher
from fplcopilot.rag.sources import Source, backfill_queries, enabled_sources, google_news_url

log = logging.getLogger("fplcopilot.ingest")

MIN_FULLTEXT_CHARS = 200  # короче — считаем, что trafilatura вытащила мусор, берём summary
MIN_PLAUSIBLE_YEAR = 2000
TRACKING_PREFIXES = ("utm_", "at_", "ns_")
TRACKING_PARAMS = frozenset({"cmp", "oc", "fbclid", "gclid", "igshid", "mc_cid", "mc_eid"})
STATUS_LABELS = {
    "a": "available",
    "d": "doubtful",
    "i": "injured",
    "s": "suspended",
    "u": "unavailable",
    "n": "not in squad",
}

_TAG_RE = re.compile(r"<[^>]+>")


# ---------- чистые функции (покрыты unit-тестами) ----------


def normalize_url(url: str) -> str:
    """Канонический URL для дедупликации: без фрагмента, трекинг-параметров и лишнего слэша."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    for default_port, sch in ((":80", "http"), (":443", "https")):
        if scheme == sch and netloc.endswith(default_port):
            netloc = netloc.removesuffix(default_port)
    query = sorted(
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not (k.lower().startswith(TRACKING_PREFIXES) or k.lower() in TRACKING_PARAMS)
    )
    path = parts.path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return urlunsplit((scheme, netloc, path, urlencode(query), ""))


def content_hash(title: str, body: str) -> str:
    """sha256 от текста, нечувствительный к регистру и пробелам/переносам."""
    norm = " ".join(f"{title}\n{body}".lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def parse_published(entry: Mapping[str, Any]) -> datetime | None:
    """Дата публикации из записи feedparser -> aware UTC; None, если даты нет/не разобрать/неправдоподобна."""
    dt = _parse_published_raw(entry)
    if dt is None:
        return None
    # Google News иногда отдаёт pubDate = 1970-01-01; будущее дальше суток — тоже мусор.
    if dt.year < MIN_PLAUSIBLE_YEAR or dt > datetime.now(UTC) + timedelta(days=1):
        return None
    return dt


def _parse_published_raw(entry: Mapping[str, Any]) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        st = entry.get(key)
        if st:
            return datetime(*st[:6], tzinfo=UTC)
    for key in ("published", "updated", "dc_date"):
        raw = entry.get(key)
        if not raw:
            continue
        dt: datetime | None = None
        try:
            dt = parsedate_to_datetime(raw)  # RFC 2822: "Wed, 16 Sep 2026 18:30:00 +0000"
        except (TypeError, ValueError):
            try:
                dt = datetime.fromisoformat(raw)  # ISO 8601, включая суффикс "Z"
            except ValueError:
                continue
        if dt is None:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    return None


def html_to_text(fragment: str | None) -> str:
    if not fragment:
        return ""
    return " ".join(html.unescape(_TAG_RE.sub(" ", fragment)).split())


def strip_publisher_suffix(title: str, publisher: str | None) -> str:
    """Google News: 'Заголовок - Reuters' -> 'Заголовок' (иначе 'Liverpool Echo' матчится как клуб)."""
    if publisher and title.endswith(f" - {publisher}"):
        return title[: -len(publisher) - 3].rstrip()
    return title


# ---------- модель и запись в БД ----------


@dataclass
class Article:
    source: str
    url: str
    title: str
    summary: str
    content: str
    published_at: datetime
    players: list[int] = field(default_factory=list)
    teams: list[int] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return content_hash(self.title, self.content)


@dataclass
class SourceStats:
    new: int = 0
    skipped: int = 0
    failed: int = 0
    no_date: int = 0
    too_old: int = 0  # published_at раньше settings.news_backfill_since
    snapshots: int = 0  # только для fpl_api

    def __iadd__(self, other: SourceStats) -> Self:
        for k, v in asdict(other).items():
            setattr(self, k, getattr(self, k) + v)
        return self

    def __str__(self) -> str:
        s = (
            f"new={self.new} skipped={self.skipped} failed={self.failed} "
            f"no_date={self.no_date} too_old={self.too_old}"
        )
        return f"{s} snapshots_new={self.snapshots}" if self.snapshots else s


# Состояние игрока в FPL API, по которому определяем «изменилось ли что-то».
StatusState = tuple[str, int | None, str, datetime | None]


def state_of(p: Player) -> StatusState:
    return (p.status, p.chance_of_playing_next_round, p.news, p.news_added)


def snapshot_needed(p: Player, last: StatusState | None) -> bool:
    """Писать снимок только при изменении относительно последнего сохранённого.

    Игрок без истории снимков, который доступен и без новости, — не интересен
    (иначе ~460 бесполезных строк). Возврат в строй (status='a', news='') после
    травмы — это изменение, и оно фиксируется: история «травма → выздоровление».
    """
    if last is None:
        return bool(p.news) or p.status != "a"
    return state_of(p) != last


_INSERT_ARTICLE = text(
    """
    INSERT INTO news_articles
        (source, url, title, summary, content, published_at, content_hash, players, teams, raw)
    VALUES
        (:source, :url, :title, :summary, :content, :published_at, :content_hash,
         :players, :teams, CAST(:raw AS jsonb))
    ON CONFLICT (url) DO NOTHING
    RETURNING id
    """
)
_INSERT_SNAPSHOT = text(
    """
    INSERT INTO player_status_snapshots (player_id, gw, status, chance_next, news, news_added)
    VALUES (:player_id, :gw, :status, :chance_next, :news, :news_added)
    RETURNING id
    """
)
_LAST_SNAPSHOTS = text(
    """
    SELECT DISTINCT ON (player_id) player_id, status, chance_next, news, news_added
    FROM player_status_snapshots
    ORDER BY player_id, snapshot_at DESC, id DESC
    """
)
_EXISTS = text("SELECT 1 FROM news_articles WHERE url = :url")


def article_exists(session: Session, url: str) -> bool:
    return session.execute(_EXISTS, {"url": url}).first() is not None


def insert_article(session: Session, art: Article) -> bool:
    """True — вставлено, False — такой url уже есть."""
    row = session.execute(
        _INSERT_ARTICLE,
        {
            "source": art.source,
            "url": art.url,
            "title": art.title,
            "summary": art.summary or None,
            "content": art.content or None,
            "published_at": art.published_at,
            "content_hash": art.content_hash,
            "players": art.players,
            "teams": art.teams,
            "raw": json.dumps(art.raw, ensure_ascii=False, default=str),
        },
    ).first()
    session.commit()
    return row is not None


def last_snapshots(session: Session) -> dict[int, StatusState]:
    """Последний снимок каждого игрока (news_added приводим к UTC для сравнения с API)."""
    out: dict[int, StatusState] = {}
    for pid, status, chance, news, added in session.execute(_LAST_SNAPSHOTS):
        out[pid] = (status, chance, news, added.astimezone(UTC) if added else None)
    return out


def insert_snapshot(session: Session, p: Player, gw: int | None) -> None:
    session.execute(
        _INSERT_SNAPSHOT,
        {
            "player_id": p.id,
            "gw": gw,
            "status": p.status,
            "chance_next": p.chance_of_playing_next_round,
            "news": p.news,
            "news_added": p.news_added,
        },
    )
    session.commit()


def record_status(session: Session, p: Player, gw: int | None, last: StatusState | None) -> bool:
    """Снимок пишется только при изменении относительно last; True — записан."""
    if not snapshot_needed(p, last):
        return False
    insert_snapshot(session, p, gw)
    return True


# ---------- источники ----------


def fetch_fulltext(http: httpx.Client, url: str) -> str | None:
    try:
        resp = http.get(url)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        log.warning("fulltext %s: %s", url, exc)
        return None
    extracted = trafilatura.extract(resp.text, url=str(resp.url), include_comments=False)
    if not extracted or len(extracted) < MIN_FULLTEXT_CHARS:
        return None
    return extracted


def ingest_feed(
    source: Source,
    session: Session,
    matcher: EntityMatcher,
    http: httpx.Client,
    *,
    feed_url: str | None = None,
    limit: int,
    delay: float,
    since: datetime,
    label: str | None = None,
) -> SourceStats:
    """Одна RSS/Atom-лента (обычная или поисковая Google News)."""
    stats = SourceStats()
    url = feed_url or source.url
    try:
        resp = http.get(url)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        log.error("%s: feed unavailable: %s", source.name, exc)
        stats.failed += 1
        return stats

    feed = feedparser.parse(resp.content)
    if not feed.entries:
        log.warning("%s: no entries (bozo=%s)", source.name, feed.bozo)
        return stats

    for entry in feed.entries:
        if stats.new >= limit:
            break
        link = entry.get("link")
        if not link:
            stats.failed += 1
            continue
        norm_url = normalize_url(link)
        if article_exists(session, norm_url):
            stats.skipped += 1
            continue
        published = parse_published(entry)
        if published is None:
            stats.no_date += 1
            continue
        if published < since:
            stats.too_old += 1
            continue

        publisher = (
            (entry.get("source") or {}).get("title") if source.kind == "google_news" else None
        )
        title = strip_publisher_suffix(html_to_text(entry.get("title")), publisher)
        summary = html_to_text(entry.get("summary") or entry.get("description"))
        content = None
        if source.fetch_fulltext:
            content = fetch_fulltext(http, link)
            time.sleep(delay)
        body = content or summary

        matched = matcher.match(f"{title}\n{body}")
        raw = {
            "link": link,
            "guid": entry.get("id"),
            "author": entry.get("author"),
            "tags": [t.get("term") for t in entry.get("tags", []) if t.get("term")],
            "fulltext": content is not None,
        }
        if publisher:
            raw["publisher"] = publisher
        if label:
            raw["query"] = label
        art = Article(
            source=source.name,
            url=norm_url,
            title=title,
            summary=summary,
            content=body,
            published_at=published,
            players=matched.players,
            teams=matched.teams,
            raw=raw,
        )
        if insert_article(session, art):
            stats.new += 1
        else:
            stats.skipped += 1
    return stats


def ingest_fpl_api(bs: Bootstrap, session: Session, *, since: datetime) -> SourceStats:
    """FPL API: снимок статуса при каждом изменении + документ в news_articles на каждую новость."""
    stats = SourceStats()
    event = bs.next_event or bs.current_event
    gw = event.id if event else None
    last = last_snapshots(session)
    for p in bs.elements:
        if record_status(session, p, gw, last.get(p.id)):
            stats.snapshots += 1
        if not p.news:
            continue

        key = (
            p.news_added.isoformat()
            if p.news_added
            else hashlib.sha1(p.news.encode()).hexdigest()[:12]
        )
        url = f"fpl://player/{p.id}/news/{key}"
        if article_exists(session, url):
            stats.skipped += 1
            continue
        if p.news_added and p.news_added < since:
            stats.too_old += 1
            continue
        team = bs.team(p.team)
        status = STATUS_LABELS.get(p.status, p.status)
        chance = (
            f", chance of playing next round {p.chance_of_playing_next_round}%"
            if p.chance_of_playing_next_round is not None
            else ""
        )
        content = (
            f"{p.full_name} ({team.name}, {p.position.short}) — FPL status: {status}{chance}. "
            f"Official FPL news: {p.news}"
        )
        art = Article(
            source="fpl_api",
            url=url,
            title=f"{p.web_name}: {p.news}",
            summary=p.news,
            content=content,
            published_at=p.news_added or datetime.now(UTC),
            players=[p.id],
            teams=[p.team],
            raw={
                "status": p.status,
                "chance_next": p.chance_of_playing_next_round,
                "news_added": p.news_added,
                "gw": gw,
                "team": team.short_name,
            },
        )
        if insert_article(session, art):
            stats.new += 1
        else:
            stats.skipped += 1
    return stats


# ---------- оркестрация ----------


def run_once(*, backfill: bool = False, limit: int | None = None) -> dict[str, SourceStats]:
    limit = limit or settings.news_limit_per_source
    delay = settings.news_fetch_delay
    since = settings.news_since_utc
    bs = FPLClient().bootstrap(refresh=True)
    matcher = EntityMatcher.from_bootstrap(bs)
    results: dict[str, SourceStats] = {}
    headers = {"User-Agent": settings.news_user_agent, "Accept-Language": "en-GB,en;q=0.9"}

    with (
        httpx.Client(
            headers=headers, timeout=settings.news_fetch_timeout, follow_redirects=True
        ) as http,
        session_scope() as session,
    ):
        for src in enabled_sources("rss"):
            results[src.name] = _guarded(
                src.name,
                session,
                lambda src=src: ingest_feed(
                    src, session, matcher, http, limit=limit, delay=delay, since=since
                ),
            )

        if backfill:
            for gsrc in enabled_sources("google_news"):
                agg = SourceStats()
                for label, query in backfill_queries(bs.teams):
                    agg += _guarded(
                        f"{gsrc.name}[{label}]",
                        session,
                        lambda gsrc=gsrc, label=label, query=query: ingest_feed(
                            gsrc,
                            session,
                            matcher,
                            http,
                            feed_url=google_news_url(query),
                            limit=limit,
                            delay=delay,
                            since=since,
                            label=label,
                        ),
                    )
                    time.sleep(delay)
                results[gsrc.name] = agg

        for fsrc in enabled_sources("fpl_api"):
            results[fsrc.name] = _guarded(
                fsrc.name, session, lambda: ingest_fpl_api(bs, session, since=since)
            )

    for name, st in results.items():
        log.info("%-18s %s", name, st)
    log.info(
        "total: new=%d skipped=%d failed=%d too_old=%d snapshots_new=%d (since %s)",
        sum(s.new for s in results.values()),
        sum(s.skipped for s in results.values()),
        sum(s.failed for s in results.values()),
        sum(s.too_old for s in results.values()),
        sum(s.snapshots for s in results.values()),
        since.date(),
    )
    return results


def rematch_all(batch: int = 500) -> int:
    """Пересчитать players/teams для всех статей текущим матчером (после правок алиасов).

    Ничего не скачивает: берёт title + content из БД. Возвращает число обновлённых строк.
    """
    bs = FPLClient().bootstrap(refresh=True)
    matcher = EntityMatcher.from_bootstrap(bs)
    updated = 0
    with session_scope() as session:
        last_id = 0
        while True:
            rows = session.execute(
                text(
                    "SELECT id, title, coalesce(content, summary, '') FROM news_articles "
                    "WHERE id > :last AND source <> 'fpl_api' ORDER BY id LIMIT :n"
                ),
                {"last": last_id, "n": batch},
            ).all()
            if not rows:
                break
            for row_id, title, body in rows:
                m = matcher.match(f"{title}\n{body}")
                res = session.execute(
                    text(
                        "UPDATE news_articles SET players = CAST(:p AS int[]), teams = CAST(:t AS int[]) "
                        "WHERE id = :id AND (players <> CAST(:p AS int[]) OR teams <> CAST(:t AS int[]))"
                    ),
                    {"p": m.players, "t": m.teams, "id": row_id},
                )
                updated += res.rowcount
                last_id = row_id
            session.commit()
    log.info("rematch: %d article(s) re-tagged", updated)
    return updated


_PROPAGATE_TAGS = text(
    """
    UPDATE news_chunks c
    SET players = a.players, teams = a.teams
    FROM news_articles a
    WHERE a.id = c.article_id
      AND (c.players IS DISTINCT FROM a.players OR c.teams IS DISTINCT FROM a.teams)
    """
)


def propagate_tags_to_chunks() -> int:
    """Скопировать players/teams статей в их чанки (после --rematch). Эмбеддинги не трогаем:
    текст чанка не меняется, меняются только фильтры сущностей. Возвращает число чанков."""
    with session_scope() as session:
        n = session.execute(_PROPAGATE_TAGS).rowcount
    log.info("propagate: %d chunk(s) updated from article tags", n)
    return int(n)


def _guarded(name: str, session: Session, fn: Any) -> SourceStats:
    """Ошибка одного источника не должна валить весь прогон (особенно в --loop)."""
    try:
        return fn()
    except SQLAlchemyError:
        session.rollback()
        log.exception("%s: database error", name)
    except Exception:  # логируем и идём дальше
        log.exception("%s: unexpected error", name)
    return SourceStats(failed=1)


def index_new_articles() -> None:
    """--index: дочанковать и заэмбеддить статьи без чанков (rag/index.py). Ошибка не валит цикл."""
    from fplcopilot.rag.index import rebuild_missing

    try:
        st = rebuild_missing()
    except Exception:  # OpenAI/БД недоступны — попробуем в следующем цикле
        log.exception("index: failed, will retry next cycle")
        return
    log.info("index: %s", st)


def refresh_signals() -> None:
    """--refresh-signals: пакетное обновление сигналов (rag/refresh.py). Ошибка не валит цикл."""
    from datetime import UTC, datetime

    from fplcopilot.rag.refresh import refresh

    try:
        rep = refresh(as_of=datetime.now(UTC), max_players=15, max_teams=4, max_usd=0.02)
    except Exception:  # OpenAI/БД недоступны — попробуем в следующем цикле
        log.exception("refresh-signals: failed, will retry next cycle")
        return
    log.info(
        "refresh-signals: %d player(s), %d club(s), %d LLM call(s), $%.4f",
        len(rep.players),
        len(rep.teams),
        rep.llm_calls,
        rep.spent_usd,
    )


def run_cycle(
    *, backfill: bool = False, limit: int | None = None, index: bool = False, refresh: bool = False
) -> bool:
    """Один цикл --loop: сбор, затем индекс и сигналы. False — сбор упал целиком.

    Ошибки вне отдельных источников (bootstrap FPL API без сети/DNS, недоступная БД) ловятся
    здесь, иначе одна потеря сети останавливает весь цикл.
    """
    ok = True
    try:
        run_once(backfill=backfill, limit=limit)
    except Exception:
        log.exception("ingest: cycle failed, will retry next cycle")
        ok = False
    if index:
        index_new_articles()
    if refresh:
        refresh_signals()
    return ok


def run_loop(
    every_min: float,
    *,
    backfill: bool = False,
    limit: int | None = None,
    index: bool = False,
    refresh: bool = False,
    cycles: int | None = None,
) -> int:
    """Бесконечный (или `cycles` раз) цикл сбора с паузой every_min минут между началами."""
    done = 0
    while cycles is None or done < cycles:
        started = time.monotonic()
        run_cycle(backfill=backfill, limit=limit, index=index, refresh=refresh)
        done += 1
        if cycles is not None and done >= cycles:
            break
        sleep_for = max(0.0, every_min * 60 - (time.monotonic() - started))
        log.info("sleeping %.0fs until next run", sleep_for)
        try:
            time.sleep(sleep_for)
        except KeyboardInterrupt:
            log.info("stopped")
            break
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FPL Copilot: сбор новостного корпуса")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="один прогон (по умолчанию)")
    mode.add_argument("--loop", action="store_true", help="бесконечный цикл с паузой --every")
    ap.add_argument("--every", type=float, default=30, help="пауза между прогонами, минуты")
    ap.add_argument("--backfill", action="store_true", help="дополнительно Google News по клубам")
    ap.add_argument("--limit", type=int, default=None, help="максимум новых записей на источник")
    ap.add_argument("--report", action="store_true", help="напечатать отчёт по корпусу")
    ap.add_argument(
        "--index",
        action="store_true",
        help="после каждого прогона чанковать и эмбеддить новые статьи (rag.index --rebuild-missing)",
    )
    ap.add_argument(
        "--rematch",
        action="store_true",
        help="пересчитать players/teams у всех статей текущим матчером (без скачивания) "
        "и скопировать теги в news_chunks",
    )
    ap.add_argument(
        "--refresh-signals",
        action="store_true",
        help="после --index обновить сигналы игроков с новыми статьями и дайджесты их клубов "
        "(rag.refresh: dense, ≤ 15 игроков, ≤ 4 клубов, ≤ $0.02 за цикл)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("httpx", "httpx2"):
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("trafilatura").setLevel(logging.ERROR)

    if args.rematch:
        n_articles = rematch_all()
        n_chunks = propagate_tags_to_chunks()
        print(f"rematch: articles re-tagged={n_articles}, chunks updated={n_chunks}")
    if not (args.once or args.loop or args.backfill):
        if args.report:
            from fplcopilot.rag.report import build_report

            print(build_report())
        return 0

    if args.loop:
        return run_loop(
            args.every,
            backfill=args.backfill,
            limit=args.limit,
            index=args.index,
            refresh=args.refresh_signals,
        )
    else:
        run_once(backfill=args.backfill, limit=args.limit)
        if args.index:
            index_new_articles()
        if args.refresh_signals:
            refresh_signals()

    if args.report:
        from fplcopilot.rag.report import build_report

        print(build_report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
