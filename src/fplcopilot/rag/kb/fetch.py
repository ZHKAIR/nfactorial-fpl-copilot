"""Загрузка документов KB: html (trafilatura -> markdown с заголовками), reddit_json
(old.reddit.com/<path>.json, только selftext поста — без комментариев), internal (файл репозитория).

Вежливость и бюджет: браузерный User-Agent для сайтов (как в news-ингесте), описательный
UA для Reddit, паузы между запросами (1 с сайты / 2 с Reddit), лимит запросов к Reddit (15)
и общий лимит сетевых загрузок за процесс (60). Все ответы кэшируются на диск
(.cache/kb/http/<sha1(url)>.json): проверка источников (--probe) и ингест (--ingest) делят
одну загрузку, повторный запуск ничего не скачивает (`offline=True` запрещает сеть вовсе).

Чистые функции (unit-тесты): reddit_json_url, reddit_post_from_payload, clean_reddit_markdown,
extract_reddit_post_links, extract_html, is_login_redirect.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
import trafilatura

from fplcopilot.config import PROJECT_ROOT, settings
from fplcopilot.rag.kb.registry import KBSource

log = logging.getLogger("fplcopilot.kb.fetch")

MIN_TEXT_CHARS = 500  # короче — считаем, что текст статьи не извлёкся (меню, заглушка, paywall)
DELAY_HTML_S = 1.0
DELAY_REDDIT_S = 2.0
MAX_REDDIT_REQUESTS = 15
MAX_TOTAL_FETCHES = 60
HTTP_TIMEOUT_S = 20.0
REDDIT_USER_AGENT = (
    "fpl-copilot-kb/0.1 (educational FPL decision-support project; reads public post text only; "
    "python-httpx)"
)
BROWSER_USER_AGENT = settings.news_user_agent

_MD_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)\s]+)(?:\s+\"[^\"]*\")?\)")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_EMPHASIS = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_REDDIT_POST_LINK = re.compile(
    r"(?:https?://(?:www\.|old\.|new\.)?reddit\.com)?/r/(?P<sub>[A-Za-z0-9_]+)/comments/"
    r"(?P<id>[a-z0-9]{5,8})\b"
)
_OG_TITLE = re.compile(r'<meta[^>]+property="og:title"[^>]+content="([^"]*)"', re.IGNORECASE)
_TITLE_TAG = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


# ---------- модель результата ----------


@dataclass
class Fetched:
    url: str  # канонический url источника (ключ kb_docs.url)
    final_url: str = ""
    status: int | None = None
    title: str = ""
    text: str = ""  # markdown-подобный текст (заголовки `#`, списки) для chunking
    method: str = "none"  # trafilatura | reddit_json | internal | none
    published_at: datetime | None = None
    error: str | None = None
    from_cache: bool = False
    linked_urls: list[str] = field(
        default_factory=list
    )  # reddit: ссылки на другие посты сабреддита

    @property
    def ok(self) -> bool:
        return self.error is None and len(self.text) >= MIN_TEXT_CHARS

    @property
    def chars(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class CachedResponse:
    url: str
    final_url: str
    status: int
    content_type: str
    body: str
    fetched_at: str
    from_cache: bool


# ---------- чистые функции ----------


def reddit_json_url(url: str) -> str:
    """https://www.reddit.com/r/FantasyPL/comments/ia4vpk/… -> https://old.reddit.com/r/FantasyPL/comments/ia4vpk/.json"""
    parts = urlsplit(url.strip())
    path = parts.path.removesuffix(".json").rstrip("/") + "/.json"
    return urlunsplit(("https", "old.reddit.com", path, "", ""))


def canonical_reddit_post_url(subreddit: str, post_id: str) -> str:
    return f"https://www.reddit.com/r/{subreddit}/comments/{post_id}/"


def is_login_redirect(final_url: str) -> bool:
    """old.reddit отдаёт 302 на /login/?reason=lor2 неавторизованным «подозрительным» клиентам."""
    return "/login" in urlsplit(final_url).path


def clean_reddit_markdown(text: str) -> str:
    """HTML-сущности, zero-width, картинки, ссылки -> текст, **жирный** -> текст; заголовки # остаются."""
    s = html.unescape(html.unescape(text or ""))  # reddit экранирует дважды: &amp;#x200B;
    s = _ZERO_WIDTH.sub("", s)
    s = _MD_IMAGE.sub("", s)
    s = _MD_LINK.sub(r"\1", s)
    s = _MD_EMPHASIS.sub(r"\2", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


@dataclass(frozen=True)
class RedditPost:
    title: str
    selftext: str
    created_utc: float | None
    permalink: str
    subreddit: str


def reddit_post_from_payload(payload: Any) -> RedditPost:
    """JSON old.reddit /comments/<id>/.json: [listing(post), listing(comments)] -> данные поста.

    Берём только children[0] первого листинга (сам пост); комментарии игнорируются.
    """
    listing = payload[0] if isinstance(payload, list) else payload
    try:
        post = listing["data"]["children"][0]["data"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("неожиданная структура JSON поста reddit") from exc
    return RedditPost(
        title=html.unescape(str(post.get("title") or "")).strip(),
        selftext=clean_reddit_markdown(str(post.get("selftext") or "")),
        created_utc=float(post["created_utc"]) if post.get("created_utc") else None,
        permalink=str(post.get("permalink") or ""),
        subreddit=str(post.get("subreddit") or ""),
    )


def extract_reddit_post_links(text: str, subreddit: str = "FantasyPL") -> list[str]:
    """Канонические ссылки на посты того же сабреддита из текста (уникальные, по порядку)."""
    out: list[str] = []
    for m in _REDDIT_POST_LINK.finditer(text or ""):
        if m.group("sub").lower() != subreddit.lower():
            continue
        url = canonical_reddit_post_url(subreddit, m.group("id"))
        if url not in out:
            out.append(url)
    return out


def html_title(body: str) -> str:
    m = _OG_TITLE.search(body) or _TITLE_TAG.search(body)
    return html.unescape(" ".join(m.group(1).split())) if m else ""


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=UTC)).astimezone(UTC)


def extract_html(body: str, url: str) -> tuple[str, str, datetime | None]:
    """(title, markdown-текст, дата публикации) из HTML; текст пустой, если trafilatura ничего не нашла."""
    text = (
        trafilatura.extract(
            body,
            url=url,
            include_comments=False,
            include_tables=True,
            include_formatting=True,
            output_format="markdown",
        )
        or ""
    )
    title, published = "", None
    try:
        meta = trafilatura.extract_metadata(body, default_url=url)
    except Exception:  # noqa: BLE001 — метаданные не критичны, текст важнее
        meta = None
    if meta is not None:
        title = (meta.title or "").strip()
        published = _parse_date(meta.date)
    return title or html_title(body), text.strip(), published


# ---------- HTTP с кэшем и бюджетом ----------


class HttpCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def path_for(self, url: str) -> Path:
        return self.directory / (hashlib.sha1(url.encode("utf-8")).hexdigest() + ".json")

    def get(self, url: str) -> CachedResponse | None:
        p = self.path_for(url)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return CachedResponse(
            url=url,
            final_url=d.get("final_url", url),
            status=int(d.get("status", 0)),
            content_type=d.get("content_type", ""),
            body=d.get("body", ""),
            fetched_at=d.get("fetched_at", ""),
            from_cache=True,
        )

    def put(self, resp: CachedResponse) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path_for(resp.url).write_text(
            json.dumps(
                {
                    "url": resp.url,
                    "final_url": resp.final_url,
                    "status": resp.status,
                    "content_type": resp.content_type,
                    "fetched_at": resp.fetched_at,
                    "body": resp.body,
                }
            ),
            encoding="utf-8",
        )

    def drop(self, url: str) -> None:
        p = self.path_for(url)
        if p.exists():
            p.unlink()


class FetchBudgetExceeded(RuntimeError):
    pass


class KBFetcher:
    """Загрузчик с дисковым кэшем, паузами по типу запроса и лимитами (см. докстринг модуля)."""

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        offline: bool = False,
        max_total: int = MAX_TOTAL_FETCHES,
        max_reddit: int = MAX_REDDIT_REQUESTS,
        delay_html: float = DELAY_HTML_S,
        delay_reddit: float = DELAY_REDDIT_S,
        timeout: float = HTTP_TIMEOUT_S,
        sleep: Callable[[float], None] = time.sleep,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.cache = HttpCache(cache_dir or settings.cache_dir / "kb" / "http")
        self.offline = offline
        self.max_total = max_total
        self.max_reddit = max_reddit
        self.delays = {"html": delay_html, "reddit": delay_reddit}
        self.timeout = timeout
        self._sleep = sleep
        self._transport = transport
        self.network_calls = 0
        self.reddit_calls = 0
        self.cache_hits = 0
        self._last_call: dict[str, float] = {}

    # -- низкий уровень --

    def _get(self, url: str, *, kind: str, user_agent: str, accept: str) -> CachedResponse:
        cached = self.cache.get(url)
        if cached is not None:
            self.cache_hits += 1
            return cached
        if self.offline:
            raise FetchBudgetExceeded(f"offline: {url} нет в кэше")
        if self.network_calls >= self.max_total:
            raise FetchBudgetExceeded(f"лимит сетевых загрузок {self.max_total} исчерпан")
        if kind == "reddit" and self.reddit_calls >= self.max_reddit:
            raise FetchBudgetExceeded(f"лимит запросов к Reddit {self.max_reddit} исчерпан")
        last = self._last_call.get(kind)
        if last is not None:
            wait = self.delays.get(kind, DELAY_HTML_S) - (time.monotonic() - last)
            if wait > 0:
                self._sleep(wait)
        self._last_call[kind] = time.monotonic()
        self.network_calls += 1
        if kind == "reddit":
            self.reddit_calls += 1
        headers = {"User-Agent": user_agent, "Accept": accept, "Accept-Language": "en-GB,en;q=0.9"}
        with httpx.Client(
            headers=headers,
            timeout=self.timeout,
            follow_redirects=True,
            transport=self._transport,
        ) as client:
            r = client.get(url)
        resp = CachedResponse(
            url=url,
            final_url=str(r.url),
            status=r.status_code,
            content_type=r.headers.get("content-type", ""),
            body=r.text,
            fetched_at=datetime.now(UTC).isoformat(),
            from_cache=False,
        )
        self.cache.put(resp)
        return resp

    # -- по видам источников --

    def fetch(self, source: KBSource) -> Fetched:
        try:
            if source.fetch == "internal":
                return self.fetch_internal(source.url, title_hint=source.title_hint)
            if source.fetch == "reddit_json":
                return self.fetch_reddit(source.url, title_hint=source.title_hint)
            return self.fetch_html(source.url, title_hint=source.title_hint)
        except FetchBudgetExceeded as exc:
            return Fetched(url=source.canonical_url, error=f"budget: {exc}")
        except httpx.HTTPError as exc:
            return Fetched(url=source.canonical_url, error=f"http: {type(exc).__name__}: {exc}")

    def fetch_html(self, url: str, *, title_hint: str = "") -> Fetched:
        from fplcopilot.rag.ingest import normalize_url

        resp = self._get(url, kind="html", user_agent=BROWSER_USER_AGENT, accept="text/html,*/*")
        out = Fetched(
            url=normalize_url(url),
            final_url=resp.final_url,
            status=resp.status,
            from_cache=resp.from_cache,
        )
        if resp.status != 200:
            out.error = f"HTTP {resp.status}"
            return out
        if is_login_redirect(resp.final_url):
            out.error = "redirect to login"
            return out
        title, text, published = extract_html(resp.body, resp.final_url)
        out.title = title or title_hint
        out.text = text
        out.published_at = published
        out.method = "trafilatura"
        if len(text) < MIN_TEXT_CHARS:
            out.error = f"text too short ({len(text)} chars) — JS-rendered page or paywall?"
        return out

    def fetch_reddit(self, url: str, *, title_hint: str = "") -> Fetched:
        from fplcopilot.rag.ingest import normalize_url

        json_url = reddit_json_url(url)
        resp = self._get(
            json_url, kind="reddit", user_agent=REDDIT_USER_AGENT, accept="application/json"
        )
        out = Fetched(
            url=normalize_url(url),
            final_url=resp.final_url,
            status=resp.status,
            from_cache=resp.from_cache,
        )
        if is_login_redirect(resp.final_url):
            out.error = "reddit: redirect to login (lor2) — unauthenticated JSON blocked"
            self.cache.drop(json_url)  # не кэшируем отказ: следующий запуск попробует снова
            return out
        if resp.status != 200:
            out.error = f"reddit: HTTP {resp.status}"
            self.cache.drop(json_url)
            return out
        if "json" not in resp.content_type:
            out.error = f"reddit: non-JSON response ({resp.content_type or 'unknown'})"
            self.cache.drop(json_url)
            return out
        try:
            post = reddit_post_from_payload(json.loads(resp.body))
        except (ValueError, json.JSONDecodeError) as exc:
            out.error = f"reddit: {exc}"
            return out
        out.title = post.title or title_hint
        out.text = post.selftext
        out.method = "reddit_json"
        out.published_at = (
            datetime.fromtimestamp(post.created_utc, UTC) if post.created_utc else None
        )
        out.linked_urls = [
            u for u in extract_reddit_post_links(post.selftext, post.subreddit or "FantasyPL")
            if normalize_url(u) != out.url
        ]  # fmt: skip
        if len(out.text) < MIN_TEXT_CHARS:
            out.error = f"reddit: selftext too short ({len(out.text)} chars)"
        return out

    def fetch_internal(self, path: str, *, title_hint: str = "") -> Fetched:
        p = Path(path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        out = Fetched(url=path, final_url=str(p), method="internal")
        if not p.is_file():
            out.error = f"internal: файл не найден: {p}"
            return out
        text = p.read_text(encoding="utf-8").strip()
        first = next((ln for ln in text.splitlines() if ln.strip()), "")
        if first.startswith("#"):
            out.title = first.lstrip("#").strip()
            text = text[len(first) :].strip()
        out.title = title_hint or out.title or p.stem  # реестр знает, как назвать внутренний файл
        out.text = text
        out.status = 200
        out.published_at = datetime.fromtimestamp(p.stat().st_mtime, UTC)
        if len(text) < MIN_TEXT_CHARS:
            out.error = f"internal: text too short ({len(text)} chars)"
        return out
