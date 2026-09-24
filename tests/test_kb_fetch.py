"""Загрузчик KB без сети: reddit JSON -> текст поста, ссылки хаба, HTML -> markdown, кэш и бюджет
(httpx.MockTransport), internal-файлы."""

import json
from datetime import UTC, date, datetime

import httpx
import pytest

from fplcopilot.rag.kb.fetch import (
    MIN_TEXT_CHARS,
    KBFetcher,
    clean_reddit_markdown,
    extract_html,
    extract_reddit_post_links,
    is_login_redirect,
    reddit_json_url,
    reddit_post_from_payload,
)
from fplcopilot.rag.kb.registry import KBSource

SELFTEXT = (
    "**Part I – The Basics**\n\n"
    "Welcome to r/FantasyPL! This guide covers the rules.&amp;#x200B;\n\n"
    "- You get **1 free transfer** per week, banked up to 5.\n"
    "- Hits cost -4 points. See [Part II](https://www.reddit.com/r/FantasyPL/comments/iaorcf/) "
    "and [Part III](/r/FantasyPL/comments/ib2xyz/part_iii/).\n"
    "- Ignore [this](https://www.reddit.com/r/soccer/comments/zzzzzz/) and "
    "https://www.reddit.com/r/FantasyPL/comments/iaorcf/ again.\n\n"
    "![img](https://i.redd.it/abc.png)\n\n"
    "Have fun &amp; good luck.\n"
)

PAYLOAD = [
    {
        "kind": "Listing",
        "data": {
            "children": [
                {
                    "kind": "t3",
                    "data": {
                        "title": "Beginners&#39; Guide Part I",
                        "selftext": SELFTEXT,
                        "created_utc": 1597510800.0,
                        "permalink": "/r/FantasyPL/comments/ia4vpk/beginners_guide_part_i/",
                        "subreddit": "FantasyPL",
                    },
                }
            ]
        },
    },
    {
        "kind": "Listing",
        "data": {
            "children": [
                {"kind": "t1", "data": {"body": "COMMENT: bet on Haaland at 3/1 — ignore me"}}
            ]
        },
    },
]


def test_reddit_json_url_variants():
    expected = "https://old.reddit.com/r/FantasyPL/comments/ia4vpk/.json"
    assert reddit_json_url("https://www.reddit.com/r/FantasyPL/comments/ia4vpk/") == expected
    assert reddit_json_url("https://www.reddit.com/r/FantasyPL/comments/ia4vpk") == expected
    assert reddit_json_url("https://old.reddit.com/r/FantasyPL/comments/ia4vpk/.json") == expected
    assert (
        reddit_json_url("https://www.reddit.com/r/FantasyPL/comments/ia4vpk/title_slug/?x=1")
        == "https://old.reddit.com/r/FantasyPL/comments/ia4vpk/title_slug/.json"
    )


def test_reddit_post_from_payload_uses_selftext_only_and_cleans_markdown():
    post = reddit_post_from_payload(PAYLOAD)
    assert post.title == "Beginners' Guide Part I"
    assert post.subreddit == "FantasyPL"
    assert datetime.fromtimestamp(post.created_utc, UTC).year == 2020
    text = post.selftext
    assert "COMMENT" not in text and "bet on" not in text  # комментарии не берём
    assert "\u200b" not in text and "&amp;" not in text and "#x200B" not in text
    assert "Have fun & good luck." in text
    assert "**" not in text and "1 free transfer" in text  # жирный -> текст
    assert "![img]" not in text and "i.redd.it" not in text  # картинки долой
    assert "See Part II and Part III." in text  # ссылки -> текст ссылки
    assert text.startswith("Part I – The Basics")  # заголовок-строка остаётся первой строкой


def test_reddit_post_links_are_canonical_unique_and_same_subreddit():
    links = extract_reddit_post_links(SELFTEXT, "FantasyPL")
    assert links == [
        "https://www.reddit.com/r/FantasyPL/comments/iaorcf/",
        "https://www.reddit.com/r/FantasyPL/comments/ib2xyz/",
    ]
    assert extract_reddit_post_links(SELFTEXT, "soccer") == [
        "https://www.reddit.com/r/soccer/comments/zzzzzz/"
    ]


def test_reddit_payload_shape_errors_and_login_redirect_detection():
    with pytest.raises(ValueError, match="структура"):
        reddit_post_from_payload({"data": {"children": []}})
    assert is_login_redirect("https://old.reddit.com/login/?reason=lor2&dest=...")
    assert not is_login_redirect("https://old.reddit.com/r/FantasyPL/comments/ia4vpk/.json")
    assert clean_reddit_markdown("a  \n\n\n\nb") == "a\n\nb"


HTML = """<!doctype html><html><head><title>FPL Chips Guide | ExampleFPL</title>
<meta property="og:title" content="FPL Chips Guide: When to Play Each Chip">
<meta property="article:published_time" content="2026-03-14T10:00:00+00:00">
</head><body><nav><a href="/">Home</a> <a href="/blog">Blog</a></nav>
<article>
<h1>FPL Chips Guide: When to Play Each Chip</h1>
<p>Every FPL manager gets eight chips per season, two of each type, one for each half of the
campaign. The first set must be used before the Gameweek 19 deadline and a fresh set unlocks at
Gameweek 20. Using them at the right time can swing your season by hundreds of thousands of
rank places, while using them at the wrong time wastes their potential entirely.</p>
<h2>Wildcard</h2>
<p>The Wildcard lets you make unlimited free transfers for one Gameweek and the changes persist.
Most guides suggest holding the first Wildcard until you have three or more problems in the
squad, because a Wildcard played on a whim rarely beats a couple of well-timed free transfers.</p>
<h2>Free Hit</h2>
<p>The Free Hit is a one-week squad: at the next deadline your team reverts to how it was. It is
played when confirming transfers and cannot be cancelled once confirmed, so plan it carefully
around blank and double Gameweeks rather than reacting to a single injury.</p>
</article><footer>Copyright ExampleFPL</footer></body></html>"""


def test_extract_html_returns_title_markdown_headings_and_date():
    title, text, published = extract_html(HTML, "https://example.com/blog/chips")
    assert title == "FPL Chips Guide: When to Play Each Chip"
    assert published is not None and published.date() == date(
        2026, 3, 14
    )  # trafilatura: только дата
    assert published.tzinfo is not None
    assert "## Wildcard" in text and "## Free Hit" in text  # заголовки сохранены для chunking
    assert "cannot be cancelled once confirmed" in text
    assert "Copyright" not in text and "Home" not in text.split("\n")[0]


def _transport(calls: list[str]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        url = str(request.url)
        if "login" in url:
            return httpx.Response(200, text="<html>Welcome to Reddit</html>")
        if "old.reddit.com" in url:
            return httpx.Response(
                302, headers={"location": "https://old.reddit.com/login/?reason=lor2"}
            )
        if "short" in url:
            return httpx.Response(200, text="<html><body><p>tiny</p></body></html>")
        if "missing" in url:
            return httpx.Response(404, text="nope")
        return httpx.Response(200, text=HTML, headers={"content-type": "text/html"})

    return httpx.MockTransport(handler)


def test_fetcher_caches_html_and_reports_failures(tmp_path):
    calls: list[str] = []
    f = KBFetcher(cache_dir=tmp_path, sleep=lambda s: None, transport=_transport(calls))
    src = KBSource("guide", "example", "https://example.com/blog/chips/", "Chips", ("chips",))
    first = f.fetch(src)
    assert first.ok and first.method == "trafilatura" and not first.from_cache
    assert first.url == "https://example.com/blog/chips"  # канонический url без слэша
    assert first.chars >= MIN_TEXT_CHARS
    second = f.fetch(src)
    assert second.ok and second.from_cache and f.network_calls == 1 and f.cache_hits == 1
    assert len(calls) == 1

    short = f.fetch(KBSource("s", "example", "https://example.com/short", "S", ("chips",)))
    assert not short.ok and "too short" in (short.error or "")
    missing = f.fetch(KBSource("m", "example", "https://example.com/missing", "M", ("chips",)))
    assert not missing.ok and missing.error == "HTTP 404" and missing.status == 404


def test_fetcher_reddit_login_redirect_is_an_error_and_not_cached(tmp_path):
    calls: list[str] = []
    f = KBFetcher(cache_dir=tmp_path, sleep=lambda s: None, transport=_transport(calls))
    src = KBSource(
        "r", "reddit", "https://www.reddit.com/r/FantasyPL/comments/ia4vpk/", "R", ("beginner",),
        fetch="reddit_json",
    )  # fmt: skip
    res = f.fetch(src)
    assert not res.ok and "login" in (res.error or "")
    assert f.reddit_calls == 1
    assert not any(tmp_path.iterdir())  # отказ не кэшируем — следующий запуск попробует снова
    f.fetch(src)
    assert f.reddit_calls == 2  # без кэша — новый запрос


def test_fetcher_budgets_and_offline_mode(tmp_path):
    calls: list[str] = []
    f = KBFetcher(
        cache_dir=tmp_path, sleep=lambda s: None, transport=_transport(calls), max_total=1
    )
    ok = f.fetch(KBSource("a", "example", "https://example.com/a", "A", ("chips",)))
    assert ok.ok
    over = f.fetch(KBSource("b", "example", "https://example.com/b", "B", ("chips",)))
    assert not over.ok and "budget" in (over.error or "") and len(calls) == 1

    offline = KBFetcher(cache_dir=tmp_path, offline=True, transport=_transport(calls))
    cached = offline.fetch(KBSource("a", "example", "https://example.com/a", "A", ("chips",)))
    assert cached.ok and cached.from_cache
    fresh = offline.fetch(KBSource("c", "example", "https://example.com/c", "C", ("chips",)))
    assert not fresh.ok and "offline" in (fresh.error or "") and len(calls) == 1


def test_fetch_internal_prefers_registry_title_and_strips_h1(tmp_path):
    path = tmp_path / "rules.md"
    body = "\n".join(f"- Rule {i}: something about FPL transfers and chips." for i in range(20))
    path.write_text(f"# Digest (v1, built-in)\n\n{body}\n", encoding="utf-8")
    f = KBFetcher(cache_dir=tmp_path / "cache", offline=True)
    res = f.fetch(
        KBSource("i", "internal", str(path), "Rules digest", ("rules",), fetch="internal")
    )
    assert res.ok and res.method == "internal" and res.title == "Rules digest"
    assert not res.text.startswith("#") and "Rule 0" in res.text
    plain = f.fetch_internal(str(path))
    assert plain.title == "Digest (v1, built-in)"  # без title_hint — из H1
    missing = f.fetch(
        KBSource("x", "internal", str(tmp_path / "no.md"), "X", ("rules",), fetch="internal")
    )
    assert not missing.ok and "не найден" in (missing.error or "")


def test_payload_roundtrip_through_json_matches_fixture():
    # то, что придёт с old.reddit — строка JSON; убеждаемся, что парсинг устойчив к сериализации
    post = reddit_post_from_payload(json.loads(json.dumps(PAYLOAD)))
    assert post.permalink.endswith("/beginners_guide_part_i/")
