"""Чистые функции ингеста: нормализация URL, даты, хэш, HTML -> текст (без сети и БД)."""

import time
from datetime import UTC, datetime

from fplcopilot.rag.ingest import (
    content_hash,
    html_to_text,
    normalize_url,
    parse_published,
    strip_publisher_suffix,
)


def test_normalize_url_strips_tracking_fragment_and_sorts_query():
    u = normalize_url(
        "HTTPS://www.BBC.co.uk/sport/football/articles/abc/?b=2&utm_source=x&at_medium=RSS&a=1#top"
    )
    assert u == "https://www.bbc.co.uk/sport/football/articles/abc?a=1&b=2"


def test_normalize_url_is_idempotent_and_keeps_meaningful_params():
    u = "https://news.google.com/rss/articles/CBMi?oc=5"
    once = normalize_url(u)
    assert once == "https://news.google.com/rss/articles/CBMi"
    assert normalize_url(once) == once
    assert normalize_url("https://x.com/a?page=2&utm_campaign=z") == "https://x.com/a?page=2"
    assert normalize_url("https://x.com:443/") == "https://x.com/"
    assert normalize_url("fpl://player/12/news/2026-09-15T19:30:09+00:00").startswith("fpl://")


def test_parse_published_prefers_struct_time_and_is_utc():
    st = time.gmtime(1789000000)  # struct_time из feedparser (UTC)
    dt = parse_published({"published_parsed": st})
    assert dt == datetime.fromtimestamp(1789000000, tz=UTC)
    assert dt.tzinfo is UTC
    # struct_time важнее строкового поля, даже если строка с другим смещением
    both = parse_published({"published_parsed": st, "published": "Wed, 16 Sep 2026 18:30:00 +0100"})
    assert both == dt


def test_parse_published_falls_back_to_strings():
    assert parse_published({"published": "Wed, 16 Sep 2026 18:30:00 +0100"}) == datetime(
        2026, 9, 16, 17, 30, tzinfo=UTC
    )
    assert parse_published({"updated": "2026-09-16T18:30:00Z"}) == datetime(
        2026, 9, 16, 18, 30, tzinfo=UTC
    )
    assert parse_published({"published": "2026-09-16 18:30:00"}) == datetime(
        2026, 9, 16, 18, 30, tzinfo=UTC
    )  # naive -> считаем UTC


def test_parse_published_returns_none_without_date():
    assert parse_published({"title": "no date here"}) is None
    assert parse_published({"published": "not a date at all"}) is None


def test_parse_published_rejects_implausible_dates():
    assert parse_published({"published_parsed": time.gmtime(0)}) is None  # 1970 из Google News
    assert parse_published({"published": "2099-01-01T00:00:00Z"}) is None


def test_content_hash_is_stable_to_whitespace_and_case():
    a = content_hash("Saka  injury", "Bukayo Saka is\nout for   two weeks.")
    b = content_hash("saka injury", "bukayo saka is out for two weeks.")
    assert a == b
    assert len(a) == 64
    assert a != content_hash("Saka injury", "Bukayo Saka is out for three weeks.")


def test_html_to_text_and_publisher_suffix():
    assert html_to_text("<p>Doku &amp; Foden <b>fit</b></p>\n<br/>") == "Doku & Foden fit"
    assert html_to_text(None) == ""
    assert strip_publisher_suffix("Joao Pedro out with injury - Reuters", "Reuters") == (
        "Joao Pedro out with injury"
    )
    assert strip_publisher_suffix("Title - Other", "Reuters") == "Title - Other"
    assert strip_publisher_suffix("Title", None) == "Title"
