"""Реестр источников KB (sources_kb.yaml): валидация без сети — уникальные url, известные теги."""

import pytest

from fplcopilot.rag.kb.registry import (
    FETCH_KINDS,
    KNOWN_TAGS,
    RegistryError,
    enabled_sources,
    load_registry,
    parse_registry,
    summarize,
)


def entry(**overrides):
    base = {
        "name": "x",
        "source": "blog",
        "url": "https://example.com/guide",
        "title_hint": "Guide",
        "tags": ["chips"],
        "fetch": "html",
        "enabled": True,
    }
    return base | overrides


def test_real_registry_loads_and_is_within_spec():
    sources = load_registry()
    enabled = enabled_sources(sources)
    assert 30 <= len(enabled) <= 50
    assert len({s.canonical_url for s in sources}) == len(sources)
    assert len({s.name for s in sources}) == len(sources)
    for s in sources:
        assert s.tags and set(s.tags) <= KNOWN_TAGS, s.name
        assert s.fetch in FETCH_KINDS
        assert s.title_hint and s.source
        if not s.enabled:
            assert s.note, f"{s.name}: выключенный источник должен объяснять почему"
    # официальные правила и внутренний дайджест присутствуют и помечены rules
    assert any(s.source == "premierleague" and "rules" in s.tags and s.enabled for s in enabled)
    assert any(s.fetch == "internal" and "rules" in s.tags for s in enabled)
    # тег rules — только официальные страницы и внутренний файл (промпт считает их авторитетом)
    assert all(s.source in {"premierleague", "internal"} for s in enabled if "rules" in s.tags)
    summary = summarize(sources)
    assert summary.enabled == len(enabled) and summary.by_tag["chips"] >= 5


def test_duplicate_url_after_normalization_is_rejected():
    rows = [
        entry(name="a", url="https://example.com/guide/?utm_source=x"),
        entry(name="b", url="https://EXAMPLE.com/guide"),
    ]
    with pytest.raises(RegistryError, match="дубликат url"):
        parse_registry({"sources": rows})


def test_duplicate_name_is_rejected():
    rows = [entry(name="a"), entry(name="a", url="https://example.com/other")]
    with pytest.raises(RegistryError, match="дубликат name"):
        parse_registry(rows)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"tags": ["chips", "gambling"]}, "неизвестные теги"),
        ({"tags": []}, "хотя бы один тег"),
        ({"tags": ["chips", "chips"]}, "повторяются"),
        ({"fetch": "rss"}, "fetch должен быть"),
        ({"url": "ftp://example.com/x"}, "http"),
        ({"url": ""}, "пустой url"),
        ({"title_hint": ""}, "title_hint"),
        ({"source": ""}, "source"),
        ({"enabled": "yes"}, "bool"),
        ({"follow_links": 3}, "только для reddit_json"),
        ({"fetch": "reddit_json", "url": "https://example.com/post"}, "reddit.com"),
        ({"follow_links": -1}, "follow_links"),
    ],
)
def test_invalid_entries_are_rejected(override, match):
    with pytest.raises(RegistryError, match=match):
        parse_registry([entry(**override)])


def test_registry_level_tag_list_is_validated_and_internal_paths_allowed():
    with pytest.raises(RegistryError, match="registry.tags"):
        parse_registry({"tags": ["chips", "nope"], "sources": [entry()]})
    (src,) = parse_registry(
        {
            "tags": sorted(KNOWN_TAGS),
            "sources": [entry(fetch="internal", url="docs/rules.md", tags=["rules"])],
        }
    )
    assert src.is_internal and src.canonical_url == "docs/rules.md"
    (reddit,) = parse_registry(
        [
            entry(
                fetch="reddit_json",
                url="https://www.reddit.com/r/FantasyPL/comments/ibh7xd/",
                follow_links=8,
                enabled=False,
                note="blocked",
            )
        ]
    )
    assert reddit.follow_links == 8 and not reddit.enabled
    assert reddit.canonical_url == "https://www.reddit.com/r/FantasyPL/comments/ibh7xd"
