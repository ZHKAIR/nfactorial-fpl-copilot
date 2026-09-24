"""Реестр источников KB — `src/fplcopilot/rag/sources_kb.yaml` (проверка от 17.09.2026 в docs/strategy_kb.md).

Каждая запись: name (слаг), source (издатель — пишется в kb_docs.source), url, title_hint,
tags ⊆ KNOWN_TAGS, fetch ∈ FETCH_KINDS, enabled, note (почему выключен / особенности),
follow_links (только reddit_json: сколько ссылок на другие посты сабреддита подтянуть из
текста — для хаба «talking points»). Валидация — чистые функции без сети (unit-тесты).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from fplcopilot.rag.ingest import normalize_url

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "sources_kb.yaml"

FetchKind = Literal["html", "reddit_json", "internal"]
FETCH_KINDS: tuple[FetchKind, ...] = ("html", "reddit_json", "internal")
KNOWN_TAGS: frozenset[str] = frozenset(
    {
        "rules",
        "chips",
        "transfers",
        "hits",
        "captaincy",
        "structure",
        "rank",
        "prices",
        "fixtures",
        "defcon",
        "beginner",
    }
)
# Теги, означающие официальные правила (а не мнение сообщества) — см. docs/strategy_kb.md
RULES_TAG = "rules"


class RegistryError(ValueError):
    """Ошибка в реестре источников — это ошибка данных, а не кода."""


@dataclass(frozen=True)
class KBSource:
    name: str
    source: str
    url: str
    title_hint: str
    tags: tuple[str, ...]
    fetch: FetchKind = "html"
    enabled: bool = True
    note: str = ""
    follow_links: int = 0  # reddit_json: сколько ссылок на посты того же сабреддита подтянуть

    @property
    def is_internal(self) -> bool:
        return self.fetch == "internal"

    @property
    def canonical_url(self) -> str:
        """Ключ дедупликации (kb_docs.url UNIQUE): для http — normalize_url, для internal — путь."""
        return self.url if self.is_internal else normalize_url(self.url)


def _as_list(value: Any, what: str, name: str) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise RegistryError(f"{name}: {what} должен быть списком строк")
    return value


def parse_source(raw: dict[str, Any]) -> KBSource:
    name = str(raw.get("name") or "").strip()
    if not name:
        raise RegistryError("источник без name")
    url = str(raw.get("url") or "").strip()
    fetch = raw.get("fetch", "html")
    if fetch not in FETCH_KINDS:
        raise RegistryError(f"{name}: fetch должен быть одним из {FETCH_KINDS}, получен {fetch!r}")
    if not url:
        raise RegistryError(f"{name}: пустой url")
    if fetch != "internal" and not url.startswith(("http://", "https://")):
        raise RegistryError(f"{name}: url должен начинаться с http(s)://, получен {url!r}")
    if fetch == "reddit_json" and "reddit.com/" not in url:
        raise RegistryError(f"{name}: reddit_json ожидает ссылку на reddit.com, получен {url!r}")
    tags = tuple(_as_list(raw.get("tags"), "tags", name))
    if not tags:
        raise RegistryError(f"{name}: нужен хотя бы один тег из {sorted(KNOWN_TAGS)}")
    unknown = sorted(set(tags) - KNOWN_TAGS)
    if unknown:
        raise RegistryError(f"{name}: неизвестные теги {unknown}; допустимы {sorted(KNOWN_TAGS)}")
    if len(set(tags)) != len(tags):
        raise RegistryError(f"{name}: теги повторяются")
    title_hint = str(raw.get("title_hint") or "").strip()
    if not title_hint:
        raise RegistryError(f"{name}: пустой title_hint")
    source = str(raw.get("source") or "").strip()
    if not source:
        raise RegistryError(f"{name}: пустой source (издатель)")
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise RegistryError(f"{name}: enabled должен быть bool")
    follow = raw.get("follow_links", 0)
    if not isinstance(follow, int) or follow < 0:
        raise RegistryError(f"{name}: follow_links должен быть целым >= 0")
    if follow and fetch != "reddit_json":
        raise RegistryError(f"{name}: follow_links поддерживается только для reddit_json")
    return KBSource(
        name=name,
        source=source,
        url=url,
        title_hint=title_hint,
        tags=tags,
        fetch=fetch,
        enabled=enabled,
        note=str(raw.get("note") or "").strip(),
        follow_links=follow,
    )


def validate_sources(sources: Iterable[KBSource]) -> list[KBSource]:
    """Уникальность name и канонического url (после normalize_url). Возвращает список."""
    out = list(sources)
    names: dict[str, KBSource] = {}
    urls: dict[str, KBSource] = {}
    for s in out:
        if s.name in names:
            raise RegistryError(f"дубликат name: {s.name}")
        names[s.name] = s
        key = s.canonical_url
        if key in urls:
            raise RegistryError(f"дубликат url: {s.url} ({s.name} и {urls[key].name})")
        urls[key] = s
    return out


def parse_registry(data: Any) -> list[KBSource]:
    """Разобранный YAML (dict с ключом sources или сразу список) -> валидный список источников."""
    if isinstance(data, dict):
        declared = data.get("tags")
        if declared is not None:
            extra = sorted(set(_as_list(declared, "tags", "registry")) - KNOWN_TAGS)
            if extra:
                raise RegistryError(f"в registry.tags неизвестные теги {extra}")
        data = data.get("sources")
    if not isinstance(data, list):
        raise RegistryError("реестр должен содержать список sources")
    return validate_sources(parse_source(raw) for raw in data)


def load_registry(path: Path | None = None) -> list[KBSource]:
    path = path or REGISTRY_PATH
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return parse_registry(data)


def enabled_sources(sources: Iterable[KBSource] | None = None) -> list[KBSource]:
    return [s for s in (sources if sources is not None else load_registry()) if s.enabled]


@dataclass
class RegistrySummary:
    total: int = 0
    enabled: int = 0
    by_fetch: dict[str, int] = field(default_factory=dict)
    by_tag: dict[str, int] = field(default_factory=dict)


def summarize(sources: Iterable[KBSource]) -> RegistrySummary:
    s = RegistrySummary()
    for src in sources:
        s.total += 1
        if src.enabled:
            s.enabled += 1
            s.by_fetch[src.fetch] = s.by_fetch.get(src.fetch, 0) + 1
            for t in src.tags:
                s.by_tag[t] = s.by_tag.get(t, 0) + 1
    return s
