"""Клиент OpenAI (эмбеддинги + структурированные ответы) и трейсинг LangSmith.

Трейсинг включается ТОЛЬКО непустым LANGSMITH_API_KEY в .env: тогда клиент оборачивается
`wrap_openai`, а функции пайплайна с `@traceable` пишут трейсы в проект LANGSMITH_PROJECT.
Без ключа — обычный клиент, никаких предупреждений; ключ можно добавить без правок кода.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from functools import lru_cache

import tiktoken
from openai import OpenAI

from fplcopilot.config import settings

log = logging.getLogger(__name__)

EMBED_BATCH = 100  # текстов на один запрос embeddings
_TOKENIZER = "cl100k_base"  # у text-embedding-3-* и gpt-4o-* совпадает для оценки длины


def configure_tracing() -> bool:
    """Выставляет окружение LangSmith из settings. True — трейсинг включён."""
    if settings.tracing_enabled:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key or ""
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
        return True
    # Ключа нет: глушим трейсинг, даже если в окружении случайно стоит LANGSMITH_TRACING=true,
    # иначе langsmith начнёт ругаться на отсутствующий ключ при каждом вызове.
    os.environ["LANGSMITH_TRACING"] = "false"
    return False


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    if not settings.openai_api_key:
        msg = "OPENAI_API_KEY не задан в .env — эмбеддинги и извлечение сигналов недоступны"
        raise RuntimeError(msg)
    # Ретраи (429/5xx/сеть) с backoff делает сам SDK.
    client = OpenAI(api_key=settings.openai_api_key, max_retries=3, timeout=60.0)
    if configure_tracing():
        from langsmith.wrappers import wrap_openai

        client = wrap_openai(client)
        log.info("LangSmith tracing on (project=%s)", settings.langsmith_project)
    return client


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(_TOKENIZER)


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))


def embed_texts(texts: Sequence[str], *, model: str | None = None) -> list[list[float]]:
    """Эмбеддинги для списка текстов, батчами по EMBED_BATCH; порядок сохраняется."""
    model = model or settings.rag_embedding_model
    client = get_openai_client()
    out: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = [t if t.strip() else " " for t in texts[start : start + EMBED_BATCH]]
        resp = client.embeddings.create(input=batch, model=model)
        out.extend(item.embedding for item in sorted(resp.data, key=lambda d: d.index))
    return out


def embed_query(text: str, *, model: str | None = None) -> list[float]:
    return embed_texts([text], model=model)[0]
