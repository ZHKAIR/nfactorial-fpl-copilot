"""Одно реальное извлечение через OpenAI (gpt-4o-mini) поверх живого индекса.

Запуск: uv run pytest -q -m llm. Исключён из дефолтного `-m "not network"`; скипается без
OPENAI_API_KEY или без БД. Проверяет инварианты, которые обязан обеспечивать валидатор.
"""

from datetime import UTC, datetime

import pytest
from sqlalchemy import text

from fplcopilot.config import settings
from fplcopilot.data import FPLClient
from fplcopilot.db import ping, session_scope
from fplcopilot.rag.extract import ZERO_MINUTES_AVAILABILITY, extract_signal, quote_in_text
from fplcopilot.rag.retrieve import find_player

pytestmark = [pytest.mark.llm, pytest.mark.network]


@pytest.fixture(scope="module", autouse=True)
def _require_env():
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY не задан")
    if not ping():
        pytest.skip("Postgres недоступен")
    with session_scope() as s:
        n = s.execute(text("SELECT count(*) FROM news_chunks WHERE embedding IS NOT NULL")).scalar()
    if not n:
        pytest.skip("индекс пуст: uv run python -m fplcopilot.rag.index --rebuild-missing")


def test_real_extraction_respects_validation_invariants():
    bs = FPLClient().bootstrap()
    player = find_player(bs, "Haaland")
    sig = extract_signal(player.id, datetime.now(UTC), save=False, bs=bs)

    assert sig.player_id == player.id
    assert sig.prompt_version == settings.rag_prompt_version  # дефолт v2 (RAG_PROMPT_VERSION)
    assert sig.model == settings.rag_llm_model
    if sig.abstained:  # нет чанков про игрока -> unknown без вызова LLM
        assert sig.availability == "unknown" and sig.confidence == 0 and not sig.evidence
    assert 0 <= sig.start_probability <= 1 and 0 <= sig.confidence <= 1
    assert 0 <= sig.expected_minutes <= 90
    if sig.availability in ZERO_MINUTES_AVAILABILITY:
        assert sig.start_probability == 0 and sig.expected_minutes == 0
    if not sig.evidence:
        assert sig.availability == "unknown" and sig.confidence == 0
    assert {e.chunk_id for e in sig.evidence} <= set(sig.retrieved_chunk_ids)

    with session_scope() as s:
        texts = dict(
            s.execute(
                text("SELECT id, text FROM news_chunks WHERE id = ANY(:ids)"),
                {"ids": sig.retrieved_chunk_ids},
            ).all()
        )
    for ev in sig.evidence:
        assert quote_in_text(ev.quote, texts[ev.chunk_id])  # цитаты — реальные подстроки
