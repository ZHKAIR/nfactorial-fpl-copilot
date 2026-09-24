"""Settings (config.py) и .env.example: пустые «KEY=» не роняют запуск.

Пользователь копирует .env.example в .env как есть: числовые ключи вроде FPL_MANAGER_ID= пустые.
Пустое значение = дефолт поля, для необязательных полей — None.
"""

from __future__ import annotations

import re

import pytest

from fplcopilot.config import PROJECT_ROOT, Settings

ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
KEYS = re.findall(
    r"^#?\s*([A-Z][A-Z0-9_]+)=", ENV_EXAMPLE.read_text(encoding="utf-8"), re.MULTILINE
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)


def test_env_example_copied_as_is_loads(tmp_path):
    env = tmp_path / ".env"
    env.write_text(ENV_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    s = Settings(_env_file=env)
    defaults = Settings(_env_file=None)
    assert s.fpl_manager_id is None and s.openai_api_key is None and not s.tracing_enabled
    assert s.database_url == defaults.database_url  # совпадает с дефолтами docker-compose.yml
    assert s.fpl_cache_ttl == 3600 and s.xpts_fixture_provider == "team_rating"


def test_every_env_example_key_left_empty_means_default(tmp_path):
    env = tmp_path / ".env"
    env.write_text("".join(f"{k}=\n" for k in KEYS), encoding="utf-8")
    s, defaults = Settings(_env_file=env), Settings(_env_file=None)
    fields = [k.lower() for k in KEYS if k.lower() in Settings.model_fields]
    assert len(fields) >= 60
    differ = {f: getattr(s, f) for f in fields if getattr(s, f) != getattr(defaults, f)}
    assert differ == {"rag_per_article_cap": None}  # .env.example: «пусто = без cap»


def test_empty_process_env_and_real_values(monkeypatch):
    monkeypatch.setenv("FPL_MANAGER_ID", "")
    monkeypatch.setenv("NEWS_BACKFILL_SINCE", " ")
    monkeypatch.setenv("AGENT_PROMPT_VERSION", "")
    s = Settings(_env_file=None)
    assert s.fpl_manager_id is None and s.agent_prompt_version == "v4"
    assert s.news_backfill_since == Settings.model_fields["news_backfill_since"].default
    monkeypatch.setenv("FPL_MANAGER_ID", "895045")
    monkeypatch.setenv("RAG_PER_ARTICLE_CAP", "3")
    s = Settings(_env_file=None)
    assert s.fpl_manager_id == 895045 and s.rag_per_article_cap == 3
