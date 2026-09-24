"""Промпты живут в файлах prompts/<version>/<name>.<role>.md и версионируются явно.

Это осознанно: на защите нужно показать эволюцию промпта (v1 -> v2 ...) diff-ом по файлам,
а в player_signals.prompt_version хранится, какой версией получен каждый сигнал.
Шаблоны — простые str.format: плейсхолдеры {name}, литеральные скобки — {{ }}; лишние
kwargs игнорируются, поэтому v1-шаблон можно рендерить с полями v2 (gw_calendar).

Версия по умолчанию — settings.rag_prompt_version (RAG_PROMPT_VERSION, сейчас v2; v1 сохранён
для A/B и истории: docs/EVALS.md «Prompt evolution»).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from fplcopilot.config import settings

PROMPTS_DIR = Path(__file__).resolve().parent
DEFAULT_VERSION = settings.rag_prompt_version


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    system: str
    user_template: str

    def render_user(self, **kwargs: object) -> str:
        return self.user_template.format(**kwargs)


@lru_cache(maxsize=16)
def load_prompt(name: str, version: str | None = None) -> Prompt:
    version = version or DEFAULT_VERSION
    base = PROMPTS_DIR / version
    if not base.is_dir():
        raise ValueError(f"prompt version {version!r} не найдена; есть: {available_versions(name)}")
    system = (base / f"{name}.system.md").read_text(encoding="utf-8").strip()
    user = (base / f"{name}.user.md").read_text(encoding="utf-8").strip()
    return Prompt(name=name, version=version, system=system, user_template=user)


def available_versions(name: str) -> list[str]:
    return sorted(p.parent.name for p in PROMPTS_DIR.glob(f"v*/{name}.system.md"))
