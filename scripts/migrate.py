"""Применяет SQL-миграции из src/fplcopilot/migrations/*.sql по порядку.

Запуск:
    uv run python scripts/migrate.py

Миграции обязаны быть идемпотентными (CREATE ... IF NOT EXISTS): скрипт применяет
все файлы при каждом запуске, а в schema_migrations лишь фиксирует факт применения.
Alembic не используем осознанно — схема маленькая, а времени до дедлайна мало.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import text

from fplcopilot.db import get_engine

log = logging.getLogger("migrate")

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "src" / "fplcopilot" / "migrations"

TRACKING_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename   text        PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""


def migrate(migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
    files = sorted(migrations_dir.glob("*.sql"))
    engine = get_engine()
    applied: list[str] = []
    with engine.begin() as conn:
        conn.execute(text(TRACKING_TABLE))
        seen = {row[0] for row in conn.execute(text("SELECT filename FROM schema_migrations"))}
        for path in files:
            conn.exec_driver_sql(path.read_text(encoding="utf-8"))
            if path.name not in seen:
                conn.execute(
                    text("INSERT INTO schema_migrations (filename) VALUES (:f)"), {"f": path.name}
                )
            log.info("%s %s", "re-applied" if path.name in seen else "applied", path.name)
            applied.append(path.name)
    return applied


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    applied = migrate()
    print(f"OK: {len(applied)} migration(s): {', '.join(applied)}")


if __name__ == "__main__":
    main()
