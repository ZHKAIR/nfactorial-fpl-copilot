"""Подключение к Postgres: SQLAlchemy 2.x поверх psycopg 3.

Схема управляется plain-SQL миграциями (src/fplcopilot/migrations/*.sql,
применяются scripts/migrate.py), ORM-моделей нет — запросы пишем через text().
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from fplcopilot.config import settings


def sqlalchemy_url(url: str) -> str:
    """`postgresql://...` -> `postgresql+psycopg://...` (иначе SQLAlchemy ищет psycopg2)."""
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url.removeprefix(prefix)
    return url


@lru_cache(maxsize=4)
def get_engine(url: str | None = None) -> Engine:
    return create_engine(sqlalchemy_url(url or settings.database_url), pool_pre_ping=True)


def session_factory(url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(url), expire_on_commit=False)


@contextmanager
def session_scope(url: str | None = None) -> Iterator[Session]:
    """Одна транзакция: commit при успехе, rollback при исключении."""
    session = session_factory(url)()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


def ping(url: str | None = None) -> bool:
    """True, если БД отвечает. Используется тестами, чтобы аккуратно скипаться без Postgres."""
    try:
        with get_engine(url).connect() as conn:
            conn.execute(text("SELECT 1"))
    except SQLAlchemyError:
        return False
    return True
