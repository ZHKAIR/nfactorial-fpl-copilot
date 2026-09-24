#!/bin/sh
# Entrypoint контейнера: дождаться Postgres, применить миграции, запустить команду
# (по умолчанию — Streamlit; сервис ingest передаёт свою команду через compose `command:`).
set -eu

MIGRATE_RETRIES="${MIGRATE_RETRIES:-30}"
MIGRATE_DELAY="${MIGRATE_DELAY:-2}"

i=1
until python scripts/migrate.py; do
  if [ "$i" -ge "$MIGRATE_RETRIES" ]; then
    echo "entrypoint: migrations failed after $i attempts, giving up" >&2
    exit 1
  fi
  echo "entrypoint: DB not ready (attempt $i/$MIGRATE_RETRIES), retrying in ${MIGRATE_DELAY}s…" >&2
  i=$((i + 1))
  sleep "$MIGRATE_DELAY"
done

exec "$@"
