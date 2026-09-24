#!/bin/sh
# Восстановить дамп (pg_dump -Fc) в базу прод-стека docker-compose.prod.yml.
#
#   scripts/db_restore.sh backups/fpl-20260925-0100.dump
#   COMPOSE_PROJECT=fplc-prodtest scripts/db_restore.sh <dump>   # другой проект compose
#
# Поднимает только db (если ещё не запущена), ждёт готовности и делает pg_restore --clean:
# таблицы из дампа заменяют существующие. Приложение после восстановления лучше перезапустить:
#   docker compose -f docker-compose.prod.yml up -d
set -eu

DUMP="${1:?укажите файл дампа: scripts/db_restore.sh backups/<file>.dump}"
[ -f "$DUMP" ] || { echo "db_restore: нет файла $DUMP" >&2; exit 1; }

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.prod.yml}"
set -- -f "$COMPOSE_FILE"
[ -n "${COMPOSE_PROJECT:-}" ] && set -- "$@" -p "$COMPOSE_PROJECT"
[ -n "${ENV_FILE:-}" ] && set -- "$@" --env-file "$ENV_FILE"

# POSTGRES_USER / POSTGRES_DB: окружение -> .env (без source: значения могут быть любыми) -> fpl
envval() { grep -E "^$1=" "${ENV_FILE:-.env}" 2>/dev/null | tail -n 1 | cut -d= -f2-; }
PGUSER="${POSTGRES_USER:-$(envval POSTGRES_USER)}"
PGUSER="${PGUSER:-fpl}"
PGDB="${POSTGRES_DB:-$(envval POSTGRES_DB)}"
PGDB="${PGDB:-fpl}"

docker compose "$@" up -d db
echo "db_restore: жду готовности Postgres…"
i=0
until docker compose "$@" exec -T db pg_isready -U "$PGUSER" -d "$PGDB" >/dev/null 2>&1; do
  i=$((i + 1))
  [ "$i" -ge 60 ] && { echo "db_restore: Postgres не поднялся" >&2; exit 1; }
  sleep 2
done

echo "db_restore: $DUMP -> $PGDB"
docker compose "$@" exec -T db pg_restore -U "$PGUSER" -d "$PGDB" --clean --if-exists \
  --no-owner --no-privileges --exit-on-error < "$DUMP"
docker compose "$@" exec -T db psql -U "$PGUSER" -d "$PGDB" -Atc \
  "select 'news_articles: ' || count(*) from news_articles union all
   select 'player_signals: ' || count(*) from player_signals union all
   select 'player_gw_history: ' || count(*) from player_gw_history"
echo "db_restore: готово"
