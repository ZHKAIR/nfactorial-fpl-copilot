#!/bin/sh
# Дамп локальной БД FPL Copilot из контейнера (только чтение): pg_dump -Fc -> backups/.
#
#   scripts/db_dump.sh                       # контейнер fpl-copilot-db (docker-compose.yml)
#   DB_CONTAINER=other-db scripts/db_dump.sh # другой контейнер
#
# Файл: backups/fpl-YYYYmmdd-HHMM.dump (каталог backups/ в .gitignore). Перенос на сервер и
# восстановление — scripts/db_restore.sh, docs/deploy.md.
set -eu

DB_CONTAINER="${DB_CONTAINER:-fpl-copilot-db}"
PGUSER="${POSTGRES_USER:-fpl}"
PGDB="${POSTGRES_DB:-fpl}"
OUT_DIR="${OUT_DIR:-backups}"

mkdir -p "$OUT_DIR"
out="$OUT_DIR/fpl-$(date +%Y%m%d-%H%M).dump"
echo "db_dump: $DB_CONTAINER ($PGDB) -> $out"
docker exec "$DB_CONTAINER" pg_dump -U "$PGUSER" -d "$PGDB" -Fc --no-owner > "$out.part"
mv "$out.part" "$out"
ls -lh "$out"
