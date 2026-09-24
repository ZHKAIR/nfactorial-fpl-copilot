#!/bin/sh
# Ежедневный бэкап прод-БД на сервере (для cron): pg_dump -Fc -> backups/, хранить N дней.
#
#   crontab -e
#   15 3 * * * cd /home/deploy/fpl-copilot && scripts/db_backup.sh >> backups/backup.log 2>&1
set -eu

KEEP_DAYS="${KEEP_DAYS:-14}"
OUT_DIR="${OUT_DIR:-backups}"
# POSTGRES_USER / POSTGRES_DB: окружение -> .env (без source: значения могут быть любыми) -> fpl
envval() { grep -E "^$1=" "${ENV_FILE:-.env}" 2>/dev/null | tail -n 1 | cut -d= -f2-; }
PGUSER="${POSTGRES_USER:-$(envval POSTGRES_USER)}"
PGUSER="${PGUSER:-fpl}"
PGDB="${POSTGRES_DB:-$(envval POSTGRES_DB)}"
PGDB="${PGDB:-fpl}"

mkdir -p "$OUT_DIR"
out="$OUT_DIR/prod-$(date +%Y%m%d-%H%M).dump"
docker compose -f docker-compose.prod.yml exec -T db pg_dump -U "$PGUSER" -d "$PGDB" -Fc --no-owner \
  > "$out.part"
mv "$out.part" "$out"
find "$OUT_DIR" -name 'prod-*.dump' -mtime +"$KEEP_DAYS" -delete
echo "$(date -Is) db_backup: $out ($(du -h "$out" | cut -f1))"
