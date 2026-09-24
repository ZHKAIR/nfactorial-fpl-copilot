#!/bin/sh
# Локальная разработка: Postgres в Docker, миграции, Streamlit на хосте (горячая перезагрузка).
#   ./scripts/dev_up.sh            # db -> migrate -> streamlit (порт 8501)
#   PORT=8600 ./scripts/dev_up.sh
set -eu
cd "$(dirname "$0")/.."

PORT="${PORT:-8501}"

docker compose up -d db
echo "dev_up: waiting for db…"
i=0
until docker compose exec -T db pg_isready -U "${POSTGRES_USER:-fpl}" -d "${POSTGRES_DB:-fpl}" >/dev/null 2>&1; do
  i=$((i + 1))
  if [ "$i" -ge 30 ]; then
    echo "dev_up: db is not ready after 60s" >&2
    exit 1
  fi
  sleep 2
done
uv run python scripts/migrate.py
exec uv run streamlit run src/fplcopilot/app/Home.py --server.port "$PORT" --browser.gatherUsageStats false
