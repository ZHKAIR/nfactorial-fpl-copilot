#!/bin/sh
# Обновление на сервере: забрать код из GitHub, пересобрать образ, перезапустить стек.
# Миграции БД применяет entrypoint контейнера (scripts/docker-entrypoint.sh) при старте.
#
#   cd ~/fpl-copilot && scripts/deploy.sh
set -eu

COMPOSE="docker compose -f docker-compose.prod.yml"

git pull --ff-only
$COMPOSE build app
$COMPOSE up -d
$COMPOSE ps
docker image prune -f >/dev/null
echo "deploy: готово — проверьте https://$(grep -E '^DOMAIN=' .env | cut -d= -f2)"
