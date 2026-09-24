# FPL Copilot: Streamlit UI (+ тот же образ для ingest-цикла). docker compose up -d -> :8501
# uv sync --frozen из uv.lock; зависимости ставятся до копирования кода (кэш слоёв).
# Всё под непривилегированным пользователем сразу: chown -R после установки удвоил бы образ.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.7.9 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_CACHE_DIR=/tmp/uv-cache \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/.cache \
    && chown -R app:app /app
WORKDIR /app
USER app

# 1) зависимости (без самого проекта) — слой переиспользуется, пока не меняются pyproject/uv.lock
COPY --chown=app:app pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/tmp/uv-cache,uid=10001,gid=10001 \
    uv sync --frozen --no-dev --no-install-project

# 2) код проекта
COPY --chown=app:app src ./src
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app skills ./skills
COPY --chown=app:app samples ./samples
RUN --mount=type=cache,target=/tmp/uv-cache,uid=10001,gid=10001 \
    uv sync --frozen --no-dev \
    && chmod +x scripts/docker-entrypoint.sh

# /app/.cache — дисковый кэш FPL API, модель flashrank, SQLite-чекпоинты агента (named volume)
VOLUME ["/app/.cache"]

EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).status == 200 else 1)"

ENTRYPOINT ["scripts/docker-entrypoint.sh"]
CMD ["streamlit", "run", "src/fplcopilot/app/Home.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true", "--browser.gatherUsageStats", "false"]
