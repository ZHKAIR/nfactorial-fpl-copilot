from datetime import UTC, date, datetime
from pathlib import Path
from types import NoneType
from typing import Any, Literal, get_args

from pydantic import ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Все настройки читаются из .env в корне проекта и из окружения."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    fpl_manager_id: int | None = None  # дефолт CLI / скриптов; в UI не подставляется
    # Предзаполнение поля ID в UI (локальная разработка / демо); строка: пусто / не число = нет
    app_default_manager_id: str | None = None
    # Прод: пароль входа в приложение (пусто — без входа, как локально и в тестах) и суточный
    # лимит запросов к LLM из UI на процесс (чат, скриншот, обновление новостей, объяснение
    # сравнения; пусто — без лимита)
    app_password: str | None = None
    app_daily_llm_limit: int | None = None
    fpl_cache_ttl: int = 3600
    cache_dir: Path = PROJECT_ROOT / ".cache"

    openai_api_key: str | None = None
    langsmith_api_key: str | None = None
    langsmith_project: str = "fpl-copilot"

    # --- Postgres (+pgvector), см. docker-compose.yml ---
    database_url: str = "postgresql://fpl:fpl@localhost:5433/fpl"

    # --- ингест новостей (fplcopilot.rag.ingest) ---
    news_user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    )
    news_fetch_timeout: float = 15.0  # сек, на один HTTP-запрос
    news_fetch_delay: float = 1.0  # сек, пауза между загрузками полного текста
    news_limit_per_source: int = 200  # максимум новых записей с одной ленты за прогон
    # Окно сезона: статьи старше не сохраняем (все источники). Дефолт — начало предсезонки 2026/27.
    news_backfill_since: date = date(2026, 7, 1)

    # --- RAG: индекс, поиск, извлечение сигналов (fplcopilot.rag.index / retrieve / extract) ---
    rag_embedding_model: str = "text-embedding-3-small"
    rag_embedding_dims: int = 1536  # должно совпадать с vector(N) в 003_news_chunks.sql
    rag_chunk_tokens: int = 300  # верхняя граница чанка (склеиваем абзацы, пока влезают)
    rag_llm_model: str = "gpt-4o-mini"
    rag_half_life_days: float = 7.0  # период полураспада для time-decay при слиянии
    rag_candidates: int = 30  # top-N кандидатов от каждого из dense/bm25 до слияния и reranker
    rag_reranker: Literal["flashrank", "none"] = "flashrank"
    rag_reranker_model: str = "ms-marco-MiniLM-L-12-v2"
    # --- RAG v2 (A/B #2, docs/rag.md): промпт, pre-LLM abstention, ручки retrieval v2 ---
    # prompts/<version>/signal_extraction.*.md; v3 = v2 + относительные сроки от даты документа
    # (A/B #3, docs/EVALS.md: точность = v2); v1 / v2 хранятся для A/B
    rag_prompt_version: str = "v3"
    rag_abstain: bool = True  # без официального fpl_api-чанка и без упоминаний/при низком score
    # Пороги score (0 = гейт выключен, решает только «нет упоминаний»). A/B #2: 0.2 в hybrid_rerank
    # дал 4 ложных отказа (Raya, Gabriel, Emersonn, Vicario: новости есть, но не про травму —
    # rerank 0.01–0.16), dense-косинус классы не разделяет (no_coverage 0.50–0.59 vs покрытые 0.42–0.61).
    rag_abstain_score: float = 0.0  # hybrid_rerank: порог max «сырого» rerank_score кандидатов
    rag_abstain_dense_score: float = 0.0  # dense: порог косинуса top-1
    rag_per_article_cap: int | None = 2  # retrieval v2: макс. чанков одной статьи в top-k
    rag_date_aware_rerank: bool = True  # retrieval v2: hybrid_rerank сортирует по rerank×date
    # Проверка summary кодом (rag/summary_check.py): даты / числа / имена только из процитированных
    # документов, FPL prior и календаря; иначе предложение убирается. Промпт v1 не проверяется (A/B).
    rag_summary_check: bool = True

    # --- xPts (fplcopilot.core): сила фикстур и букмекерские коэффициенты ---
    # team_rating — рейтинги FPL + xG/xGC сезона (без внешних API); odds — The Odds API v4 (рынок
    # h2h soccer_epl) для ближайших туров + team_rating для остальных (гибрид); без ключа/сети odds
    # целиком падает на team_rating. Коэффициенты — только внутренний вход модели, пользователю не
    # показываются (наружу — FSI 1–5, xG, p(clean sheet), источник).
    xpts_fixture_provider: Literal["team_rating", "odds"] = "team_rating"
    odds_api_key: str | None = None  # https://the-odds-api.com, бесплатный тариф 500 запросов/мес
    odds_cache_ttl: int = 21600  # сек, кэш .cache/odds/epl_h2h.json (6 ч ≈ 120 запросов/мес)

    # --- внешняя стата (Understat + поля FPL bootstrap): слабый аддитивный слой поверх xPts v0 ---
    # Пенальтист order=1: +XPTS_SETPIECE_WEIGHT xPts за тур (масштабируется минутами / 90).
    # Штатник прямых штрафных order=1: +XPTS_FREEKICK_WEIGHT. Оба по умолчанию маленькие.
    xpts_setpiece_weight: float = 0.12
    xpts_freekick_weight: float = 0.03
    # 25 % Understat xG90/xA90 + 75 % наши; сдвиг per-90 не больше ±XPTS_UNDERSTAT_SHIFT_CAP.
    xpts_understat_blend: float = 0.25
    xpts_understat_shift_cap: float = 0.15
    # Сейвы вратаря уже в компоненте saves (E[floor(λ/3)]). Этот вес — доп. член; 0 = не двоить.
    xpts_saves_weight: float = 0.0
    understat_enabled: bool = True
    understat_season: int = 2026  # Understat: стартовый год сезона 2026/27
    understat_cache_ttl: int = 43200  # сек, кэш .cache/understat/epl_{season}.json (12 ч)

    # --- «Почему»: раскладка очков за последние туры (только объяснение, не xPts) ---
    why_form_last_n: int = 3
    why_luck_over: float = 5.0  # голы+ассисты − моменты ≥ порога → везение (не xPts)
    why_luck_under: float = -4.0  # ≤ порога → недобор
    why_repeatable_share: float = 0.50  # доля сухих / защиты / сейвов (не выход на поле)
    why_repeatable_min_points: int = 8
    why_pen_xg: float = 0.76
    why_pen_shrink_k: float = 10.0
    why_pen_league_avg_fallback: float = 0.14
    why_pen_standout_conceded: int = 2
    why_pen_min_p: float = 0.30  # ниже — не пишем про пенальти (кроме выделяющегося соперника)
    why_opp_ratio: float = 1.3  # соперник vs среднее лиги: ≥ratio дырявый, ≤1/ratio глухой
    why_opp_min_matches: int = 5  # меньше матчей — не сравниваем с лигой
    why_llm_timeout_s: float = 8.0
    why_llm_model: str = "gpt-4o"  # кэш на диск; русский у mini был водянистый

    # --- Оптимизатор (fplcopilot.core.optimizer): MILP через PuLP ---
    # highs — HiGHS через highspy (в комплекте); cbc — CBC из PuLP (на macOS arm64 бинарь Intel,
    # не запускается без Rosetta); auto — HiGHS, если доступен, иначе CBC.
    optimizer_solver: Literal["auto", "highs", "cbc"] = "auto"
    optimizer_time_limit_s: float = 90.0  # лимит на один solve
    optimizer_pool_size: int = 120  # кандидатов вне состава в многотуровом плане
    optimizer_mip_gap: float = 0.001  # относительный MIP gap, при котором solve останавливается

    # --- Агент (fplcopilot.agent): LangGraph-граф, см. docs/agent.md ---
    agent_router_model: str = "gpt-4o-mini"  # роутер и грейдер: structured output, temperature 0
    agent_explain_model: str = "gpt-4o-mini"  # объяснение: числа приходят готовыми из инструментов
    # Температура объяснителя: 0 по docs/LLM_CHOICE.md §4.2 (валидатор с первого раза 0.92 vs 0.79
    # при 0.2; длина/judge/цена не меняются). До 17.09 дефолт был 0.2 (демо v1/v2 и A/B — на 0.2).
    agent_explain_temperature: float = 0.0
    agent_signal_max_age_h: float = 12.0  # сигнал старше (часов) -> извлечь заново (rag.extract)
    # Новости в советах (узел candidate_news после compute, docs/agent.md): сигналы игроков, которых
    # рекомендует / ранжирует оптимизатор, и дайджесты их клубов (rag/team_news.py). Сначала
    # сохранённое моложе AGENT_SIGNAL_MAX_AGE_H, иначе извлечение в пределах бюджета на запрос;
    # сверх бюджета — сохранённое любого возраста с пометкой возраста.
    agent_candidate_news: bool = True
    agent_candidate_max_players: int = 6  # кандидатов с сигналом на запрос
    agent_candidate_routes: int = 3  # покупки / продажи скольких лучших маршрутов
    agent_candidate_ranking_top: int = 3  # строк рейтинга player_ranking
    agent_team_news_max_clubs: int = 3  # клубов с дайджестом на запрос
    agent_news_max_llm_calls: int = 3  # извлечений (игроки + клубы) на запрос: ≈ $0.0025, 12–15 с
    agent_news_mode: str = "hybrid_rerank"  # интерактивный путь (docs/EVALS.md §3.3)
    agent_news_reoptimize: bool = True  # покупка недоступна по новости -> один пересчёт без неё
    agent_news_exclude_min_confidence: float = 0.5  # порог уверенности для исключения кандидата
    # Версия промптов агента (agent/prompts/<version>/): v2 — язык ответа, horizon только явный,
    # явная схема таблиц, Sources только по обсуждаемым игрокам, rules_context; v3 — + блок NEWS
    # (новости кандидатов и клубов), роутер: разбор состава -> transfer; v4 — чат: история диалога,
    # интенты squad_review / fixtures / chips / general_fpl, фишки и целевой тур в плане, гибкий
    # формат ответа (docs/EVALS.md «Chat A/B: v3 → v4»); v1–v3 хранятся для A/B
    agent_prompt_version: str = "v4"
    # SQLite-чекпоинты LangGraph: HITL resume между запусками CLI
    agent_checkpoint_path: Path = PROJECT_ROOT / ".cache" / "agent_checkpoints.sqlite"

    # --- MCP-сервер fpl-intelligence (fplcopilot.mcp_server, docs/mcp.md) ---
    # stdio — транспорт по умолчанию (Cursor / Claude Desktop); host/port — для streamable-http
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8765

    # --- Vision (fplcopilot.vision): состав из скриншота Pick Team / Transfers, см. docs/vision.md ---
    vision_model: str = "gpt-4o-mini"  # мультимодальная модель, structured output, temperature 0
    vision_detail: Literal["low", "high", "auto"] = "high"  # detail изображения в запросе OpenAI
    vision_prompt_version: str = "v2"  # vision/prompts/<version>/; v1 хранится для A/B

    @field_validator("*", mode="before")
    @classmethod
    def _empty_means_unset(cls, value: Any, info: ValidationInfo) -> Any:
        """«KEY=» (скопированный .env.example): необязательное поле -> None, остальные -> дефолт."""
        if not (isinstance(value, str) and not value.strip()) or info.field_name is None:
            return value
        field = cls.model_fields[info.field_name]
        if NoneType in get_args(field.annotation):
            return None
        return field.get_default(call_default_factory=True)

    @property
    def news_since_utc(self) -> datetime:
        return datetime.combine(self.news_backfill_since, datetime.min.time(), tzinfo=UTC)

    @property
    def tracing_enabled(self) -> bool:
        """LangSmith включается только непустым ключом — без него клиент OpenAI «голый»."""
        return bool(self.langsmith_api_key and self.langsmith_api_key.strip())


settings = Settings()
