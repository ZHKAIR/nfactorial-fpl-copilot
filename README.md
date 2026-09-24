# FPL Copilot

> Рабочее название. «FPL Copilot» также называется не связанный с проектом коммерческий сайт —
> перед публикацией проект может быть переименован.

Decision-support для Fantasy Premier League: система, которая перед каждым дедлайном отвечает
менеджеру FPL на вопросы «кого продать», «кого поставить капитаном», «стоит ли брать −4»,
«что делать в следующие 5 туров» — числами из детерминированных моделей (ожидаемые очки xPts,
MILP-оптимизатор состава, новостные сигналы с цитатами), а не мнением языковой модели.

**Пользователь** — менеджер FPL, у которого раз в неделю есть один дедлайн и несколько
решений с реальной ценой ошибки (травмированный игрок в старте — минус очки и минус ранг).
**Боль**: информация разбросана — флаги FPL, пресс-конференции, новости о травмах, календарь,
цены; готовые «модели xPts» в сообществе платные и непрозрачные, а спросить чат-бота напрямую
нельзя — он уверенно выдумывает цифры. Здесь LLM только маршрутизирует вопрос, извлекает факты
из новостей и объясняет ответ; считает всё код (главный принцип исходного описания проекта).

Финальный проект курса LLM Engineering. Состояние на **24 сентября 2026**: сыграны GW1–5,
дедлайн GW6 — 10.10 10:00 UTC (пауза на матчи сборных). Живой эксперимент: прогноз xPts v0 на
GW5 сохранён 17.09 (`xpts_predictions`, 659 игроков) рядом со снимком официального `ep_next` FPL;
тур сыгран и разобран 24.09 (`scripts/gw_review.py --gw 5`, `docs/xpts.md` «Итог GW5»): в среднем
v0 вровень с `ep_next` (MAE 1.146 против 1.166 — в пределах шума), точнее среди сыгравших, но top-20
у `ep_next` лучше; сухари модель занижает на треть. Для GW6 рядом с
`v0` сохранён прогноз по букмекерским коэффициентам (`v0-odds`, `docs/xpts.md`). Что сделано и что
осталось до сдачи — [`docs/PLAN_STATUS.md`](docs/PLAN_STATUS.md).

## Live demo

**https://fpl-copilot.duckdns.org** — развёрнутая версия (VPS, Docker Compose, HTTPS через Caddy).
Вход по паролю: пароль выдаётся проверяющим отдельно. После входа введите в сайдбаре свой ID
менеджера FPL (число из адреса `fantasy.premierleague.com/entry/<ID>/`) или откройте демо-команду
`https://fpl-copilot.duckdns.org/?manager=895045`. Стартовая страница — «Брифинг»: главный трансфер,
капитан, проблемы состава и лента новостей; дальше — поле состава, план на 3–6 туров с фишками,
карточка игрока, сравнение и чат с агентом. Как развернуть самому — [`docs/deploy.md`](docs/deploy.md).

## 60-second pitch

Каждую неделю каждый менеджер Fantasy Premier League перед дедлайном решает одно и то же: кого
продать, кого поставить капитаном, стоит ли платить 4 очка за лишний трансфер. Ответы
разбросаны по новостям, флагам официального API и таблицам сообщества. FPL Copilot собирает это
в одном месте, но не так, как чат-бот: новости превращаются в структурированный сигнал с
цитатами, ожидаемые очки считает прозрачная компонентная модель, состав и трансферы подбирает
математический оптимизатор с правилами FPL, а языковая модель лишь понимает вопрос и объясняет
решение — и её ответ проверяется кодом. Чат помнит разговор, понимает русский и английский, умеет
планировать фишки. Один вопрос стоит в среднем четверть цента и занимает 7–9 секунд (медиана
на 60 вопросах). Те же инструменты доступны через MCP в Cursor или Claude Desktop, а когда у API
нет вашего состава — его можно сфотографировать.

## Что умеет: страницы UI

Streamlit-приложение (`src/fplcopilot/app/`, [`docs/ui.md`](docs/ui.md)) — семь экранов, все
работают через один набор инструментов `LiveTools`, тот же, что использует агент и MCP-сервер.

| экран | что показывает | инструменты |
|---|---|---|
| **Мой состав** (`1_squad.py`, стартовая) | шапка команды (очки сезона, ранг, банк, бесплатные трансферы, чипы, прогноз очков на тур); именная строка серьёзных проблем (статус FPL / ротация); HTML-таблица 15 игроков по образцу smartplay Players: роль, Player (имя-ссылка на страницу «Игрок» `/player?pid=…`, значок у проблемных, клуб · поз · TSB второй строкой), Price, TSB%, Form, Pts/Game, xMins, xPts, 3GW xPts, FRR#, цветной календарь на 5 туров с легендой сложности, разделитель «Bench»; «Проблемы состава» с цитатами; **загрузка скриншота Pick Team** в expander | `get_gameweek_context`, `predict_player`, `analyze_player_risk`, `vision.squad_from_image` |
| **К дедлайну** (`2_deadline.py`) | лучшие 11 и скамейка (MILP), капитан/вице, xPts лучших 11 против текущих; варианты капитана safe / balanced / differential по владению; top-3 маршрутов трансфера с Δ на тур и горизонт, хитом и вердиктом go / hit_not_worth / hold и блоком «Почему» (факты считает код, текст склеивает `gpt-4o` с проверкой и запасным текстом); переключатель «разрешить −4» | `optimize_team`, `recommend_transfers`, `core/why_facts.py` |
| **План** (`3_plan.py`) | план на 3–6 туров: ходы с причиной, FT/хиты/банк по турам, план против «без трансферов», альтернатива Wildcard с рекомендацией; **фишка в туре — Bench Boost / Triple Captain** (`app/plan_chips.py`; недоступная фишка — понятная причина); «Что изменилось» — diff с сохранённым снимком | `build_gameweek_plan(chips=…)` |
| **Игрок** (`4_player.py`) | поиск (неоднозначное имя → выбор), xPts по турам с компонентами, фикстуры с индексом силы FSI 1–5, стандарты и Understat (xG/xA), новостной сигнал с цитатами и подблоком **«Форма и контекст»** (только у сигналов промпта v4), блок **«Новости клуба»** (выбывшие одноклубники по данным FPL с туром возвращения + дайджест с цитатами, кнопка «Обновить новости клуба»), кнопка «Обновить сигнал» с числом LLM-вызовов, токенами и $ | `predict_player`, `analyze_player_risk`, `team_news` |
| **Сравнение** (`7_compare.py`) | два игрока бок о бок (поиск или из состава, адрес `?a=<id>&b=<id>`): таблица метрик на ближайший тур и на 3 тура, календарь на 6 туров, новости, объяснение `gpt-4o-mini` по готовым фактам (факты видны без ИИ) | `predict_player`, `analyze_player_risk`, `app/compare.py` |
| **Чат** (`5_chat.py`) | LangGraph-агент (промпты **v4**): помнит 6 последних реплик — уточнения «а на GW7?», «а если без хита?» переписываются в самостоятельный вопрос; оценка состава, календарь клубов, фишки моего состава, разбор сыгранного тура по фактам, общие вопросы про FPL вместо ложных отказов; фишка / целевой тур / запрет хитов / «не продавать X» из фразы уходят в план; прогресс по узлам, markdown-ответ на языке вопроса; футер «Как получен ответ» с интентом, инструментами, LLM-вызовами, стоимостью, валидацией, версией промптов и вопросом с учётом истории; **human-in-the-loop** — карточка «требует подтверждения: хит / Wildcard» с кнопками Подтвердить / Отклонить и карточка уточнения имени с кнопками-кандидатами (второй HITL); 6 кнопок-примеров | агент (все инструменты `TOOL_NAMES` + внутренние `team_fixtures`, `chips_status`, `rank_forecast`, `transfer_trends`, `team_news`, `review_gameweek`) |
| **О системе** (`6_about.py`) | граф агента, построенный из скомпилированного LangGraph (подсвечены оба узла-прерывания), схема потока данных, таблица «страница → инструменты» | — |

Общий сайдбар: ID менеджера (живёт в адресе `?manager=<id>`, после «Сохранить» — в cookie браузера,
«Забыть» стирает; `APP_DEFAULT_MANAGER_ID` предзаполняет поле только для локального демо,
`app/manager_state.py`), стратегия conservative / balanced / aggressive (меняет коэффициенты цели
оптимизатора, а не тон текста), обратный отсчёт до дедлайна, свежесть данных, статус LangSmith,
источник состава («из FPL за GW5» / «со скриншота» / «нет — загрузите скриншот»).

### Скриншоты

Чат v4 (23–24.09.2026, manager 6856911 и 895045, GW6; сценарий владельца «Bench Boost в GW7,
João Pedro не продавать, платные трансферы нельзя»):

| | |
|---|---|
| ![Чат v4: Bench Boost в GW7](docs/screenshots/chat_v4_01_bb_gw7_6856911.png) План с Bench Boost в GW7 без хитов и с João Pedro (6856911) | ![Чат v4: причины и источники](docs/screenshots/chat_v4_02_bb_gw7_details_6856911.png) Скамейка BB, причины с цитатами новостей `[fpl_api, 16.09]`, источник правила о фишках из KB |
| ![Чат v4: уточнение](docs/screenshots/chat_v4_04_followup_gw7_6856911.png) Уточнение следующей репликой — чат помнит тур, фишку и ограничения | ![Чат v4: футер](docs/screenshots/chat_v4_03_footer_prompt_version.png) Футер «Как получен ответ»: интент plan, 19 вызовов инструментов, 2 LLM-вызова, ≈ $0.0028, промпты v4, вопрос с учётом истории |
| ![Чат v4: BB недоступен, до правки](docs/screenshots/chat_v4_06_bb_gw7_895045_unavailable.png) 895045: «Bench Boost недоступен» — до правки таблица xPts по игрокам была выдумана (одинаковые 4.11) | ![Чат v4: BB недоступен, после правки](docs/screenshots/chat_v4_07_bb_gw7_895045_postfix.png) После правки: xPts каждого игрока итогового состава на GW7 — из фактов плана |
| ![Чат v4: сила состава](docs/screenshots/chat_v4_05_squad_strength_6856911.png) `squad_review`: проблемы состава, xPts текущих 11 против лучших | ![Чат v4: уточнение «а кого тогда продать?»](docs/screenshots/chat_v4_08_followup_sell_895045.png) 895045, 24.09: уточнение «а кого тогда продать?» после оценки состава — маршрут без хита с новостями по каждому игроку |
| ![Чат v4: футер и LangSmith](docs/screenshots/chat_v4_09_footer_langsmith_895045.png) Футер: «Промпты: v4», «Вопрос с учётом истории: Кого продать из моего состава?», 2 LLM-вызова ≈ $0.0030; в сайдбаре — «LangSmith: выключен — месячная квота трейсов исчерпана» | |

Остальные новые экраны 24.09 (финальный код, 8501):

| | |
|---|---|
| ![План с Bench Boost](docs/screenshots/plan_chip_bb_gw7_6856911.png) «План», 6856911: Bench Boost в GW7 — +7.57 xPts скамейки, колонки «Фишка» / «Вклад фишки» | ![Игрок: новости клуба](docs/screenshots/player_club_news_saka.png) «Игрок» `?pid=12` (Saka): вердикт по новостям с цитатами и «Новости клуба» — выбывшие Arsenal по данным FPL |

Остальные экраны — 17.09.2026, manager 895045, GW5 (с тех пор «Обзор» заменён стартовой страницей
«Мой состав», таблица состава переделана 22.09, добавлена страница «Сравнение», `docs/ui.md`):

| | |
|---|---|
| ![Обзор](docs/screenshots/home.png) Обзор: GW5, дедлайн, граф агента | ![Мой состав](docs/screenshots/squad.png) Мой состав: xPts, сигналы, диагностика |
| ![Скриншот состава](docs/screenshots/squad_screenshot.png) Состав со скриншота: 15/15 карточек, $0.0066 | ![К дедлайну](docs/screenshots/deadline.png) К дедлайну: 3-4-3, 56.14 xPts, маршруты |
| ![К дедлайну, override](docs/screenshots/deadline_override.png) Те же инструменты на составе со скриншота | ![План](docs/screenshots/plan.png) План на 5 туров: 307.5 против 283.1 |
| ![Игрок](docs/screenshots/player.png) Игрок: xPts по турам, FSI, сигнал | ![Чат HITL](docs/screenshots/chat_hitl.png) Чат: прерывание перед хитом |
| ![Чат reject](docs/screenshots/chat_reject.png) Чат: «Отклонить» → пересчёт без хита | |

## Архитектура — коротко

Подробно — [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (слои, путь запроса, модель данных,
таблица решений, отказы, стоимость, гипотезы); исходники диаграмм — [`docs/diagrams/`](docs/diagrams/).

```mermaid
flowchart LR
    U["Пользователь<br/>UI · CLI · Cursor/Claude через MCP"] --> A["LangGraph-агент, 14 узлов<br/>router (история) → signals ⟲ retry → compute<br/>→ candidate_news (⟲ 1 пересчёт) → HITL → explain → validate"]
    A --> T["LiveTools — 11 инструментов<br/>(те же в UI и MCP; в MCP +3 справочных)"]
    T --> X["xPts v0 + MILP HiGHS (фишки BB / TC)<br/>core/ — без LLM"]
    T --> R["News RAG + новости клуба<br/>pgvector + BM25 → RRF → flashrank<br/>→ gpt-4o-mini → валидатор цитат и summary"]
    T --> V["Vision: скриншот → Squad"]
    X --> D["Postgres + pgvector<br/>FPL API с кэшем"]
    R --> D
```

Слои: интерфейсы (Streamlit, CLI, MCP-сервер `fpl-intelligence` + Skill) → оркестрация
(LangGraph: 14 узлов, ветвление по 16 интентам, ограниченный цикл повторного поиска ≤ 2, узел
`candidate_news` с одним пересчётом, если рекомендованная покупка недоступна по новости, два
`interrupt_before` — перед платным действием и перед выбором игрока при неоднозначном имени;
промпты v4: история диалога, ответ на языке вопроса) → инструменты (`agent/tools.py`: 11 общих с
MCP, включая стратегическую KB и рейтинг, + внутренние для графа: календари, фишки, прогнозный
рейтинг, тренды трансферов, новости клуба, разбор сыгранного тура) → детерминированное ядро (`core/`: компонентная модель
xPts со слабым слоем Understat, модель минут, сила фикстур, MILP на PuLP + HiGHS с фишками
Bench Boost / Triple Captain; `rag/`: новостной RAG, дайджест новостей клуба, вечнозелёная база
знаний по стратегии; `vision/`) → данные (FPL API с дисковым кэшем, 4 RSS-источника + Google News +
статусы FPL, Understat, OpenAI `gpt-4o-mini` и `text-embedding-3-small`, flashrank) → хранение
(Postgres 16 + pgvector, миграции 001–010, SQLite-чекпоинты). Путь одного запроса от вопроса до
ответа — `docs/ARCHITECTURE.md` §3 и [`docs/diagrams/request_path.mmd`](docs/diagrams/request_path.mmd).

## Как запустить

### Docker (однокомандный запуск)

```bash
cp .env.example .env
# в .env обязателен только OPENAI_API_KEY (эмбеддинги, извлечение сигналов, чат, скриншот);
#   остальные ключи можно оставить пустыми («KEY=» = значение по умолчанию):
#   POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB / POSTGRES_PORT -> fpl / fpl / fpl / 5433 и
#   DATABASE_URL -> postgresql://fpl:fpl@localhost:5433/fpl (свои POSTGRES_* -> свой DATABASE_URL
#   postgresql://<user>:<password>@localhost:<port>/<db>; внутри compose URL переопределяется),
#   FPL_MANAGER_ID — менеджер по умолчанию для CLI и скриптов (например 895045; в UI не
#   подставляется — предзаполнение поля для локального демо: APP_DEFAULT_MANAGER_ID),
#   LANGSMITH_API_KEY — трейсинг включается, когда ключ задан.
docker compose up -d            # db (pgvector/pgvector:pg16) + app (Streamlit) + ingest (новости каждые 30 мин)
open http://localhost:8501
```

Образ `python:3.12-slim` + `uv sync --frozen`, 1.37 GB (замер 17.09); `PYTHONPATH=/app/src` задан в
`Dockerfile`; entrypoint ждёт Postgres и применяет все миграции `001–010` (`scripts/migrate.py`
идемпотентно берёт каждый `*.sql` по порядку); UI отвечает на `/_stcore/health` через ~4 с после
старта ([`docs/ui.md`](docs/ui.md), «Docker»). У всех трёх сервисов `restart: unless-stopped`.

### Выкладка на сервер (VPS)

Прод-стек `docker-compose.prod.yml` (Postgres и Streamlit закрыты, наружу — только Caddy с
авто-HTTPS, вход в приложение по паролю `APP_PASSWORD`, суточный лимит ИИ `APP_DAILY_LLM_LIMIT`),
перенос базы `scripts/db_dump.sh` → `scripts/db_restore.sh`, обновление `scripts/deploy.sh`,
бэкапы `scripts/db_backup.sh` — пошагово для Hetzner / Ubuntu 24.04 в
[`docs/deploy.md`](docs/deploy.md).

Первичное наполнение данных (один раз; новости дальше собирает сервис `ingest`):

```bash
docker compose exec app python -m fplcopilot.rag.ingest --once --backfill   # RSS + Google News + статусы FPL
docker compose exec app python -m fplcopilot.rag.index --rebuild-missing    # чанки + эмбеддинги (~27 с, ~$0.004)
docker compose exec app python -m fplcopilot.core.history --sync            # 659 element-summary -> история (~135 с)
docker compose exec app python -m fplcopilot.core.xpts --save               # прогноз xPts на следующий тур (+ снимок ep_next)
docker compose exec app python -m fplcopilot.rag.kb --ingest                # база знаний по стратегии (41 URL)
```

**Перед демо** (свежие данные; иначе история отстаёт на тур, а часть новостей кандидатов придёт из
кэша с пометкой возраста — бюджет чата 3 извлечения на запрос):

```bash
docker compose exec app python -m fplcopilot.core.history --sync                        # история за сыгранный тур
docker compose exec app python -m fplcopilot.rag.refresh --manager 895045 --dry-run     # план и верхняя оценка цены
docker compose exec app python -m fplcopilot.rag.refresh --manager 895045 --max-players 40 --max-usd 0.10
# в UI — кнопка «Обновить данные» в сайдбаре (сбрасывает кэши страниц); после правки .env —
# docker compose up -d app (пересоздаёт контейнер: restart переменные из .env не перечитывает)
```

### Локальная разработка

```bash
uv sync --extra dev                                       # Python 3.12, зависимости из uv.lock (+ pytest, ruff)
cp .env.example .env                                      # обязателен только OPENAI_API_KEY (порт db по умолчанию 5433)
docker compose up -d db                                   # только Postgres + pgvector
uv run python scripts/migrate.py                          # идемпотентные миграции 001–010
uv run python -m fplcopilot.rag.ingest --once --backfill  # новости (один раз), затем:
uv run python -m fplcopilot.rag.index --rebuild-missing   #   индекс
uv run python -m fplcopilot.core.history --sync           # история матчей и прошлых сезонов (после каждого тура)
uv run python -m fplcopilot.core.xpts --save              # прогноз на следующий тур
uv run python -m fplcopilot.rag.kb --ingest               # Strategy KB
uv run python -m fplcopilot.rag.refresh --manager 895045 --max-players 40 --max-usd 0.10   # сигналы и новости клубов перед демо
uv run streamlit run src/fplcopilot/app/Home.py           # http://localhost:8501
# или всё вместе: ./scripts/dev_up.sh  (db -> migrate -> streamlit)
# фоновый сбор новостей (+ обновление сигналов игроков с новыми статьями, ≤ $0.02 за цикл):
#   uv run python -m fplcopilot.rag.ingest --loop --every 30 --index --refresh-signals
uv run pytest -q -m "not network"                         # 812 тестов без сети/LLM на 24.09 (db-тесты скипаются без Postgres)
```

`PYTHONPATH=src` локально не нужен: `uv sync` ставит пакет `fplcopilot` в `.venv` в editable-режиме
(hatchling, `packages = ["src/fplcopilot"]`), поэтому `uv run streamlit …` и `uv run python -m
fplcopilot…` находят код из `src/`. После правок кода или `.env` Streamlit нужно перезапустить:
настройки читаются один раз при импорте, а агент кэшируется `st.cache_resource` (из-за этого 23.09 UI
работал на промптах v2, `docs/chat_diagnosis.md` §4).

## CLI cheat-sheet

```bash
# Агент (LangGraph), docs/agent.md
uv run python -m fplcopilot.agent.cli --manager 895045 --strategy balanced "Should I sell Palmer?"
uv run python -m fplcopilot.agent.cli --manager 895045 "What if I take a -4 to bring in Saka and Guéhi?"
uv run python -m fplcopilot.agent.cli --manager 895045 "Стоит ли продавать Палмера?"      # ответ на русском, кириллица в именах
uv run python -m fplcopilot.agent.cli --thread <id> --decision confirm|reject     # resume после HITL
uv run python -m fplcopilot.agent.cli --thread <id> --choose <player_id>          # resume после уточнения имени
uv run python -m fplcopilot.agent.cli --manager 895045 "Who should I captain?" --prompt-version v3   # A/B версий промптов
uv run python scripts/agent_demo.py --ab docs/agent_prompt_ab.md                  # 11 сценариев -> docs/agent_demo_output_v2.md (+ A/B промптов)

# Оптимизатор (MILP), docs/optimizer.md
uv run python -m fplcopilot.core.optimizer --manager 895045 --gw 5 --xi
uv run python -m fplcopilot.core.optimizer --manager 895045 --gw 5 --transfer --allow-hit --horizon 3
uv run python -m fplcopilot.core.optimizer --manager 895045 --gw 5 --plan --horizon 5 --save
uv run python -m fplcopilot.core.optimizer --manager 6856911 --plan --chip 7:bboost [--chip 8:3xc]   # фишки BB / TC в туре
uv run python -m fplcopilot.core.optimizer --manager 895045 --gw 5 --diagnose

# xPts v0, docs/xpts.md
uv run python -m fplcopilot.core.xpts --gw 5 --top 25 --position MID
uv run python -m fplcopilot.core.xpts --player Haaland --horizon 3
uv run python -m fplcopilot.core.xpts --explain "João Pedro"
uv run python -m fplcopilot.core.backtest --gws 2,3,4                             # -> docs/xpts_backtest.json
uv run python scripts/gw_review.py --gw 5 --sync                                  # после финального свистка GW5

# News RAG, docs/rag.md
uv run python -m fplcopilot.rag.query "Is João Pedro fit for GW5?" --player "João Pedro" --timings
uv run python -m fplcopilot.rag.signal --player "João Pedro" --timings
uv run python -m fplcopilot.rag.signal --players "João Pedro,Caicedo,Foden,Haaland"
uv run python -m fplcopilot.rag.signal --player Saka --prompt v4 --no-save --timings   # form_notes (промпт v4, опция)
uv run python -m fplcopilot.rag.refresh --manager 895045 --dry-run                # пакетное обновление: план и цена
uv run python scripts/ingest_report.py                                            # сводка по корпусу

# Strategy KB (RAG #2), docs/strategy_kb.md
uv run python -m fplcopilot.rag.kb --query "when should I use my wildcard?" --tags chips -k 6
uv run python -m fplcopilot.rag.kb --answer "how many free transfers can I bank?"

# MCP-сервер, docs/mcp.md
uv run python -m fplcopilot.mcp_server                                            # stdio
uv run python -m fplcopilot.mcp_server --transport streamable-http --port 8765    # http://127.0.0.1:8765/mcp
uv run python scripts/mcp_smoke.py                                                # клиент: list + 4 вызова + ресурс
npx -y @modelcontextprotocol/inspector@latest --cli uv run python -m fplcopilot.mcp_server -- --method tools/list

# Vision, docs/vision.md
uv run python scripts/squad_from_screenshot.py samples/pick_team_valid.png
uv run python scripts/squad_from_screenshot.py shot.png --squad --manager-id 123 --json

# Evals, docs/EVALS.md
uv run python -m evals.run_rag --suite all --modes dense,hybrid,hybrid_rerank --k 8   # корпус по умолчанию = на as_of
uv run python -m evals.run_rag --suite signals --modes hybrid_rerank --prompt v4 --no-judge   # A/B #4 (v3 vs v4)
uv run python -m evals.run_chat --manager 895045 --label after-x                  # чат: 60 вопросов, ≈ $0.15
uv run python -m evals.run_chat --router-probe 1 --label after-x                  # только роутинг, ≈ $0.05
uv run python -m evals.run_chat --rows evals/golden/chat_holdout.jsonl --label holdout
uv run python -m evals.run_kb --modes dense,hybrid,hybrid_rerank -k 6
uv run python -m evals.run_hparams --suite all --budget-usd 1.0
uv run python -m evals.run_models --budget-usd 0.5
```

## MCP + Skill в Cursor и Claude Desktop

Сервер `fpl-intelligence` ([`docs/mcp.md`](docs/mcp.md)) выставляет 14 инструментов: 11
инструментов агента (`get_gameweek_context`, `predict_player`, `compare_players`,
`analyze_player_risk`, `optimize_team`, `recommend_transfers`, `build_gameweek_plan` — в том числе
с фишками Bench Boost / Triple Captain в названном туре, `simulate_scenario`, `diagnose_squad`,
`search_strategy_kb` — стратегическая база знаний с цитатами, `rank_players`) и 3 справочных
(`get_player_advanced_stats`, `get_player_points_breakdown`, `get_team_defensive_profile` —
Understat xG/xA, разбор очков по категориям, xGA команды), 5 ресурсов
(`fpl://gameweek/current`, `fpl://manager/{id}/squad`, `fpl://manager/{id}/plan`,
`fpl://player/{id}/signal`, `fpl://kb/stats`) и промпт `pre_deadline_review`. Он отдаёт
**решения и их основания**, а не факты REST — факты читает свой клиент FPL API с кэшем.

**Cursor.** В корне репозитория лежит `.cursor/mcp.json` без абсолютных путей (Cursor
подставляет `${workspaceFolder}`) — откройте `fpl-copilot/` как workspace, включите сервер в
Settings → MCP (должно появиться 14 инструментов). Postgres должен быть запущен. Если сервер не
стартует — подставьте абсолютный путь вместо `${workspaceFolder}` (подробнее в `docs/mcp.md`):

```json
{
  "mcpServers": {
    "fpl-intelligence": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--project", "${workspaceFolder}", "python", "-m", "fplcopilot.mcp_server"]
    }
  }
}
```

**Claude Desktop.** Тот же блок под ключом `mcpServers` в
`~/Library/Application Support/Claude/claude_desktop_config.json`, но с абсолютными путями:
`"command": "/opt/homebrew/bin/uv"`, `"--project", "<ABSOLUTE_PATH_TO_REPO>"` (интерполяции
переменных там нет, а PATH GUI-приложения короче терминального).

**Skill `fpl-transfer-analyst`** ([`skills/fpl-transfer-analyst/SKILL.md`](skills/fpl-transfer-analyst/SKILL.md),
[`skills/README.md`](skills/README.md)) — frontmatter с триггерами («should I sell / buy / keep»,
«who to captain», «take a -4», «wildcard», «starting XI», «is X fit»), таблица «тип вопроса →
порядок инструментов», шаблон ответа, жёсткие правила (не считать самому, без ставок, спросить
перед хитом, `ambiguous` → уточнить) и `references/` (правила FPL 2026/27, шаблон вывода). Копия
для Cursor лежит в репозитории — `.cursor/skills/fpl-transfer-analyst/` (побайтно совпадает с
`skills/`, синхронизация `rsync`, см. `skills/README.md`); для Claude Code ту же папку нужно
скопировать в `.claude/skills/` — в репозитории её нет.
Проверка: спросить в чате Cursor «Should I sell João Pedro? manager 895045» и убедиться, что
инструменты вызываются в порядке Skill'а, а перед хитом агент спрашивает подтверждение.

## Evals — коротко

Полностью — [`docs/EVALS.md`](docs/EVALS.md) (golden dataset, метрики и что они не показывают,
A/B #1–#4, эволюция промпта, §7 гиперпараметры, §8 сравнение моделей, §9 A/B чата v3 → v4),
[`docs/chat_diagnosis.md`](docs/chat_diagnosis.md) (диагностика чата до v4) и
[`docs/LLM_CHOICE.md`](docs/LLM_CHOICE.md) (выбор модели по семи ролям: измерения, стоимость,
дизайн фоллбэков).

- **Golden dataset — 55 размеченных строк RAG/KB + 75 вопросов чата.** RAG и KB — с фиксированным `as_of = 2026-09-17T09:00Z`:
  27 игроков (`evals/golden/signals.jsonl`: ожидаемый класс доступности, диапазоны p(start) и
  минут, `return_gw`, обязательность evidence; теги `conflict`, `no_coverage`, `ambiguous_name`,
  `return_date`…), 18 поисковых запросов с релевантностью на уровне статей
  (`retrieval.jsonl`, 3 — без ответа по построению), 10 вопросов по стратегии
  (`strategy.jsonl`). Протокол разметки и слабости — `evals/golden/README.md`, изменения — `CHANGELOG.md`.
  Чат: `evals/golden/chat.jsonl` — 60 живых вопросов (50 RU / 10 EN, разговорные, с опечатками;
  7 многоходовых, 5 off-topic / prompt injection) с допустимыми интентами, ожидаемым поведением и
  языком ответа; `chat_holdout.jsonl` — 15 новых вопросов, написанных после последней правки
  промптов и прогнанных один раз.
- **Метрики**: retrieval P@8 / R@8 / hit@8 / MRR на уровне статей, свежесть top-1, латентность;
  сигналы — accuracy strict/lenient, macro-F1, попадание в диапазоны, evidence present,
  false-evidence на `no_coverage`, `return_gw` exact/±1, число исправлений валидатора,
  abstention (в т.ч. ложные), стоимость; **faithfulness** — LLM-as-judge (gpt-4o-mini, T = 0) по
  цитатам; KB — hit@6 по домену, keyword recall, валидность цитат, поведение «not covered»; чат —
  точность интента, ложные отказы, отказ настоящего off-topic, язык ответа, регенерации валидатора,
  ручная оценка «полезен / частично / криво» с причиной по каждой строке, LLM-судья только как тренд
  (совпадает с ручной оценкой на 43/60), латентность и стоимость.
- **A/B #1 (режимы поиска)**: `hybrid_rerank` лучше по поиску (recall@8 0.837 vs 0.794 у dense,
  свежесть top-1 4.9 vs 10 дней) и даёт сигнал для abstention, но **не** лучше по downstream
  точности (24/27 vs 26/27). Решение: rerank для интерактивного пути, `dense` для батча,
  `hybrid` без ранкера — не выпускать (худшая рука).
- **A/B #2 (промпт v2 + retrieval v2 + abstention)**: `return_gw` 0/8 → **8/8**, исправлений
  валидатора 1.4–1.6 → 0.04–0.41 на строку, faithfulness 0.70–0.73 → 0.75–0.82, 4 LLM-вызова
  сэкономлено без ложных отказов; порог rerank-score 0.2 **отвергнут** (4 ложных отказа).
- **A/B #3 (23.09: проверка summary кодом + промпт сигнала v3)**: первый прогон показал падение
  0.852 → 0.741, и его сначала списали на «вырос корпус» — построчный разбор нашёл **утечку из
  будущего** (prior FPL по `news_added` брал снимки после `as_of`, −3 строки; статьи с задним
  `pubDate`, −1 строка). После исправления (`snapshot_at`, `--corpus-cutoff as_of` по умолчанию) на
  одинаковых данных v2 / v2 + check / v3 + check = 25 / 24 / 25 из 27 — разница в пределах шума;
  v3 + проверка summary — дефолт.
- **A/B #4 (23.09: `form_notes`, промпт сигнала v4)**: цитаты формы в доказательствах 4/78 → 2/73,
  завышенный confidence fit-игроков 0.93–0.98 → 0.83, но точность 25/27 → 23/27 в обоих повторах
  (Reece James) — **v4 не дефолт**, остаётся опцией.
- **A/B чата v3 → v4 (24.09, 60 вопросов, тот же код)**: интент 45 → 59 из 60, ложные отказы
  10/55 → 0/55 (off-topic по-прежнему 5/5), язык ответа 46 → 58 из 60, ручная оценка «полезен /
  криво» 17 / 26 → 49 / 4, p95 32.7 → 25.5 с, $0.151 за 60 вопросов; holdout 15/15 по интенту,
  13 полезных, 0 кривых. **v4 — дефолт.** Подтверждающий прогон 4 на финальном коде (+ `gw_review`,
  «игрок не из состава», цены помечены как непрогнозируемые): «полезен / криво» 54 / 2, многоходовые
  6/7, язык 60/60, интент 58/60, $0.157 за 60 вопросов. Остались: c12 (бюджет в прозе) и c55
  (уточнение иногда наследует совет ассистента).
- **Гиперпараметры** (`EVALS.md` §7, `LLM_CHOICE.md` §3.1, §4.2): температура извлечения
  T ∈ {0, 0.3, 0.7} не влияет на точность при n = 27, но влияет на воспроизводимость (T = 0:
  22/23 меток и 12/23 текстов совпадают между прогонами; T = 0.7: 21/23 и 0/23); для объяснения
  `max_tokens` 300 обрезает 100 % ответов, 600 — 25 %, 1000+ — 0 % → выпущенные 1400 дают
  двукратный запас (с промптов v3 — 2000: русский ответ с таблицей и блоком NEWS упирался в 1400);
  T = 0 проходит валидатор с первого раза чаще (0.92 vs 0.79 при 0.2) — explain переведён на T = 0
  17.09.
- **Выбор модели** (`EVALS.md` §8, `LLM_CHOICE.md`): gpt-4o-mini против gpt-4.1-nano, gpt-4.1-mini,
  gpt-5.4-mini, gpt-4.1 на тех же промптах — nano непригоден для извлечения (15/27), gpt-5.4-mini
  лучше по F1 (0.92 vs 0.73) при 5.7× цене, gpt-4.1 та же точность при 15× цене; в объяснениях
  сильные модели чаще вписывают производные числа, которые ловит валидатор. Решение —
  gpt-4o-mini остаётся; судья answer-vs-facts — gpt-4.1-mini (согласие с gpt-4o-mini 53 %).
- **xPts v0 против бейзлайнов** (walk-forward GW2–4, `docs/xpts.md`): MAE 1.07–1.19 против
  1.27–1.42 у form3/ppg, Spearman 0.67–0.74 против 0.61–0.69. **Живой тест GW5 против `ep_next` FPL**
  (`EVALS.md` §10): MAE 1.146 против 1.166 (разница −0.02, 95 % ДИ [−0.08; +0.04] — ничья), ρ 0.740
  против 0.734; среди 301 сыгравшего v0 точнее (MAE 2.02 против 2.22, ДИ не включает 0), в 25 из 35
  сильных расхождений ближе к факту; но top-20 `ep_next` набрал больше (4.5 против 4.0 очка, 3 против 2
  попаданий) — утверждение «v0 лучше официального прогноза» не подтверждено; сухари занижены
  (121 против 184).
- **Vision**: `detail=high` против `low` — 15/15 карточек и цены против 1/15 цен и подмены
  игрока; `high` — дефолт (`docs/vision.md`).
- **KB**: hit@6 = 1.0 во всех режимах, keyword recall 1.0, 16/17 цитат valid, «not covered» 2/2
  ($0.0041 за 10 ответов, `docs/strategy_kb.md`).

## Документы

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — слои, путь одного запроса, модель данных, решения
  и альтернативы, отказы, утечки из будущего, стоимость, гипотезы; диаграммы — [`docs/diagrams/`](docs/diagrams/).
- [`docs/EVALS.md`](docs/EVALS.md) — golden, метрики, A/B #1–#4, гиперпараметры, модели, A/B чата;
  [`docs/LLM_CHOICE.md`](docs/LLM_CHOICE.md) — выбор LLM по ролям; [`docs/chat_diagnosis.md`](docs/chat_diagnosis.md) — диагностика чата.
- [`docs/DEFENSE_QA.md`](docs/DEFENSE_QA.md) — питч, сценарий демо, ответы на вопросы §6.2;
  [`docs/PLAN_STATUS.md`](docs/PLAN_STATUS.md) — план, статус и риски.
- Компоненты: [`docs/agent.md`](docs/agent.md), [`docs/rag.md`](docs/rag.md),
  [`docs/strategy_kb.md`](docs/strategy_kb.md), [`docs/xpts.md`](docs/xpts.md),
  [`docs/optimizer.md`](docs/optimizer.md), [`docs/mcp.md`](docs/mcp.md), [`docs/vision.md`](docs/vision.md),
  [`docs/ui.md`](docs/ui.md), [`docs/sources.md`](docs/sources.md).

## Структура проекта

```
fpl-copilot/
├── README.md, docker-compose.yml, Dockerfile, pyproject.toml, uv.lock, .env.example
├── src/fplcopilot/
│   ├── config.py, db.py                 # pydantic-settings из .env; SQLAlchemy session_scope
│   ├── data/                            # FPLClient (публичный API, дисковый кэш), pydantic-схемы
│   ├── rag/                             # ингест новостей, чанкинг, индекс pgvector, гибридный поиск,
│   │   ├── kb/                          #   извлечение PlayerSignal, summary_check, новости клуба
│   │   │                                #   (team_news), пакетный refresh; Strategy KB (RAG #2)
│   │   └── sources.py, sources_kb.yaml  # реестры источников
│   ├── prompts/v1 … v4/                 # промпты сигнала (дефолт v3, v4 — опция) и дайджеста клуба (v1)
│   ├── core/                            # xPts v0, модель минут, сила фикстур (рейтинги / The Odds API),
│   │                                    #   Understat, MILP с фишками BB / TC, стратегии, кандидаты,
│   │                                    #   планы, бэктест, факты «Почему» — без LLM (кроме why_narrate)
│   ├── agent/                           # LangGraph-граф, инструменты LiveTools, резолвер имён и
│   │   └── prompts/v1 … v4/             #   транслит, чат (история), валидатор, трейсинг; промпты агента
│   ├── mcp_server/                      # fpl-intelligence: 14 tools, 5 resources, 1 prompt
│   ├── vision/                          # скриншот -> JSON -> резолюция -> правила FPL -> Squad
│   ├── app/                             # Streamlit: Home.py (навигация) + views/1_squad … 7_compare
│   └── migrations/001–010.sql           # идемпотентные миграции
├── evals/                               # golden/ (+ chat.jsonl, chat_holdout.jsonl), run_rag.py, run_kb.py,
│                                        #   run_chat.py, run_hparams.py, run_models.py, judge.py,
│                                        #   metrics.py, prompts/, results/*.json
├── skills/fpl-transfer-analyst/         # SKILL.md + references/ (копия в .cursor/skills/)
├── scripts/                             # migrate, dev_up, agent_demo, gw_review, mcp_smoke,
│                                        #   squad_from_screenshot, ingest_report, smoke_fpl
├── samples/                             # синтетические скриншоты Pick Team
├── tests/                               # 822 теста (812 без сети) на 24.09: фейки, unit, db, llm
└── docs/                                # ARCHITECTURE, EVALS, DEFENSE_QA, PLAN_STATUS, документы
                                         #   компонентов, screenshots/, diagrams/
```

## Обязательные модули и чек-лист задания (§3, §8) — где в проекте

Строки 1–10 — обязательные модули §3 (без любого из них проект не допускается к защите),
11–16 — артефакты сдачи. ✅ сделано · ⚠️ частично / ожидает действия · ⏳ не сделано

| # | пункт | статус | evidence |
|---|---|---|---|
| 1 | Собственный MCP-сервер с 2+ tools | ✅ 14 tools (11 агента, incl. `search_strategy_kb`, `rank_players`; +3 справочных), 5 resources, 1 prompt; stdio + streamable-http; контракт ошибок | `src/fplcopilot/mcp_server/`, [`docs/mcp.md`](docs/mcp.md), `tests/test_mcp_*.py` (22) |
| 2 | Собственный Skill с SKILL.md | ✅ frontmatter с триггерами, workflow по типам вопросов, шаблон, правила, references | [`skills/fpl-transfer-analyst/SKILL.md`](skills/fpl-transfer-analyst/SKILL.md), `.cursor/skills/` |
| 3 | LangGraph с многошаговым workflow | ✅ 14 узлов; ветвление после `router` (16 интентов) / `resolve_clarification` / `candidate_news` / `check_action` / `confirm_action`; цикл `grade_signals ⟲ rewrite_retry` ≤ 2 и один пересчёт `candidate_news → compute`; два HITL `interrupt_before` (хит/Wildcard; выбор игрока) с SQLite-чекпоинтами; история диалога; промпты v4 + A/B v3 → v4 на 60 вопросах | `src/fplcopilot/agent/graph.py`, [`docs/agent.md`](docs/agent.md), [`docs/agent_prompt_ab.md`](docs/agent_prompt_ab.md), [`docs/EVALS.md`](docs/EVALS.md) §9, `tests/test_agent_*.py` (70), `test_chat_v4.py` + `test_candidate_news.py` + `test_cyrillic_names.py` (58) |
| 4 | RAG с обоснованным выбором компонентов | ✅ два RAG: parent-child чанки ≤ 300 токенов с заголовком, `text-embedding-3-small`, pgvector HNSW, BM25 + RRF + time-decay, flashrank cross-encoder, abstention, валидатор цитат и проверка summary кодом; новости клуба (выбывшие — из данных FPL, контекст — дайджест с проверенными цитатами); срок действия сигнала по содержанию | [`docs/rag.md`](docs/rag.md), [`docs/strategy_kb.md`](docs/strategy_kb.md), [`docs/EVALS.md`](docs/EVALS.md) |
| 5 | Обработка документов / скрапинг | ✅ RSS + полный текст trafilatura, Google News backfill, статусы FPL API; HTML → markdown для KB; недоступные JS/Cloudflare-сайты задокументированы | [`docs/sources.md`](docs/sources.md), `src/fplcopilot/rag/ingest.py`, `rag/kb/fetch.py` |
| 6 | Мультимодальность | ✅ vision: скриншот Pick Team → строгий JSON → детерминированная резолюция → правила FPL → тот же `Squad`, что из API; используется в UI как источник состава | [`docs/vision.md`](docs/vision.md), `src/fplcopilot/vision/`, `tests/test_vision_*.py` (71); ⚠️ реальный скриншот ещё не прогонялся |
| 7 | LangSmith с реальными трейсами | ⚠️ проводка сделана (`wrap_openai`, `@traceable`, callbacks графа с тегами `intent:*`, `model:*`, `strategy:*`), ключ задан; на 24.09 **месячная квота трейсов проекта исчерпана (429)** — чтобы показать дашборд на защите, нужно решение владельца (новый проект / организация, платный план или Langfuse); приложение при отказах LangSmith само выключает трейсинг и работает дальше | `src/fplcopilot/rag/llm.py`, `agent/tracing.py`, [`docs/PLAN_STATUS.md`](docs/PLAN_STATUS.md) §4 |
| 8 | Golden dataset 10+ и автоматизированные evals | ✅ 55 строк RAG/KB + 60 вопросов чата + 15 holdout; `evals.run_rag` (с воспроизводимым корпусом `--corpus-cutoff`), `run_kb`, `run_chat`, `run_hparams`, `run_models`; результаты — JSON в `evals/results/` | `evals/golden/`, [`docs/EVALS.md`](docs/EVALS.md), `tests/test_evals_*.py` |
| 9 | A/B эксперимент с выводами | ✅ A/B #1 режимы поиска; A/B #2 промпт v2 + retrieval v2 + abstention; A/B #3 проверка summary + промпт v3 (и найденная утечка из будущего); A/B #4 `form_notes` (отклонён по точности); A/B чата v3 → v4; vision high/low; гиперпараметры T / top_p / max_tokens с повторами; сравнение пяти моделей | [`docs/EVALS.md`](docs/EVALS.md) §3–4, §7–9, [`docs/vision.md`](docs/vision.md), `evals/results/` |
| 10 | Обоснованный выбор LLM и гиперпараметров | ✅ по семи ролям: модель, альтернативы, стоимость/латентность/качество, дизайн фоллбэка; дефолты в коде (gpt-4o-mini; T = 0 везде, объяснение переведено с 0.2 на 0 по §4.2; max_tokens 400 / 120 / 1400 (v3+: 2000) / 800); ⚠️ исключение 23.09 — текст «Почему» на `gpt-4o` (≈ $0.006 за вызов против ≈ $0.0004 на mini): описан в `LLM_CHOICE.md`, но A/B mini против 4o не делался — кандидат на проверку | [`docs/LLM_CHOICE.md`](docs/LLM_CHOICE.md), `src/fplcopilot/agent/llm.py`, `config.py`, [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §6 |
| 11 | Веб-фронтенд | ✅ Streamlit, 7 экранов (+ «Сравнение»), чат с историей и HITL (подтверждение хита/WC и выбор игрока), фишки в плане, новости клуба в карточке игрока, скриншот состава, ID менеджера в адресе и cookie | [`docs/ui.md`](docs/ui.md), `docs/screenshots/`, `tests/test_app_*.py` (53) |
| 12 | GitHub-репозиторий с README и инструкцией | ✅ публичный репозиторий [ZHKAIR/nfactorial-fpl-copilot](https://github.com/ZHKAIR/nfactorial-fpl-copilot); README — этот файл, запуск — «Как запустить», сервер — [`docs/deploy.md`](docs/deploy.md) | [`docs/PLAN_STATUS.md`](docs/PLAN_STATUS.md) |
| 13 | ARCHITECTURE.md или mindmap | ✅ | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/diagrams/`](docs/diagrams/) |
| 14 | EVALS.md с метриками и результатами | ✅ golden, метрики, A/B #1–4 и чата, эволюция промпта, гиперпараметры, модели; результаты — JSON в `evals/results/` | [`docs/EVALS.md`](docs/EVALS.md) |
| 15 | Презентация | ⏳ не начата; структура и тайминг — в [`docs/DEFENSE_QA.md`](docs/DEFENSE_QA.md) | — |
| 16 | Развёрнутое демо или однокомандный локальный запуск | ✅ публичный URL https://fpl-copilot.duckdns.org (вход по паролю, прод-стек `docker-compose.prod.yml` + Caddy HTTPS); локально — `docker compose up -d` (db + app + ingest) | `docker-compose.prod.yml`, `docker-compose.yml`, `Dockerfile`, [`docs/deploy.md`](docs/deploy.md) |

Рекомендуемые пункты §4: **guardrails** — есть (доменный отказ betting/off-topic и просьб
игнорировать инструкции без LLM-объяснения, документы как ДАННЫЕ против prompt injection, валидатор
имён, чисел, туров, капитана и кириллических имён в ответе, проверка summary сигнала кодом,
детерминированная проверка правил FPL для скриншота); **кэширование** — есть (дисковый кэш FPL API
и Understat, сигналы и дайджесты клубов ≤ 12 ч, `st.cache_data`, кэш входов оптимизатора,
`inputs_hash` планов, http-кэш KB, дисковый кэш текстов «Почему»); **Docker compose** — есть;
**собственный eval-фреймворк / нестандартные метрики** — есть (walk-forward бэктест xPts против
бейзлайнов, `validation_fixes`, false-evidence, ложные abstention, воспроизводимый корпус на `as_of`,
раннер чата с ручной оценкой по строкам и пробником стабильности роутера, hallucination через
`price_mismatch`/`club_mismatch` в vision); **внешний API** — FPL API, The Odds API, Understat. Нет:
fallback между моделями (есть только повтор того же вызова при 429), CI/CD, роль-модель
пользователей (есть только общий пароль входа `APP_PASSWORD`), голос, fine-tuning, внешние
пользователи (см. Roadmap). **Деплой на публичный URL** — есть (https://fpl-copilot.duckdns.org,
[`docs/deploy.md`](docs/deploy.md)); **суточный лимит запросов к LLM** — `APP_DAILY_LLM_LIMIT`.

## Roadmap — что сознательно вырезано и почему

Порядок работ переставлен под критерии оценки (после разбора расхождений исходного плана с
заданием 17.09; рабочие заметки студента в репозиторий не входят, итог — в
[`docs/PLAN_STATUS.md`](docs/PLAN_STATUS.md) §1–2): всё, что оценивается — агент, MCP, Skill, RAG,
evals — сделано раньше «ML-фаз» первоначального плана из 17 фаз. Что не вошло в MVP:

| отложено | почему сейчас нет | где задел |
|---|---|---|
| Free Hit и автоматический выбор тура фишки | Bench Boost / Triple Captain в **заданном пользователем** туре сделаны 23.09 (цель тура меняется, новых ограничений нет); Free Hit — состав на один тур с возвратом, отдельная ветка модели; тур фишки оптимизатор не выбирает | `docs/optimizer.md`, «Фишки в туре», «Что дальше» |
| Mini-league advisor | нужны picks соперников и метрики EO — отдельный модуль, не влияющий на обязательные критерии | исходный план проекта (вне репозитория), `docs/PLAN_STATUS.md` §2 |
| Odds как дефолтный провайдер сложности | провайдер работает с 22.09 (The Odds API, 20/20 матчей GW6–GW7 сопоставлены), но дефолт — `team_rating`, пока разбор GW6 не покажет, чей FSI ближе к факту (`v0-odds` сохранён рядом с `v0`); коэффициенты — только вход модели, пользователю не показываются | `core/odds.py`, `docs/xpts.md`, `XPTS_FIXTURE_PROVIDER=odds` |
| LightGBM xPts v1 | 4 тура данных — обучать нечего; v0 без ML уже бьёт бейзлайны; v1 — на vaastav 2016–2026 после сезона данных | `docs/xpts.md`, «Идеи для v1» |
| Telegram-напоминания о дедлайне | удобство, не интеллект; UI и MCP закрывают доступ к системе | — |
| Деплой на публичный URL, CI/CD с evals на PR | инфраструктурные шаги после стабилизации кода; на 24.09 не сделаны; LangSmith-дашборд упирается в исчерпанную квоту трейсов | `docs/PLAN_STATUS.md` |
| Fallback на другую модель | один провайдер (structured outputs OpenAI); при отказе LLM есть повтор при 429 и детерминированные ответы | `docs/ARCHITECTURE.md` §8 |
| Проверка логики прозы объяснителя целиком | валидатор проверяет числа, имена, туры, капитана, скамейку, «игрока не из состава», но не рассуждение: в прогоне 4 объяснитель назвал Gabriel (£8.0) подходящим под бюджет £7m, хотя маршруты бюджет соблюдают (c12); уточнение иногда наследует совет ассистента из истории (c55) | `docs/EVALS.md` §9 |
| Внешние пользователи | демо идут на публичных менеджерах 895045 и 6856911; реальные вопросы из UI (137 тредов) стали основой диагностики и golden чата, но фидбека участников мини-лиги пока нет | `docs/chat_diagnosis.md` |

## Лицензия и благодарности

Лицензия не выбрана (учебный проект; будет добавлена при публикации). Проект опирается на:

- [sertalpbilal/FPL-Optimization-Tools](https://github.com/sertalpbilal/FPL-Optimization-Tools) —
  формулировка MILP (`dev/solver.py`): цель с `decay_base`, `bench_weights`, `ft_value`, `itb_value`,
  динамика бесплатных трансферов; наши λ/ω/пороги и no-good cuts — свои.
- [flashrank](https://github.com/PrithivirajDamodaran/FlashRank) — cross-encoder `ms-marco-MiniLM-L-12-v2` на ONNX.
- [HiGHS](https://highs.dev) через `highspy` и [PuLP](https://github.com/coin-or/pulp) — солвер и моделирование.
- [LangGraph](https://github.com/langchain-ai/langgraph), [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk),
  [Streamlit](https://streamlit.io), [pgvector](https://github.com/pgvector/pgvector), [rank_bm25](https://github.com/dorianbrown/rank_bm25),
  [trafilatura](https://trafilatura.readthedocs.io), [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz), OpenAI API.
- Данные: официальный публичный FPL API (`fantasy.premierleague.com/api`); новости — BBC Sport,
  Sky Sports, The Guardian, Fantasy Football Scout (RSS), Google News; база знаний —
  premierleague.com (The Scout), LiveFPL, FPLWatch, FPL Pilot, Fantasy Football Fix, Fantasy
  Football Scout, Draft Fantasy, GoalIQ (полный реестр с статусами — `docs/sources.md`,
  `docs/strategy_kb.md`). Тексты используются для извлечения сигналов и цитирования с указанием
  источника; проект не даёт советов по ставкам.
- [The Odds API v4](https://the-odds-api.com) (`soccer_epl`, рынок h2h, бесплатный тариф 500 запросов/мес, кэш 6 ч) — сложность матчей и ожидаемые голы для xPts при `XPTS_FIXTURE_PROVIDER=odds`; наружу только FSI 1–5 / xG / p(clean sheet), коэффициенты не показываются (`docs/xpts.md`).
- [Understat](https://understat.com) (EPL 2026/27, без ключа, `GET /getLeagueData/EPL/2026`, кэш 12 ч) — xG/xA/удары игрока и командный xGA; слабый blend в xPts (`XPTS_UNDERSTAT_BLEND`) и блок на карточке игрока. Фолы не берём.
