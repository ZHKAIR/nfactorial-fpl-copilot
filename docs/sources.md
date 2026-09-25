# Источники новостей

Для каждого источника проверены: HTTP-статус с браузерным User-Agent, парсинг feedparser, наличие дат публикации,
извлечение полного текста статьи trafilatura. Реестр в коде — `src/fplcopilot/rag/sources.py`.

| Источник | URL | Статус | Записей / даты | Полный текст | Заметки |
|---|---|---|---|---|---|
| BBC Sport football | `feeds.bbci.co.uk/sport/football/rss.xml` | **включён** | 79, все с датой (~2.5 мес.) | да (3–6k символов) | ссылки с `?at_medium=RSS&at_campaign=rss` — трекинг срезается при нормализации URL |
| Sky Sports football | `skysports.com/rss/11095` | **включён** | 20, все с датой (~последние 2 мес.) | да; видео-страницы дают <200 символов → берём summary | 20 записей — окно короткое, нужен регулярный опрос |
| Sky Sports News (все виды спорта) | `skysports.com/rss/12040` | выключен | 20, с датой | да | общая лента (MMA, крикет, F1) — мало футбола, дублирует 11095 |
| The Guardian football | `theguardian.com/football/rss` | **включён** | 59, все с датой | да (2–7k символов) | без paywall; в ленте попадаются вечнозелёные материалы старых лет |
| Fantasy Football Scout | `fantasyfootballscout.co.uk/feed/` | **включён** | 12, все с датой (~2 дня) | частично (1–3k символов; часть контента только для подписчиков) | самый релевантный для FPL: пресс-конференции, травмы, «Scout Notes» |
| premierinjuries.com | `/feed/`, `/rss`, `/injury-table.php` | выключен | — | — | 403 «Just a moment…» — Cloudflare JS-challenge на всём сайте; без headless-браузера недоступен |
| premierleague.com news | `/rss`, `/news/rss`, `/news` | выключен | — | — | RSS нет (404); страница новостей и «Injuries» рендерятся JS |
| Google News RSS (search) | `news.google.com/rss/search?q=...&hl=en-GB&gl=GB&ceid=GB:en` | **включён только для `--backfill`** | до 100 на запрос, все с pubDate (глубина до ~9 мес.) | **нет** | ссылки — JS-редиректы `news.google.com/rss/articles/<id>`; новый формат id не декодируется base64 (нужен batchexecute) → храним заголовок + издателя из `<source>`; суффикс « - Издатель» в заголовке срезаем |
| FPL API (`bootstrap-static`) | `fantasy.premierleague.com/api/bootstrap-static/` | **включён** | все игроки; новость — у игроков с непустым `news` (дата — `news_added`) | структурированные поля | источник `fpl_api`: `player_status_snapshots` + документ `fpl://player/{id}/news/{news_added}`; авторитетный статус/шанс сыграть |

Запросы backfill: `"Premier League injury news"` и по одному на клуб
`"<полное название>" injury OR injured OR doubt OR fit OR press conference` (полные названия — `SEARCH_NAMES`).

## Правила ингеста

- Дедупликация по `news_articles.url` (UNIQUE) после нормализации: нижний регистр схемы/хоста,
  без фрагмента, без трекинг-параметров (`utm_*`, `at_*`, `ns_*`, `CMP`, `oc`, `fbclid`, …),
  без завершающего слэша, параметры отсортированы.
- `published_at` — только timezone-aware UTC; записи без даты или с неправдоподобной датой
  (1970, будущее) пропускаются (`no_date` в логе).
- **Окно сезона** `NEWS_BACKFILL_SINCE` (дефолт `2026-07-01`, начало предсезонки 2026/27):
  статьи старше не сохраняются ни из одного источника (`too_old` в логе). Без окна Google News
  и «вечнозелёные» материалы Guardian приносят заголовки с 2017 года (957 таких записей в backfill).
- Полный текст — trafilatura; короче 200 символов считаем неудачей и берём RSS summary.
- Вежливость: User-Agent браузера, таймаут 15 с, пауза 1 с между загрузками, лимит новых
  записей на источник за прогон (`NEWS_LIMIT_PER_SOURCE`, по умолчанию 200).
- `content_hash` (sha256 нормализованного заголовка+текста) — задел на дедупликацию одинаковых
  статей с разных URL при чанкинге.

## Снимки статусов FPL API (`player_status_snapshots`)

Источник `fpl_api` делает два дела: (1) документ в `news_articles` на каждую новость игрока
(`fpl://player/{id}/news/{news_added}`), (2) **историю переходов** в `player_status_snapshots`.
Снимок пишется, когда `(status, chance_next, news, news_added)` игрока отличается от его
последнего сохранённого снимка — включая возврат в строй (`status='a'`, `news=''`). Игроки без
истории, доступные и без новости, не пишутся (иначе ~460 пустых строк). Состояние может законно
повторяться (травма → здоров → травма → здоров), поэтому UNIQUE-ограничения нет — дедуп это
сравнение с последним снимком (`snapshot_needed`), повторный прогон без изменений даёт 0 строк.
Так RAG и evals получают цепочку «травма → выздоровление», а не только «травма».

## Матчер сущностей (`rag/entity_matcher.py`)

Детерминированный, без LLM. Текст очищается от диакритики (`Ødegaard` → `Odegaard`), токенизируется,
n-граммы ищутся в словаре алиасов (жадно, от длинных к коротким).

- **Клубы:** `name`, `short_name` (только в верхнем регистре; коды-омонимы слов `NEW SUN EVE TOT CRY`
  исключены) и прозвища по `short_name` (`Spurs`, `Man City`, `Forest`, `Villa`, `Toon`, …).
  Намеренно нет `United`, `City`, `Blues`, `Reds` — неоднозначны.
- **Игроки:** `web_name` (и без инициалов: `M.Salah` → `Salah`), `second_name` целиком, его первая и
  последняя фамилии, полные имена (`Bukayo Saka`, `Gabriel Magalhaes`, `Gabriel Martinelli`).
- **Регистр:** заглавная буква алиаса требует заглавной в тексте (`Wolves`/`Forest`/`White` не
  срабатывают на `wolves`/`forest`/`white`), строчная принимает любой — ALL-CAPS заголовки проходят.
- **Правило неоднозначности.** Алиас неоднозначен, если указывает на нескольких игроков (`Silva`,
  `Palmer`; `Gabriel` — web_name одного и first_name других) или является обычным английским словом
  (`White`, `Wood`, `Rice`, …). Такой алиас присваивается **только** если: (а) в тексте есть полное
  имя игрока; (б) ровно один из кандидатов уже найден по однозначному алиасу; (в) ровно один
  кандидат играет за клуб, упомянутый в этом же тексте. Иначе упоминание игнорируется:
  пропустить упоминание дешевле, чем приписать травму другому игроку.
  Пример: «Arsenal's Gabriel is a doubt» → никому (Магальяес, Мартинелли и Жезус — все Arsenal);
  «Gabriel Magalhaes is a doubt» → Магальяес; «Bournemouth defender Silva» → António Silva.
- **Чужие имена.** Однословный алиас, к которому через пробел примыкает другое слово с заглавной
  буквы, считается другим человеком/объектом и пропускается: «Enzo Maresca», «Rúben Amorim»,
  «Keith Andrews», «Marco Silva», «Old Trafford». Исключения: сосед — часть названия клуба, часть
  имени самого игрока, ALL-CAPS или обычное слово заголовка (`Injury`, `Update`, `Boost`, дни/месяцы,
  страны — `HEADLINE_WORDS`). Знаки препинания и перенос строки между словами снимают правило
  («Doku, Sarr, Foden» матчится). Без этого правила в топе упоминаний оказываются Enzo/Rúben/
  Andrews/Trafford — менеджеры и стадион.
- **Известное ограничение:** знаменитые однофамильцы без соседнего имени («the Ferguson era»,
  «Carrick» как тренер) остаются шумом; возможное решение — контекст клуба/роли или LLM-верификация.
- После правок алиасов корпус перетегируется без скачивания:
  `uv run python -m fplcopilot.rag.ingest --rematch`.

## Как запустить

```bash
docker compose up -d db                                   # Postgres 16 + pgvector
uv run python scripts/migrate.py                          # идемпотентные SQL-миграции
uv run python -m fplcopilot.rag.ingest --once --backfill  # первый прогон: RSS + Google News + FPL API
uv run python -m fplcopilot.rag.ingest --loop --every 30 --index  # фоновый сбор каждые 30 минут + эмбеддинг нового
docker compose up -d --no-deps ingest                     # то же в контейнере (restart: unless-stopped); логи: docker compose logs -f ingest
uv run python scripts/ingest_report.py                    # сводка по корпусу (= ingest --report)
uv run pytest -q -m "not network"                         # unit + db (db скипается без Postgres)
```

Дальше по корпусу работает RAG (чанки, эмбеддинги, гибридный поиск, сигналы) — см. `docs/rag.md`.
