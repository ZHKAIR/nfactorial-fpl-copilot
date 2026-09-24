# MCP-сервер `fpl-intelligence` и Skill `fpl-transfer-analyst`

Шаг 8b (после vision 8a, до Strategy KB 8c). `src/fplcopilot/mcp_server/` выставляет по
Model Context Protocol **14 инструментов, 5 ресурсов (2 статических + 3 шаблона) и 1 промпт**, чтобы
ими пользовался любой MCP-хост — Cursor, Claude Desktop, Inspector, свой клиент. Инструменты: 11
инструментов агента (`agent/tools.py`, `TOOL_NAMES`; десятый — `search_strategy_kb`,
стратегическая база знаний с цитатами, `docs/strategy_kb.md`, плюс ресурс `fpl://kb/stats`;
одиннадцатый — `rank_players`) и 3 справочных: `get_player_advanced_stats`,
`get_player_points_breakdown`, `get_team_defensive_profile`. `skills/fpl-transfer-analyst/` — Skill
(SKILL.md + references), который объясняет хосту, *в каком порядке* звать эти инструменты и *как*
оформлять ответ.

```bash
uv run python -m fplcopilot.mcp_server                                    # stdio (Cursor / Claude Desktop)
uv run python -m fplcopilot.mcp_server --transport streamable-http --port 8765   # http://127.0.0.1:8765/mcp
uv run python scripts/mcp_smoke.py                                        # клиент stdio: list + 5 вызовов + 2 ресурса
npx @modelcontextprotocol/inspector@latest --cli uv run python -m fplcopilot.mcp_server -- --method tools/list
uv run pytest -q tests/test_mcp_server.py                                 # 21 unit-тест без сети/БД
uv run pytest -q tests/test_mcp_integration.py -m db                      # spawn через stdio + 3 вызова
```

## Зачем свой MCP: факты против решений

Публичные FPL MCP-серверы (обёртки над `fantasy.premierleague.com/api`) отдают **факты**:
bootstrap, фикстуры, picks, историю игрока. Мы их не дублируем — наш `data/fpl_client.py` те же
эндпоинты читает напрямую с дисковым кэшем. `fpl-intelligence` отдаёт **решения и их
основания**: xPts v0 по турам с компонентами и дисперсией, MILP-маршруты трансфера с вердиктом
по хиту, многотуровый план с альтернативой Wildcard, лучшие 11 и капитанские опции, новостной
сигнал с цитатами. Это то, чего нет ни у FPL API, ни у чужих MCP, и то, что LLM сама не посчитает
без ошибок (см. валидатор в docs/agent.md: объяснитель складывал числа). Букмекерские
коэффициенты — отдельный внешний MCP/API, который позже станет *входом* модели xPts
(`XPTS_FIXTURE_PROVIDER=odds`), но никогда не выйдет наружу: инструменты о ставках не говорят.

Принцип не меняется: **LLM-хост объясняет, сервер считает**. Каждый инструмент — тонкая обёртка
над `LiveTools`; логика не дублируется, MCP лишь (1) разрешает имена игроков в id, (2) собирает
pydantic-вход, (3) отдаёт `model_dump()` с округлением до 2 знаков и обрезкой длинных списков.

## Инструменты

Docstring функции = описание инструмента для LLM (когда звать, что вернёт, единицы). Все
числовые поля округлены до 0.01; списки длиннее 25 элементов обрезаются (`<key>_omitted = N`),
строки — до 400 символов (цитаты evidence целиком).

| tool | зачем | вход (JSON) | что под капотом | LLM | латентность* |
|---|---|---|---|---|---|
| `get_gameweek_context` | точка входа: тур, дедлайн, as-of, состав/банк/FT/чипы/issues | `manager_id?`, `gw?` (только следующий тур), `strategy` | `LiveTools.get_gameweek_context` -> bootstrap, `load_inputs` (3 тура xPts, best_xi, diagnose) | нет | 0.9 с холодный, 30 мс тёплый |
| `predict_player` | xPts игрока на 1–8 туров: компоненты, sd, p_start, минуты, фикстуры (FSI, xG) | `player` (имя/id), `horizon=1` | резолвер -> `PredictionStore.get(gw)` (core/xpts) | нет | 0.5 с на 3 тура |
| `compare_players` | ранжирование 2+ игроков по сумме xPts | `players[]`, `horizon=3` | `predict_player` × N + сортировка | нет | 0.2 с/игрок |
| `analyze_player_risk` | доступность/ротация из новостей + статус FPL, цитаты `[source, dd.mm]` | `player` | последний `player_signals` (≤ 12 ч) иначе `rag.extract.extract_signal` (hybrid_rerank k=8) | **да, только при извлечении** (gpt-4o-mini, ~4 с, ≈ $0.001) | 6 мс cached / 3–5 с extracted |
| `optimize_team` | лучшие 11, скамейка, капитан/вице, top-3 капитанов с тегом по владению | `manager_id`, `strategy` | `best_xi` (MILP, 1 тур) + `_lineup_player` | нет | 0.3–1 с |
| `recommend_transfers` | top-3 маршрутов трансфера с вердиктом по хиту; принудительная продажа/покупка + `alternative` + `verdict_on_forced_sale` | `manager_id`, `strategy`, `horizon=3`, `allow_hit?`, `sell[]`, `buy[]`, `keep[]`, `exclude[]` | `single_transfer` или `constrained_routes` (та же MILP + no-good cuts) | нет | 1.3 с |
| `build_gameweek_plan` | план на N туров, FT копятся, WC-альтернатива, фишки BB / TC в названном туре, diff с прошлым планом, снимок в БД | `manager_id`, `horizon=5`, `strategy`, `allow_hits=true`, `chips?` = `[{gw, chip}]`, chip ∈ {`bboost`, `3xc`} | `plan_transfers` (+ `allow_hits` — Task 0, `chips` -> `validate_chip_plan`, docs/optimizer.md), `latest_plan`/`diff_plans`/`save_plan` | нет | 5–15 с (солвер) |
| `simulate_scenario` | what-if: продать/купить X, хит, Wildcard сейчас | `manager_id`, `sell[]`, `buy[]`, `keep[]`, `allow_hit?`, `use_wildcard?`, `strategy`, `horizon=3` | `constrained_routes` (+ «версия с хитом»), при WC — `plan_transfers(["wildcard"])` | нет | 1–3 с, WC ~10 с |
| `diagnose_squad` | проблемы состава: kind/severity/detail, `problem_players` | `manager_id`, `strategy` | `candidates.diagnose_squad` через `load_inputs` | нет | 0.5 с тёплый |
| `search_strategy_kb` | стратегическая KB: правила, чипы, хиты, FT, капитанство, цены — чанки с `title/url/source/tags/text/score` для цитируемого ответа «как играть»; хост отвечает **только** по чанкам, `rules`-теги авторитетны | `query`, `tags[]` ⊆ {rules, chips, transfers, hits, captaincy, structure, rank, prices, fixtures, defcon, beginner}, `k=6` (1–12) | `LiveTools.search_strategy_kb` -> `rag.kb.retrieve.KBRetriever.search` (dense + BM25 -> RRF -> flashrank, cap 2 на документ; reranker общий с новостным Retriever) | нет (эмбеддинг запроса) | 0.3–1 с тёплый; первый вызов 3–4 с (загрузка ранкера) |
| `rank_players` | рейтинг лиги без названного игрока: pts/£m, стабильность, очки, форма; сезон или окно прошлых туров — не прогноз | `metric`, `position?`, `max_price?`, `min_price?`, `limit=10` (1–25), `min_minutes?`, `gw_from?` / `gw_to?` или `last_n_gws?` | `LiveTools.rank_players` (bootstrap + `player_gw_history` для окна; нет истории -> `history_unavailable`) | нет | ~1 с |
| `get_player_advanced_stats` | xG/xA/удары Understat + стандарты FPL (пенальтист / штатник / сейвы); без коэффициентов; `team_id` для профиля клуба | `player_id` | `core/ext_stats.player_advanced_stats` | нет | 0.2 с тёплый кэш |
| `get_player_points_breakdown` | очки за последние N туров по категориям (выход, голы, ассисты, сухие, сейвы, бонус, DefCon, карточки, `other`) + метка lucky / unlucky / repeatable; сумма = `total_points`; объяснение, не вход xPts | `player_id`, `last_n=3` (1–8) | `core/points_form.points_breakdown` над `player_gw_history` (туры < следующего) | нет | < 0.1 с |
| `get_team_defensive_profile` | командный xGA Understat («мало пропускает»); фолы не отдаём | `team_id` | `core/ext_stats.team_defensive_profile` | нет | 0.2 с |

Три справочных инструмента (`get_player_advanced_stats`, `get_player_points_breakdown`,
`get_team_defensive_profile`) не входят в `TOOL_NAMES` агента (граф их не зовёт) и принимают id,
а не имя: id игрока — из `predict_player` / `get_gameweek_context`, id клуба — `team_id` из
`get_player_advanced_stats`. Фолы / стандарты соперника Understat стабильно не публикует — FBref отложен.

Фишки в `build_gameweek_plan`: без `chips` план прежний (без фишек). С `chips` солвер
оптимизирует трансферы под Bench Boost / Triple Captain в указанном туре; в ответе `chips`
[{gw, chip, name, points, players}] (вклад фишки уже в `xi_points_by_gw` / `expected_total`) и
`notes` (фишка в первом туре отключает WC-альтернативу). Схема пускает только `bboost` / `3xc` и
тур 1–38 (иное SDK отклоняет как `isError`); правила сезона — в `validate_chip_plan`:
`ChipPlanError` -> `{"error": "invalid_chip_plan", "hint": ..., "chips_available_by_gw": {"6":
["bboost", "3xc"], ...}}` — какие BB / TC у менеджера есть по турам горизонта.

\* M-серия, тёплый дисковый кэш FPL API, Postgres локально; первый вызов процесса + ~1 с на
bootstrap/фикстуры. Замеры — из smoke ниже и stderr-лога сервера (`tool <name> {...} ok in N ms`).

`search_strategy_kb` отдаёт чанки, а не готовый ответ: LLM-хост пишет ответ сам и цитирует
`[source]` / `[n]`. Внутри агента тот же поиск обёрнут ещё и в `answer_strategy_question`
(gpt-4o-mini + детерминированная проверка цитат, docs/strategy_kb.md §4) — через MCP он не
выставлен намеренно: хост сам является LLM, второй LLM внутри инструмента был бы лишним.
Пустой результат (узкий тег) приходит с `note` «say the topic is not covered», неизвестный тег
или `k` вне 1–12 — `invalid_input`.

Имена игроков резолвит тот же детерминированный `agent/resolve.PlayerResolver`, что и граф:
web name, полное имя, инициалы, контекст клуба, «тот, что в твоём составе» (для инструментов с
`manager_id`), доминирование по владению; числовая строка — id из bootstrap. Неоднозначность
никогда не разрешается угадыванием:

```json
{"error": "ambiguous", "field": "player", "mention": "Gabriel",
 "candidates": [{"id": 3, "name": "Gabriel", "full_name": "Gabriel dos Santos Magalhães", "team": "ARS", "position": "DEF", "price": 6.3, "ownership": 41.2}, ...],
 "hint": "'Gabriel' is a first name shared by 4 players; ask the user which one and retry with the full name or the numeric id"}
```

## Ресурсы и промпт

| URI | что отдаёт (JSON) |
|---|---|
| `fpl://gameweek/current` | `GameweekContext` без менеджера: `gw`, `current_gw`, `deadline`, `as_of` |
| `fpl://manager/{manager_id}/squad` | состав (picks последнего завершённого тура) с ролями и xPts, банк, FT, чипы, issues |
| `fpl://manager/{manager_id}/plan` | последний сохранённый план на предстоящий тур из `plan_snapshots` (ходы по турам, хиты, FT, рекомендация, WC-альтернатива, `created_at`) или `null` |
| `fpl://player/{player_id}/signal` | последний `PlayerSignal` из `player_signals` с evidence; `null`, если нет. Извлечение **не** запускает |
| `fpl://kb/stats` | размер стратегической KB: `docs`, `chunks`, `chunks_embedded`, `tokens`, `docs_by_source`, `tags{tag: {docs, chunks}}`, `max_fetched_at` (снимок 17.09.2026: 41 документ, 622 чанка, 9 источников) |

Промпт `pre_deadline_review(manager_id, strategy="balanced")` возвращает одно user-сообщение с
последовательностью вызовов (context -> diagnose -> risk/predict для проблемных -> recommend ->
optimize -> plan -> при хите/чипе `search_strategy_kb(tags=["hits"|"chips"], k=2)` с одной
цитатой правила) и правилами вывода — то же, что Skill, но доступно любому MCP-клиенту через
`prompts/get`.

## Реализация

- SDK: `mcp[cli] >= 2.2` (mcp 2.2.0). Класс высокого уровня — `MCPServer` (в SDK 1.x назывался
  `FastMCP`; в 2.x модуль `mcp.server.fastmcp` удалён, транспортные параметры переехали в
  `run()`). Декораторы `@mcp.tool() / @mcp.resource() / @mcp.prompt()` без изменений.
- Транспорты: `stdio` (по умолчанию) и `streamable-http` (`--host/--port`, дефолты
  `MCP_HOST=127.0.0.1`, `MCP_PORT=8765` в `config.py`/.env). SSE не включаем (legacy).
- Жизненный цикл: один `LiveTools` на процесс, создаётся лениво при первом вызове (поэтому
  `tools/list` мгновенный), закрывается в `lifespan` при остановке (`LiveTools.close()`: ONNX-
  ранкер, HTTP-клиент). Sync-инструменты SDK 2.x выполняет в worker-потоках — вызовы
  сериализованы `RLock` (`LiveTools` и PuLP не потокобезопасны).
- Логи — только в stderr (`logging.basicConfig(force=True)` заменяет RichHandler SDK на
  однострочный формат): `tool recommend_transfers {'manager_id': 895045, ...} ok in 1223 ms`.
  В stdio-транспорте stdout занят протоколом; SDK 2.x дополнительно переводит fd 1 в stderr.
- Контракт ошибок (`runtime.error_payload`): инструмент **никогда** не бросает исключение
  наружу, всегда dict:

| `error` | когда | `hint` |
|---|---|---|
| `invalid_input` | horizon вне 1–8, manager_id ≤ 0, < 2 игроков в compare, пустой сценарий, gw ≠ следующий | что исправить |
| `ambiguous` / `unknown_player` | резолвер (см. выше) | кандидаты / проверить написание |
| `squad_unavailable` | нет публичных picks (команда стартует позже, 404) | текст `SquadUnavailable`; player-level инструменты работают |
| `infeasible` | `ScenarioInfeasible` / `InfeasibleError` (нужно 3 трансфера при запрете хитов и т.п.) | ослабить ограничения |
| `invalid_chip_plan` (`chips_available_by_gw`) | `ChipPlanError` в `build_gameweek_plan`: фишки нет у менеджера, тур вне горизонта, две фишки в туре, одна фишка дважды за половину сезона | текст правила + какие BB / TC остались по турам |
| `history_unavailable` | `rank_players` с окном туров без `player_gw_history` | убрать окно (сезонные итоги), окно не аппроксимируется |
| `fpl_api_error` (`status`) / `fpl_api_unreachable` | HTTP-ошибка FPL API / сеть | 404 -> менеджер не найден; иначе повторить |
| `db_unavailable` | `SQLAlchemyError` (Postgres) | `docker compose up -d db`, `DATABASE_URL` |
| `internal` (`type`) | всё остальное | первые 300 символов сообщения; stack trace — в stderr |

  Ошибки валидации схемы (неверный `strategy`, не тот тип) отдаёт сам SDK как `isError=true`
  с текстом pydantic — LLM видит enum в схеме и обычно не промахивается.

## Подключение

**Cursor** — `.cursor/mcp.json` в корне проекта, без абсолютных путей: Cursor подставляет
`${workspaceFolder}` (корень, где лежит `.cursor/mcp.json`) в `command` / `args` / `env`
([docs: Config interpolation](https://cursor.com/docs/mcp)); аргументы заданы массивом, поэтому
пробел в пути не мешает:

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

Cursor читает проектный `.cursor/mcp.json` из **корня workspace**: откройте `fpl-copilot/` как
workspace (если workspace — родительская папка, скопируйте блок в `~/.cursor/mcp.json`, заменив
`${workspaceFolder}` абсолютным путём). `.env` проекта подхватывается pydantic-settings по
абсолютному пути модуля, так что переменные окружения в конфиге не нужны. **Если сервер не
стартует** (в логе MCP `No such file` / `uv: command not found` / литеральное `${workspaceFolder}`
в старой версии Cursor или в CLI-агенте): подставьте абсолютный путь к репозиторию вместо
`${workspaceFolder}` и полный путь к `uv` (`/opt/homebrew/bin/uv`) вместо `uv`. Skill лежит в
`.cursor/skills/fpl-transfer-analyst/` (копия `skills/…`, синхронизировать `rsync -a --delete
skills/fpl-transfer-analyst/ .cursor/skills/fpl-transfer-analyst/`); для Claude Code — тот же
контент в `.claude/skills/`.

**Claude Desktop** — `~/Library/Application Support/Claude/claude_desktop_config.json`, тот же
блок под ключом `mcpServers`; интерполяции там нет, путь абсолютный:

```json
{
  "mcpServers": {
    "fpl-intelligence": {
      "command": "/opt/homebrew/bin/uv",
      "args": ["run", "--project", "<ABSOLUTE_PATH_TO_REPO>", "python", "-m", "fplcopilot.mcp_server"]
    }
  }
}
```

(`uv` должен быть доступен приложению по полному пути — PATH у GUI-приложений короче, чем в
терминале.)

**Inspector** — UI: `npx @modelcontextprotocol/inspector uv run python -m fplcopilot.mcp_server`
(или к уже запущенному `--transport streamable-http` по URL `http://127.0.0.1:8765/mcp`).
CLI v2 требует разделитель `--` перед своими флагами, иначе `-m` из `python -m` трактуется как
`--method`:

```bash
npx -y @modelcontextprotocol/inspector@latest --cli uv run python -m fplcopilot.mcp_server -- --method tools/list --format json
npx -y @modelcontextprotocol/inspector@latest --cli uv run python -m fplcopilot.mcp_server -- --method tools/call \
    --tool-name predict_player --tool-args-json '{"player":"Haaland","horizon":1}' --format json
npx -y @modelcontextprotocol/inspector@latest --cli uv run python -m fplcopilot.mcp_server -- --method resources/read --uri fpl://gameweek/current
npx -y @modelcontextprotocol/inspector@latest --cli uv run python -m fplcopilot.mcp_server -- --method prompts/get \
    --prompt-name pre_deadline_review --prompt-args manager_id=895045
# v1 (deprecated) понимает форму из задания без «--»:
npx -y @modelcontextprotocol/inspector@1 --cli uv run python -m fplcopilot.mcp_server --method tools/list
```

## Проверка (17.09.2026, GW5, manager 895045)

`uv run python scripts/mcp_smoke.py` — клиент `stdio_client` + `ClientSession` из SDK спавнит
сервер, делает `initialize`, списки, пять вызовов и чтение двух ресурсов (сокращено; прогон
после добавления 10-го инструмента, 12:33Z):

```
server: fpl-intelligence v0.1.0 protocol 2025-11-25 | instructions 938 chars | 1678 ms
tools (10, 2 ms): get_gameweek_context, predict_player, compare_players, analyze_player_risk,
  optimize_team, recommend_transfers, build_gameweek_plan, simulate_scenario, diagnose_squad,
  search_strategy_kb
resources: ['fpl://gameweek/current', 'fpl://kb/stats']
resource templates: ['fpl://manager/{manager_id}/squad', 'fpl://manager/{manager_id}/plan', 'fpl://player/{player_id}/signal']
prompts: [('pre_deadline_review', ['manager_id', 'strategy'])]

== get_gameweek_context({}) 953 ms
  GW5 deadline 2026-09-18T17:30:00Z | as_of 2026-09-17T10:59:28Z
== predict_player({"player": "Haaland", "horizon": 3}) 486 ms
  Haaland (MCI, FWD, £15.5, status a) total_xpts 20.74 over 3 GWs
    GW5: xPts 7.13 ±4.08 p_start 0.98 min 84 | vSUN(FSI 1)
    GW6: xPts 5.72 ±3.34 p_start 0.98 min 84 | @LIV(FSI 3)
    GW7: xPts 7.89 ±4.43 p_start 0.98 min 84 | vIPS(FSI 1)
== recommend_transfers({"manager_id": 895045, "strategy": "balanced"}) 1253 ms
  GW5 FT 2 bank £0.1 | baseline XI 56.14 | recommendation: route 1
  #1 Konsa, João Pedro -> Barry, Guéhi | hit 0 | Δnext +4.03 | Δhor +9.78 | go
  #2 Palmer, João Pedro -> Barry, B.Fernandes | hit 0 | Δnext +3.56 | Δhor +9.72 | go
  #3 Konsa, Palmer -> Guéhi, Gibbs-White | hit 0 | Δnext +4.3 | Δhor +8.64 | go
== analyze_player_risk({"player": "João Pedro"}) 6 ms
  João Pedro: FPL status d (75%) | signal doubtful conf 0.7 p_start 0.6 | origin cached age 2.6 h | llm_calls 0
  summary: João Pedro is doubtful for Gameweek 5 due to an unspecified injury, with a 75% chance of playing ...
    [ffscout, 16.09] Joao Pedro (£7.8m) has pulled out of the Brazil squad for the upcoming September internationals ...
    [fpl_api, 16.09] João Pedro: Unspecified injury - 75% chance of playing
== search_strategy_kb({"query": "when is a -4 points hit worth it", "tags": ["hits"], "k": 3}) 3923 ms
  3 chunks (tags ['hits']):
    [internal] FPL 2026/27 rules digest (internal, fpl-transfer-analyst Ski ['rules', 'transfers', 'hits', 'chips', 'defcon', 'structure'] score 0.59
      Transfers, free transfers, hits - After GW1 one free transfer (FT) is added per gameweek; unused FTs bank up to a maximu…
    [premierleague] FPL Rules Copilot ['rules', 'transfers', 'hits', 'chips', 'beginner'] score 0.01
      Transfer rules After selecting your squad you can buy and sell players in the transfer market. Unlimited transfers can b…
      https://www.premierleague.com/en/news/4661029
    [internal] FPL 2026/27 rules digest ... score 0.0
      Strategy heuristics used by the tools (not official rules) - Hit verdict: `hit_marginal_gain` = extra discounted gain ov…
== read_resource(fpl://manager/895045/squad) 3 ms
  squad GW4 picks: Raya, Gabriel, De Cuyper, Konsa, Palmer(C), Rogers, Szoboszlai, Schade, João Pedro, Haaland, Wissa, Forster*, Hall*, O'Shea*, Slater*
  bank £0.1 FT 2 chips [] issues 4
== read_resource(fpl://kb/stats) 19 ms
  docs 41 chunks 622 (embedded 622) tokens 98152 | sources 9 | tags chips:208, beginner:147, rank:146, captaincy:121, transfers:117, prices:99
== get_prompt(pre_deadline_review) 1 ms: role user, 1879 chars

latencies (ms): {"initialize": 1678, "list_tools": 2, "get_gameweek_context": 70, "predict_player": 285,
  "recommend_transfers": 1308, "analyze_player_risk": 5, "search_strategy_kb": 3923, "read_resource": 3,
  "read_kb_stats": 19, "get_prompt": 1} | wall 7709 ms
```

Совпадает с оптимизатором из docs/optimizer.md (маршруты 1–2, baseline 56.14). Сигнал João
Pedro пришёл из кэша (4.1 ч < 12 ч) — LLM не вызывался; с `origin: extracted` вызов стоит
≈ $0.001 и 3–5 с. `search_strategy_kb` с тегом `hits` (в корпусе 2 документа / 43 чанка с этим
тегом) вернул официальную страницу правил и внутренний дайджест — именно их агент подшивает как
`rules_context` к решениям о хите; 3.9 с — первый вызов процесса (загрузка ONNX-ранкера flashrank
и эмбеддинг запроса), тёплый ~0.3–1 с. Inspector v2 CLI (`--method tools/list --format json`) вернул те же имена (на 17.09 — 10),
`tools/call predict_player` — `structuredContent` с `total_xpts 7.13`, `tools/call
predict_player {"player": "Gabriel"}` — ошибку `ambiguous` с четырьмя кандидатами,
`resources/read fpl://gameweek/current` — `application/json` с `gw 5`. Транспорт streamable-http
проверен клиентом `mcp.Client("http://127.0.0.1:8765/mcp")`: протокол 2026-07-28, все инструменты в списке,
`get_gameweek_context` за 83 мс.

**Повторная проверка stdio (23.09.2026, GW6)** — сервер запущен так же, как в `.cursor/mcp.json`
(`uv run --project <repo> python -m fplcopilot.mcp_server`), клиент `stdio_client` SDK:

```
tools (14): get_gameweek_context, predict_player, compare_players, analyze_player_risk, optimize_team,
  recommend_transfers, build_gameweek_plan, simulate_scenario, diagnose_squad, search_strategy_kb,
  rank_players, get_player_advanced_stats, get_player_points_breakdown, get_team_defensive_profile
resources: fpl://gameweek/current, fpl://kb/stats | templates: manager/{id}/squad, manager/{id}/plan,
  player/{id}/signal | prompts: pre_deadline_review
get_gameweek_context(6856911): chips_available [wildcard, freehit, bboost, 3xc]; (895045): []
get_player_advanced_stats(411): Haaland MCI team_id 15, Understat xG90 0.95, «пенальтист №1»
get_team_defensive_profile(15): xGA/матч 1.44 < медиана 1.72 -> concedes_little
build_gameweek_plan(6856911, horizon 5, chips=[{gw 7, bboost}]) 6.5 с: transfers, expected 308.28 vs
  baseline 274.13; chips: Bench Boost GW7 +7.31 (Dubravka, João Pedro, Ajayi, Kusi-Asare); GW7 XI 73.66
build_gameweek_plan(895045, chips=[{gw 7, bboost}]) 0.4 с: invalid_chip_plan «Bench Boost недоступен
  менеджеру в GW7 …», chips_available_by_gw {"6": [], …, "10": []}
build_gameweek_plan(6856911, horizon 3, chips=[{gw 11, bboost}]): invalid_chip_plan «тур вне горизонта
  плана GW6–GW8»; chips=[{gw 6, freehit}] -> isError (схема: 'bboost' or '3xc')
```

## Skill `fpl-transfer-analyst` (`skills/`, копия в `.cursor/skills/`)

```
skills/
├── README.md                                  # что такое Skill, почему не промпт, как тестировать
└── fpl-transfer-analyst/
    ├── SKILL.md                               # 146 строк: триггеры, workflow по типам вопросов (+ strategy question, фишки в плане, рейтинг, разбор очков), шаблон, правила, примеры
    └── references/
        ├── fpl_rules_2026_27.md               # правила 2026/27: скоринг с DefCon, FT/хиты, два набора чипов и GW19, дедлайны (индексирован в KB как источник internal)
        └── output_template.md                 # шаблон ответа + откуда брать каждую колонку + заполненный пример
```

Frontmatter: `name: fpl-transfer-analyst`, `description` с триггерами («should I sell/buy/keep»,
«who to captain», «take a -4 / points hit», «wildcard / free hit», «starting XI / bench order»,
«plan my transfers», «is X fit/injured», общие вопросы о правилах и стратегии);
`disable-model-invocation` не выставлен — Skill должен подхватываться по контексту. Тело: когда
применять; таблица «тип вопроса -> порядок инструментов» (status -> `analyze_player_risk` +
`predict_player`; transfer -> `diagnose_squad` -> `recommend_transfers` -> подтверждение при
хите/WC; plan -> `build_gameweek_plan` (+ `chips` для названного тура BB / TC); captain/lineup ->
`optimize_team`; league ranking -> `rank_players`; past points -> `get_player_points_breakdown` /
`get_player_advanced_stats` / `get_team_defensive_profile`; what-if ->
`simulate_scenario`; **strategy question -> `search_strategy_kb` -> ответ только по чанкам с
цитатами, «not covered» вместо выдумки**); правило «при хите/чипе — одна цитата правила из
`search_strategy_kb(tags=["hits"|"chips"], k=2)` как `[source]`»; шаблон вывода (+ формат ответа
на стратегический вопрос: `[n]` + нумерованные Sources); жёсткие правила (не считать самому, без
ставок и коэффициентов, as-of и «состав = picks последнего завершённого тура», `ambiguous` ->
спросить, правила только из KB, ответ на языке пользователя); мини-примеры: два хода про João
Pedro/Saka и стратегический вопрос на русском. Копия в `.cursor/skills/` синхронизирована
(`diff -rq` пуст). Тестирование — в `skills/README.md`.

## Тесты

`tests/test_mcp_server.py` (21, без сети/БД/LLM): реестр — ровно 14 имён по явному списку
`MCP_TOOLS` (первые 11 = `TOOL_NAMES`; `search_strategy_kb`: `required=["query"]`, `k` по
умолчанию 6, read-only), описания > 80 символов, JSON-схемы (`required`, enum стратегий,
`allow_hits`, `chips` — необязательный список `{gw, chip}` с enum `bboost|3xc`), аннотации read-only;
`build_gameweek_plan` без `chips` передаёт пустой список, с `chips` — типизированные условия,
`ChipPlanError` -> `invalid_chip_plan` с `chips_available_by_gw`, `freehit` / `gw=0` отклоняет
схема до инструмента; справочные инструменты на фейках (Understat-индекс из снимка, история
туров): xG90, `team_id`, `concedes_little` по медиане лиги, сумма категорий = `total_points`,
метка `lucky`, пустая история, ошибки входа; `rank_players` (валидация, окно, `history_unavailable`);
ресурсы (`fpl://gameweek/current`, `fpl://kb/stats`) / шаблоны / промпт; промпт содержит
инструменты в порядке Skill'а и шаг `search_strategy_kb`; форма ошибок резолвера (`ambiguous` с
кандидатами, `unknown_player`, id-строки, «Palmer» -> тот, что в составе); инструмент возвращает
`ambiguous` **до** вызова `LiveTools` (фейк); обёртка округляет до 0.01 и режет списки
(`notes_omitted`); `search_strategy_kb` валидирует пустой запрос / `k` / неизвестные теги до
инструмента и режет текст чанка до 400 символов; `fpl://kb/stats` читает `LiveTools.kb_stats`;
контракт ошибок (`squad_unavailable`, `internal`, `invalid_input`) без stack trace;
`error_payload` по типам исключений; `verdict_on_forced_sale`; `Runtime.close`.
`tests/test_mcp_integration.py` (1, `db`): спавн `python -m fplcopilot.mcp_server` через
`stdio_client`, `initialize`, `tools/list` (14 имён = `MCP_TOOLS`), ресурсы / шаблоны / промпт,
`get_gameweek_context`, `predict_player("Haaland")`, `get_player_points_breakdown`,
`fpl://gameweek/current`; скип без Postgres или без bootstrap.

## Ограничения

- Моделируется только предстоящий тур (`gw` в `get_gameweek_context` — валидация, не выбор);
  состав = picks последнего завершённого тура, цена продажи = текущая, FT — оценка по истории
  (docs/optimizer.md). Состав со скриншота (docs/vision.md) через MCP пока не передаётся —
  следующий шаг: инструмент `set_squad_override`.
- `analyze_player_risk` — единственный инструмент с LLM и записью в БД (`player_signals`);
  `build_gameweek_plan` пишет `plan_snapshots` (в том числе план с фишками). Остальное read-only.
- Фишки в плане — только Bench Boost и Triple Captain в туре, который назвал пользователь; Free
  Hit не моделируется, лучший тур для фишки сервер сам не ищет (перебор туров — N solve'ов).
- Вызовы сериализованы: параллельные запросы от хоста ждут друг друга (план 5 туров — до 15 с).
  `read_timeout` клиента должен быть ≥ 60 с для `build_gameweek_plan`.
- Аутентификации нет: streamable-http слушает `127.0.0.1`; для удалённого доступа нужен
  reverse-proxy с токеном (в SDK есть `auth=`, не включено).
- Инспектор v2 при `--cli` без `--` ошибочно съедает `-m`; v1 (`@1`) устарел, но понимает
  форму без разделителя.
