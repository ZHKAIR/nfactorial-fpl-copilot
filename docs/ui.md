# UI: Streamlit-приложение и Docker

Пакет `src/fplcopilot/app/` — тонкий интерфейс над теми же инструментами, что использует
агент и MCP-сервер (`agent/tools.py: LiveTools`), и над LangGraph-агентом (`agent/graph.py`).
**В UI нет бизнес-логики**: страницы вызывают инструменты и форматируют результат помощниками
`app/format.py` (чистые функции, `tests/test_app_format.py`). Все числа — из инструментов, уже округлённые,
поэтому совпадают с CLI (`python -m fplcopilot.core.optimizer …`, `python -m fplcopilot.agent.cli …`).

```bash
uv run streamlit run src/fplcopilot/app/Home.py          # локально, http://localhost:8501
./scripts/dev_up.sh                                       # db (docker) -> migrate -> streamlit
docker compose up -d                                      # db + app (+ ingest) -> http://localhost:8501
uv run pytest -q tests/test_app_*.py                      # 94: format + ui_kit/briefing + вход по паролю + AppTest на фейках (Брифинг, HITL: хит/WC и уточнение имени) + squad_override
uv run pytest -q tests/test_compare.py tests/test_plan_chips.py tests/test_manager_state.py tests/test_player_*.py   # ещё 73: сравнение, фишки плана, ID менеджера, карточка игрока
```

## Страницы ↔ инструменты

| Страница (`app/views/`) | Что показывает | Инструменты |
|---|---|---|
| **Брифинг** (`0_briefing.py`, стартовая `/`) | по референсу «The Briefing»: приветствие по имени менеджера (FPL `entry.player_first_name`, время суток), подзаголовок «До дедлайна нужно сделать трансфер, выбрать капитана и …», карточка дедлайна; **главный ход** — рекомендуемый маршрут (продаём → покупаем аватарами клубов, выигрыш за 3 тура, «почему» — детерминированный `format.route_card`, таблетки: вердикт, хит, Δ ближайшего тура, шанс старта покупаемых, банк после; «Что может пойти не так» — риски покупки из `risk_note`) или «держать»; **капитан** — два лучших варианта «VS» (очки с повязкой); **«Следить до дедлайна»** — серьёзные проблемы состава с сохранённым разбором новостей (источник · дата); **прогноз на тур** — лучший состав из ваших 15 и ваш нынешний старт (оба из `optimize_team`) и место в общем зачёте («топ N %» от `bootstrap.total_players`); прогноз по турам с трансферами и без — на «Плане»; лента «Новости за 10 дней» — важные новости состава и до 5 заголовков лиги (подробнее ниже); внизу строка вопроса в чат. Действия — только переходы (`st.switch_page`: «К дедлайну», «Сравнение» с `a`/`b`) и вопрос в чат (`queued_prompt`). Логика — чистые функции `app/briefing.py` | `get_gameweek_context`, `recommend_transfers(horizon=3, allow_hit=False)`, `optimize_team`, `predict_player(horizon=5 / 3)`, `analyze_player_risk(cached_only=True)`, свежие статьи корпуса RAG (`news_articles`, SQL без LLM) |
| **Мой состав** (`1_squad.py`, `/squad`) | вкладки **«Поле»** (футболки клубов по схеме, C / V, точка проблемы, матч и прогноз на тур, скамейка; имя — ссылка на «Игрок», `ui_kit.pitch_html`) и **«Таблица»** (HTML-таблица smartplay, описана в этой же ячейке ниже); **шапка команды** (6 метрик: Очки сезона · Общий ранг · Банк · Бесплатных трансферов · Чипы · Прогноз очков на GW — сумма xPts стартовых 11, капитан ×2); **строка серьёзных проблем** (`format.serious_problems` -> `problems_line`, `st.warning`): именной список только по статусу FPL (под вопросом / травма / дисквалификация / недоступен, с текстом новости FPL) и риску ротации (`diagnose_squad: not_playing` + p(start) из прогноза) — «Haaland — под вопросом (75 %), колено · Saka — риск ротации (выйдет в старте 55 %)»; тяжёлый календарь сюда не входит (виден цветом); без серьёзных проблем строки нет; **HTML-таблица 15 игроков по образцу smartplay Players** (`format.squad_table_rows` -> `format.squad_table_html`, `st.markdown(unsafe_allow_html=True)`): старт GKP→DEF→MID→FWD, строка-разделитель «Bench», затем скамейка; колонки `# · Роль (C/VC/—/Зап1…4) · Player · Price · TSB% · Form · Pts/Game · xMins GW6 · xPts GW6 · 3GW xPts · FRR# · GW6…GW10`; ячейка Player в две строки — имя 15px/600 **ссылкой на страницу «Игрок» `/player?pid=<element id>`** (`format.PLAYER_URL`) + значок только у проблемных (🟡 под вопросом / ротация, 🔴 травма / дисквалификация / недоступен; `title` = «Под вопросом (75 %) — новость FPL»), ниже серым «MUN · MID · 38.8%»; числа вправо; календарь «TOT (H)» с фоном по сложности 1–5 (`format.FSI_COLORS`) и легендой «Fixture difficulty: Very Easy … Very Tough»; подсказки — `title=` на `<th>` и серый «?» (`format.SQUAD_COLUMN_HELP`, по-русски); FRR# — ранг календаря клуба среди 20 по средней сложности 5 туров (`app/player_card.team_fixture_run_rank`, прогноз одного представителя на клуб); стили нейтральные (rgba серого), читаются в светлой и тёмной теме; **«Проблемы состава»** — expander только по игрокам из строки проблем: замечания `diagnose_squad` и цитаты (источник, дата, ссылка); **«Скриншот состава»** — expander внизу (раскрыт, только если состава нет): одна подсказка, uploader, «Распознать», итог «Распознано 15 игроков, капитан Saka» / «Не сошлось: 14 игроков вместо 15» (`format.parsed_result_text`), таблица карточек, «Использовать этот состав», подробности (экран, модель, токены, $) — в popover «Подробности» (`format.parsed_details`); кнопки-примеры `samples/` — только при `?demo=1` (`demo_mode()`, для демонстрации и AppTest) | `get_gameweek_context` (+ `diagnose_squad` внутри `load_inputs`), `predict_player(horizon=5)`, `analyze_player_risk(cached_only=True)` только для проблемных, bootstrap из кэша `LiveTools` (TSB% / Form / Pts/Game), `vision.squad_from_image` / `to_squad` |
| К дедлайну (`2_deadline.py`, `/deadline`) | сначала ответ — карточка **«План на GW N»** (`app/deadline_plan.py`, без LLM): нумерованные шаги — трансфер из рекомендованного маршрута (бесплатный / платный, банк после, зачем), кто уходит на скамейку, **«не продаём»** для ценных травмированных игроков (цена продажи против цены обратного выкупа по истории трансферов), смена старта, схема и капитан; затем лучшие 11 (вкладки «Поле» / «Таблица», xPts лучших 11 vs вашего старта), карточки «Проверка состава» и «Варианты капитана» (safe / balanced / differential по владению); top-3 маршрутов трансфера карточками: out → in, Δ след. тур, Δ горизонт, хит, вердикт (go / hit_not_worth / hold), «Почему» — факты кода (календарь, история, пенальти), формулировка LLM с запасным шаблоном (`core/why_narrate.py`); переключатель «Разрешить платный трансфер (минус 4 очка)» | `optimize_team`, `recommend_transfers(horizon=3, allow_hit)`, `predict_player(horizon=3)` для игроков маршрутов, цены продажи из истории трансферов (`FPLClient.sale_prices`) |
| План (`3_plan.py`, `/plan`) | сверху «Итог плана» (подробнее — раздел про «План» ниже); настройки одной строкой: горизонт 3 / 4 / 5 / 6 GW, платные трансферы вкл/выкл, **чипы** (`app/plan_view.py`, `app/plan_chips.py`) — popover, где для каждого доступного менеджеру Bench Boost / Triple Captain выбирается тур горизонта (сыгранные в этой половине сезона не предлагаются; доступных нет — подпись вместо выбора); с фишкой — строка «Bench Boost в GW7: +6.27 xPts скамейки (…)» / «Triple Captain в GW7: +N xPts — капитан X ×3», колонки «Фишка» / «Вклад фишки» в таблице по турам, обе суммы (план и «без трансферов») считаны с фишкой; нарушение правил фишки (`ChipPlanError`) — «Фишку нельзя запланировать» с причиной вместо traceback; Wildcard в туре фишки — «не сравнивался»; ходы по турам с причиной (`out_problem`), FT/хит/банк/XI/капитан по турам, целевой состав, план vs «без трансферов», альтернатива Wildcard и рекомендация; **«Что изменилось»** — diff с последним сохранённым снимком (смена фишек — строка «фишки»); кнопка «Сохранить снимок» (с теми же фишками) | `build_gameweek_plan(save=False, chips=[…])`; внутри — `plan.latest_plan` -> `diff_plans`; кнопка — тот же инструмент с `save=True` (`save_plan`); с фишкой — свой `st.cache_data`, ключ включает фишки |
| Игрок (`4_player.py`, `/player`) | раскладка SmartPlay «Player report» (раздел «Карточка игрока по SmartPlay» ниже); поиск по bootstrap (`find_players`; неоднозначно -> radio; прямая ссылка `?pid=<id>`); в expander'ах — xPts по турам (столбики, горизонт 5), компоненты xPts, календарь («Календарь: следующие 5 туров»): соперник, дома/в гостях, **сложность матча FSI 1–5 с подписью — коэффициенты не показываются**; стандарты и Understat (xG/xA); новостной сигнал: доступность, p(start), минуты, ротация, confidence, цитаты со ссылками и датами; подблок **«Форма и контекст»** (`app/form_notes_view.py`: форма / роль / позиция / стандарты из того же сигнала с цитатами — только у сигналов промпта v4, `player_signals.form_notes`; в вердикт доступности и xPts не входит, у сигналов v1–v3 подблока нет); кнопка «Обновить сигнал» — принудительное извлечение с числом LLM-вызовов, токенами, $ и мс; блок **«Новости клуба»** (`app/team_news_view.py`): таблица «Недоступны или под вопросом (данные FPL)» — игрок, позиция, статус FPL с шансом, «Вернётся GWN» по календарю, новость FPL, сигнал новостей — и дайджест клуба (ротация / слова тренера / форма) с цитатами и ссылками; при открытии — сохранённый дайджест любого возраста без LLM, кнопка «Обновить новости клуба» — извлечение (время, ≈ $); подпись «не доказательство доступности самого игрока» | `predict_player(horizon=5)`, `analyze_player_risk(cached_only=True)` / `analyze_player_risk(force=True)`, `team_news(cached_only=True)` / `team_news(force=True)` |
| **Сравнение** (`7_compare.py`, `/compare`) | два игрока бок о бок: выбор «Поиск» (как на «Игрок», radio при неоднозначности) или «Из состава» (два селекта), адрес `?a=<id>&b=<id>` сохраняет пару; кнопка «Сравнить»; HTML-таблица метрик в колонках smartplay (Price, TSB%, Form, Pts/Game, Pts, xMins и xPts ближайшего тура, 3GW / 5GW xPts, Start%, FRR#, Pens, Understat xG90 / Shots/90) и календарь на 6 туров с FSI; «Объяснение» — вердикт на ближайший тур и на 3 тура, «Почему», «Из новостей», оговорки, «Подробности» (модель, токены, $, мс); факты собирает код (`app/compare.py: build_compare_facts`), текст — `gpt-4o-mini` (`agent/prompts/v2/compare.system.md`), кэш в `session_state` по (a, b, стратегия); без `OPENAI_API_KEY` — те же факты в expander «Факты (без ИИ)» | `predict_player(horizon=6)` для пары и представителей клубов (FRR#), `analyze_player_risk(cached_only=True)`, `core/ext_stats` (Understat) |
| Чат (`5_chat.py`, `/chat`) | история `st.chat_message`; **промпты v4** (`AGENT_PROMPT_VERSION`, по умолчанию `v4`): в `Agent.stream(..., history=…)` уходят 6 последних реплик (`agent/chat.history_from_messages`), уточнения «а на GW7?», «а если без хита?», «а кого тогда продать?» роутер переписывает в самостоятельный вопрос; во время прогона — спиннер «Думаю…», затем markdown-ответ на языке вопроса. Лог узлов и сводка прогона (интент, инструменты, LLM-вызовы и стоимость, валидация, версия промптов, `standalone_query`) пользователю не показываются — они остаются в состоянии графа, evals и LangSmith; **HITL**: при прерывании — карточка действия (вид, детали, цена) и кнопки «Подтвердить» / «Отклонить» -> `Agent.stream_resume(thread_id, decision)`, карточка уточнения имени с кнопками-кандидатами; каждый запрос и каждое продолжение списываются с суточного лимита LLM (`app/llm_budget.py`); 6 кнопок-примеров на русском (`format.example_prompts(ui.gw)`: состав, продажа без хита, Bench Boost через тур, капитан, разбор прошлого тура, календарь; тур подставляется из контекста, без контекста — без номера); «Очистить историю» | LangGraph-агент (все инструменты `TOOL_NAMES` через `compute` + внутренние `team_fixtures`, `chips_status`, `rank_forecast`, `transfer_trends`, `team_news`, `review_gameweek`) |
| О системе (`6_about.py`, `/about`) | только архитектура: **граф агента из скомпилированного LangGraph** (`agent.graph.get_graph()` -> `format.agent_graph_dot`; подсвечены оба узла-прерывания `confirm_action` и `resolve_clarification`), схема потока данных (`format.pipeline_dot`; число инструментов — `len(TOOL_NAMES)`), таблица «Страница → инструменты» (`format.PAGE_TOOLS`). Пояснения — в `help` подзаголовков; метрик, команд запуска и ссылок на docs на экране нет | — |

Принцип текстов на «Мой состав» и «О системе»: на экране — действие и результат, объяснения —
в help (?); без упоминаний моделей, стоимости, docs и «синтетических эталонов».

`Home.py` — `st.set_page_config`, экран входа (при заданном `APP_PASSWORD`, см. «Прод: вход и
лимит LLM») и `st.navigation` с вкладками в шапке: 8 экранов — Брифинг (`/`, стартовый) · Мой
состав (`/squad`) · К дедлайну (`/deadline`) · План (`/plan`) · Игрок (`/player`) · Сравнение
(`/compare`) · Чат (`/chat`) · О системе (`/about`).

Дизайн — по референсу Figma «FPL Copilot UI/UX Exploration» (три концепции, смешаны):
«Брифинг» — The Briefing; «Мой состав» и «К дедлайну» — Tactics Room (поле + справа карточки
«Проверка состава» — кого выпустить / убрать и смена капитана — и «Варианты капитана» 1–3 с
тегом по владению); «План» — Season Canvas (переключатель 3 / 4 / 5 / 6 GW, карточка на каждый
тур: ходы Out → In с Δ и причиной, «−4», прогноз, капитан, FT, банк; целевой состав чипами,
Wildcard, «Что изменилось» — полоса-наблюдение + таблица diff; таблицы плана — в expander).
Из референса **не перенесено** (нет данных или действия): «confidence %» у трансфера, тактика
соперника («38 % шансов слева»), «Save team / Apply change / Set captain» (записи в FPL нет),
8 GW, перетаскивание карточек, «On track / Flexibility», выдуманные источники новостей, окно
What-if и боковая панель разбора.

### Система дизайна

- Светлая «рама»: светлые шапка, сайдбар и фон `#F4F7F9`, белые карточки, акцент — синий
  эмблемы `#06A8FD`, кнопки `#0277B8`. Семантика цвета: синий — рекомендация и главное,
  зелёный — плюс, жёлтый — риск, красный — проблема. Токены — `app/theme.py` (`TOKENS_CSS`,
  `light-dark()`), цвета Streamlit — `app/.streamlit/config.toml`.
- Тема по умолчанию — **светлая**: Streamlit следует настройке ОС, поэтому
  `theme.DEFAULT_LIGHT_JS` один раз на адрес записывает «Light» в тот же ключ localStorage, что
  меню ⋮ (`stActiveTheme-<path>-v2`, метка `fplc-theme-default-<path>`); тёмная — ⋮ → Dark,
  выбор пользователя дальше не трогается.
- Шрифты с кириллицей (OFL, `app/static/`, `[[theme.fontFaces]]` latin / latin-ext / cyrillic):
  **Manrope** — весь текст (одна строка — один шрифт), **JetBrains Mono** — только числа и
  короткие метки. Иерархия: подпись секции 11.5 px капсом → заголовок карточки 22 px → текст
  14.5 px → крупное число 30 px. Эмблема в сайдбаре 150 px по центру (надписи на кольце герба
  читаются, отдельного вордмарка нет), в шапке при свёрнутом сайдбаре — 46 px.
- Компоненты — `app/ui_kit.py` (чистые функции, `tests/test_app_ui_kit.py`): подписи, заголовок
  карточки, крупное число, таблетки, аватары с цветом клуба, строка трансфера, «VS», новость,
  блок риска, наблюдение, столбики и парные столбики, рейтинг, поле (нападающие вверху, вратарь
  внизу, скамейка под полем), карточки туров с бейджем чипа. CSS — `ui_kit.KIT_CSS`, внедряется
  из `Home.py` (`theme.inject`). Карточки с виджетами — `st.container(border=True, key="card_…")`,
  шапка страницы — `common.page_head`. Тексты UI — естественный русский, числа очков через
  `format.points_text` («1 очко / 2.5 очка / 5 очков»); «Спросить ассистента» вместо «Copilot».

### Карточка игрока по SmartPlay и лента новостей

- «Игрок» повторяет раскладку smartplayfpl.com/players/N: шапка «Player report» (метки цена ·
  позиция · клуб, крупное имя, клуб и статус, крупный xPts, строка xPts · Points · Form · 3GW
  xPts · Own% · Net Tx), секции парами — Overview, Minutes, Attacking Output (+xGI), Advanced
  Attacking (Creativity, Threat, xG/90; Shots и npxG Understat — только у сопоставленных),
  Defence & Reliability (у вратаря — Saves) — строками «число + полоска перцентиля»,
  Ownership & Transfers полосой, Fixtures (next 6 GWs), похожие игроки: Competes for minutes,
  **Alternatives at the price** (`player_card.alternatives_at_price`: та же позиция, другой клуб,
  ±£0.5m, статус «a»), The differential bridge — с кнопкой «Compare» (открывает «Сравнение»).
  Наше — ниже в expander'ах: прогноз xPts по турам (HTML-столбики, подписи горизонтально) и
  компоненты, рейтинг календаря, «Откуда очки» списком, стандарты и Understat, новости.
- Синяя точка у метрики — входит в прогноз xPts (`Metric.model`, подсказка «как»): модель минут,
  xG / xA / xGI / xG/90, защитные действия, сейвы, очередь пенальти и штрафных, blend Understat.
  Справка (в прогноз не идут): очки, форма, фактические голы / ассисты, Creativity / Threat,
  сухие матчи и xGC игрока, владение, трансферы, удары, npxG.
- «Следить до дедлайна» на Брифинге — строка на игрока (точка, имя-ссылка, короткий статус,
  источник · дата) и только игроки вашего старта или лучшего состава тура
  (`briefing.watch_filter`): запасной с риском ротации составу тура не мешает.
- Лента «Новости за 10 дней» (`briefing.briefing_news`, без LLM при показе): сверху важные
  новости по игрокам состава из сохранённых разборов (`analyze_player_risk(cached_only)`,
  `briefing.is_important` / `news_feed`) — выделены меткой «важно» и цветом статуса; ниже — до 5
  заголовков лиги из корпуса RAG (`briefing.league_headlines`: без статусов FPL API, предпочтение
  BBC / Sky / Guardian / FFScout и темам травм, капитана, календаря). Медленная бесконечная
  прокрутка с паузой при наведении и затуханием; `prefers-reduced-motion` — статичный список.
- Строка вопроса ассистенту — компактная форма с синей рамкой внизу экрана (не
  `st.chat_input`: тот прокручивает страницу вниз при загрузке).

### «План»: ответ сначала и чипы

- Сверху — «Итог плана»: фраза «В GW6 сделайте трансфер A → B и ещё N ходов. План на 5 туров
  даст X очка — на Y больше, чем если ничего не менять» (`app/plan_view.py`), три числа (по
  плану / ничего не менять / разница), эффект чипов. Ниже — настройки одной строкой (горизонт
  3–6, платные трансферы, чипы), карточки туров (ходы — главное; прогноз, капитан, FT, банк —
  мелко), «Прогноз по турам» двумя столбиками и детали в expander'ах.
- «Если ничего не менять» — лучший состав из нынешних 15 на каждый тур без трансферов и чипов
  (`PlanOut.baseline_xi_points_by_gw`, та же baseline-модель, что `baseline_total`); «по плану»
  — `xi_points_by_gw` минус −4 за платные трансферы тура. Оба — из одного вызова
  `build_gameweek_plan`.
- Чипы (`app/plan_chips.py` + `app/plan_view.py`): Bench Boost и Triple Captain — для каждого
  доступного менеджеру (`GameweekContext.chips_available`: окна bootstrap минус сыгранные, по два
  набора на сезон) выбирается тур горизонта; в один тур — один чип. План пересчитывается с
  `BuildPlanInput.chips=[PlanChipIn(gw, chip)]`, нарушение правил — понятная ошибка
  (`ChipPlanError`); эффект — разница с тем же планом без чипов (оба вызова кэшируются). MILP
  (`core/optimizer.py`): Bench Boost — очки всех 15 (старт и скамейка — как их поставит
  менеджер), Triple Captain — капитан ×3; baseline «без трансферов» — с теми же чипами. Free Hit
  не моделируется, Wildcard план сравнивает сам (альтернатива в первом туре).

Сайдбар общий для всех страниц (`common.sidebar()`) — панель управления, объяснения только в
`help` (?): «ID менеджера FPL» (`app/manager_state.py`) — поле в `st.form`, кнопки «Сохранить» и
«Забыть»; Enter в поле = «Сохранить» (первая кнопка формы): только цифры, пробелы обрезаются, не
число -> «Введите число», нет в FPL (404 entry/{id}) -> «Команда с ID N не найдена в FPL» — в обоих
случаях прежний id остаётся; успех -> ID применён, записан в адрес и в cookie `fplc_manager`
(`st.html(..., unsafe_allow_javascript=True)`, 365 дней), toast «ID сохранён на этом устройстве»;
«Забыть» (или «Сохранить» с пустым полем) стирает ID из сессии, адреса и cookie. Источник ID:
`?manager=<id>` в адресе → `session_state["manager_id"]` (int, контракт страниц) → cookie
(`st.context.cookies` новой сессии) → `APP_DEFAULT_MANAGER_ID` (только локальное демо) → пусто;
`FPL_MANAGER_ID` в UI не подставляется. Меню `st.navigation` очищает query params — сайдбар на
каждом запуске возвращает `manager`, не трогая `pid`, `a`/`b`, `demo`; мусор в `?manager=` молча
убирается, 404 — убирается с сообщением. Без ID — подсказка «Где взять ID: сайт FPL → Points —
число в адресе `/entry/<ID>/`». Под полем подтверждение
«<команда> · 56 очков · ранг 10 724 305» из `FPLClient.entry` (`st.cache_data(ttl=600)`; 404 ->
«Менеджер не найден в FPL»; 0 -> «Без менеджера»); «Стратегия» — «Осторожная / Сбалансированная /
Агрессивная» (`format.STRATEGY_LABELS` / `STRATEGY_HELP`, внутренние значения conservative /
balanced / aggressive) + caption «Применено: сбалансированная»; строка «**GW6** · дедлайн 10 окт,
15:00 · через **17 д 20 ч**» (местное время); короткий «Состав: из FPL за GW5» с help про скриншот /
«Состав: со скриншота (GW6)» + кнопка «Сбросить» / warning «Состав: нет — загрузите скриншот»
(`format.squad_source_text`);
две строки свежести «Новости: 11 мин назад» и «Разбор новостей об игроках: 3 дн назад» (⚠ и help,
если старше 24 ч; `format.freshness_lines`), красный «Прогноз на GW6 не рассчитан» только когда в
`xpts_predictions` нет строк на тур; «Обновить данные» (help «Перечитать данные FPL и новости»);
внизу мелко честный статус трейсинга (`common.langsmith_status_text` по `agent/tracing.tracing_status`):
«LangSmith: включён», «LangSmith: выключен» (нет ключа), «выключен — месячная квота трейсов
исчерпана» или «выключен — LangSmith отклоняет трейсы (429)» — после отказов загрузки процесс сам
выключает трейсинг и работает дальше (`tracing._LangSmithFailureGuard`, `docs/agent.md`). Предупреждения про OPENAI_API_KEY и недоступную БД —
только при проблеме. В текстах сайдбара нет слов «сигнал», «БД», «строк», «фикстура», «picks».

Файлы страниц лежат в `app/views/` (не `pages/`): каталог `pages/` Streamlit подхватывает как
«старую» многостраничность и до первого вызова `st.navigation` показывает сырые имена файлов;
`st.navigation` в `Home.py` даёт русские названия, иконки и чистые URL (`/` — «Брифинг»,
`/squad`, `/deadline`, `/plan`, `/player`, `/compare`, `/chat`, `/about`).

Язык таблицы состава: метки колонок смartplay (`Player`, `Price`, `TSB%`, `Form`, `Pts/Game`,
`xMins`, `xPts`, `3GW xPts`, `FRR#`) и легенда сложности оставлены по-английски намеренно
(референс скопирован буквально); всё остальное — шапка, строка проблем,
значки, expander'ы, сайдбар, подсказки (?) — по-русски. Таблица — HTML (не `st.dataframe`):
только так получаются ячейка игрока в две строки, ссылки на страницу «Игрок» и фон ячеек
календаря; сортировки по колонкам нет (15 строк).

## Источник состава: picks или скриншот

1. Если у менеджера есть публичные picks (`entry/{id}/event/{gw}/picks/` за последний
   завершённый тур) — они и используются; в сайдбаре «Состав: из FPL за GW N». Все страницы явно
   говорят, что это picks прошлого тура (трансферы текущего тура публичный API не отдаёт до
   дедлайна).
2. Если picks нет (404: команда стартует в этом туре — как 10835228 до дедлайна GW5 — или ID без
   истории) — страница «Мой состав» показывает причину из `SquadUnavailable`
   («manager 10835228 ('Nfactorial_test') has no public picks yet: the team starts in GW5…»),
   остальные страницы — предупреждение и подсказку загрузить скриншот. Уровень игроков (страница
   «Игрок», чат про статус/сравнение) работает без состава.
3. Скриншот (PNG/JPG/WEBP экрана Pick Team / Transfers, или один из двух синтетических примеров в
   `samples/`) -> `vision.squad_from_image` (gpt-4o-mini, ≈ $0.007) -> таблица карточек
   (на экране -> игрок FPL, позиция, клуб, цена на экране vs FPL, уверенность, метод сопоставления)
   и список нарушений правил (`blocking` / предупреждение). Кнопка **«Использовать этот состав»**
   появляется только при `is_valid` (нет blocking-нарушений); она делает `to_squad(...)` ->
   `LiveTools.squad_override_from(...)` -> `st.session_state["squad_override"]` (`SquadOverride`).
4. Дальше **каждый инструмент уровня состава получает `squad_override=`**: контекст, диагностика,
   лучшие 11, трансферы, план, сценарии, а чат передаёт его в `Agent.stream(..., squad_override=)`.
   Ключи кэша страниц включают отпечаток состава (`SquadOverride.fingerprint()`), поэтому числа
   для picks и для скриншота не смешиваются. Проверено на 895045: picks -> 3-4-3, 56.14 xPts,
   капитан Haaland, маршрут #1 Konsa, João Pedro -> Guéhi, Barry; скриншот
   `pick_team_valid.png` -> 5-4-1, 52.44 xPts, капитан B.Fernandes, FT 1, банк £0.5m, маршрут
   #1 Senesi -> Guéhi (`docs/screenshots/deadline_override.png`).

### Поддержка override в инструментах и агенте (необязательный параметр, по умолчанию — picks)

`agent/tools.py`:

- `class SquadOverride(BaseModel)` — компактное JSON-описание состава не из API: `source`, `gw`,
  `picks: list[Pick]`, `bank`, `team_value`, `free_transfers`, `chips_used`;
  `from_squad(squad)`, `to_squad(bs, manager_id)`, `fingerprint()`. `SquadLike = Squad | SquadOverride`,
  `as_override(...)`.
- `LiveTools.get_gameweek_context / diagnose_squad / optimize_team / recommend_transfers /
  build_gameweek_plan / simulate_scenario(inp, *, squad_override: SquadLike | None = None)`;
  `LiveTools.inputs(..., squad_override=None)` подменяет `client.squad()` прокси `_ClientWithSquad`,
  так что `optimizer.load_inputs` получает состав со скриншота без изменений в `core/`.
  `LiveTools.override_squad(manager_id, so)`, `LiveTools.squad_override_from(squad, manager_id=)`
  (сыгранные чипы — из публичной истории менеджера, если есть; иначе Wildcard считается доступным).
- `PlayerRiskInput.cached_only: bool = False` — только сохранённый сигнал любого возраста
  (с пометкой `stale …`), без LLM-извлечения; для таблиц UI.
- `PlanOut.target_squad: list[str]` — целевой состав к концу горизонта.

`agent/graph.py` / `agent/state.py`: `AgentState.squad_override: dict | None` (JSON
`SquadOverride`, живёт в чекпоинтах — resume после HITL видит тот же состав);
`Agent.run(...) / Agent.stream(..., squad_override: SquadLike | None = None)`; узлы вызывают
инструменты уровня состава через `squad_tool(fn, state)` — kwarg передаётся только при наличии
override, поэтому фейки в тестах без этого параметра работают как есть. `live_deps(tools=)` /
`live_agent(tools=)` — UI делит один `LiveTools` между страницами и агентом.

## HITL в UI

`Agent.stream` (на экране — спиннер «Думаю…») -> `Agent.snapshot(thread_id)`. Если
`interrupted` и есть `pending_action` (хит или Wildcard в основной рекомендации), сообщение
ассистента показывает карточку «Требует подтверждения: платный трансфер (хит) — Konsa, Palmer,
João Pedro → Saka, Barry, Guéhi (GW5), цена 4 очков» и две кнопки. Нажатие ->
`Agent.stream_resume(thread_id, "confirm" | "reject")` (тот же чекпоинтер SQLite
`.cache/agent_checkpoints.sqlite`, что у CLI: тред из UI можно продолжить командой
`python -m fplcopilot.agent.cli --thread <id> --decision …`) -> ответ рендерится под пометкой
«Решение: подтверждено / отклонено — пересчёт без хита/Wildcard». Проверено вживую на 895045
(`docs/screenshots/chat_hitl.png`, `chat_reject.png`): reject -> «Scenario feasible, no hit needed …
Gabriel, Rogers → Saka, Guéhi, +2.96»; confirm -> «Taking a -4 hit … +5.46 next GW, +14.15 over the
horizon» — те же числа, что в `docs/agent_demo_output.md`. AppTest воспроизводит обе ветки на фейках.

Второй вид прерывания (агент v2): если `interrupted` и есть `pending_clarification` (неоднозначное
имя — «Should I sell Gabriel?»), сообщение показывает карточку «Уточнение: какого **Gabriel** вы
имеете в виду?» и кнопку на каждого кандидата (полное имя, клуб, позиция, цена, «· в составе»)
плюс «Отменить уточнение». Нажатие -> `Agent.stream_resume(thread_id, player_id=…)` (или
`"cancel"`) -> граф продолжает с выбранным игроком до полного ответа под пометкой «Уточнение:
выбран игрок id 4»; тот же тред можно продолжить из CLI `--thread <id> --choose <player_id>`.
AppTest: клик по кандидату -> `recommend_transfers(sell=[4])`, кнопка отмены -> отказ без
инструментов состава.

## Прод: вход и лимит LLM

- **Вход** (`app/auth.py`): при непустом `APP_PASSWORD` `Home.py` до навигации показывает экран
  входа; до верного пароля не выполняются ни страницы, ни сайдбар, ни вызовы OpenAI. После входа —
  cookie `fplc_auth` (HMAC от пароля, 14 дней), потому что ссылки на карточку игрока — обычные
  переходы и открывают новую сессию Streamlit; после неверного пароля — растущая задержка. Пустой
  `APP_PASSWORD` — вход не нужен (локально, тесты).
- **Суточный лимит LLM** (`app/llm_budget.py`): `APP_DAILY_LLM_LIMIT` запросов в сутки (UTC) на
  процесс — чат и продолжение после подтверждения, распознавание скриншота, обновление новостей
  игрока и клуба, объяснение сравнения; сверх лимита — сообщение, страницы без LLM работают. Пусто —
  без лимита. Счётчик в памяти и обнуляется при перезапуске, поэтому жёсткий потолок трат — лимит в
  кабинете OpenAI. Развёртывание — docs/deploy.md.

## Кэширование

| что | как | ключ / TTL |
|---|---|---|
| `LiveTools` (FPL-клиент с дисковым кэшем, `PredictionStore`, входы оптимизатора, reranker) | `st.cache_resource` — один экземпляр на процесс | — |
| LangGraph-агент | `st.cache_resource`, `live_agent(tools=get_tools())` | — |
| контекст тура, лучшие 11, маршруты, план, прогнозы игроков | `st.cache_data(ttl=600)`; pydantic-модели пиклятся | (manager, gw, strategy, horizon, allow_hit(s), отпечаток override) |
| сохранённые сигналы (`cached_only`) | `st.cache_data(ttl=60)` | (ids, минута) |
| свежесть данных (3 SQL-запроса) | `st.cache_data(ttl=60)` | gw |
| карточка менеджера `entry/{id}` для «<команда> · очки · ранг» (и отрицательный результат 404) | `st.cache_data(ttl=600)` | manager_id |
| «Обновить данные» | `tools.invalidate()` + `bootstrap(refresh=True)` + `st.cache_data.clear()` | — |
| «Обновить сигнал» на странице игрока | после извлечения — `tools.invalidate()` + очистка кэша, чтобы новый сигнал попал в модель минут / xPts | — |

Значения виджетов сайдбара (менеджер, стратегия) переживают переход между страницами
(`common._persist`): id виджета в Streamlit включает hash скрипта страницы, поэтому ключ
«трогается» в `session_state` до создания виджета.

## Ошибки

`common.guarded(fn, …)` -> `st.error` с заголовком и подсказкой вместо traceback:
`SquadUnavailable` (нет picks -> загрузите скриншот), `SQLAlchemyError` (БД недоступна: `docker
compose up -d db`, `scripts/migrate.py`; **без Postgres недоступны xPts, оптимизатор, сигналы,
планы и KB** — `core/xpts.make_context` читает историю игроков из `player_gw_history` /
`player_season_history`; без БД работают только состав из FPL API (bootstrap, picks) и
детерминированные отказы чата), `httpx.HTTPStatusError` / `HTTPError` (FPL API: проверьте ID, кэш `.cache/fpl`),
`ScenarioInfeasible` / `InfeasibleError`, ошибки OpenAI (`OPENAI_API_KEY`). Без ключа OpenAI
чат, извлечение сигналов и распознавание скриншота отключены явным сообщением; остальное работает.

## Docker

```bash
docker compose build app          # python:3.12-slim + uv sync --frozen --no-dev (uv.lock), non-root
docker compose up -d              # db (pgvector) + app :8501 + ingest (цикл новостей каждые 30 мин)
docker compose up -d app          # только UI, если ingest уже крутится на хосте
docker compose stop app           # остановить UI; db не трогать — в нём корпус новостей
```

- `Dockerfile`: `python:3.12-slim`, `uv` 0.7.9, зависимости отдельным слоем (`--no-install-project`),
  затем `src scripts skills samples`; пользователь `app` (uid 10001) с самого начала (иначе
  `chown -R` после установки удваивает образ: 2.48 GB -> 1.37 GB); `PYTHONPATH=/app/src`;
  `EXPOSE 8501`; `HEALTHCHECK` на `/_stcore/health`; entrypoint `scripts/docker-entrypoint.sh`
  ждёт Postgres (`scripts/migrate.py` с повторами до 30 × 2 с) и запускает Streamlit
  `--server.address 0.0.0.0`.
- `docker-compose.yml`: сервис `db` (pgvector); `app` (build ., `env_file: .env`,
  `DATABASE_URL=postgresql://fpl:fpl@db:5432/fpl` поверх .env, `depends_on: db healthy`, порт
  `${APP_PORT:-8501}:8501`, том `appcache:/app/.cache` — кэш FPL API, модель flashrank, SQLite-чекпоинты)
  и `ingest` (тот же образ, `python -m fplcopilot.rag.ingest --loop --every 30 --index`,
  `restart: unless-stopped`).
- `.dockerignore`: `.venv .cache .git .env .cursor docs/screenshots evals/runs …`; `samples/` (104 KB)
  входит в образ — кнопки «Пример» на странице «Мой состав» (`/?demo=1`) работают и в контейнере.
- Замер на Apple Silicon (arm64): первая сборка 1 мин 13 с (все колёса — highspy, onnxruntime,
  psycopg, pyarrow — есть под linux/aarch64), пересборка с кэшем 46 с; образ **1.37 GB** (venv 811 MB:
  pyarrow 142, pandas 75, onnxruntime 59, numpy 68, pulp 36, streamlit 35); `docker compose up -d app`
  -> миграции применены и `/_stcore/health` = 200 через **4 с**; страница «Мой состав» для 895045
  из контейнера — picks GW4, диагностика, сигналы из БД.
- Сервис `ingest` проверен одним циклом из того же образа (при работающем хост-цикле, без
  пересоздания `db`): `docker compose run --rm --no-deps ingest python -m
  fplcopilot.rag.ingest --once --limit 3` -> entrypoint применил 8 миграций идемпотентно
  (`re-applied 001…008`), пять источников опрошены (`bbc_football skipped=81`, `sky_football 20`,
  `guardian_football 46, too_old=10`, `ffscout 12`, `fpl_api 198`), `total: new=0 skipped=357
  failed=0` — новых статей нет, потому что хост-цикл (`--loop --every 30`) уже забрал их;
  6.1 с на весь прогон. Штатный запуск `docker compose up -d ingest` использует тот же образ и
  команду `--loop --every 30 --index`.

## Локальная разработка

```bash
cp .env.example .env               # OPENAI_API_KEY, FPL_MANAGER_ID, POSTGRES_*, DATABASE_URL
docker compose up -d db && uv run python scripts/migrate.py
uv run streamlit run src/fplcopilot/app/Home.py        # или ./scripts/dev_up.sh (всё вместе)
uv run pytest -q -m "not network and not llm"          # весь набор без сети/LLM: 853 passed, 14 skipped (db), 10 deselected
```

`AppTest` (`tests/test_app_smoke.py`) запускает страницы с подменёнными `common.get_tools` /
`common.get_agent` (фейки из `tests/test_agent_fakes.py` + реальный LangGraph на `MemorySaver`):
Home и все страницы без исключений; чат — кнопки HITL при прерывании, обе ветки (reject ->
`recommend_transfers(allow_hit=False)`, confirm — без пересчёта); `squad_override` из
`session_state` доходит до `optimize_team` / `recommend_transfers`, числа меняются;
страница без picks предлагает скриншот; без ключа OpenAI — подсказка.

## Проверка вживую (manager 895045, `docs/screenshots/`)

**GW5: числа страниц против CLI.** Скриншоты этой таблицы показывают более ранний вид страниц;
числа от вида не зависят.

| страница | результат | файл |
|---|---|---|
| Мой состав | 15 игроков, Palmer 4.76 / Haaland 7.13 / João Pedro 3.30 (как CLI), expander João Pedro с цитатами ffscout/fpl_api | `squad.png` |
| Мой состав, скриншот | `pick_team_valid.png`: 15/15 карточек, 39 780 + 989 токенов, $0.0066, 8.3 с, valid -> override | `squad_screenshot.png` |
| К дедлайну | 3-4-3, 56.14 xPts, Haaland (C) / Szoboszlai; маршруты #1 Konsa, João Pedro → Guéhi, Barry +4.03/+9.78 go; #2 Palmer, João Pedro → Barry, B.Fernandes +3.56/+9.72 — совпадает с docs/optimizer.md | `deadline.png` |
| К дедлайну, скриншот-состав | 5-4-1, 52.44 xPts, B.Fernandes / Mbeumo, FT 1, банк £0.5m, #1 Senesi → Guéhi | `deadline_override.png` |
| План | 307.54 vs 283.13 (+24.41), GW5 Konsa→Guéhi +13.45, João Pedro→Barry +2.49 (под вопросом), GW6 Palmer→Saka, GW7 Schade→Tavernier, GW8 De Cuyper→Rúben, GW9 roll; diff со снимком, сохранённым агентом, — без изменений; «Сохранить» -> plan_snapshots id=17 | `plan.png` |
| Игрок | «Palmer» -> radio (Cole / Alex); xPts по турам 4.76 … 25.21 за 5; календарь с FSI; сохранённый сигнал 1.4 ч; «Обновить сигнал» -> 1 LLM-вызов, 4 441 + 247 токенов, $0.0008, 6.2 с | `player.png` |
| Чат | «What if I take a -4 to bring in Saka and Guéhi?» -> прерывание, кнопки; reject -> free route +2.96; confirm -> hit route +5.46; стоимость: router $0.0003, explain $0.0008 | `chat_hitl.png`, `chat_reject.png` |

Стоимость всей проверки: 2 прогона чата с продолжениями (≈ $0.005), одно извлечение сигнала
($0.0008), один vision-вызов ($0.0066) — **≈ $0.013**.

### GW6: смоук страниц

Страницы из таблицы для `?manager=895045` и страница без ID (новый пользователь: cookie стёрта
«Забыть») открываются без исключений и красных ошибок Streamlit.

| страница | результат | файл |
|---|---|---|
| Мой состав | 895045: шапка, строка проблем (Palmer 75 %); без ID — подсказка «Укажите ID менеджера FPL…» | — |
| К дедлайну | 3-4-3, 60.69 xPts, C Saka / VC Haaland; маршрут #1 Palmer, O'Shea → Gabriel, Groß +2.0 / +7.3; смоук нашёл, что запасной текст «Почему» терял имя продаваемого («Плюс под вопросом…»), — исправлено | — |
| План | 6856911, Bench Boost в GW7: +7.57 xPts скамейки, «Фишка» / «Вклад фишки» в таблице по турам | `plan_chip_bb_gw7_6856911.png` |
| Игрок `?pid=12` | Saka: сигнал с цитатами, «Новости клуба» (выбывшие Arsenal из FPL, сохранённый дайджест) | `player_club_news_saka.png` |
| Сравнение `?a=12&b=154` | Saka против Palmer: таблица, календарь 6 туров, объяснение gpt-4o-mini | — |
| Чат | «как думаешь на сколько силен мой состав?» → `squad_review`; «а кого тогда продать?» → `transfer` («Кого продать из моего состава?»); «кого поставить капитаном в этом туре?» → `captain`, Saka 7.43; во всех ответах русский язык, промпты v4, валидация с первой попытки; ≈ $0.0084 за три хода | `chat_v4_08_followup_sell_895045.png` |
| О системе | граф агента (14 узлов, оба interrupt) и поток данных | — |

## Ограничения

- Один `LiveTools` на процесс: одновременные пользователи делят кэши и солвер (для демо и одного
  менеджера — нормально; для многопользовательского режима нужен воркер на запрос).
- Состав со скриншота живёт в `session_state` (вкладка браузера); перезагрузка страницы = снова
  picks. Сыгранные чипы со скриншота не видны — берутся из истории менеджера, иначе Wildcard
  считается доступным; цена продажи = цена на экране.
- `st.cache_data(ttl=600)`: до 10 минут страницы могут показывать прогноз, посчитанный до нового
  сигнала, если сигнал извлёк не UI (например, ingest-цикл); кнопка «Обновить данные» сбрасывает.
- «Сохранить снимок» пересчитывает план (5–15 с) — `PlanOut` не несёт полный `TransferPlan`.
- Чат отвечает на языке вопроса (промпты агента v2+, по умолчанию v4: русский вопрос → русский ответ, имена и числа без изменений); подписи UI — русские.
- Настройки (`AGENT_PROMPT_VERSION` и др.) читаются один раз при старте, агент кэшируется
  `st.cache_resource`: после правки кода или `.env` процесс Streamlit нужно перезапустить, иначе он
  продолжает работать на прежней версии промптов (такой случай разобран в `docs/chat_diagnosis.md`
  §4). Версия, на которой получен ответ, записана в `llm_calls[*].prompt_version` состояния графа.
- История чата — последние 6 реплик одной вкладки (`session_state`); каждый ход — свой тред
  LangGraph, HITL-решение относится к своему ходу.
- Streamlit-графики: `st.graphviz_chart` рендерит DOT в браузере (viz.js), локальный graphviz не нужен.
- `ingest` в compose проверен одним циклом (`--once`, см. «Docker»); непрерывный `--loop` в
  контейнере параллельно с хост-циклом не запускали — два цикла на одну БД безопасны
  (идемпотентные upsert), но бессмысленны.
