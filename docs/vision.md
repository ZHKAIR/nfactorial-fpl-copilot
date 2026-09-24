# Vision: состав со скриншота «Pick Team» → строгий JSON → валидированный `Squad`

Шаг 8a (после агента 7, до MCP 8b).

Мультимодальный компонент проекта (`src/fplcopilot/vision/`). Не декорация: без него приложение
не знает **текущий** состав пользователя.

## Зачем

Публичный FPL API отдаёт `entry/{id}/event/{gw}/picks/` только для **завершённых** туров. До дедлайна
черновик состава (после трансферов, с новым капитаном) доступен только под логином, а логин в
приложении мы не просим. Поэтому пользователь делает скриншот экрана «Pick Team» / «Transfers»
в приложении или на сайте FPL, vision-модель превращает его в строгий JSON, а детерминированный
код сопоставляет имена с живым `bootstrap-static` и проверяет правила FPL. На выходе — тот же
`fplcopilot.data.schemas.Squad`, что отдаёт `FPLClient.squad()`, поэтому оптимизатор и агент
работают с ним без изменений.

Принцип тот же, что в RAG-извлечении сигналов (`docs/rag.md`): **модель читает, код решает**.
Модель не знает id игроков и не резолвит имена — она OCR-подобный читатель. Всё, что можно
проверить детерминированно (имена против bootstrap, 2/5/5/3, лимит клуба, капитан), проверяется
кодом, а не доверяется модели.

## Пайплайн

```
image (png/jpg/webp, bytes|path)
  └─ load_image: mime по magic bytes; > 2048 px по стороне -> уменьшить (Pillow) -> PNG
  └─ call_vision_llm: system-промпт vision/prompts/v2/squad_extraction.system.md
        + image как data-URL (detail = VISION_DETAIL, дефолт high), gpt-4o-mini (VISION_MODEL),
        response_format = ScreenshotSquadRaw (strict structured output), temperature 0
  └─ ScreenshotSquadRaw: layout (скретчпад), players[{name_as_shown, position, club_hint,
        price_as_shown, is_captain, is_vice_captain, is_bench, bench_order, is_flagged}],
        bank_as_shown, free_transfers_as_shown, screen_type, notes
  └─ PlayerResolver (resolve.py): нормализация -> алиасы -> точное / rapidfuzz >= 88 ->
        тай-брейки позиция -> клуб -> цена ±0.3 -> resolved | ambiguous | unresolved
  └─ validate_squad (validate.py): правила FPL -> issues[{code, message, blocking}]
  └─ ParsedSquad (+ raw для отладки, + usage: токены, $ и мс) -> is_valid = нет blocking
  └─ to_squad (to_squad.py): ParsedSquad -> data.schemas.Squad (picks 1..15, multiplier, bank, team_value, gw)
```

`squad_from_image(image, bootstrap, as_of=None)` — весь конвейер; `parsed_from_raw(raw, bootstrap)` —
детерминированная часть отдельно (тесты, evals, повторная резолюция без нового вызова модели).
LangSmith: клиент OpenAI из `rag/llm.py` (`wrap_openai` при наличии ключа), стадии — `@traceable`.

## Схемы (`vision/schemas.py`)

**`ScreenshotSquadRaw`** — контракт structured output. Все поля обязательны (strict-режим OpenAI не
допускает default), «неизвестно» = `null`. Поле `layout` идёт **первым** намеренно: модель
генерирует JSON по порядку схемы, и описание экрана «ряды + скамейка + счёт карточек» до списка
игроков работает как структурированный chain-of-thought (см. промпт v2 ниже).

**`ParsedSquad`** — результат кода: `players: list[ResolvedPlayer]` (`player_id`, `web_name`,
`position`/`position_as_shown`, `team_short`, `price`/`price_as_shown`, `match_confidence` 0..1,
`match_method` exact | exact+hints | fuzzy | fuzzy+hints | ambiguous | unresolved, `candidates`),
`bank`, `free_transfers`, `gw` (ближайший непрошедший дедлайн на `as_of`), `captain_id`, `vice_id`,
`starting_ids`, `bench_order`, `issues: list[SquadIssue{code, message, blocking}]`, `is_valid`,
`raw`, `usage`.

## Промпт: v1 → v2

Файлы `vision/prompts/<version>/squad_extraction.system.md`, версия — `VISION_PROMPT_VERSION`
(дефолт v2, v1 хранится для A/B). Общее для обеих версий: читать только видимое, имена ровно как
напечатано («Haaland», «B.Fernandes», «João Pedro» — не раскрывать инициалы, не дописывать имена),
позиция по ряду на поле, цена из «£7.8m», бейджи C/V, скамейка — полоса под полем (4 карточки,
первая — вратарь), флаги травм, «In the bank»/«ITB», не выдумывать недостающих игроков —
вернуть меньше и объяснить в `notes`.

**Что сломалось в v1.** На эталоне `pick_team_broken` (14 карточек) модель прочитала 10: ряд FWD
она сочла скамейкой (`is_bench=true`, `bench_order` 1–2 у Haaland и João Pedro), а настоящую полосу
«Substitutes» не прочитала вовсе, `notes` пустой. Валидация состав заблокировала («squad size 10»),
но по неверной причине — пользователь получил бы бесполезный список ошибок.

**Что изменилось в v2.** (1) Явная процедура: экран = шапка + поле (до 4 рядов) + **отдельная**
полоса запасных вне поля на сером фоне с подписями; «ряд форвардов — это ещё поле, не скамейка».
(2) Поле `layout` заполняется первым: счёт карточек по рядам и в полосе, итог; затем «список игроков
должен совпасть с итогом». (3) Точнее описан значок травмы (маленький круг/треугольник в левом
верхнем углу футболки, жёлтый/красный, «!»; бейдж C/V — не флаг). (4) Текст пользователя тоже
просит прочитать «шапку, все ряды И полосу запасных». Результат — таблица ниже: 14/14 на сломанном
эталоне с ровно ожидаемыми нарушениями, флаг травмы найден.

Арифметика скретчпада при этом **ненадёжна**: на валидном эталоне модель написала «MID 5, FWD 3 = 13;
… total 17» при 15 верных карточках. Ценность `layout` — в том, что он заставляет посмотреть на весь
экран, а не в его числах; поэтому предупреждение `layout_mismatch` выдаётся только когда карточек
меньше 15 (объясняет недостачу), а не при любом расхождении.

## Резолюция (`vision/resolve.py`) — детерминированно, без LLM

1. Нормализация обеих сторон: NFKD без диакритики («João» → «joao»), нижний регистр, точки/дефисы/
   апострофы → пробел («B.Fernandes» → «b fernandes», «Gibbs-White» → «gibbs white»). Мусор, который
   модель может приклеить к имени, срезается: «Haaland (C)», «Gvardiol £5.7m».
2. Алиасы игрока: `web_name`; `web_name` без инициалов («J.Timber» → «timber»); `second_name`
   целиком; последняя заглавная часть фамилии («dos Santos Magalhães» → «magalhaes»); «имя фамилия» и
   полное имя; инициал + фамилия («b fernandes»); **однословное** имя («gabriel», «pedro»).
   Многословное имя («João Pedro» у Costinha) алиасом не становится — такое написание на карточке
   всегда `web_name` другого игрока.
3. Точное совпадение → кандидаты со score 100. Иначе `rapidfuzz.fuzz.ratio` по всем алиасам,
   порог **88** («Haland» → «haaland» 92 ✓, «Fernandez» → «fernandes» 89 ✓, «Mboumo» → «mbeumo» 83 ✗
   — лучше unresolved, чем чужой игрок).
4. Кандидатов > 1 → тай-брейки по подсказкам со скриншота, каждый применяется, только если оставляет
   хотя бы одного: **позиция** (ряд на поле) → **клуб** (`club_hint` против short_name/name/прозвищ из
   `TEAM_ALIASES`, fuzz ≥ 85) → **цена** ±0.3 (цена на скриншоте может отставать от `now_cost`).
5. Один кандидат → resolved (`match_confidence` 1.0 exact, 0.9 если понадобились подсказки, score/100
   для fuzzy); ноль → unresolved; несколько → **ambiguous** с перечнем кандидатов — blocking-нарушение
   `ambiguous: Gabriel — candidates [Gabriel (ARS DEF £8.0m), Martinelli (ARS MID £6.3m), G.Jesus
   (ARS FWD £5.9m), Gudmundsson (LEE DEF £4.5m)]`. Лучше спросить пользователя, чем оптимизировать
   чужой состав.

Примеры на bootstrap 17.09.2026 (тесты `test_vision_resolve.py` воспроизводят их на мини-bootstrap):

| На карточке | подсказки | результат |
|---|---|---|
| `Joao Pedro` / `JOÃO PEDRO` | — | João Pedro (CHE FWD), exact 1.0 — акценты и регистр не важны |
| `B.Fernandes` / `B. Fernandes` / `Bruno Fernandes` | — | B.Fernandes (MUN), exact |
| `Fernandes` | — | ambiguous: B.Fernandes (MUN), Fernandes (TOT) |
| `Fernandes` | клуб «Man United» или цена 12.0 | B.Fernandes, exact+hints 0.9 |
| `Gabriel` | — | ambiguous ×4 |
| `Gabriel` | ряд DEF | ambiguous: Gabriel (ARS), Gudmundsson (LEE) — позиции мало |
| `Gabriel` | ARS + £8.0m (или DEF + ARS) | Gabriel (4), exact+hints |
| `Gabriel` | ряд MID + ARS | Martinelli |
| `Silva` | DEF | ambiguous: Silva (BOU), Morato (NFO) → нужен клуб |
| `Palmer` | MID / GKP | Cole Palmer (CHE) / Alex Palmer (IPS) |
| `Van Dijk` / `Virgil` | — | Virgil (LIV) |
| `Haland` | — | Haaland, fuzzy 0.92 |
| `Zzyzx` | — | unresolved |

## Валидация (`vision/validate.py`) — каждое нарушение → `SquadIssue`

| Правило | code | blocking |
|---|---|---|
| карточка не найдена / неоднозначна | `unresolved` / `ambiguous` | да |
| один игрок дважды | `duplicate` | да |
| карточек ≠ 15 | `squad_size` | да |
| не 2 GKP / 5 DEF / 5 MID / 3 FWD (позиция bootstrap, для нераспознанных — ряд на экране) | `position_count` | да |
| > 3 игроков одного клуба | `club_limit` | да |
| скамейка не распознана (экран Transfers) — старт выберет `choose_default_lineup` | `no_bench` | нет |
| старт ≠ 11; схема вне 1 / 3–5 / 2–5 / 1–3 | `xi_count` / `formation` | нет |
| капитан / вице не ровно один, на скамейке, один и тот же игрок | `captain` / `vice` | нет |
| вратарь скамейки не первый | `bench_gk` | нет |
| позиция / цена (> ±0.3) / клуб на экране не сходятся с bootstrap | `position_mismatch` / `price_mismatch` / `club_mismatch` | нет |
| сумма цен на экране vs bootstrap > £1.0m (устаревший скриншот) | `team_value` | нет |
| скретчпад модели насчитал больше карточек, чем вернул (при < 15) | `layout_mismatch` | нет |
| `notes` модели непустой | `model_note` | нет |

`is_valid = нет blocking`. Blocking — то, что оптимизатор исправить не может (неизвестно, кто в
составе); предупреждения о капитане/схеме — может (он сам выбирает старт и капитана), поэтому не
блокируем. Предупреждения о расхождениях подсказок — сигнал галлюцинации модели (см. `detail=low`
ниже: «Livramento» вместо Virgil выдали именно они).

## `to_squad` (`vision/to_squad.py`)

Валидный `ParsedSquad` → `Squad(manager_id=0|заданный, gw=parsed.gw, players=[SquadPlayer×15],
bank=с экрана|0.0, team_value=Σ цен (с экрана, иначе now_cost)+bank, active_chip=None,
free_transfers=с экрана|None)`. Picks 1..11 — старт в порядке FPL (GKP, DEF, MID, FWD), 12..15 —
скамейка в порядке `bench_order`; `multiplier` 2/1/0. Капитан на скамейке или два капитана —
берётся первый капитан из старта, иначе без капитана; вице ≠ капитан. Без скамейки — детерминированный
старт: запасной вратарь — более дешёвый GKP, на скамейку уходят самые дешёвые полевые, пока схема
валидна. С blocking-нарушениями — `ValueError` (или `allow_invalid=True` для отладки).

## Эталоны (`samples/`, `vision/render.py`)

Два синтетических «Pick Team» на Pillow из живого bootstrap (≈ 49 KB PNG каждый, 900×1280):
поле с разметкой, ряды по позициям, карточка = футболка цвета клуба с кодом клуба + белая плашка с
`web_name` + фиолетовая плашка «£7.8m», бейджи C/V, значок «!», полоса «Substitutes» с подписями
«GKP / 1. DEF / 2. MID / 3. FWD», шапка с «In the bank £0.5m» и «Free Transfers 1».

- `pick_team_valid.png` — 4-4-2: Raya; Gabriel, Virgil, Gvardiol (!), Senesi; B.Fernandes (C), Semenyo,
  Mbeumo, Eze; João Pedro (V), Watkins; скамейка Sels, Gudmundsson, Rogers, Wood. Банк £0.5m, FT 1.
  Нарочно: «João Pedro» с акцентом, «B.Fernandes» с инициалом, неоднозначный «Gabriel», Gudmundsson
  (имя Gabriel) в том же составе, ровно 3 игрока ARS.
- `pick_team_broken.png` — 14 игроков (4 MID), два капитана (B.Fernandes, Haaland), 4 игрока Arsenal
  (Raya, Gabriel, Saka, J.Timber), старт 10. Банк £1.2m, FT 2.

Шрифт: встроенный в Pillow Aileron не содержит «ã», «é», «£», «—» (рисует квадраты), поэтому
`render.py` берёт системный TrueType (Arial/Helvetica на macOS, DejaVu/Liberation на Linux;
`FPL_RENDER_FONT` для переопределения) и падает на дефолт только если ничего не нашёл.
Перегенерация: `uv run python scripts/squad_from_screenshot.py --render-samples`.

## Результаты smoke-прогонов (17.09.2026, gpt-4o-mini, T=0)

9 реальных вызовов, суммарно ≈ **$0.047**. «Цены верно» — `price_as_shown` совпал с ценой на
карточке; «резолв» — карточка сопоставлена с правильным игроком bootstrap.

| Эталон | промпт | detail | карточек | имён verbatim | резолв верно | цены верно | C / V | скамейка | флаг «!» | банк / FT | итог | токены in+out | $ | мс |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| valid | v1 | high | 15/15 | 15 | 15/15 | 15/15 | ✓ / ✓ | ✓ | ✗ | ✓ | valid, 0 issues | 38 669 + 1 346 | 0.0066 | 10 343 |
| broken | v1 | high | **10/14** | 10 | 10/10 | 10/10 | ✓ / ✓ | ✗ ряд FWD принят за скамейку, полоса Substitutes не прочитана, notes пустой | — | ✓ | invalid, но по неверной причине | 38 669 + 914 | 0.0063 | 7 471 |
| valid | **v2** | high | 15/15 | 15 | 15/15 | 15/15 | ✓ / ✓ | ✓ | ✓ | ✓ | valid, 0 issues | 39 780 + 989 | 0.0066 | 7 724 |
| broken | **v2** | high | 14/14 | 14 | 14/14 | 14/14 | ✓ / ✓ | ✓ | — | ✓ | blocking: squad 14≠15, MID 4≠5, ARS 4>3; warn: 2 captains, XI 10 — **ровно ожидаемые** | 39 780 + 931 | 0.0065 | 6 921 |
| valid | v2 | **low** | 15/15 | 14 («Mboumo») | 14/15 | **1/15** | ✗ / ✗ | ✓ | ✗ | ✓ | invalid (unresolved) + 17 warnings: цены и клубы выдуманы (BRE, BOU, WAT…) | 5 778 + 981 | 0.0015 | 7 592 |
| broken | v2 | **low** | 14/14 | 11 («Sensei», «NFO», «Livramento») | 11/14, **1 ложный** (Virgil → Livramento, реальный игрок NEW) | 1/14 | ✓ / ✗ | ✗ | — | ✓ | invalid; галлюцинацию выдали только `price_mismatch`/`club_mismatch` | 5 778 + 921 | 0.0014 | 6 760 |
| valid, обрезан (без полосы Substitutes) | v2 | high | 10/11 (Senesi пропущен) | 10 | 10/10 | 10/10 | ✓ / ✓ | ✗ ряд FWD как скамейка | — | ✓ | blocking squad 10; layout «total 14» ≠ 10, notes пустой | 28 446 + 689 | 0.0047 | 6 736 |
| valid, повтор (детерминизм) | v2 | high | 15/15 | 15 | 15/15 | 15/15 | ✓ / ✓ | ✓ | **✗** (в 1-м прогоне ✓) | ✓ | valid; layout «total 17» | 39 780 + 989 | 0.0066 | 7 340 |
| valid, CLI `--squad` | v2 | high | 15/15 | 15 | 15/15 | 15/15 | ✓ / ✓ | ✓ | ✓ | ✓ | valid → Squad GW5, £105.7m, C B.Fernandes | 39 780 + 989 | 0.0066 | 7 163 |

**high vs low (гиперпараметр для EVALS).** Токены картинки у gpt-4o-mini: high = 2 833 + 5 667 ×
тайлы (900×1280 → 768×1092 → 2×3 = 6 тайлов → 36 835), low = 2 833. Отсюда 4.5× разница в цене
(**$0.0066 против $0.0015** за скриншот) при одинаковой латентности (~7 с). Но на low модель не
видит мелкий текст: цены угадывает (1/15), бейджи C/V не видит, по одному имени на экран искажает
(«Mboumo», «Sensei» — порог 88 честно отбрасывает их как unresolved), а один раз **подменяет игрока
реальным другим** («Livramento» вместо Virgil) — худший, тихий режим отказа, который резолюция по
имени поймать не может. Вывод: `detail=high` — дефолт, `low` не годится даже как черновик; экономия
$0.005 на скриншоте не стоит одного чужого игрока в составе. Для evals полезно то, что все
расхождения low-режима уже видны в `issues` (`price_mismatch`, `club_mismatch`, `unresolved`) —
метрика «число предупреждений» отделяет режимы без разметки.

**Детерминизм.** При T=0 два одинаковых вызова дали одинаковые 15 карточек, но `is_flagged`
у Gvardiol переключился ✓ → ✗, а арифметика `layout` отличалась. Поля-«иконки» нестабильны;
имена/цены/бейджи C/V на high стабильны во всех 5 прогонах.

**Стоимость на защите.** < 1 цента за скриншот на high; 100 скриншотов ≈ $0.66.

## Реальный скриншот

На момент написания в папке с реальными скриншотами (вне репозитория) реального
скриншота нет — **прогон на реальном экране FPL ожидается**. Когда появится:
`uv run python scripts/squad_from_screenshot.py "<путь>" --json` и дописать строку в таблицу выше.
Ожидаемые отличия от синтетики: фото игроков вместо кода клуба на футболке (→ `club_hint` будет
null, тай-брейк клуба недоступен — остаются позиция и цена), тёмная тема, оверлеи («Substitution»,
баннеры), обрезка экрана телефона, точки «•» в именах, цены с «m» без «£».

## Ограничения (промпт v2)

- **Синтетика проще реальности.** Эталоны контрастные и без фото; реальные скриншоты не проверены.
- **Обрезанный экран.** Модель не сообщает об отсутствии полосы запасных (`notes` пустой), может
  назвать ряд FWD скамейкой и потерять карточку в полном ряду. Валидация блокирует результат
  (`squad_size`), `layout_mismatch` объясняет недостачу — но исправить это может только пользователь
  (загрузить полный экран).
- **`is_flagged` ненадёжен** (1 из 2 на high, 0 из 2 на low) — только информация для таблицы; статус
  доступности берётся из FPL API и новостного сигнала, не с карточки.
- **`detail=low` подменяет игроков** — см. выше; резолюция по имени не защищает от правдоподобной
  галлюцинации, только предупреждения о цене/клубе.
- **Тёмная тема** приложения и **экран Transfers** (все 15 без скамейки, selling price ≠ current
  price, отрицательный банк) не проверены; код готов (`no_bench`, `choose_default_lineup`), промпт —
  предположительно.
- **Цены.** `team_value` считается по ценам с экрана (на Pick Team это текущая цена, не selling
  price), `now_cost` — запасной вариант; это оценка, а не точная стоимость команды.
- **Неоднозначные фамилии без подсказок** («Gabriel» на экране Transfers без цены) остаются
  blocking — по дизайну; в чате агента неоднозначное имя из текста вопроса уже решается HITL-прерыванием
  с кнопками-кандидатами (`docs/agent.md`), для vision-кандидатов такой же выбор — кандидат на следующую итерацию.
- Порог fuzzy 88 не спасает от опечаток в коротких фамилиях («Mboumo» 83) — это осознанно.

## Как запустить

```bash
uv run python scripts/squad_from_screenshot.py samples/pick_team_valid.png          # таблица + нарушения + $
uv run python scripts/squad_from_screenshot.py samples/pick_team_broken.png --json  # полный ParsedSquad
uv run python scripts/squad_from_screenshot.py shot.png --detail low                # A/B по detail
uv run python scripts/squad_from_screenshot.py shot.png --squad --manager-id 123    # + Squad
uv run python scripts/squad_from_screenshot.py --render-samples                     # samples/*.png
uv run pytest -q -m "not network" tests/test_vision_*.py                            # 71 тест (1 помечен llm и деселектится без ключа); актуальное число — см. `uv run pytest --collect-only -q tests/test_vision_*.py`
uv run pytest -q -m llm tests/test_vision_pipeline.py                               # 1 реальный вызов (~$0.007)
```

```python
from fplcopilot.data import FPLClient
from fplcopilot.vision import squad_from_image, to_squad

bs = FPLClient().bootstrap()
parsed = squad_from_image("shot.png", bs)  # ParsedSquad: players, issues, is_valid, raw, usage
if parsed.is_valid:
    squad = to_squad(parsed, bs, manager_id=0)  # -> data.schemas.Squad для оптимизатора/агента
else:
    print([str(i) for i in parsed.blocking_issues])  # что уточнить у пользователя
```

Настройки (`.env`): `VISION_MODEL` (gpt-4o-mini), `VISION_DETAIL` (high | low | auto),
`VISION_PROMPT_VERSION` (v2 | v1). Зависимости: `pillow` (рендер эталонов, подготовка картинки),
`rapidfuzz` (нечёткое сопоставление).

## Что дальше (EVALS)

Мини-golden для vision: для каждого эталона — ожидаемые карточки (имя, позиция, цена, C/V, скамейка,
флаг) уже лежат в `render.sample_specs()`; метрики — card recall, name verbatim, price exact,
badge accuracy, resolved accuracy, **hallucination rate** (резолв в чужого игрока — главный риск),
число предупреждений; оси A/B — `detail`, версия промпта, модель (gpt-4o-mini vs gpt-4.1-mini),
реальные скриншоты (светлая/тёмная тема, телефон/десктоп, Pick Team/Transfers).
