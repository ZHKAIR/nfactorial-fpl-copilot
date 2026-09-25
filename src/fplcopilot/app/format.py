"""Чистые помощники форматирования для Streamlit-страниц (без импорта streamlit — unit-тесты).

Числа приходят из инструментов уже округлёнными (xPts — 2 знака, цены — 1 знак); здесь они
только превращаются в строки и табличные строки. Русские подписи UI, имена игроков/клубов — как есть.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, tzinfo
from typing import Any

from fplcopilot.agent.tools import (
    CaptainOption,
    EvidenceItem,
    FixtureBrief,
    GameweekContext,
    LineupOut,
    PlanOut,
    PlayerPrediction,
    PlayerRisk,
    RouteOut,
    RoutesOut,
)
from fplcopilot.app import theme
from fplcopilot.data.schemas import Bootstrap, Player

STRATEGIES: tuple[str, ...] = ("conservative", "balanced", "aggressive")
# Подписи в UI — без английских слов и чисел; внутренние значения (core/strategy.PRESETS)
# и тесты/агент работают с conservative / balanced / aggressive.
STRATEGY_LABELS: dict[str, str] = {
    "conservative": "Осторожная",
    "balanced": "Сбалансированная",
    "aggressive": "Агрессивная",
}
# help селекта «Стратегия»; соответствует core/strategy.PRESETS: hit_threshold 2.0 / 1.0 / 0.25,
# ownership_weight +0.005 / 0 / −0.005 (шаблон / нейтрально / дифференциалы).
STRATEGY_HELP = (
    "Осторожная — надёжные игроки, платный трансфер только при явном выигрыше. "
    "Сбалансированная — разумный риск. Агрессивная — редкие игроки, чаще платные трансферы."
)


def strategy_label(strategy: str) -> str:
    return STRATEGY_LABELS.get(strategy, strategy)


STATUS_RU: dict[str, str] = {
    "a": "доступен",
    "d": "под вопросом",
    "i": "травма",
    "s": "дисквалификация",
    "u": "недоступен",
    "n": "не в заявке",
}
AVAILABILITY_RU: dict[str, str] = {
    "fit": "в строю",
    "doubtful": "под вопросом",
    "injured": "травма",
    "suspended": "дисквалификация",
    "unavailable": "недоступен",
    "unknown": "нет данных",
}
ISSUE_KIND_RU: dict[str, str] = {
    "injured": "травма",
    "suspended": "дисквалификация",
    "unavailable": "недоступен",
    "doubtful": "под вопросом",
    "not_playing": "ротация",
    "low_xpts": "слабый прогноз",
    "fixtures": "календарь",
    "blank": "нет матча",
}
FSI_LABELS: dict[int, str] = {
    1: "очень лёгкий",
    2: "лёгкий",
    3: "средний",
    4: "тяжёлый",
    5: "очень тяжёлый",
}
VERDICT_RU: dict[str, str] = {
    "go": "Делать",
    "hit_not_worth": "Платный трансфер не окупается",
    "hold": "Держать",
}
# На карточке «go» не пишем: рекомендация уже в заголовке «рекомендовано».
VERDICT_NOTE: dict[str, str] = {
    "hold": "Лучше подождать, выигрыш слишком маленький.",
    "hit_not_worth": "За минус 4 очка этот трансфер не окупается.",
}
CAPTAIN_TAG_RU: dict[str, str] = {
    "safe": "надёжный",
    "balanced": "обычный",
    "differential": "редкий",
}
# Подпись над таблицей вариантов капитана — пороги владения в help, не в теге.
CAPTAIN_TAG_HELP = (
    "Тег по владению: больше 30 процентов менеджеров — надёжный, от 10 до 30 — обычный, "
    "меньше 10 — редкий."
)
RECOMMENDATION_RU: dict[str, str] = {
    "transfers": "трансферы",
    "wildcard": "Wildcard",
    "hold": "держать (roll)",
}
PENDING_KIND_RU: dict[str, str] = {"hit": "платный трансфер (хит)", "wildcard": "Wildcard"}
CHIP_RU: dict[str, str] = {
    "wildcard": "Wildcard",
    "freehit": "Free Hit",
    "bboost": "Bench Boost",
    "3xc": "Triple Captain",
}
CHIP_SHORT: dict[str, str] = {
    "wildcard": "WC",
    "freehit": "FH",
    "bboost": "BB",
    "3xc": "TC",
}

def example_prompts(next_gw: int | None) -> tuple[str, ...]:
    """Кнопки-примеры страницы «Чат»: состав, трансфер без хита, фишка, капитан, разбор прошлого
    тура, календарь. Тур — ближайший дедлайн из контекста; без контекста — без номера тура."""
    before = f"перед GW{next_gw}" if next_gw else "перед ближайшим туром"
    on = f"на GW{next_gw}" if next_gw else "на ближайший тур"
    chip_gw = f"в GW{next_gw + 1}" if next_gw and next_gw < 38 else "через тур"
    return (
        "Насколько силён мой состав и где слабые места?",
        f"Кого продать {before} без платного трансфера?",
        f"Хочу сыграть Bench Boost {chip_gw} — какой состав собрать без хитов?",
        f"Кого поставить капитаном {on}?",
        "Разбери мой прошлый тур: кто подвёл и почему?",
        "У каких клубов самый лёгкий календарь на ближайшие 3 тура?",
    )


# Страницы UI и инструменты агента, которые они вызывают (docs/ui.md, страница «О системе»).
PAGE_TOOLS: tuple[tuple[str, str], ...] = (
    (
        "Брифинг",
        (
            "get_gameweek_context (+ diagnose_squad), recommend_transfers (горизонт 3), "
            "optimize_team, predict_player, analyze_player_risk (сохранённые вердикты)"
        ),
    ),
    (
        "Мой состав",
        (
            "get_gameweek_context (+ diagnose_squad), predict_player (горизонт 5), "
            "analyze_player_risk (сохранённые вердикты), vision.squad_from_image"
        ),
    ),
    ("К дедлайну", "optimize_team, recommend_transfers"),
    ("План", "build_gameweek_plan (+ latest_plan / diff_plans / save_plan)"),
    ("Игрок", "predict_player, analyze_player_risk (force = разбор новостей заново)"),
    ("Сравнение", "predict_player, analyze_player_risk (сохранённые вердикты), объяснение LLM"),
    (
        "Чат",
        (
            "LangGraph-агент: все инструменты через compute; прерывания confirm_action "
            "(платный трансфер / Wildcard) и resolve_clarification (уточнение имени)"
        ),
    ),
)


# ---------- примитивы ----------


def money(x: float | None) -> str:
    return "—" if x is None else f"£{x:.1f}m"


def signed(x: float | None, digits: int = 2) -> str:
    return "—" if x is None else f"{x:+.{digits}f}"


def num(x: float | None, digits: int = 2) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def pct(x: float | None, digits: int = 0) -> str:
    return "—" if x is None else f"{x:.{digits}f} %"


def plural(n: int | None, one: str, few: str, many: str) -> str:
    """Русское склонение с числом: 1 статья, 2 статьи, 5 статей, 11 статей, 21 статья."""
    n = int(n or 0)
    tail, tail2 = abs(n) % 10, abs(n) % 100
    if tail == 1 and tail2 != 11:
        return f"{n} {one}"
    if 2 <= tail <= 4 and not 12 <= tail2 <= 14:
        return f"{n} {few}"
    return f"{n} {many}"


def points_text(x: float, digits: int = 1, *, signed: bool = False) -> str:
    """Очки с правильным словом: «1 очко», «2 очка», «5 очков», «2.5 очка» (дробное — всегда
    «очка»); signed — со знаком «+2 очка» / «−1.5 очка». Целое печатается без «.0»."""
    value = round(float(x), digits)
    sign = ("+" if value > 0 else "−" if value < 0 else "") if signed else ("−" if value < 0 else "")
    a = abs(value)
    if a == int(a):
        n = int(a)
        word = plural(n, "очко", "очка", "очков").split(" ", 1)[1]
        return f"{sign}{n} {word}"
    return f"{sign}{a:.{digits}f} очка"


def fsi_label(fsi: int | None) -> str:
    if fsi is None:
        return "—"
    return f"FSI {fsi} · {FSI_LABELS.get(int(fsi), '?')}"


def fixture_text(fx: FixtureBrief) -> str:
    where = "дома" if fx.is_home else "в гостях"
    return f"{fx.opponent} ({where}, {fsi_label(fx.fsi)})"


def fixtures_text(fixtures: Sequence[FixtureBrief]) -> str:
    return "; ".join(fixture_text(f) for f in fixtures) if fixtures else "blank (нет матча)"


ROTATION_RU: dict[str, str] = {
    "low": "низкий",
    "medium": "средний",
    "high": "высокий",
    "unknown": "нет данных",
}

# Шаблонные английские summary / note кода (rag/extract.py, rag/team_news.py, agent/tools.py):
# в БД и для LLM остаются английскими, в интерфейсе — по-русски. Текст модели не переводится.
_TEMPLATES_RU: tuple[tuple[re.Pattern[str], Any], ...] = (
    (re.compile(r"No club news found\."), lambda m: "Новостей о клубе не найдено."),
    (
        re.compile(
            r"No player-specific news found\."
            r"(?: FPL API status (\w+) \([^)]*\) is the only signal\.)?"
        ),
        lambda m: (
            "Новостей именно об игроке не найдено"
            + (f"; есть только статус FPL: {STATUS_RU.get(m[1], m[1])}." if m[1] else ".")
        ),
    ),
    (
        re.compile(
            r"No evidence about (.+?) in the retrieved documents; "
            r"FPL API status (\w+) \([^)]*\) is the only signal\."
        ),
        lambda m: (
            f"В найденных статьях нет сведений о {m[1]}; "
            f"есть только статус FPL: {STATUS_RU.get(m[2], m[2])}."
        ),
    ),
    (
        re.compile(r"FPL status (\w+) is the only signal\."),
        lambda m: f"Есть только статус FPL: {STATUS_RU.get(m[1], m[1])}.",
    ),
    (re.compile(r"no saved (?:news signal|club news digest) \(cached_only\)"), lambda m: ""),
    (
        re.compile(r"stale (signal|digest) \(([\d.]+) h > ([\d.]+) h\), not refreshed"),
        lambda m: (
            ("разбор" if m[1] == "signal" else "дайджест")
            + f" старше срока ({m[2]} ч > {m[3]} ч), не обновлялся"
        ),
    ),
    (
        re.compile(r"(signal|club news) extraction failed: (.*?)( \(stale cached signal used\))?"),
        lambda m: (
            ("обновить разбор" if m[1] == "signal" else "обновить дайджест клуба")
            + f" не удалось ({m[2]})"
            + ("; показан прежний сохранённый" if m[3] else "")
        ),
    ),
)


def template_ru(text: str | None) -> str:
    """Шаблонная английская строка кода -> русская; всё остальное (текст модели) — как есть."""
    s = (text or "").strip()
    for rx, ru in _TEMPLATES_RU:
        if m := rx.fullmatch(s):
            return ru(m)
    return text or ""


def status_text(status: str, chance: int | None = None) -> str:
    label = STATUS_RU.get(status, status)
    if status == "d" and chance is not None:
        return f"{label} ({chance} %)"
    return label


def availability_text(risk: PlayerRisk | None) -> str:
    """Вердикт из новостей об игроке: «в строю · уверенность 0.80» / «нет разбора новостей»."""
    if risk is None or risk.origin == "unavailable":
        return "нет разбора новостей"
    label = AVAILABILITY_RU.get(risk.availability, risk.availability)
    return f"{label} · уверенность {risk.confidence:.2f}"


def age_text(hours: float | None) -> str:
    if hours is None:
        return "—"
    if hours < 1:
        return f"{round(hours * 60)} мин назад"
    if hours < 48:
        return f"{hours:.1f} ч назад"
    days = hours / 24
    return f"{days:.0f} дн назад" if abs(days - round(days)) < 0.05 else f"{days:.1f} дн назад"


def since_text(ts: datetime | None, now: datetime) -> str:
    if ts is None:
        return "нет данных"
    ts = ts if ts.tzinfo else ts.replace(tzinfo=UTC)
    return age_text((now - ts).total_seconds() / 3600)


def deadline_countdown(deadline: datetime | None, now: datetime, *, short: bool = False) -> str:
    """«1 д 7 ч 30 мин»; short=True — без минут, когда до дедлайна больше суток («17 д 20 ч»)."""
    if deadline is None:
        return "—"
    delta = deadline - now
    total = int(delta.total_seconds())
    if total <= 0:
        return "дедлайн прошёл"
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    parts = []
    if days:
        parts.append(f"{days} д")
    if days or hours:
        parts.append(f"{hours} ч")
    if not (short and days):
        parts.append(f"{minutes} мин")
    return " ".join(parts)


MONTHS_RU_SHORT = (
    "янв",
    "фев",
    "мар",
    "апр",
    "мая",
    "июн",
    "июл",
    "авг",
    "сен",
    "окт",
    "ноя",
    "дек",
)


def deadline_text(deadline: datetime, tz: tzinfo | None = None) -> str:
    """Дедлайн в местном времени: «10 окт, 16:30» (tz=None — часовой пояс системы)."""
    local = deadline.astimezone(tz)
    return f"{local.day} {MONTHS_RU_SHORT[local.month - 1]}, {local:%H:%M}"


def player_label(p: Player, bs: Bootstrap) -> str:
    own = float(p.selected_by_percent or 0.0)
    return (
        f"{p.full_name} ({bs.team(p.team).short_name}, {p.position.short}, "
        f"£{p.price:.1f}m, {own:.1f} %)"
    )


# ---------- состав ----------


def issues_by_player(issues: Iterable[Mapping[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for i in issues:
        out.setdefault(int(i["player_id"]), []).append(dict(i))
    for lst in out.values():
        lst.sort(key=lambda i: -int(i.get("severity", 0)))
    return out


# Серьёзные проблемы для строки над таблицей и значка у имени: статус FPL и ротация.
# Календарь / низкий прогноз сюда не входят — они видны цветом в календаре и в xPts.
STATUS_PROBLEM_RU: dict[str, tuple[str, str]] = {  # статус FPL -> (тон, подпись)
    "d": ("warn", "под вопросом"),
    "i": ("bad", "травма"),
    "s": ("bad", "дисквалификация"),
    "u": ("bad", "недоступен"),
    "n": ("bad", "не в заявке"),
}
ROTATION_FLAG = "warn"
# Тон проблемы -> цветная иконка Material в markdown-подписи expander (эмодзи не используем).
FLAG_ICON: dict[str, str] = {
    "warn": ":orange[:material/warning:]",
    "bad": ":red[:material/error:]",
}


def _p_start(pred: PlayerPrediction | None) -> float | None:
    return pred.by_gw[0].p_start if pred and pred.by_gw else None


def serious_problems(
    ctx: GameweekContext,
    preds: Mapping[int, PlayerPrediction],
    issues: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Игроки с серьёзными проблемами в порядке состава: [{id, name, flag, label, news, text,
    title}]. Статус FPL (под вопросом / травма / дисквалификация / недоступен / не в заявке —
    из `SquadPlayerRow.status`) или риск ротации (`diagnose_squad`: kind = not_playing, с
    p(start) из прогноза). `text` — для строки «Haaland — под вопросом (75 %), колено»;
    `title` — для подсказки значка «Под вопросом (75 %) — колено»."""
    by_player = issues_by_player(ctx.issues if issues is None else issues)
    out: list[dict[str, Any]] = []
    for p in ctx.squad or []:
        flag: str | None = None
        label = ""
        news = (p.news or "").strip()
        if p.status in STATUS_PROBLEM_RU:
            flag, label = STATUS_PROBLEM_RU[p.status]
            if p.status == "d" and p.chance is not None:
                label += f" ({p.chance} %)"
        elif any(i.get("kind") == "not_playing" for i in by_player.get(p.id, [])):
            flag, label = ROTATION_FLAG, "риск ротации"
            ps = _p_start(preds.get(p.id))
            if ps is not None:
                label += f" (выйдет в старте {ps * 100:.0f} %)"
            news = ""
        if flag is None:
            continue
        out.append(
            {
                "id": p.id,
                "name": p.name,
                "flag": flag,
                "label": label,
                "news": news,
                "text": f"{p.name} — {label}" + (f", {news}" if news else ""),
                "title": label[0].upper() + label[1:] + (f" — {news}" if news else ""),
            }
        )
    return out


def problems_line(problems: Sequence[Mapping[str, Any]]) -> str | None:
    """«Haaland — под вопросом (75 %), колено · Saka — риск ротации (выйдет в старте 55 %)»;
    None — серьёзных проблем нет (строка не показывается)."""
    if not problems:
        return None
    return " · ".join(str(p["text"]) for p in problems)


# ---------- таблица «Мой состав» (по образцу smartplay Players, HTML) ----------

POSITION_ORDER: dict[str, int] = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
# Сложность матча 1–5 красится классами theme.fsi_class (одна палитра для всех страниц,
# светлая и тёмная тема). Легенда сложности — как на smartplay Players («Fixture difficulty: Very Easy … Very Tough»).
FSI_LEGEND: dict[int, str] = {
    1: "Very Easy",
    2: "Easy",
    3: "Medium",
    4: "Tough",
    5: "Very Tough",
}
FRR_HORIZON = 5  # туров в календаре и в рейтинге календаря FRR#
PLAYER_URL = "/player?pid={pid}"  # страница «Игрок» читает st.query_params["pid"] (element id)

# Метки колонок — как на smartplay Players (решение студента); «Роль» — наше дополнение.
# Подсказки (?) — по-русски, адаптированы из подсказок smartplay.
SQUAD_COLUMNS: tuple[str, ...] = (
    "#",
    "Роль",
    "Player",
    "Price",
    "TSB%",
    "Form",
    "Pts/Game",
    "xMins",  # в UI подпись «xMins GW6»
    "xPts",  # «xPts GW6»
    "3GW xPts",
    "FRR#",
)
NUMERIC_COLUMNS: frozenset[str] = frozenset(SQUAD_COLUMNS[3:])
SQUAD_COLUMN_HELP: dict[str, str] = {
    "#": "Порядковый номер: 1–11 — стартовый состав, 12–15 — скамейка",
    "Роль": (
        "C — капитан (очки ×2), VC — вице-капитан, «—» — в старте, Зап1…Зап4 — скамейка "
        "в порядке выхода на замену"
    ),
    "Player": (
        "Игрок: имя как в FPL; ниже — клуб · позиция (GKP — вратарь, DEF — защитник, "
        "MID — полузащитник, FWD — нападающий) · владение. Точка у имени — проблема: "
        "жёлтая — под вопросом / риск ротации, красная — травма, дисквалификация или "
        "недоступен (наведите — детали). Клик по имени открывает страницу «Игрок»"
    ),
    "Price": "Текущая цена в FPL, £ млн; растёт и падает в зависимости от трансферов менеджеров",
    "TSB%": (
        "Team Selected By — доля менеджеров FPL, у которых есть этот игрок. Высокое владение — "
        "безопаснее для капитана, но меньше шанс обогнать соперников"
    ),
    "Form": "Средние очки за матч за последние 30 дней — главный индикатор текущей отдачи",
    "Pts/Game": (
        "Средние очки за сыгранный матч в сезоне; удобно сравнивать игроков с разным числом минут"
    ),
    "xMins": "Ожидаемые минуты в GW{gw} по нашей модели (статус FPL, новости об игроке, история выходов)",
    "xPts": "Прогноз очков на GW{gw}: минуты, форма (xG, xA, бонусы), сложность матча и оборона соперника",
    "3GW xPts": "Сумма прогноза очков на 3 ближайших тура",
    "FRR#": (
        "Рейтинг календаря команды среди 20 клубов по средней сложности матчей на 5 ближайших "
        "туров (1 = самый лёгкий календарь, 20 = самый тяжёлый)"
    ),
    "GW": (
        "Соперник в туре GW{gw}: (H) — дома, (A) — в гостях; два матча — через «·». Цвет — "
        "сложность матча 1–5 по xG-модели: зелёный — очень лёгкий, красный — очень тяжёлый"
    ),
}


def column_help(column: str, gw: int) -> str:
    """Подсказка колонки с подставленным туром («xMins GW6»)."""
    return SQUAD_COLUMN_HELP[column].format(gw=gw)


def column_label(column: str, gw: int) -> str:
    return {"xMins": f"xMins GW{gw}", "xPts": f"xPts GW{gw}"}.get(column, column)


def calendar_cell(fixtures: Sequence[FixtureBrief] | None) -> tuple[str, int | None]:
    """(«TOT (H)», сложность 1–5) для ячейки календаря — формат smartplay «Next Opp»;
    нет матча -> («—», None); два матча -> «TOT (H) · LIV (A)» и средняя сложность."""
    if not fixtures:
        return "—", None
    text = " · ".join(f"{f.opponent} ({'H' if f.is_home else 'A'})" for f in fixtures)
    return text, round(sum(int(f.fsi) for f in fixtures) / len(fixtures))


def fsi_css(fsi: int | None) -> str:
    """CSS-класс чипа календаря по сложности 1–5; пусто, если сложности нет."""
    if fsi is None or not 1 <= int(fsi) <= 5:
        return ""
    return theme.fsi_class(int(fsi))


LEGEND_CSS = f"""
<style>
{theme.FSI_CSS_RULES}
.fpl-legend {{
  display: flex; flex-wrap: wrap; align-items: center; gap: 6px;
  margin: 0.6rem 0 1rem; font-size: 12px; color: var(--fpl-muted);
}}
.fpl-legend .fpl-fx {{ min-width: 0; padding: 3px 9px; }}
.fpl-fx {{
  display: inline-block; min-width: 62px; padding: 4px 7px; border-radius: 6px;
  font-size: 11.5px; font-weight: 600; line-height: 1.25; text-align: center;
  white-space: nowrap; font-variant-numeric: tabular-nums;
}}
</style>
"""


def fsi_legend_html() -> str:
    """Легенда сложности одной строкой, как на smartplay («Fixture difficulty: Very Easy …
    Very Tough»), теми же чипами, что в календаре (st.markdown с unsafe_allow_html)."""
    chips = "".join(
        f'<span class="fpl-fx {theme.fsi_class(k)}">{label}</span>'
        for k, label in FSI_LEGEND.items()
    )
    return LEGEND_CSS + f'<div class="fpl-legend">Fixture difficulty: {chips}</div>'


def _xpts_next(p: Any, pred: PlayerPrediction | None) -> float | None:
    if pred and pred.by_gw:
        return pred.by_gw[0].xpts
    return p.xpts_next


def squad_table_rows(
    ctx: GameweekContext,
    preds: Mapping[int, PlayerPrediction],
    bs: Bootstrap | None,
    *,
    gws: Sequence[int],
    frr: Mapping[str, int] | None = None,
    problems: Sequence[Mapping[str, Any]] | None = None,
    xpts_horizon: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(старт, скамейка) для страницы «Мой состав» — колонки `SQUAD_COLUMNS` + календарь
    GW{gws}. Старт — GKP -> DEF -> MID -> FWD, роли C / VC / —, # 1–11; скамейка — в порядке
    picks, роли Зап1…Зап4, # 12–15. Числа — числами (формат в `squad_table_html`); TSB% /
    Form / Pts/Game — из bootstrap (`bs`), если он есть; FRR# — из `frr` (клуб -> ранг).
    Служебные ключи: `_id`, `_team`, `_pos`, `_flag` (значок, подсказка) для проблемных
    игроков из `problems`, `_fsi` (сложность по колонкам календаря для раскраски)."""
    squad = list(ctx.squad or [])
    starters = sorted(
        (p for p in squad if p.is_starting),
        key=lambda p: (POSITION_ORDER.get(p.position, 9), squad.index(p)),
    )
    bench = [p for p in squad if not p.is_starting]
    frr = frr or {}
    flags = {int(pr["id"]): (str(pr["flag"]), str(pr["title"])) for pr in problems or []}

    def row(no: int, p: Any, role: str) -> dict[str, Any]:
        pred = preds.get(p.id)
        by_gw = {g.gw: g for g in pred.by_gw} if pred else {}
        nxt = by_gw.get(ctx.gw)
        first = [g for g in (pred.by_gw if pred else []) if g.gw < ctx.gw + xpts_horizon]
        pl: Player | None = None
        if bs is not None:
            try:
                pl = bs.player(p.id)
            except KeyError:
                pl = None
        tsb = float(pl.selected_by_percent) if pl and pl.selected_by_percent else p.ownership
        out: dict[str, Any] = {
            "#": no,
            "Роль": role,
            "Player": p.name,
            "Price": p.price,
            "TSB%": tsb,
            "Form": float(pl.form) if pl and pl.form is not None else None,
            "Pts/Game": float(pl.points_per_game)
            if pl and pl.points_per_game is not None
            else None,
            "xMins": round(nxt.exp_minutes) if nxt else None,
            "xPts": _xpts_next(p, pred),
            "3GW xPts": round(sum(g.xpts for g in first), 2) if first else None,
            "FRR#": frr.get(p.team),
        }
        fsi: dict[str, int | None] = {}
        for g in gws:
            gp = by_gw.get(g)
            text, level = calendar_cell(gp.fixtures) if gp else ("—", None)
            out[f"GW{g}"] = text
            fsi[f"GW{g}"] = level
        out["_id"] = p.id
        out["_team"] = p.team
        out["_pos"] = p.position
        out["_flag"] = flags.get(p.id)
        out["_fsi"] = fsi
        return out

    xi_rows = [
        row(i, p, "C" if p.is_captain else "VC" if p.is_vice else "—")
        for i, p in enumerate(starters, start=1)
    ]
    bench_rows = [row(len(xi_rows) + i, p, f"Зап{i}") for i, p in enumerate(bench, start=1)]
    return xi_rows, bench_rows


def cell_text(column: str, value: Any) -> str:
    """Текст числовой ячейки: £4.6 · 42.2% · 6.0 · 90 · #1; None -> «—»."""
    if value is None:
        return "—"
    if column == "Price":
        return f"£{float(value):.1f}"
    if column == "TSB%":
        return f"{float(value):.1f}%"
    if column == "xMins":
        return f"{float(value):.0f}"
    if column == "FRR#":
        return f"#{int(value)}"
    if column in NUMERIC_COLUMNS:
        return f"{float(value):.1f}"
    return str(value)


# CSS-подсказка вместо HTML title= (браузерный title медленный, не стилизуется, в тёмной
# теме почти нечитаем). Инверсия к теме: тёмная плашка на светлой, светлая на тёмной.
TIP_CSS_RULES = """
.tip { position: relative; display: inline-block; cursor: help; }
.tip .tip-text {
  display: none;
  position: absolute; z-index: 10000; top: 100%; left: 50%; transform: translateX(-50%);
  margin-top: 6px; width: 260px; max-width: 280px; box-sizing: border-box;
  padding: 8px 10px; border-radius: 8px;
  background: var(--fpl-tip-bg, #17181B); color: var(--fpl-tip-fg, #F3F4F2);
  font-size: 12px; font-weight: 400; line-height: 1.4; text-transform: none;
  white-space: normal; text-align: left; letter-spacing: normal;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
  pointer-events: none;
}
.tip:hover .tip-text { display: block; }
.tip.tip-left .tip-text { left: 0; transform: none; }
.tip.tip-right .tip-text { left: auto; right: 0; transform: none; }
.tip.tip-block { display: block; width: 100%; }
.tip.q {
  margin-left: 4px; width: 14px; height: 14px; line-height: 12px; border-radius: 50%;
  text-align: center; font-size: 9.5px; font-weight: 600; flex-shrink: 0;
  color: var(--fpl-faint, #9A9CA3); border: 1px solid var(--fpl-line-strong, #DADBD6);
  box-sizing: border-box; vertical-align: 1px;
}
.tip.q:hover { color: var(--fpl-text, inherit); border-color: var(--fpl-muted, #63666D); }
.tip.th {
  text-decoration: underline dotted var(--fpl-faint, #9A9CA3); text-underline-offset: 3px;
}
.tip.th:hover { color: var(--fpl-text, inherit); }
"""

TIP_CSS = f"<style>\n{TIP_CSS_RULES}\n</style>\n"

# Метрики-карточки и подписи виджетов со значком «?».
TIP_WIDGET_CSS = f"""
<style>
{TIP_CSS_RULES}
.fpl-metric {{
  margin: 0 0 0.75rem 0; padding: 14px 18px 13px; border-radius: 12px;
  background: var(--fpl-surface, transparent); border: 1px solid var(--fpl-line-strong, transparent);
  box-shadow: var(--fpl-shadow, none);
  min-height: 88px; box-sizing: border-box; min-width: 0;
  display: flex; flex-direction: column; justify-content: space-between;
  overflow: hidden;
}}
.fpl-metric-label {{
  font-size: 12px; font-weight: 700; line-height: 1.35;
  letter-spacing: 0.06em; text-transform: uppercase; color: var(--fpl-muted, #63666D);
  display: flex; align-items: flex-start; gap: 2px; min-height: 1.1rem; overflow: visible;
}}
.fpl-metric-label .tip.q {{ margin-top: 1px; letter-spacing: 0; }}
.fpl-metric-value {{
  font-family: var(--fpl-mono, inherit); font-size: 1.45rem; font-weight: 600; line-height: 1.2;
  letter-spacing: -0.02em;
  margin-top: 6px; white-space: nowrap; overflow: visible; text-overflow: clip;
  font-variant-numeric: tabular-nums;
}}
.fpl-metric-value.wrap {{
  white-space: normal; overflow: hidden; font-size: 1.05rem; line-height: 1.3;
}}
.fpl-stats {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); gap: 10px;
  margin: 0.25rem 0 1rem;
}}
.fpl-stats .fpl-metric {{ margin: 0; height: 100%; }}
.fpl-route-metrics {{
  display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 0.25rem 0 0.75rem;
}}
.fpl-route-metrics .fpl-metric {{ margin: 0; min-height: 0; padding: 10px 12px; }}
.fpl-route-metrics .fpl-metric-value {{ font-size: 1.3rem; }}
.fpl-route-metrics .hit {{ grid-column: 1 / -1; }}
.fpl-metric-delta {{
  font-size: 12px; margin-top: 4px; color: var(--fpl-muted, inherit);
  font-variant-numeric: tabular-nums;
}}
.fpl-field-label {{
  font-size: 0.875rem; margin-bottom: 0.25rem; display: flex; align-items: center; gap: 4px;
  white-space: pre-wrap;
}}
.fpl-field-label.heading {{
  font-size: 1.08rem; font-weight: 600; letter-spacing: -0.02em; color: inherit;
  margin: 1.25rem 0 0.5rem;
}}
</style>
"""


def _esc(s: Any) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def tip(
    text: str,
    *,
    mark: str = "?",
    align: str = "center",
    css_class: str = "q",
    escape_mark: bool = True,
) -> str:
    """Значок/метка с CSS-подсказкой (без title= и без JS — Streamlit вырезает script).

    align: center | left | right — у крайних колонок left/right, чтобы не уезжала за экран.
    escape_mark=False — mark уже безопасный HTML (обёртка ячейки календаря и т.п.).
    """
    classes = ["tip"]
    if css_class:
        classes.append(css_class)
    if align == "left":
        classes.append("tip-left")
    elif align == "right":
        classes.append("tip-right")
    visible = _esc(mark) if escape_mark else mark
    return (
        f'<span class="{" ".join(classes)}">{visible}'
        f'<span class="tip-text">{_esc(text)}</span></span>'
    )


def metric_html(
    label: str,
    value: str,
    *,
    help_text: str | None = None,
    delta: str | None = None,
) -> str:
    """Метрика с CSS-подсказкой у подписи (вместо st.metric(..., help=) — тот залипает по клику)."""
    tip_h = tip(help_text) if help_text else ""
    delta_h = f'<div class="fpl-metric-delta">{_esc(delta)}</div>' if delta else ""
    return (
        TIP_WIDGET_CSS
        + '<div class="fpl-metric"><div class="fpl-metric-label">'
        + f'<span>{_esc(label)}</span>{tip_h}</div>'
        + f'<div class="fpl-metric-value">{_esc(value)}</div>{delta_h}</div>'
    )


def stat_strip_html(cells: Sequence[tuple[str, ...]]) -> str:
    """Ряд метрик одной сеткой [(подпись, значение, help[, класс значения])]: карточки одной
    высоты, перенос по ширине экрана — вместо st.columns, где длинная подпись растягивает одну
    карточку. Класс `wrap` — длинный текст (чипы) остаётся внутри карточки."""
    items = []
    for cell in cells:
        label, value, help_text, *rest = cell
        extra = f" {rest[0]}" if rest else ""
        tip_h = tip(help_text) if help_text else ""
        items.append(
            f'<div class="fpl-metric"><div class="fpl-metric-label"><span>{_esc(label)}</span>'
            f'{tip_h}</div><div class="fpl-metric-value{extra}">{_esc(value)}</div></div>'
        )
    return TIP_WIDGET_CSS + '<div class="fpl-stats">' + "".join(items) + "</div>"


def label_with_tip(label: str, help_text: str, *, heading: bool = False) -> str:
    """Подпись виджета / заголовка со значком «?» (вместо help= у Streamlit)."""
    cls = "fpl-field-label heading" if heading else "fpl-field-label"
    return (
        TIP_WIDGET_CSS
        + f'<div class="{cls}"><span>{_esc(label)}</span>{tip(help_text)}</div>'
    )


# Таблицы данных: тонкие линии без заливки шапки, строка ~48px, имя 14px/600 + вторая строка
# 12px, числа моноширинными цифрами вправо; цвета — токены темы (theme.TOKENS_CSS) с запасными
# нейтральными значениями, поэтому читаются и в светлой, и в тёмной теме.
DATA_TABLE_CSS_RULES = """
.fpl-table-wrap { width: 100%; overflow: visible; }
table.fpl-table {
  width: 100%; border-collapse: separate; border-spacing: 0; overflow: visible; border: none;
  font-size: 13px; line-height: 1.25; font-variant-numeric: tabular-nums;
}
table.fpl-table th, table.fpl-table td { border: none; }
table.fpl-table th {
  text-align: left; font-size: 11.5px; font-weight: 600; color: var(--fpl-muted, #63666D);
  padding: 0 7px 9px; border-bottom: 1px solid var(--fpl-line-strong, rgba(128,128,128,0.35));
  white-space: nowrap; background: transparent;
}
table.fpl-table td {
  padding: 7px; border-bottom: 1px solid var(--fpl-line, rgba(128,128,128,0.2));
  white-space: nowrap; overflow: visible; vertical-align: middle;
}
table.fpl-table tbody tr:hover td { background: var(--fpl-hover, rgba(128,128,128,0.05)); }
.fpl-table th.num, .fpl-table td.num { text-align: right; }
.fpl-table th.cal, .fpl-table td.cal { text-align: center; }
.fpl-table td.muted, .fpl-table td.no { color: var(--fpl-faint, #9A9CA3); }
.fpl-table td.no { text-align: right; width: 1%; }
.fpl-role {
  display: inline-block; min-width: 22px; padding: 1px 6px; border-radius: 999px;
  font-size: 11px; font-weight: 700; text-align: center; color: var(--fpl-faint, #9A9CA3);
}
.fpl-role.c { background: var(--fpl-brand, #06A8FD); color: #03121C; }
.fpl-role.vc { color: var(--fpl-text, inherit); box-shadow: inset 0 0 0 1px var(--fpl-line-strong, #DADBD6); }
.fpl-role.bench { font-weight: 600; }
.fpl-dot {
  display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-left: 6px;
  vertical-align: 1px;
}
.fpl-dot.warn { background: var(--fpl-warn, #B7791F); }
.fpl-dot.bad { background: var(--fpl-bad, #C8373E); }
table.fpl-table tr.sep td {
  padding: 14px 7px 6px; font-size: 11px; font-weight: 600; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--fpl-faint, #9A9CA3); background: transparent;
  border-bottom: 1px solid var(--fpl-line-strong, rgba(128,128,128,0.35));
}
table.fpl-table tbody tr.sep:hover td { background: transparent; }
"""

# Таблица «Мой состав» (по образцу smartplay Players).
SQUAD_TABLE_CSS = f"""
<style>
{TIP_CSS_RULES}
{DATA_TABLE_CSS_RULES}
{theme.FSI_CSS_RULES}
.fpl-squad td {{ height: 48px; }}
.fpl-squad td.player {{ min-width: 150px; }}
.fpl-squad .name {{ font-size: 14px; font-weight: 600; }}
.fpl-squad .name a {{ color: inherit; text-decoration: none; }}
.fpl-squad .name a:hover {{ color: var(--fpl-primary, inherit); }}
.fpl-squad .sub {{ font-size: 11.5px; color: var(--fpl-muted, #8a8f98); margin-top: 3px; }}
.fpl-squad td.xpts {{ font-weight: 700; }}
.fpl-squad td.cal {{ padding-left: 3px; padding-right: 3px; }}
.fpl-squad tr.bench td {{ color: var(--fpl-muted, inherit); }}
.fpl-squad tr.bench td.player .name {{ font-weight: 550; }}
.fpl-fx {{
  display: inline-block; min-width: 62px; padding: 4px 7px; border-radius: 6px;
  font-size: 11.5px; font-weight: 600; line-height: 1.25; text-align: center;
}}
@media (max-width: 1360px) {{ .fpl-squad-wrap {{ overflow-x: auto; }} }}
</style>
"""


def role_html(role: str) -> str:
    """Роль в составе таблеткой: C — синяя (цвет эмблемы), VC — контур, остальное — серым."""
    cls = {"C": " c", "VC": " vc", "—": ""}.get(role, " bench")
    return f'<span class="fpl-role{cls}">{_esc(role)}</span>'


def squad_table_html(
    xi: Sequence[Mapping[str, Any]],
    bench: Sequence[Mapping[str, Any]],
    *,
    gw: int,
    gws: Sequence[int],
    player_url: str = PLAYER_URL,
) -> str:
    """HTML-таблица состава для st.markdown(unsafe_allow_html=True): подсказка — на самом
    заголовке (пунктир, CSS `tip`), ячейка Player в две строки (имя-ссылка на страницу «Игрок»
    `/player?pid=<id>` + цветная точка проблемы; ниже «MUN · MID · 38.8%»), числа вправо,
    календарь чипами по сложности, разделитель «Bench» перед скамейкой."""
    cal_cols = [f"GW{g}" for g in gws]
    n_cols = len(SQUAD_COLUMNS) + len(cal_cols)
    head = []
    for i, col in enumerate(SQUAD_COLUMNS):
        cls = "num" if col in NUMERIC_COLUMNS else "no" if col == "#" else ""
        align = "left" if i < 3 else "right" if i >= n_cols - 3 else "center"
        head.append(
            f'<th class="{cls}">'
            f"{tip(column_help(col, gw), mark=column_label(col, gw), align=align, css_class='th')}"
            "</th>"
        )
    for j, g in enumerate(gws):
        i = len(SQUAD_COLUMNS) + j
        align = "right" if i >= n_cols - 3 else "center"
        mark = tip(column_help("GW", g), mark=f"GW{g}", align=align, css_class="th")
        head.append(f'<th class="cal">{mark}</th>')

    def tr(r: Mapping[str, Any], *, bench_row: bool = False) -> str:
        tds = [f'<td class="no">{r["#"]}</td>', f'<td class="role">{role_html(r["Роль"])}</td>']
        flag = r.get("_flag")
        badge = ""
        if flag:
            dot = f'<span class="fpl-dot {flag[0]}"></span>'
            badge = tip(flag[1], mark=dot, css_class="flag", escape_mark=False)
        url = player_url.format(pid=r["_id"])
        sub = f"{_esc(r['_team'])} · {_esc(r['_pos'])} · {cell_text('TSB%', r['TSB%'])}"
        tds.append(
            f'<td class="player"><div class="name"><a href="{url}" target="_self">'
            f'{_esc(r["Player"])}</a>{badge}</div><div class="sub">{sub}</div></td>'
        )
        for col in SQUAD_COLUMNS[3:]:
            cls = "num xpts" if col == "xPts" else "num"
            tds.append(f'<td class="{cls}">{cell_text(col, r.get(col))}</td>')
        for col in cal_cols:
            fx_cls = fsi_css((r.get("_fsi") or {}).get(col))
            text = _esc(r.get(col, "—"))
            inner = f'<span class="fpl-fx {fx_cls}">{text}</span>' if fx_cls else text
            tds.append(f'<td class="cal">{inner}</td>')
        tr_cls = ' class="bench"' if bench_row else ""
        return f"<tr{tr_cls}>" + "".join(tds) + "</tr>"

    body = [tr(r) for r in xi]
    if bench:
        body.append(f'<tr class="sep"><td colspan="{n_cols}">Bench</td></tr>')
        body += [tr(r, bench_row=True) for r in bench]
    return (
        SQUAD_TABLE_CSS
        + '<div class="fpl-table-wrap fpl-squad-wrap">'
        + '<table class="fpl-table fpl-squad"><thead><tr>'
        + "".join(head)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def squad_header(
    ctx: GameweekContext,
    entry: Mapping[str, Any] | None,
    preds: Mapping[int, PlayerPrediction],
) -> list[tuple[str, str, str]]:
    """Шапка команды: [(подпись, значение, help[, класс])] — очки сезона, общий ранг (FPL
    entry), банк, бесплатные трансферы, чипы (короткие WC/FH/BB/TC, иначе наезжают на соседнюю
    карточку), прогноз очков на тур (старт, капитан ×2)."""
    points = entry.get("points") if entry else None
    rank = entry.get("rank") if entry else None
    total = 0.0
    have_pred = False
    for p in ctx.squad or []:
        if not p.is_starting:
            continue
        x = _xpts_next(p, preds.get(p.id))
        if x is None:
            continue
        have_pred = True
        total += x * (2 if p.is_captain else 1)
    names = [CHIP_RU.get(c, c) for c in ctx.chips_available]
    chips = " · ".join(CHIP_SHORT.get(c, c) for c in ctx.chips_available) or "—"
    chips_help = (
        "Неиспользованные чипы: " + ", ".join(names) if names else "Неиспользованные чипы"
    )
    return [
        (
            "Очки сезона",
            "—" if points is None else str(int(points)),
            "Очки команды за сезон по FPL",
        ),
        (
            "Общий ранг",
            "—" if rank is None else f"{int(rank):,}".replace(",", " "),
            "Место среди всех менеджеров FPL",
        ),
        ("Банк", money(ctx.bank), "Свободные деньги, £ млн"),
        (
            "Бесплатных трансферов",
            "—" if ctx.free_transfers is None else str(ctx.free_transfers),
            "Трансферы без потери очков на этот тур; каждый сверх — платный (−4)",
        ),
        ("Чипы", chips, chips_help, "wrap"),
        (
            f"Прогноз очков на GW{ctx.gw}",
            num(total, 1) if have_pred else "—",
            "Сумма прогноза очков стартового состава на тур, капитан ×2",
        ),
    ]


def evidence_rows(evidence: Sequence[EvidenceItem]) -> list[dict[str, Any]]:
    return [
        {"Источник": e.source, "Дата": e.date, "Цитата": e.quote, "Ссылка": e.url} for e in evidence
    ]


# ---------- к дедлайну ----------


def signed_words(x: float | None, digits: int = 2) -> str:
    """«плюс 1.91» / «минус 0.50» — без знаков +/− на экране."""
    if x is None:
        return "—"
    if abs(x) < 5 * 10 ** (-(digits + 1)):
        return f"{0:.{digits}f}"
    return f"{'плюс' if x > 0 else 'минус'} {abs(x):.{digits}f}"


def vs_best_label(gap: float | None, digits: int = 2) -> str:
    """Подпись к «Ваш старт»: gap = лучший состав − ваш старт.
    Отставание — «−5.91 к лучшему составу»."""
    if gap is None:
        return "—"
    return f"{signed_compact(-gap, digits)} к лучшему составу"


def _best_label(gap: float | None, digits: int = 2) -> str:
    """Алиас `vs_best_label` — страница «К дедлайну» вызывает `fmt._best_label`."""
    return vs_best_label(gap, digits)


def signed_compact(x: float | None, digits: int = 1) -> str:
    """«+1.9» / «−0.5» / «0.0» — короткое значение для узких метрик, без слова «плюс»."""
    if x is None:
        return "—"
    if abs(x) < 5 * 10 ** (-(digits + 1)):
        return f"{0:.{digits}f}"
    sign = "+" if x > 0 else "−"
    return f"{sign}{abs(x):.{digits}f}"


def hit_compact(hit_cost: int | None) -> str:
    """Платный трансфер в метрике: «нет» / «−4»."""
    return "нет" if not hit_cost else f"−{int(hit_cost)}"


def route_verdict_note(verdict: str) -> str | None:
    """Строка-предупреждение вместо бейджа; None для «go» (рекомендация уже в заголовке)."""
    return VERDICT_NOTE.get(verdict)


def bank_millions(x: float | None) -> str:
    """«3.0 миллиона» — банк без символа валюты и без «m»."""
    return "—" if x is None else f"{x:.1f} миллиона"


def percent_amount(n: float) -> str:
    """«51 процент» / «40 процентов» — число и слово без знака %."""
    return plural(round(float(n)), "процент", "процента", "процентов")


# Короткие русские имена клубов по FPL short_name (для «Лидс дома», «Вест Хэм в гостях»).
TEAM_NAME_RU: dict[str, str] = {
    "ARS": "Арсенал",
    "AVL": "Астон Вилла",
    "BOU": "Борнмут",
    "BRE": "Брентфорд",
    "BHA": "Брайтон",
    "BUR": "Бернли",
    "CHE": "Челси",
    "COV": "Ковентри",
    "CRY": "Кристал Пэлас",
    "EVE": "Эвертон",
    "FUL": "Фулхэм",
    "HUL": "Халл",
    "IPS": "Ипсвич",
    "LEE": "Лидс",
    "LEI": "Лестер",
    "LIV": "Ливерпуль",
    "LUT": "Лутон",
    "MCI": "Сити",
    "MUN": "Юнайтед",
    "NEW": "Ньюкасл",
    "NFO": "Ноттингем",
    "SHU": "Шеффилд Юнайтед",
    "SOU": "Саутгемптон",
    "SUN": "Сандерленд",
    "TOT": "Тоттенхэм",
    "WHU": "Вест Хэм",
    "WOL": "Вулверхэмптон",
}


def club_name_ru(opponent: str | None) -> str:
    """Код/имя соперника -> короткое русское имя клуба; неизвестное оставляем как есть."""
    if not opponent:
        return "соперник"
    key = opponent.strip()
    return TEAM_NAME_RU.get(key.upper(), key)


# Творительный падеж для «сыграет с Борнмутом» (ключи — русские имена из TEAM_NAME_RU).
CLUB_INSTR: dict[str, str] = {
    "Арсенал": "Арсеналом",
    "Астон Вилла": "Астон Виллой",
    "Борнмут": "Борнмутом",
    "Брентфорд": "Брентфордом",
    "Брайтон": "Брайтоном",
    "Бернли": "Бернли",
    "Челси": "Челси",
    "Ковентри": "Ковентри",
    "Кристал Пэлас": "Кристал Пэласом",
    "Эвертон": "Эвертоном",
    "Фулхэм": "Фулхэмом",
    "Халл": "Халлом",
    "Ипсвич": "Ипсвичем",
    "Лидс": "Лидсом",
    "Лестер": "Лестером",
    "Ливерпуль": "Ливерпулем",
    "Лутон": "Лутоном",
    "Сити": "Сити",
    "Юнайтед": "Юнайтед",
    "Ньюкасл": "Ньюкаслом",
    "Ноттингем": "Ноттингемом",
    "Шеффилд Юнайтед": "Шеффилд Юнайтед",
    "Саутгемптон": "Саутгемптоном",
    "Сандерленд": "Сандерлендом",
    "Тоттенхэм": "Тоттенхэмом",
    "Вест Хэм": "Вест Хэмом",
    "Вулверхэмптон": "Вулверхэмптоном",
}


def club_instr(name: str) -> str:
    """Творительный: «Борнмут» -> «Борнмутом»; неизвестное — как есть."""
    return CLUB_INSTR.get(name, name)


@dataclass(frozen=True)
class WhyMatch:
    """Один матч для текста «Почему»: имя клуба, дом/выезд, опционально P(win) и xG."""

    opponent_name: str
    is_home: bool
    win_prob: float | None = None  # 0..1; None -> не выдумываем проценты победы
    xg_for: float | None = None
    xg_against: float | None = None
    fsi: int | None = None


def fixture_match_label(fixture: str | None) -> str:
    """Матч в таблице состава: `vTOT (FSI 4)` / `@LIV` -> «TOT (д)» / «LIV (в)»;
    уже «TOT (H)» / «TOT (д)» тоже нормализуем."""
    if not fixture or fixture.strip() in ("—", "-"):
        return "—"
    parts: list[str] = []
    for chunk in re.split(r"\s*·\s*", fixture.strip()):
        chunk = chunk.strip()
        m = re.match(r"^([v@])([A-Za-z]{2,4})\b", chunk)
        if m:
            parts.append(f"{m.group(2).upper()} ({'д' if m.group(1) == 'v' else 'в'})")
            continue
        m = re.match(r"^([A-Za-z]{2,4})\s*\(([HhAaДдВв])\)", chunk)
        if m:
            mark = m.group(2).lower()
            where = "д" if mark in ("h", "д") else "в"
            parts.append(f"{m.group(1).upper()} ({where})")
            continue
        parts.append(chunk)
    return " · ".join(parts)


def _join_ru(parts: Sequence[str]) -> str:
    """«A», «A и B», «A, B и C»."""
    items = [p for p in parts if p]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} и {items[1]}"
    return f"{', '.join(items[:-1])} и {items[-1]}"


def fixtures_calendar_phrase(matches: Sequence[WhyMatch]) -> str:
    """«Лидс дома, затем Брайтон в гостях и Вест Хэм дома»."""
    parts = [
        f"{m.opponent_name} {'дома' if m.is_home else 'в гостях'}" for m in matches
    ]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} и {parts[1]}"
    return f"{parts[0]}, затем {_join_ru(parts[1:])}"


def fixtures_stats_phrase(matches: Sequence[WhyMatch]) -> str:
    """Шансы на победу, если все win_prob заданы; иначе ожидаемые голы; иначе пусто.
    Проценты победы не выдумываем."""
    if not matches:
        return ""
    if all(m.win_prob is not None for m in matches):
        nums = [str(round(float(m.win_prob) * 100)) for m in matches]  # type: ignore[arg-type]
        return f"шансы команды на победу {_join_ru(nums)} процентов"
    if all(m.xg_for is not None and m.xg_against is not None for m in matches):
        scored = [f"{float(m.xg_for):.1f}" for m in matches]  # type: ignore[arg-type]
        conceded = [f"{float(m.xg_against):.1f}" for m in matches]  # type: ignore[arg-type]
        return (
            f"команда забивает в среднем {_join_ru(scored)}, "
            f"пропускает {_join_ru(conceded)}"
        )
    return ""


def _mean_fsi(matches: Sequence[WhyMatch]) -> float | None:
    vals = [int(m.fsi) for m in matches if m.fsi is not None]
    return sum(vals) / len(vals) if vals else None


def _calendar_hardness(matches: Sequence[WhyMatch]) -> str:
    mean = _mean_fsi(matches)
    if mean is None:
        return "сложный"
    if mean >= 4.0:
        return "тяжёлый"
    if mean <= 2.0:
        return "лёгкий"
    return "средний"


_COUNT_RU = {1: "один", 2: "два", 3: "три", 4: "четыре"}


def _venue_word(is_home: bool) -> str:
    return "дома" if is_home else "в гостях"


def _match_with_venue(m: WhyMatch) -> str:
    return f"{m.opponent_name} {_venue_word(m.is_home)}"


def _count_matches(n: int, one: str, few: str, many: str) -> str:
    word = _COUNT_RU.get(n, str(n))
    tail, tail2 = n % 10, n % 100
    if tail == 1 and tail2 != 11:
        noun = one
    elif 2 <= tail <= 4 and not 12 <= tail2 <= 14:
        noun = few
    else:
        noun = many
    return f"{word} {noun}"


def _unique_hardest(matches: Sequence[WhyMatch]) -> WhyMatch | None:
    """Единственный матч с максимальной сложностью ≥ 4; иначе None."""
    if not matches:
        return None
    best = max((m.fsi or 0) for m in matches)
    if best < 4:
        return None
    tops = [m for m in matches if (m.fsi or 0) == best]
    return tops[0] if len(tops) == 1 else None


def sell_ahead_phrase(matches: Sequence[WhyMatch]) -> str:
    """«впереди Сити дома и два выездных матча» / «впереди Лидс дома, затем Брайтон и Вест Хэм»."""
    if not matches:
        return ""
    standout = _unique_hardest(matches)
    if standout is not None:
        rest = [m for m in matches if m is not standout]
        head = f"впереди {_match_with_venue(standout)}"
        if not rest:
            return f"{head} — самый тяжёлый"
        if all(not m.is_home for m in rest):
            return f"{head} и {_count_matches(len(rest), 'выездной матч', 'выездных матча', 'выездных матчей')}"
        if all(m.is_home for m in rest):
            return (
                f"{head} и {_count_matches(len(rest), 'домашний матч', 'домашних матча', 'домашних матчей')}"
            )
        return f"{head} — самый тяжёлый"
    first, rest = matches[0], list(matches[1:])
    head = f"впереди {_match_with_venue(first)}"
    if not rest:
        return head
    if all(not m.is_home for m in rest):
        return f"{head} и {_count_matches(len(rest), 'выездной матч', 'выездных матча', 'выездных матчей')}"
    if all(m.is_home for m in rest):
        return f"{head} и {_count_matches(len(rest), 'домашний матч', 'домашних матча', 'домашних матчей')}"
    return f"{head}, затем {_join_ru([m.opponent_name for m in rest])}"


def buy_fixtures_phrase(matches: Sequence[WhyMatch]) -> str:
    """«с Борнмутом дома, Эвертоном и Тоттенхэмом»."""
    if not matches:
        return ""
    first = f"с {club_instr(matches[0].opponent_name)} {_venue_word(matches[0].is_home)}"
    rest = [club_instr(m.opponent_name) for m in matches[1:]]
    if not rest:
        return first
    if len(rest) == 1:
        return f"{first} и {rest[0]}"
    return f"{first}, {_join_ru(rest)}"


def _first_flag(flags: Mapping[str, Sequence[str]], name: str) -> str:
    items = [f for f in flags.get(name, ()) if f]
    return items[0] if items else ""


_FIXTURES_DETAIL = re.compile(
    r"mean fixture strength ([\d.]+)/5 over GW(\d+)[–\-](\d+)", re.IGNORECASE
)
_DOUBTFUL_DETAIL = re.compile(r"FPL doubtful (\d+)%", re.IGNORECASE)
_NOT_PLAYING_DETAIL = re.compile(r"p_start ([\d.]+)", re.IGNORECASE)
_LOW_XPTS_DETAIL = re.compile(
    r"([\d.]+) xPts over horizon vs position median ([\d.]+)", re.IGNORECASE
)
_OUT_NOTE = re.compile(r"^out (.+): (\w+) \((.+)\)\s*$", re.IGNORECASE)
_IN_STATUS = re.compile(r"^in (.+): FPL status (\w+)\s*$", re.IGNORECASE)
_IN_VARIANCE = re.compile(r"^in (.+): high variance \(sd ([\d.]+)\)\s*$", re.IGNORECASE)
_IN_DIFF = re.compile(r"^in (.+): differential \(([\d.]+)% owned\)\s*$", re.IGNORECASE)
_IN_TEMPLATE = re.compile(r"^in (.+): template \((\d+)% owned\)\s*$", re.IGNORECASE)


def _ownership_from_risk_note(risk_note: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in risk_note.split(";"):
        part = part.strip()
        m = _IN_TEMPLATE.match(part) or _IN_DIFF.match(part)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def _soft_flags_from_risk_note(risk_note: str) -> dict[str, list[str]]:
    """Имя -> короткие вставки (статус, разброс) без жаргона."""
    flags: dict[str, list[str]] = {}
    for part in risk_note.split(";"):
        part = part.strip()
        m = _IN_STATUS.match(part)
        if m:
            label = STATUS_RU.get(m.group(2), "под вопросом")
            flags.setdefault(m.group(1), []).append(f"{label} по статусу FPL")
            continue
        m = _IN_VARIANCE.match(part)
        if m:
            flags.setdefault(m.group(1), []).append("очки нестабильны")
            continue
        m = _OUT_NOTE.match(part)
        if m and m.group(2).lower() == "doubtful":
            dm = _DOUBTFUL_DETAIL.search(m.group(3))
            if dm:
                flags.setdefault(m.group(1), []).append(
                    f"под вопросом по статусу FPL, шанс сыграть {percent_amount(int(dm.group(1)))}"
                )
            else:
                flags.setdefault(m.group(1), []).append("под вопросом по статусу FPL")
    return flags


def why_matches_from_prediction(
    pred: PlayerPrediction, *, horizon: int = 3
) -> list[WhyMatch]:
    """Матчи из predict_player на горизонт; win_prob в FixtureBrief нет — не подставляем."""
    out: list[WhyMatch] = []
    for g in pred.by_gw[: max(1, horizon)]:
        for f in g.fixtures:
            out.append(
                WhyMatch(
                    opponent_name=club_name_ru(f.opponent),
                    is_home=f.is_home,
                    win_prob=None,
                    xg_for=f.xg_for,
                    xg_against=f.xg_against,
                    fsi=f.fsi,
                )
            )
    return out


def route_calendars_from_preds(
    route: RouteOut,
    preds: Mapping[int, PlayerPrediction],
    *,
    horizon: int = 3,
) -> tuple[dict[str, list[WhyMatch]], dict[str, float]]:
    """Календари и владение по out_ids/in_ids маршрута (parallel lists)."""
    calendars: dict[str, list[WhyMatch]] = {}
    ownership: dict[str, float] = {}
    pairs = list(zip(route.out_ids, route.out, strict=False)) + list(
        zip(route.in_ids, route.in_, strict=False)
    )
    for pid, name in pairs:
        pred = preds.get(int(pid))
        if pred is None:
            continue
        calendars[name] = why_matches_from_prediction(pred, horizon=horizon)
        ownership[name] = float(pred.player.ownership or 0.0)
    return calendars, ownership


def _coerce_facts(items: Sequence[Any]) -> list[Any]:
    """WhyFact или dict из кэша Streamlit — одинаковый интерфейс kind/clause/role."""
    from fplcopilot.core.why_facts import WhyFact

    out: list[Any] = []
    for f in items:
        if isinstance(f, WhyFact):
            out.append(f)
        elif isinstance(f, Mapping) and f.get("kind") and f.get("clause"):
            out.append(
                WhyFact(
                    str(f["kind"]),
                    str(f["clause"]),
                    str(f.get("role") or "any"),
                    int(f.get("priority") or 50),
                    int(f["player_id"]) if f.get("player_id") is not None else None,
                    int(f["points"]) if f.get("points") is not None else None,
                )
            )
    return out


def facts_by_route_name(
    route: RouteOut, facts_by_id: Mapping[int, Sequence[Any]]
) -> dict[str, list[Any]]:
    """out_ids/in_ids -> имя на карточке."""
    out: dict[str, list[Any]] = {}
    pairs = list(zip(route.out_ids, route.out, strict=False)) + list(
        zip(route.in_ids, route.in_, strict=False)
    )
    for pid, name in pairs:
        out[name] = _coerce_facts(facts_by_id.get(int(pid), ()))
    return out


def _force_status(
    name: str, facts: Mapping[str, Sequence[Any]], extra: Sequence[Any]
) -> list[Any]:
    """Статус не должен выпадать из слота interesting_limit."""
    hit = next(
        (f for f in facts.get(name, ()) if getattr(f, "kind", "") == "status"),
        None,
    )
    out = list(extra)
    if hit is not None and not any(getattr(f, "kind", "") == "status" for f in out):
        out.append(hit)
    return out


def _picked_facts(
    names: Sequence[str],
    facts: Mapping[str, Sequence[Any]],
    role: str,
) -> list[Any]:
    from fplcopilot.core.why_facts import pick_facts

    collected: list[Any] = []
    for name in names:
        collected.extend(_coerce_facts(facts.get(name, ())))
    return pick_facts(collected, role=role)


def _sell_sentence(
    names: Sequence[str],
    calendars: Mapping[str, Sequence[WhyMatch]],
    flags: Mapping[str, Sequence[str]],
    extra_facts: Sequence[Any] = (),
) -> str:
    """«Продаём X: впереди соперник дома и два выездных матча.»"""
    if not names:
        return ""
    who = _join_ru(list(names))
    fx: list[WhyMatch] = []
    flag = ""
    for name in names:
        if not fx:
            fx = list(calendars.get(name, ()))
        if not flag:
            flag = _first_flag(flags, name)
    ahead = sell_ahead_phrase(fx)
    extra = ""
    for f in extra_facts:
        if getattr(f, "kind", "") in (
            "form_unlucky",
            "form_lucky",
            "form_repeatable",
            "few_chances",
            "low_minutes",
            "status",
        ):
            extra = f.clause
            break
    bits = [p for p in (ahead, extra, flag) if p]
    if bits:
        return f"Продаём {who}: {'; '.join(bits)}."
    return f"Продаём {who}."


def _buy_sentences(
    names: Sequence[str],
    calendars: Mapping[str, Sequence[WhyMatch]],
    own: Mapping[str, float],
    flags: Mapping[str, Sequence[str]],
    out_fsi: float | None,
    extra_facts: Sequence[Any] = (),
) -> list[str]:
    """1–2 фразы про покупку: стата, затем календарь."""
    if not names:
        return []
    who = _join_ru(list(names))
    fx: list[WhyMatch] = []
    flag = ""
    own_v: float | None = None
    for name in names:
        if not fx:
            fx = list(calendars.get(name, ()))
        if not flag:
            flag = _first_flag(flags, name)
        if own_v is None and name in own:
            own_v = own[name]
    form = next(
        (f for f in extra_facts if getattr(f, "kind", "").startswith("form_")),
        None,
    )
    moments = next(
        (f for f in extra_facts if getattr(f, "kind", "") == "moments_no_goals"),
        None,
    )
    if form is not None:
        moments = None
    contrast = next(
        (f for f in extra_facts if getattr(f, "kind", "") == "penalty_contrast"),
        None,
    )
    leaky = next((f for f in extra_facts if getattr(f, "kind", "") == "leaky_opp"), None)
    strat = next((f for f in extra_facts if getattr(f, "kind", "") in ("strategy", "ownership")), None)
    lead = form or moments

    sentences: list[str] = []
    if lead is not None:
        sentences.append(f"{who} {lead.clause}.")
    if contrast is not None:
        clause = contrast.clause
        sentences.append(clause[:1].upper() + clause[1:] + ".")

    cal = buy_fixtures_phrase(fx)
    if cal:
        head = f"Сыграет {cal}" if lead is not None else f"{who} сыграет {cal}"
    elif lead is None:
        head = f"Покупаем {who}"
    else:
        head = ""
    tail: list[str] = []
    in_fsi = _mean_fsi(fx)
    if out_fsi is not None and in_fsi is not None and in_fsi <= out_fsi - 0.7:
        tail.append("календарь спокойнее")
    elif fx and _calendar_hardness(fx) == "лёгкий":
        tail.append("календарь спокойный")
    if leaky is not None:
        tail.append(leaky.clause)
    if strat is None:
        if own_v is not None and own_v >= 40:
            tail.append(f"его держат {percent_amount(own_v)} менеджеров")
        elif own_v is not None and own_v < 5:
            tail.append(f"его держат всего {percent_amount(own_v)} менеджеров")
    if flag:
        tail.append(flag)
    if head:
        if not tail:
            sentences.append(f"{head}.")
        elif len(tail) == 1:
            sentences.append(f"{head} — {tail[0]}.")
        else:
            sentences.append(f"{head} — {tail[0]}, плюс {tail[1]}.")
    if strat is not None:
        clause = strat.clause
        sentences.append(clause[:1].upper() + clause[1:] + ".")
    return sentences


def _buy_sentence(
    names: Sequence[str],
    calendars: Mapping[str, Sequence[WhyMatch]],
    own: Mapping[str, float],
    flags: Mapping[str, Sequence[str]],
    out_fsi: float | None,
) -> str:
    """Одна строка без фактов — для тестов старого пути."""
    return " ".join(_buy_sentences(names, calendars, own, flags, out_fsi))


def _has_penalty_fact(names: Sequence[str], facts: Mapping[str, Sequence[Any]]) -> bool:
    return any(
        getattr(f, "kind", "") in ("penalty", "penalty_outlook")
        for name in names
        for f in facts.get(name, ())
    )


def _penalty_contrast(
    out_names: Sequence[str],
    in_names: Sequence[str],
    facts: Mapping[str, Sequence[Any]],
) -> Any | None:
    """Только контраст + посчитанный outlook. Голая фраза «бьёт пенальти» не проходит."""
    outlook = next(
        (
            f
            for name in in_names
            for f in facts.get(name, ())
            if getattr(f, "kind", "") == "penalty_outlook"
        ),
        None,
    )
    if outlook is None:
        return None
    if _has_penalty_fact(out_names, facts):
        return None
    return outlook


def _clause(fact: Any) -> str:
    from fplcopilot.core.why_narrate import has_stamp

    clause = getattr(fact, "clause", "") or ""
    return "" if has_stamp(clause) else clause


def _sell_stance(
    kinds: Mapping[str, Any],
    sell_cal: str,
    flag: str,
) -> tuple[str, str, bool]:
    """Одна позиция продажи. few_chances бьёт form_ok; оба сразу не отдаём."""
    if "low_minutes" in kinds or "status" in kinds:
        fact = kinds.get("low_minutes") or kinds.get("status")
        return "minutes_risk", _clause(fact) or flag, False
    if "form_unlucky" in kinds:
        return "unlucky", _clause(kinds["form_unlucky"]), False
    if "few_chances" in kinds:
        return "weak_form", _clause(kinds["few_chances"]), False
    if "form_repeatable" in kinds or "form_ok_sell" in kinds:
        return ("solid_form_bad_fixtures" if sell_cal else "neutral"), "", True
    if flag:
        return "minutes_risk", flag, False
    if sell_cal:
        return "calendar", "", False
    return "neutral", "", False


def _buy_stance(kinds: Mapping[str, Any]) -> tuple[str, int | None, str | None, str | None, str | None]:
    """Одна позиция покупки. lucky / repeatable / unlucky / moments / neutral."""
    lucky_pts = None
    if "form_lucky" in kinds:
        m = re.search(r"(\d+)", _clause(kinds["form_lucky"]) or getattr(kinds["form_lucky"], "clause", "") or "")
        lucky_pts = int(m.group(1)) if m else None
        return "lucky_but_fixtures", lucky_pts, None, None, None
    if "form_repeatable" in kinds:
        return "repeatable", None, _clause(kinds["form_repeatable"]) or "стабильных источников", None, None
    if "form_unlucky" in kinds:
        return "unlucky", None, None, _clause(kinds["form_unlucky"]), None
    if "moments_no_goals" in kinds:
        return "moments", None, None, _clause(kinds["moments_no_goals"]), None
    status = _clause(kinds["status"]) if "status" in kinds else None
    return "neutral", None, None, None, status


def _points_from_kinds(kinds: Mapping[str, Any]) -> int | None:
    """Очки окна формы только из факта этого игрока (не сумма чужих)."""
    for key in ("form_lucky", "form_unlucky", "form_repeatable", "form_ok_sell"):
        fact = kinds.get(key)
        if fact is None:
            continue
        pts = getattr(fact, "points", None)
        if pts is not None:
            return int(pts)
        clause = getattr(fact, "clause", "") or ""
        m = re.search(r"(\d+)\s+очк", clause)
        if m:
            return int(m.group(1))
        m = re.search(r"очков мало \((\d+)", clause)
        if m:
            return int(m.group(1))
        m = re.search(r"\((\d+)\s+за\s+", clause)
        if m:
            return int(m.group(1))
    return None


def _interesting_clause(kinds: Mapping[str, Any], buy_stance: str) -> str | None:
    """Нумерованный факт сверх календаря: сухие, моменты, дырявая/глухая оборона."""
    order = ("form_repeatable", "moments_no_goals", "leaky_opp", "tight_opp")
    for key in order:
        if key not in kinds:
            continue
        if buy_stance == "lucky_but_fixtures" and key == "form_repeatable":
            continue
        clause = _clause(kinds[key])
        if clause and re.search(r"\d", clause):
            return clause
    return None


def _sell_block(
    name: str,
    pid: int | None,
    calendars: Mapping[str, Sequence[WhyMatch]],
    flags: Mapping[str, Sequence[str]],
    extra: Sequence[Any],
) -> dict[str, Any]:
    fx = list(calendars.get(name, ()))[:3] if name else []
    cal = sell_ahead_phrase(fx)
    kinds = {getattr(f, "kind", ""): f for f in extra}
    flag = _first_flag(flags, name) if name else ""
    stance, detail, form_ok = _sell_stance(kinds, cal, flag)
    reason = detail if stance in ("weak_form", "minutes_risk", "unlucky") else ""
    return {
        "name": name or "",
        "names": [name] if name else [],
        "id": pid,
        "stance": stance,
        "calendar": cal,
        "opponents": [m.opponent_name for m in fx],
        "detail": detail,
        "reason": reason,
        "form_ok": form_ok,
        "points": _points_from_kinds(kinds),
    }


def _buy_block(
    name: str,
    pid: int | None,
    calendars: Mapping[str, Sequence[WhyMatch]],
    own: Mapping[str, float],
    extra: Sequence[Any],
    *,
    sell_has_pen: bool,
) -> dict[str, Any]:
    from fplcopilot.core.why_narrate import has_stamp

    fx = list(calendars.get(name, ()))[:3] if name else []
    cal = buy_fixtures_phrase(fx)
    opps = [m.opponent_name for m in fx]
    kinds = {getattr(f, "kind", ""): f for f in extra}
    stance, lucky_pts, repeatable, moments, _stance_status = _buy_stance(kinds)
    status = _clause(kinds["status"]) if "status" in kinds else _stance_status
    leaky = None
    for key in ("leaky_opp", "tight_opp"):
        fact = kinds.get(key)
        clause = getattr(fact, "clause", "") or "" if fact is not None else ""
        if fact is not None and not has_stamp(clause):
            leaky = clause
            break
    outlook = kinds.get("penalty_outlook") if not sell_has_pen else None
    pen = getattr(outlook, "clause", None) if outlook is not None else None
    pen_opp = None
    if pen:
        for opp in opps:
            if opp and opp in pen:
                pen_opp = opp
                break
    pts = _points_from_kinds(kinds)
    if lucky_pts is None and stance == "lucky_but_fixtures":
        lucky_pts = pts
    interesting = _interesting_clause(kinds, stance)
    if stance == "repeatable" and repeatable and re.search(r"\d", str(repeatable)):
        interesting = interesting or repeatable
    return {
        "name": name or "",
        "names": [name] if name else [],
        "id": pid,
        "stance": stance,
        "calendar": cal,
        "opponents": opps,
        "lucky_points": lucky_pts,
        "repeatable": repeatable,
        "moments": moments,
        "leaky": leaky,
        "interesting": interesting,
        "penalty": pen,
        "penalty_opponent": pen_opp,
        "status": status,
        "points": pts,
        "ownership": round(own[name]) if name and name in own else None,
    }


def compose_why_payload(
    route: RouteOut,
    *,
    horizon: int = 3,
    calendars: Mapping[str, Sequence[WhyMatch]] | None = None,
    ownership: Mapping[str, float] | None = None,
    facts: Mapping[str, Sequence[Any]] | None = None,
) -> dict[str, Any]:
    """Факты строго по игроку (id/имя). Двойной трансфер — отдельные пары sell_i→buy_i."""
    calendars = dict(calendars or {})
    own = dict(_ownership_from_risk_note(route.risk_note))
    own.update(dict(ownership or {}))
    flags = _soft_flags_from_risk_note(route.risk_note)
    out_names = list(route.out)
    in_names = list(route.in_)
    fact_map = dict(facts or {})
    n = max(len(out_names), len(in_names))
    pairs: list[dict[str, Any]] = []
    for i in range(n):
        sell_name = out_names[i] if i < len(out_names) else ""
        buy_name = in_names[i] if i < len(in_names) else ""
        sell_id = int(route.out_ids[i]) if i < len(route.out_ids) else None
        buy_id = int(route.in_ids[i]) if i < len(route.in_ids) else None
        sell_extra = _picked_facts([sell_name], fact_map, "sell") if sell_name else []
        buy_extra = list(_picked_facts([buy_name], fact_map, "buy")) if buy_name else []
        if sell_name:
            sell_extra = _force_status(sell_name, fact_map, sell_extra)
        if buy_name:
            buy_extra = _force_status(buy_name, fact_map, buy_extra)
        sell_pen = bool(sell_name) and _has_penalty_fact([sell_name], fact_map)
        if buy_name and not sell_pen:
            outlook = next(
                (
                    f
                    for f in fact_map.get(buy_name, ())
                    if getattr(f, "kind", "") == "penalty_outlook"
                ),
                None,
            )
            if outlook is not None and not any(
                getattr(f, "kind", "") == "penalty_outlook" for f in buy_extra
            ):
                buy_extra.append(outlook)
        pairs.append(
            {
                "sell": _sell_block(sell_name, sell_id, calendars, flags, sell_extra),
                "buy": _buy_block(
                    buy_name, buy_id, calendars, own, buy_extra, sell_has_pen=sell_pen
                ),
            }
        )
    first = pairs[0] if pairs else {"sell": {}, "buy": {}}
    xi = route.xi_points_after
    xi_s = None if xi is None else num(xi, 1)
    bank_s = None if route.new_bank is None else f"{float(route.new_bank):.1f}"
    ids = tuple(int(x) for x in (*route.out_ids, *route.in_ids))
    variant = (sum(ids) + int(horizon)) % 3 if ids else 0
    return {
        "pairs": pairs,
        "sell": first.get("sell") or {},
        "buy": first.get("buy") or {},
        "after": {
            "xi_points": xi_s,
            "bank": bank_s,
            "horizon": int(horizon),
            "hit": int(route.hit_cost or 0),
        },
        "variant": variant,
    }


def route_why(
    route: RouteOut,
    *,
    horizon: int = 3,
    calendars: Mapping[str, Sequence[WhyMatch]] | None = None,
    ownership: Mapping[str, float] | None = None,
    facts: Mapping[str, Sequence[Any]] | None = None,
) -> str:
    """2–4 коротких предложения. Детерминированный запасной текст (без LLM)."""
    from fplcopilot.core.why_narrate import fallback_why

    payload = compose_why_payload(
        route, horizon=horizon, calendars=calendars, ownership=ownership, facts=facts
    )
    return fallback_why(payload)


def risk_note_line_ru(part: str) -> str:
    """Одна часть risk_note -> короткая русская фраза (для тестов/отладки; UI берёт route_why)."""
    part = part.strip()
    if not part:
        return ""
    m = _OUT_NOTE.match(part)
    if m:
        kind, detail = m.group(2).lower(), m.group(3)
        if kind == "fixtures":
            fm = _FIXTURES_DETAIL.search(detail)
            if fm:
                n = int(fm.group(3)) - int(fm.group(2)) + 1
                tours = plural(n, "тур", "тура", "туров").split(" ", 1)[1]
                strength = float(fm.group(1))
                s_txt = f"{strength:.0f}" if strength == int(strength) else f"{strength:.1f}"
                return (
                    f"{m.group(1)}: тяжёлый календарь, средняя сложность {s_txt} из 5 "
                    f"на ближайшие {n} {tours}"
                )
            return f"{m.group(1)}: тяжёлый календарь на ближайшие туры"
        if kind == "doubtful":
            return f"{m.group(1)}: под вопросом по статусу FPL"
        if kind == "not_playing":
            return f"{m.group(1)}: риск ротации"
        if kind == "blank":
            return f"{m.group(1)}: нет матча в ближайшем туре"
        if kind == "low_xpts":
            return f"{m.group(1)}: слабый прогноз очков на ближайшие туры"
        if kind == "injured":
            return f"{m.group(1)}: травма по статусу FPL"
        if kind == "suspended":
            return f"{m.group(1)}: дисквалификация по статусу FPL"
        if kind == "unavailable":
            return f"{m.group(1)}: недоступен по статусу FPL"
        return f"{m.group(1)}: {ISSUE_KIND_RU.get(kind, kind)}"
    m = _IN_STATUS.match(part)
    if m:
        return f"{m.group(1)}: {STATUS_RU.get(m.group(2), m.group(2))} по статусу FPL"
    m = _IN_VARIANCE.match(part)
    if m:
        return f"{m.group(1)}: нестабильные очки, разброс около {m.group(2)} очков"
    m = _IN_DIFF.match(part)
    if m:
        return (
            f"{m.group(1)}: его выбрали {percent_amount(float(m.group(2)))} менеджеров, "
            f"это редкий выбор"
        )
    m = _IN_TEMPLATE.match(part)
    if m:
        return (
            f"{m.group(1)}: его выбрали {percent_amount(int(m.group(2)))} менеджеров, "
            f"это надёжный выбор"
        )
    return re.sub(r"^(out|in)\s+", "", part, flags=re.IGNORECASE)


LINEUP_COLUMN_HELP: dict[str, str] = {
    "Роль": "C — капитан, VC — вице-капитан, «—» — в старте, B1… — скамейка",
    "Игрок": "Имя игрока; метка «продать» / «купить» — по рекомендуемому маршруту",
    "Поз": "Позиция: GKP вратарь, DEF защита, MID полузащита, FWD нападение",
    "Клуб": "Клуб игрока",
    "Цена": "Цена в FPL, миллионы фунтов",
    "Прогноз": "Ожидаемые очки в ближайшем туре по нашей модели",
    "разброс": "Насколько прогноз может отклониться: чем меньше, тем надёжнее",
    "шанс старта": "Вероятность выйти в стартовом составе по нашей модели минут",
    "Матч": "Соперник: (д) — дома, (в) — в гостях",
}


def lineup_rows(
    lu: LineupOut,
    *,
    sell: Sequence[str] = (),
    buy: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Строки таблицы лучшего состава: без колонки статуса, матч «TOT (д)», шанс старта в %."""
    sell_set = {s.casefold() for s in sell}
    buy_set = {s.casefold() for s in buy}
    rows: list[dict[str, Any]] = []

    def one(role: str, s: Any, *, tag: str | None = None) -> dict[str, Any]:
        name = s.name
        mark = tag
        if mark is None and name.casefold() in sell_set:
            mark = "продать"
        elif mark is None and name.casefold() in buy_set:
            mark = "купить"
        return {
            "Роль": role,
            "Игрок": name,
            "Поз": s.position,
            "Клуб": s.team,
            "Цена": f"{s.price:.1f}",
            "Прогноз": num(s.xpts),
            "разброс": num(s.sd, 1),
            "шанс старта": pct(
                (s.p_start * 100) if s.p_start is not None and s.p_start <= 1 else s.p_start,
                0,
            ),
            "Матч": fixture_match_label(s.fixture),
            "_tag": mark,
            "_id": getattr(s, "id", None),
        }

    for s in lu.starters:
        role = "C" if s.name == lu.captain else "VC" if s.name == lu.vice else "—"
        rows.append(one(role, s))
    for i, b in enumerate(lu.bench, start=1):
        rows.append(one(f"B{i}", b))
    return rows


def captain_rows(options: Sequence[CaptainOption]) -> list[dict[str, Any]]:
    return [
        {
            "Игрок": o.name,
            "Клуб": o.team,
            "Прогноз": num(o.xpts),
            "Как капитан": num(o.captain_points),
            "разброс": num(o.sd, 1),
            "Владение": pct(o.ownership, 1),
            "Тип": CAPTAIN_TAG_RU.get(o.tag, o.tag),
            "Матч": fixture_match_label(o.fixture),
        }
        for o in options
    ]


# Лучший состав с отметками трансфера — inline HTML (Styler в тёмной теме часто не красит).
LINEUP_TABLE_CSS = f"""
<style>
{TIP_CSS_RULES}
{DATA_TABLE_CSS_RULES}
.fpl-lineup tr.sell td {{ color: var(--fpl-faint, #9A9CA3); }}
.fpl-lineup tr.sell td:first-child, .fpl-lineup tr.buy td:first-child {{
  box-shadow: inset 2px 0 0 var(--fpl-faint, #9A9CA3);
}}
.fpl-lineup tr.buy td:first-child {{ box-shadow: inset 2px 0 0 var(--fpl-good, #16804A); }}
.fpl-lineup .tag-sell, .fpl-lineup .tag-buy, .fpl-lineup-buys .chip {{
  display: inline-block; margin-left: 7px; padding: 1px 7px; border-radius: 999px;
  font-size: 11px; font-weight: 600; letter-spacing: 0.01em; vertical-align: 1px;
}}
.fpl-lineup .tag-sell {{
  color: var(--fpl-muted, #63666D); box-shadow: inset 0 0 0 1px var(--fpl-line-strong, #DADBD6);
}}
.fpl-lineup .tag-buy, .fpl-lineup-buys .chip {{
  color: var(--fpl-good, #16804A); background: var(--fpl-good-bg, rgba(22, 128, 74, 0.12));
}}
.fpl-lineup-buys {{
  display: flex; flex-wrap: wrap; align-items: center; gap: 6px;
  margin: 0.25rem 0 0.75rem; font-size: 13px; color: var(--fpl-muted, inherit);
}}
.fpl-lineup-buys .chip {{ margin: 0; padding: 3px 10px; font-size: 12.5px; }}
</style>
"""


def buy_chips_html(names: Sequence[str]) -> str:
    """Зелёные метки «купить» над таблицей состава (игроков ещё нет в лучшем составе)."""
    if not names:
        return ""
    chips = "".join(
        f'<span class="chip">{_esc(n)} · купить</span>' for n in names
    )
    return LINEUP_TABLE_CSS + f'<div class="fpl-lineup-buys">Покупаем: {chips}</div>'


def lineup_table_html(
    lu: LineupOut,
    *,
    sell: Sequence[str] = (),
    buy: Sequence[str] = (),
) -> str:
    """HTML лучшего состава с приглушёнными «продать» и зелёной меткой у имени, если купить уже в старте."""
    rows = lineup_rows(lu, sell=sell, buy=buy)
    cols = ("Роль", "Игрок", "Поз", "Клуб", "Цена", "Прогноз", "разброс", "шанс старта", "Матч")
    num_cols = {"Цена", "Прогноз", "разброс", "шанс старта"}
    n_cols = len(cols)
    head = []
    for i, col in enumerate(cols):
        help_t = LINEUP_COLUMN_HELP.get(col, "")
        cls = "num" if col in num_cols else ""
        align = "left" if i < 2 else "right" if i >= n_cols - 2 else "center"
        label = tip(help_t, mark=col, align=align, css_class="th") if help_t else _esc(col)
        head.append(f'<th class="{cls}">{label}</th>')

    body = []
    for r in rows:
        tag = r.get("_tag")
        tr_cls = "sell" if tag == "продать" else "buy" if tag == "купить" else ""
        badge = ""
        if tag == "продать":
            badge = '<span class="tag-sell">продать</span>'
        elif tag == "купить":
            badge = '<span class="tag-buy">купить</span>'
        tds = []
        for col in cols:
            cls = "num" if col in num_cols else ""
            if col == "Игрок":
                tds.append(f'<td>{_esc(r[col])}{badge}</td>')
            elif col == "Роль":
                tds.append(f"<td>{role_html(str(r.get(col, '—')))}</td>")
            else:
                tds.append(f'<td class="{cls}">{_esc(r.get(col, "—"))}</td>')
        body.append(f'<tr class="{tr_cls}">' + "".join(tds) + "</tr>")

    return (
        LINEUP_TABLE_CSS
        + '<div class="fpl-table-wrap fpl-lineup-wrap"><table class="fpl-table fpl-lineup"><thead><tr>'
        + "".join(head)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def route_title(route: RouteOut) -> str:
    outs = ", ".join(route.out) if route.out else "—"
    inns = ", ".join(route.in_) if route.in_ else "—"
    return f"Продаём {outs}, покупаем {inns}."


def route_metrics_html(
    *,
    next_label: str,
    next_value: str,
    horizon_label: str,
    horizon_value: str,
    hit_label: str,
    hit_value: str,
) -> str:
    """Три метрики карточки без text-overflow: две сверху, платная — на всю ширину."""

    def cell(label: str, value: str, extra: str = "") -> str:
        cls = f' class="{extra}"' if extra else ""
        return (
            f"<div{cls}><div class=\"fpl-metric\"><div class=\"fpl-metric-label\">"
            f"<span>{_esc(label)}</span></div>"
            f"<div class=\"fpl-metric-value\">{_esc(value)}</div></div></div>"
        )

    return (
        TIP_WIDGET_CSS
        + '<div class="fpl-route-metrics">'
        + cell(next_label, next_value)
        + cell(horizon_label, horizon_value)
        + cell(hit_label, hit_value, "hit")
        + "</div>"
    )


def route_card(
    route: RouteOut,
    *,
    horizon: int = 3,
    calendars: Mapping[str, Sequence[WhyMatch]] | None = None,
    ownership: Mapping[str, float] | None = None,
    facts: Mapping[str, Sequence[Any]] | None = None,
) -> dict[str, Any]:
    why = route_why(
        route, horizon=horizon, calendars=calendars, ownership=ownership, facts=facts
    )
    return {
        "rank": route.rank,
        "title": route_title(route),
        "delta_next": signed_compact(route.gain_next_gw),
        "delta_horizon": signed_compact(route.gain_horizon),
        "hit": hit_compact(route.hit_cost),
        "verdict": route.verdict,
        "verdict_label": VERDICT_RU.get(route.verdict, route.verdict),
        "verdict_note": route_verdict_note(route.verdict),
        "why": why,
        "out": list(route.out),
        "in": list(route.in_),
        "horizon_label": f"За {plural(horizon, 'тур', 'тура', 'туров')}",
    }


def routes_caption(rt: RoutesOut, *, horizon: int | None = None) -> str:
    """Подпись над карточками — связная фраза во втором лице, без FT/xPts/route."""
    h = int(horizon if horizon is not None else rt.horizon)
    ft = int(rt.free_transfers or 0)
    bank = bank_millions(rt.bank)
    if ft == 0:
        start = f"У вас нет бесплатных трансферов, в банке {bank}"
    elif ft == 1:
        start = f"У вас один бесплатный трансфер и {bank} в банке"
    elif ft == 2:
        start = f"У вас два бесплатных трансфера и {bank} в банке"
    else:
        start = (
            f"У вас {plural(ft, 'бесплатный трансфер', 'бесплатных трансфера', 'бесплатных трансферов')} "
            f"и {bank} в банке"
        )
    tours = plural(h, "тур", "тура", "туров").split(" ", 1)[1]
    rec = (rt.recommendation or "").strip().lower()
    if rec == "hold" or rt.recommended_rank is None:
        rec_part = "рекомендуем держать трансфер"
    else:
        m = re.match(r"route\s+(\d+)", rec)
        n = m.group(1) if m else str(rt.recommended_rank)
        rec_part = f"рекомендуем вариант {n}"
    return (
        f"{start}; считаем на {h} {tours}. "
        f"Без трансферов ваш лучший состав наберёт {points_text(rt.baseline_xi_points, 2)} — "
        f"{rec_part}."
    )


# ---------- план ----------


def plan_move_rows(plan: PlanOut) -> list[dict[str, Any]]:
    rows = []
    for gw_s, moves in plan.moves_by_gw.items():
        if not moves:
            rows.append(
                {
                    "GW": int(gw_s),
                    "Out": "—",
                    "In": "—",
                    "Δ xPts горизонт": "",
                    "Хит": "",
                    "Причина": "roll (без трансфера)",
                }
            )
        for m in moves:
            reason = ISSUE_KIND_RU.get(m.out_problem or "", m.out_problem or "")
            rows.append(
                {
                    "GW": m.gw,
                    "Out": f"{m.out} (£{m.price_out:.1f})",
                    "In": f"{m.in_} (£{m.price_in:.1f})",
                    "Δ xPts горизонт": signed(m.delta_xpts_horizon),
                    "Хит": "−4" if m.paid else "",
                    "Причина": reason or "апгрейд по xPts",
                }
            )
    return rows


def plan_gw_rows(plan: PlanOut) -> list[dict[str, Any]]:
    gws = sorted(plan.xi_points_by_gw, key=int)
    return [
        {
            "GW": int(g),
            "FT": plan.ft_by_gw.get(g, ""),
            "Хит": plan.hits_by_gw.get(g, 0),
            "Банк": money(plan.bank_by_gw.get(g)),
            "Прогноз состава": num(plan.xi_points_by_gw.get(g)),
            "Капитан": plan.captain_by_gw.get(g, ""),
        }
        for g in gws
    ]


def plan_summary(plan: PlanOut) -> dict[str, str]:
    gain = plan.expected_total - plan.baseline_total
    out = {
        "Горизонт": f"GW{plan.from_gw}–GW{plan.from_gw + plan.horizon - 1}",
        "План": f"{num(plan.expected_total)} xPts",
        "Без трансферов": f"{num(plan.baseline_total)} xPts",
        "Выигрыш": f"{signed(gain)}",
        "Рекомендация": RECOMMENDATION_RU.get(plan.recommendation, plan.recommendation),
        "Солвер": f"{plan.solver} {plan.runtime_s:.1f} с"
        + (" (лимит времени)" if plan.time_limit_hit else ""),
    }
    if plan.wildcard:
        out["Wildcard"] = (
            f"{num(plan.wildcard.get('expected_total'))} xPts "
            f"({signed(plan.wildcard.get('delta_vs_plan'))} к плану)"
        )
    else:
        out["Wildcard"] = "недоступен"
    return out


def snapshot_time_text(value: str | datetime | None, tz: tzinfo | None = None) -> str:
    """Время снимка плана «2026-09-23T20:49:16+00:00» -> «23.09 в 01:49» (местное время);
    не разобралось — как есть."""
    if value is None:
        return "—"
    try:
        ts = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    except ValueError:
        return str(value)
    if ts.tzinfo is not None:
        ts = ts.astimezone(tz)
    return f"{ts:%d.%m} в {ts:%H:%M}"


def diff_rows(diff: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    kinds = {
        "added": "новый ход",
        "removed": "ход исчез",
        "changed_target": "другая цель",
        "moved_gw": "другой тур",
        "recommendation": "рекомендация",
    }
    return [
        {
            "GW": c.get("gw"),
            "Изменение": kinds.get(str(c.get("kind")), str(c.get("kind"))),
            "Детали": c.get("detail"),
        }
        for c in diff or []
    ]


# ---------- игрок ----------


def xpts_series(pred: PlayerPrediction) -> dict[str, float]:
    return {f"GW{g.gw}": g.xpts for g in pred.by_gw}


def components_rows(pred: PlayerPrediction) -> list[dict[str, Any]]:
    labels = {
        "appearance": "появление",
        "goals": "голы",
        "assists": "ассисты",
        "clean_sheet": "сухой матч",
        "goals_conceded": "пропущенные",
        "saves": "сейвы",
        "defcon": "DefCon",
        "bonus": "бонус",
        "cards": "карточки",
    }
    rows = []
    for g in pred.by_gw:
        row: dict[str, Any] = {"GW": g.gw, "xPts": num(g.xpts), "p(start)": num(g.p_start)}
        row["мин"] = round(g.exp_minutes)
        for k, v in g.components.items():
            row[labels.get(k, k)] = num(v)
        rows.append(row)
    return rows


def fixture_rows(pred: PlayerPrediction) -> list[dict[str, Any]]:
    rows = []
    for g in pred.by_gw:
        if not g.fixtures:
            rows.append({"GW": g.gw, "Соперник": "blank", "Где": "—", "Сложность": "—"})
        for f in g.fixtures:
            rows.append(
                {
                    "GW": g.gw,
                    "Соперник": f.opponent,
                    "Где": "дома" if f.is_home else "в гостях",
                    "Сложность": fsi_label(f.fsi),
                }
            )
    return rows


def risk_summary(risk: PlayerRisk) -> dict[str, str]:
    return {
        "Доступность": AVAILABILITY_RU.get(risk.availability, risk.availability),
        "p(start)": num(risk.start_probability),
        "Ожид. минуты": str(risk.expected_minutes),
        "Ротация": risk.rotation_risk,
        "Уверенность": num(risk.confidence),
        "Возвращение": f"GW{risk.return_gw}" if risk.return_gw else "—",
        "Источник": {
            "cached": f"сохранённый сигнал ({age_text(risk.age_h)})",
            "extracted": "извлечён сейчас",
            "unavailable": "нет сигнала",
        }.get(risk.origin, risk.origin),
        "FPL": status_text(risk.fpl_status, risk.fpl_chance)
        + (f" — {risk.fpl_news}" if risk.fpl_news else ""),
    }


def extraction_cost_text(risk: PlayerRisk, cost_usd: float) -> str:
    return (
        f"{risk.llm_calls} LLM-вызов(ов), {risk.prompt_tokens}+{risk.completion_tokens} токенов "
        f"≈ ${cost_usd:.4f}, {risk.latency_ms:.0f} мс, модель {risk.model or '—'}, режим {risk.mode} k={risk.k}"
    )


# ---------- чат / агент ----------


def tool_log_rows(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for t in state.get("tool_log") or []:
        if t.get("tool") in ("-", None):
            continue
        rows.append(
            {
                "Узел": t.get("node"),
                "Инструмент": t.get("tool"),
                "мс": t.get("latency_ms", 0),
                "OK": "✓" if t.get("ok", True) else "✗",
                "Заметка": (t.get("note") or "")[:160],
            }
        )
    return rows


def llm_call_rows(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "Узел": c.get("node"),
            "Назначение": c.get("purpose"),
            "Модель": c.get("model"),
            "Токены in/out": f"{c.get('prompt_tokens', 0)}/{c.get('completion_tokens', 0)}",
            "мс": c.get("latency_ms", 0),
            "$": f"{float(c.get('cost_usd') or 0):.4f}",
        }
        for c in state.get("llm_calls") or []
    ]


def validation_text(state: Mapping[str, Any]) -> str:
    v = state.get("validation") or {}
    if state.get("answer") is None:
        return "n/a (ответа ещё нет)"
    if not v:
        return "n/a (детерминированный ответ без LLM)"
    if v.get("passed"):
        return (
            f"пройдена: имён {v.get('checked_names', 0)}, чисел {v.get('checked_numbers', 0)}, "
            f"попыток {v.get('attempts')}"
        )
    return (
        f"НЕ пройдена после {v.get('attempts')} попыток: имена {v.get('unknown_names')}, "
        f"числа {v.get('unknown_numbers')}"
    )


def run_summary(state: Mapping[str, Any], *, elapsed_s: float | None = None) -> dict[str, str]:
    calls = state.get("llm_calls") or []
    p_tok = sum(int(c.get("prompt_tokens") or 0) for c in calls)
    c_tok = sum(int(c.get("completion_tokens") or 0) for c in calls)
    out = {
        "Интент": str(state.get("intent")),
        "Стратегия": str(state.get("strategy")),
        "GW": str(state.get("gw")),
        "Инструментов": str(len(tool_log_rows(state))),
        "LLM-вызовов": f"{len(calls)} (токены {p_tok}/{c_tok})",
        "Стоимость": f"≈ ${float(state.get('cost_estimate') or 0):.4f}",
        "Повторов поиска": str(state.get("retries") or 0),
        "Валидация": validation_text(state),
        "thread_id": str(state.get("thread_id")),
    }
    # версия промптов роутера / объяснителя (AGENT_PROMPT_VERSION процесса) — видно, что крутится
    versions = sorted({str(c.get("prompt_version")) for c in calls if c.get("purpose") in (
        "route", "explain") and c.get("prompt_version")})
    if versions:
        out["Промпты"] = ", ".join(versions)
    if state.get("standalone_query"):
        out["Вопрос с учётом истории"] = str(state["standalone_query"])
    if elapsed_s is not None:
        out["Время"] = f"{elapsed_s:.1f} с"
    return out


def pending_action_text(action: Mapping[str, Any]) -> str:
    kind = PENDING_KIND_RU.get(str(action.get("kind")), str(action.get("kind")))
    cost = action.get("cost") or 0
    cost_txt = f", цена {cost} очков" if cost else ""
    return f"Требует подтверждения: {kind} — {action.get('detail')}{cost_txt}."


# ---------- сайдбар ----------


# Сайдбар — панель управления, не документация: короткие строки, объяснения только в help (?).

MANAGER_HELP = "Число из адреса вашей команды: fantasy.premierleague.com/entry/ID/"
SIGNALS_HELP = (
    "Вердикты по игрокам (играет / травма / под вопросом) из новостей. Обновляются при вопросе "
    "в чате или на странице «Игрок»."
)
SIGNALS_STALE_H = 24.0  # разбор новостей старше суток помечаем «устарел»


def parse_manager_id(raw: str | None) -> int | None:
    """Текст поля «ID менеджера FPL» -> int; пусто / «0» -> 0 (без менеджера); не число -> None."""
    s = (raw or "").strip()
    if not s:
        return 0
    return int(s) if s.isdigit() else None


def manager_caption(manager_id: int | None, entry: Mapping[str, Any] | None) -> str:
    """Подтверждение применённого ID под полем: «<команда> · 56 очков · ранг 10 724 305».
    entry — карточка FPL API `entry/{id}` (team, points, rank) или None (404 / сеть)."""
    if not manager_id:
        return "Без менеджера"
    if not entry:
        return "Менеджер не найден в FPL"
    parts = [str(entry.get("team") or f"команда {manager_id}")]
    if entry.get("points") is not None:
        parts.append(plural(int(entry["points"]), "очко", "очка", "очков"))
    if entry.get("rank") is not None:
        parts.append(f"ранг {int(entry['rank']):,}".replace(",", " "))
    return " · ".join(parts)


def squad_source_text(
    ctx: GameweekContext | None, override_gw: int | None, *, has_manager: bool = True
) -> tuple[str, str, str | None]:
    """Короткая строка «Состав: …» для сайдбара: (текст, вид markdown/warning, help)."""
    if override_gw is not None:
        return f"Состав: со скриншота (GW{override_gw})", "markdown", None
    if ctx is not None and ctx.squad:
        help_text = (
            f"FPL отдаёт состав только за завершённый тур. Сделали трансферы на GW{ctx.gw}? "
            "Загрузите скриншот на странице «Мой состав»."
        )
        return f"Состав: из FPL за GW{ctx.squad_gw}", "markdown", help_text
    if not has_manager:
        return "Состав: нет — укажите ID менеджера или загрузите скриншот", "warning", None
    return "Состав: нет — загрузите скриншот", "warning", None


def freshness_lines(fr: Mapping[str, Any], now: datetime) -> dict[str, Any]:
    """Две строки свежести для сайдбара и флаги: news, signals («· устарел», если старше суток),
    signals_stale, xpts_missing (в БД нет прогноза на следующий тур)."""
    signal_at = fr.get("signal_at")
    stale = False
    if signal_at is not None:
        ts = signal_at if signal_at.tzinfo else signal_at.replace(tzinfo=UTC)
        stale = (now - ts).total_seconds() / 3600 > SIGNALS_STALE_H
    signals = f"Разбор новостей об игроках: {since_text(signal_at, now)}" + (
        " · устарел" if stale else ""
    )
    return {
        "news": f"Новости: {since_text(fr.get('news_at'), now)}",
        "signals": signals,
        "signals_stale": stale,
        "xpts_missing": not fr.get("xpts_rows"),
    }


# ---------- страница «О системе» ----------

# Узлы, перед которыми граф прерывается и ждёт человека (agent/graph.py: interrupt_before).
INTERRUPT_NODES: dict[str, str] = {
    "confirm_action": "платный трансфер / Wildcard",
    "resolve_clarification": "уточнение имени игрока",
}


def agent_graph_dot(graph: Any) -> str:
    """Graphviz DOT из langchain `Graph` (nodes/edges) — граф агента на странице «О системе».
    Оба узла-прерывания (`INTERRUPT_NODES`) подсвечены."""
    lines = [
        "digraph agent {",
        "rankdir=LR; nodesep=0.25; ranksep=0.4;",
        'node [shape=box, style="rounded,filled", fillcolor="#f2f0ff", fontname=Helvetica, fontsize=11];',
    ]
    for name in graph.nodes:
        label = name
        style = ""
        if name in ("__start__", "__end__"):
            style = ', shape=ellipse, fillcolor="#bfb6fc"'
        if name in INTERRUPT_NODES:
            label = f"{name}\\n(interrupt before: {INTERRUPT_NODES[name]})"
            style = ', fillcolor="#ffe8b3"'
        lines.append(f'"{name}" [label="{label}"{style}];')
    for e in graph.edges:
        attrs = " [style=dashed]" if getattr(e, "conditional", False) else ""
        lines.append(f'"{e.source}" -> "{e.target}"{attrs};')
    lines.append("}")
    return "\n".join(lines)


def pipeline_dot(n_tools: int, n_pages: int) -> str:
    """Graphviz DOT схемы потока данных: источники -> Postgres / RAG / xPts / MILP / vision ->
    LiveTools (число инструментов — из `agent.tools.TOOL_NAMES`) -> агент / UI / MCP."""
    tools_label = plural(n_tools, "инструмент", "инструмента", "инструментов")
    pages_label = plural(n_pages, "страница", "страницы", "страниц")
    return f"""
digraph pipeline {{
  rankdir=LR; node [shape=box, style="rounded,filled", fontname=Helvetica, fontsize=11];
  subgraph cluster_data {{ label="данные"; style=dashed;
    fpl [label="FPL API\\n(bootstrap, состав, календарь)", fillcolor="#e8f4ff"];
    rss [label="Новости: RSS / Google News\\n(ingest loop)", fillcolor="#e8f4ff"];
    shot [label="Скриншот Pick Team", fillcolor="#e8f4ff"];
  }}
  pg [label="Postgres + pgvector\\nnews_articles, news_chunks,\\nplayer_signals, xpts_predictions,\\nplan_snapshots", fillcolor="#fff3d6"];
  rag [label="RAG: hybrid retrieval\\n(dense + BM25 + rerank)\\n-> LLM: вердикт по игроку из новостей", fillcolor="#f2f0ff"];
  xpts [label="xPts v0 (core/xpts)\\nминуты x компоненты x сложность матчей", fillcolor="#e9f9e6"];
  milp [label="MILP HiGHS (core/optimizer)\\nлучший состав, трансферы, план, Wildcard", fillcolor="#e9f9e6"];
  vision [label="Vision LLM -> JSON\\n-> резолюция по bootstrap\\n-> правила FPL -> Squad", fillcolor="#f2f0ff"];
  tools [label="LiveTools: {tools_label}\\n(pydantic in/out)", fillcolor="#ffe8b3"];
  agent [label="LangGraph-агент\\nrouter -> signals loop -> compute\\n-> HITL -> explain -> validate", fillcolor="#f2f0ff"];
  ui [label="Streamlit UI\\n({pages_label})", fillcolor="#ffd6e0"];
  mcp [label="MCP server\\nfpl-intelligence", fillcolor="#ffd6e0"];
  rss -> pg; pg -> rag; rag -> pg [label="вердикт", fontsize=9];
  fpl -> xpts; pg -> xpts [label="новости -> минуты", fontsize=9]; xpts -> milp;
  shot -> vision; vision -> tools [label="squad_override", fontsize=9];
  fpl -> tools; xpts -> tools; milp -> tools; rag -> tools; pg -> tools;
  tools -> agent; tools -> ui; agent -> ui; tools -> mcp;
}}
"""


def parsed_squad_rows(parsed: Any) -> list[dict[str, Any]]:
    rows = []
    for p in parsed.players:
        role = (
            "C"
            if p.is_captain
            else "VC"
            if p.is_vice_captain
            else (
                f"B{p.bench_order}" if p.is_bench and p.bench_order else "B" if p.is_bench else "—"
            )
        )
        rows.append(
            {
                "Роль": role,
                "На экране": p.name_as_shown,
                "→ FPL": p.web_name or "?",
                "Поз": p.position or "?",
                "Клуб": p.team_short or "?",
                "Цена экран": money(p.price_as_shown),
                "Цена FPL": money(p.price),
                "Уверенность": num(p.match_confidence),
                "Метод": p.match_method,
                "Флаг": "!" if p.is_flagged else "",
            }
        )
    return rows


SQUAD_SIZE = 15


def parsed_result_text(parsed: Any) -> tuple[str, str]:
    """Короткий итог распознавания: («Распознано 15 игроков, капитан Saka», "success") /
    («Не сошлось: 14 игроков вместо 15», "error") / («Не сошлось: состав нарушает правила FPL»,
    "error") — детали (модель, токены, $) отдельно в `parsed_details`."""
    n = len(parsed.players)
    if parsed.is_valid:
        captain = next((p.label for p in parsed.players if p.is_captain), None)
        text = f"Распознано {plural(n, 'игрок', 'игрока', 'игроков')}"
        return (text + (f", капитан {captain}" if captain else ""), "success")
    if n != SQUAD_SIZE:
        return (
            f"Не сошлось: {plural(n, 'игрок', 'игрока', 'игроков')} вместо {SQUAD_SIZE}",
            "error",
        )
    return ("Не сошлось: состав нарушает правила FPL", "error")


def parsed_details(parsed: Any) -> dict[str, str]:
    """Подробности распознавания для свёрнутого блока: экран, сопоставлено, банк, FT, тур,
    модель, токены, стоимость, время."""
    ft = parsed.free_transfers
    out = {
        "Экран": str(parsed.screen_type),
        "Сопоставлено с FPL": f"{parsed.resolved_count}/{len(parsed.players)}",
        "Банк": money(parsed.bank),
        "Бесплатных трансферов": "—" if ft is None else str(ft),
        "Тур": f"GW{parsed.gw}" if parsed.gw else "—",
    }
    if parsed.usage:
        u = parsed.usage
        out["Модель"] = f"{u.model} (detail={u.detail})"
        out["Токены"] = f"{u.prompt_tokens} + {u.completion_tokens}"
        out["Стоимость"] = f"≈ ${u.cost_usd:.4f}"
        out["Время"] = f"{u.latency_ms:.0f} мс"
    return out


def parsed_issue_rows(parsed: Any) -> list[dict[str, Any]]:
    return [
        {
            "Уровень": "БЛОК" if i.blocking else "предупреждение",
            "Код": i.code,
            "Сообщение": i.message,
        }
        for i in parsed.issues
    ]
