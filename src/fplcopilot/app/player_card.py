"""Чистые помощники карточки игрока (страница «Игрок»; референс — smartplayfpl.com/players/N).

Без импорта streamlit — unit-тесты в tests/test_player_card.py. Названия секций и метрик
скопированы со smartplay (английские), подсказки (?) и строки сравнения — русские.

- перцентили: доля игроков той же позиции с игровым временем (>= MIN_POOL_MINUTES за сезон),
  у которых значение строго хуже; сам игрок входит в пул, поэтому максимум — 99 %;
- FRR (fixture run rank): ранг клуба по средней сложности матчей (FSI 1–5) на RUN_GWS туров,
  1 — самый лёгкий календарь; тур без матча считается как BLANK_FSI;
- похожие игроки: конкуренты за минуты (тот же клуб и позиция) и «редкая альтернатива»
  (та же позиция, владение < DIFFERENTIAL_OWNERSHIP %, цена не выше, календарь не тяжелее) —
  top-N по xPts за SIMILAR_GWS туров, только bootstrap + прогноз, без LLM.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fplcopilot.agent.tools import PlayerPrediction, PlayerRisk
from fplcopilot.app import format as fmt
from fplcopilot.app import theme
from fplcopilot.config import settings
from fplcopilot.data.schemas import Bootstrap, Player, PlayerGWHistory, Position

MIN_POOL_MINUTES = 90  # порог минут за сезон для пула перцентилей
FULL_MATCH_MINUTES = 60  # «полный» матч для Mins%
DIFFERENTIAL_OWNERSHIP = 10.0  # владение, ниже которого игрок — редкая альтернатива
RUN_GWS = 5  # горизонт FRR
SIMILAR_GWS = 3  # горизонт xPts / сложности для похожих игроков
FIXTURES_SHOWN = 6  # «Fixtures (next 6 GWs)»
BLANK_FSI = 5.0  # тур без матча = максимальная сложность

BLANK_CLASS = theme.fsi_class(0)  # тур без матча — приглушённый чип

POSITION_GENITIVE: dict[str, str] = {
    "GKP": "вратарей",
    "DEF": "защитников",
    "MID": "полузащитников",
    "FWD": "нападающих",
}


# ---------- примитивы ----------


def percentile(
    pool: Iterable[float | None], x: float | None, *, higher_is_better: bool = True
) -> int | None:
    """Процентиль x в пуле: доля значений строго хуже x, 0–100. None — нет данных или пула."""
    vals = [float(v) for v in pool if v is not None]
    if x is None or not vals:
        return None
    worse = sum(1 for v in vals if (v < x if higher_is_better else v > x))
    return round(100 * worse / len(vals))


def position_pool(
    bs: Bootstrap, position: Position, *, min_minutes: int = MIN_POOL_MINUTES
) -> list[Player]:
    """Игроки той же позиции с игровым временем — база для перцентилей."""
    return [p for p in bs.elements if p.element_type == position and p.minutes >= min_minutes]


def full_match_share(
    rows: Iterable[PlayerGWHistory], *, full_minutes: int = FULL_MATCH_MINUTES
) -> dict[int, float]:
    """player_id -> доля сыгранных матчей (минуты > 0) с >= full_minutes минут."""
    played: dict[int, int] = {}
    full: dict[int, int] = {}
    for r in rows:
        if r.minutes <= 0:
            continue
        played[r.element] = played.get(r.element, 0) + 1
        if r.minutes >= full_minutes:
            full[r.element] = full.get(r.element, 0) + 1
    return {pid: full.get(pid, 0) / n for pid, n in played.items()}


def signed_int(n: int | None) -> str:
    """+15 419 / −51 160 / 0 — разряды пробелом, минус типографский."""
    if n is None:
        return "—"
    n = int(n)
    if n == 0:
        return "0"
    body = f"{abs(n):,}".replace(",", " ")
    return ("+" if n > 0 else "−") + body


def money_delta(delta: float) -> str:
    """Разница в цене: «£2.2m дешевле» / «£0.5m дороже» / «та же цена»."""
    if abs(delta) < 0.05:
        return "та же цена"
    return f"£{abs(delta):.1f}m {'дешевле' if delta < 0 else 'дороже'}"


# ---------- шапка и ключевые числа ----------


@dataclass(frozen=True)
class Header:
    meta: str  # «Player report · £8.0 · DEF · ARS»
    name: str
    club: str
    status_tone: str
    status_text: str
    news: str


def status_badge(status: str, chance: int | None = None) -> tuple[str, str]:
    """(тон, русский текст) статуса FPL; тон — цвет бейджа Streamlit (green / orange / red /
    gray), см. `badge_md`."""
    if status == "a":
        return "green", "Играет"
    if status == "d":
        return "orange", f"Под вопросом {chance} %" if chance is not None else "Под вопросом"
    if status == "i":
        return "red", "Травма"
    if status == "s":
        return "red", "Дисквалификация"
    if status == "u":
        return "red", "Недоступен"
    if status == "n":
        return "gray", "Не в заявке"
    return "gray", fmt.status_text(status, chance)


def badge_md(tone: str, text: str) -> str:
    """Markdown-бейдж Streamlit «:green-badge[Играет]» (квадратные скобки в тексте экранируются)."""
    safe = text.replace("[", "(").replace("]", ")")
    return f":{tone}-badge[{safe}]"


def header(player: Player, bs: Bootstrap) -> Header:
    tone, text = status_badge(player.status, player.chance_of_playing_next_round)
    team = bs.team(player.team)
    return Header(
        meta=f"Player report · £{player.price:.1f} · {player.position.short} · {team.short_name}",
        name=player.full_name,
        club=team.name,
        status_tone=tone,
        status_text=text,
        news=player.news or "",
    )


@dataclass(frozen=True)
class KeyNumber:
    label: str
    value: str
    help: str
    delta: str | None = None


def xpts_over(pred: PlayerPrediction | None, n_gws: int = SIMILAR_GWS) -> float:
    if pred is None:
        return 0.0
    return round(sum(g.xpts for g in pred.by_gw[:n_gws]), 2)


def gw_difficulty(fsis: Sequence[int]) -> float:
    """Сложность тура для команды: среднее FSI её матчей; без матча — BLANK_FSI."""
    return sum(fsis) / len(fsis) if fsis else BLANK_FSI


def fsi_over(pred: PlayerPrediction | None, n_gws: int = SIMILAR_GWS) -> float:
    """Сумма сложности календаря за ближайшие n_gws туров (blank = BLANK_FSI)."""
    if pred is None:
        return BLANK_FSI * n_gws
    gws = pred.by_gw[:n_gws]
    total = sum(gw_difficulty([f.fsi for f in g.fixtures]) for g in gws)
    total += BLANK_FSI * max(0, n_gws - len(gws))  # туров в прогнозе меньше горизонта
    return round(total, 1)


def key_numbers(player: Player, pred: PlayerPrediction | None) -> list[KeyNumber]:
    """Строка ключевых чисел как у smartplay: xPts, Points, Form, 3GW xPts, Own%, Net Tx."""
    nxt = pred.by_gw[0] if pred and pred.by_gw else None
    net = int(player.transfers_in_event) - int(player.transfers_out_event)
    return [
        KeyNumber(
            "xPts",
            fmt.num(nxt.xpts) if nxt else "—",
            (
                f"Прогноз очков нашей модели на GW{nxt.gw}; ± — разброс (одно стандартное "
                "отклонение)."
                if nxt
                else "Прогноза на ближайший тур нет."
            ),
            delta=f"±{nxt.sd:.1f} sd" if nxt else None,
        ),
        KeyNumber("Points", str(player.total_points), "Очки FPL за сезон."),
        KeyNumber(
            "Form",
            fmt.num(player.form, 1),
            "Форма по FPL: средние очки за матчи последних 30 дней.",
        ),
        KeyNumber(
            "3GW xPts",
            fmt.num(xpts_over(pred, SIMILAR_GWS)) if pred and pred.by_gw else "—",
            f"Сумма прогноза очков на ближайшие {SIMILAR_GWS} тура.",
        ),
        KeyNumber(
            "Own%",
            f"{float(player.selected_by_percent or 0):.1f}%",
            "Владение: доля менеджеров FPL, у которых игрок в составе.",
        ),
        KeyNumber(
            "Net Tx",
            signed_int(net),
            "Чистые трансферы за текущий тур: сколько менеджеров купили минус сколько продали.",
        ),
    ]


# ---------- секции статистики с перцентилями ----------


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    text: str  # отформатированное значение
    pct: int | None  # процентиль 0–100 среди игроков позиции с игровым временем; None — нет
    help: str
    model: str | None = None  # как метрика влияет на прогноз xPts; None — только справка


@dataclass(frozen=True)
class Section:
    key: str
    title: str
    metrics: tuple[Metric, ...]
    collapsed: bool = False  # Defence & Reliability для MID/FWD — свернуть


@dataclass(frozen=True)
class _Spec:
    key: str
    label: str
    getter: Callable[[Player], float | None]
    fmt: Callable[[float | None], str]
    help: str
    higher_is_better: bool = True
    ranked: bool = True  # считать ли перцентиль
    model: str | None = None  # входит ли в прогноз xPts и как (см. MODEL_*)


def _int(x: float | None) -> str:
    return "—" if x is None else f"{x:.0f}"


def _f1(x: float | None) -> str:
    return "—" if x is None else f"{x:.1f}"


def _f2(x: float | None) -> str:
    return "—" if x is None else f"{x:.2f}"


def _pct0(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.0f}%"


def _avg_minutes(p: Player) -> float | None:
    return p.minutes / p.starts if p.starts else None


def _mins_share_fallback(p: Player) -> float | None:
    """Без истории туров: доля возможных минут = минуты ÷ (матчи × 90), не больше 100 %."""
    return min(1.0, p.minutes / (90 * p.starts)) if p.starts else None


# Что из метрик карточки реально входит в прогноз xPts (core/xpts.py, core/minutes.py):
# ставки per-90 по истории туров (xG, xA, BPS, сейвы, CBIT/отборы/подборы, жёлтые) со сжатием к
# приору и слабым blend с Understat, модель минут (история минут и выходов в старте + статус FPL
# и новости), очередь пенальти / прямых штрафных, сила соперника. НЕ входят: фактические голы и
# ассисты, очки, форма, Creativity / Threat / ICT, сухие матчи и xGC игрока (сухой матч — по силе
# соперника), владение, трансферы, удары, npxG.
MODEL_MINUTES = "модель минут: история минут и выходов в старте, статус FPL и новости"
MODEL_XG = "компонент «голы»: xG за 90 минут по истории туров (сжатие к приору, blend с Understat)"
MODEL_XA = (
    "компонент «ассисты»: xA за 90 минут по истории туров (сжатие к приору, blend с Understat)"
)
MODEL_DEFCON = "компонент «защитные действия»: CBIT (+ подборы у полузащиты) за 90 минут"
MODEL_SAVES = "компонент «сейвы»: сейвы за 90 минут с поправкой на силу атаки соперника"


def _xgi(p: Player) -> float | None:
    if p.expected_goal_involvements is not None:
        return p.expected_goal_involvements
    if p.expected_goals is None and p.expected_assists is None:
        return None
    return (p.expected_goals or 0.0) + (p.expected_assists or 0.0)


def _xg90(p: Player) -> float | None:
    if not p.minutes or p.expected_goals is None:
        return None
    return p.expected_goals * 90.0 / p.minutes


def _sections(
    mins_share: Callable[[Player], float | None], history_based: bool, position: Position
) -> list[tuple[str, str, list[_Spec]]]:
    mins_help = (
        f"Доля сыгранных матчей, в которых игрок провёл не меньше {FULL_MATCH_MINUTES} минут "
        "(по истории туров)."
        if history_based
        else "Доля возможных минут: минуты ÷ (матчи в старте × 90)."
    )
    defence: list[_Spec] = [
        _Spec(
            "cs",
            "Clean S.",
            lambda p: float(p.clean_sheets),
            _int,
            "Сухие матчи: команда не пропустила, а игрок провёл не меньше 60 минут.",
        ),
        _Spec(
            "xgc",
            "xGC",
            lambda p: p.expected_goals_conceded,
            _f1,
            "Ожидаемые пропущенные голы за сезон, пока игрок на поле. Меньше — лучше.",
            higher_is_better=False,
        ),
    ]
    if position == Position.GKP:
        defence.append(
            _Spec(
                "saves",
                "Saves",
                lambda p: float(p.saves),
                _int,
                "Сейвы за сезон.",
                model=MODEL_SAVES,
            )
        )
    else:
        defence.append(
            _Spec(
                "defcon",
                "Def. contr.",
                lambda p: p.defensive_contribution,
                _int,
                "Защитные действия FPL за сезон: выносы, блоки, перехваты, отборы (у полузащиты "
                "— и подборы).",
                model=MODEL_DEFCON,
            )
        )
    return [
        (
            "overview",
            "Overview",
            [
                _Spec("pts", "Pts", lambda p: float(p.total_points), _int, "Очки FPL за сезон."),
                _Spec(
                    "bonus",
                    "Bonus",
                    lambda p: float(p.bonus),
                    _int,
                    "Бонусные очки за сезон: 3/2/1 лучшим в матче по системе BPS.",
                ),
                _Spec(
                    "ppg",
                    "Pts/Game",
                    lambda p: p.points_per_game,
                    _f1,
                    "Средние очки за сыгранный матч (по данным FPL).",
                ),
                _Spec(
                    "form",
                    "Form",
                    lambda p: p.form,
                    _f1,
                    "Форма по FPL: средние очки за матчи последних 30 дней.",
                ),
            ],
        ),
        (
            "minutes",
            "Minutes",
            [
                _Spec(
                    "mins",
                    "Mins",
                    lambda p: float(p.minutes),
                    _int,
                    "Минуты на поле за сезон.",
                    model=MODEL_MINUTES,
                ),
                _Spec(
                    "matches",
                    "Matches",
                    lambda p: float(p.starts),
                    _int,
                    "Матчей в стартовом составе за сезон.",
                    model=MODEL_MINUTES,
                ),
                _Spec(
                    "avg_mins",
                    "Avg Mins",
                    _avg_minutes,
                    _int,
                    "Средние минуты за матч в старте: минуты ÷ матчи.",
                    model=MODEL_MINUTES,
                ),
                _Spec("mins_share", "Mins%", mins_share, _pct0, mins_help, model=MODEL_MINUTES),
            ],
        ),
        (
            "attack",
            "Attacking Output",
            [
                _Spec("goals", "Goals", lambda p: float(p.goals_scored), _int, "Голы за сезон."),
                _Spec("assists", "Assists", lambda p: float(p.assists), _int, "Ассисты за сезон."),
                _Spec(
                    "xg",
                    "xG",
                    lambda p: p.expected_goals,
                    _f1,
                    "Ожидаемые голы за сезон — по качеству ударов игрока.",
                    model=MODEL_XG,
                ),
                _Spec(
                    "xa",
                    "xA",
                    lambda p: p.expected_assists,
                    _f1,
                    "Ожидаемые ассисты за сезон — по качеству созданных моментов.",
                    model=MODEL_XA,
                ),
                _Spec(
                    "xgi",
                    "xGI",
                    _xgi,
                    _f1,
                    "Ожидаемое участие в голах: xG + xA.",
                    model="компоненты «голы» и «ассисты» (xG и xA за 90 минут)",
                ),
            ],
        ),
        (
            "advanced",
            "Advanced Attacking",
            [
                _Spec(
                    "creativity",
                    "Creativity",
                    lambda p: p.creativity,
                    _f1,
                    "Индекс FPL Creativity — созидание: передачи под удар, навесы, ключевые пасы.",
                ),
                _Spec(
                    "threat",
                    "Threat",
                    lambda p: p.threat,
                    _f1,
                    "Индекс FPL Threat — угроза воротам: удары и позиции для гола.",
                ),
                _Spec(
                    "xg90",
                    "xG/90",
                    _xg90,
                    _f2,
                    "Ожидаемые голы на 90 минут за сезон (FPL).",
                    model=MODEL_XG,
                ),
            ],
        ),
        ("defence", "Defence & Reliability", defence),
        (
            "ownership",
            "Ownership & Transfers",
            [
                _Spec(
                    "tsb",
                    "TSB%",
                    lambda p: float(p.selected_by_percent or 0),
                    lambda x: "—" if x is None else f"{x:.1f}%",
                    "Владение (team selected by): доля менеджеров FPL, у которых игрок в составе.",
                    ranked=False,
                ),
                _Spec(
                    "tin",
                    "Transf. In",
                    lambda p: float(p.transfers_in_event),
                    lambda x: signed_int(int(x or 0)),
                    "Сколько менеджеров купили игрока в текущем туре.",
                    ranked=False,
                ),
                _Spec(
                    "tout",
                    "Transf. Out",
                    lambda p: -float(p.transfers_out_event),
                    lambda x: signed_int(int(x or 0)),
                    "Сколько менеджеров продали игрока в текущем туре.",
                    ranked=False,
                ),
                _Spec(
                    "tnet",
                    "Net Transf.",
                    lambda p: float(p.transfers_in_event - p.transfers_out_event),
                    lambda x: signed_int(int(x or 0)),
                    "Купили минус продали за текущий тур.",
                    ranked=False,
                ),
            ],
        ),
    ]


def _understat_specs() -> list[_Spec]:
    """Метрики Understat для «Advanced Attacking» (только у сопоставленных игроков)."""
    return [
        _Spec(
            "u_shots",
            "Shots",
            lambda row: float(row.shots),
            _int,
            "Удары за сезон (Understat). В прогноз не входят — только контекст.",
        ),
        _Spec(
            "u_npxg",
            "npxG",
            lambda row: float(row.npxg),
            _f2,
            "Ожидаемые голы без пенальти за сезон (Understat).",
        ),
    ]


def percentile_help(pct: int | None, position: Position, *, higher_is_better: bool) -> str:
    """Фраза о процентиле: «88 % — лучше 88 % защитников с игровым временем (≥ 90 мин)»."""
    if pct is None:
        return ""
    pos = POSITION_GENITIVE.get(position.short, "игроков позиции")
    word = "лучше" if higher_is_better else "меньше, чем у"
    return f"{pct} % — {word} {pct} % {pos} с игровым временем (≥ {MIN_POOL_MINUTES} мин за сезон)."


def stat_sections(
    player: Player,
    bs: Bootstrap,
    *,
    mins_share: Mapping[int, float] | None = None,
    min_minutes: int = MIN_POOL_MINUTES,
    ext_row: Callable[[int], Any | None] | None = None,
) -> list[Section]:
    """Секции карточки: значение + процентиль среди игроков той же позиции с игровым временем.

    mins_share — player_id -> доля матчей с >= 60 мин из player_gw_history; None — без истории,
    тогда Mins% = минуты ÷ (матчи × 90) из bootstrap. ext_row — player_id -> строка Understat
    (или None): Shots и npxG в «Advanced Attacking» с перцентилем среди сопоставленных."""
    pool = position_pool(bs, player.position, min_minutes=min_minutes)
    if mins_share is not None:
        share: Callable[[Player], float | None] = lambda p: mins_share.get(p.id)
    else:
        share = _mins_share_fallback
    out: list[Section] = []

    def metric(s: _Spec, value: float | None, pool_values: Iterable[float | None]) -> Metric:
        pct = None
        if s.ranked:
            pct = percentile(pool_values, value, higher_is_better=s.higher_is_better)
        help_text = s.help
        tail = percentile_help(pct, player.position, higher_is_better=s.higher_is_better)
        if tail:
            help_text = f"{help_text} {tail}"
        return Metric(s.key, s.label, s.fmt(value), pct, help_text, model=s.model)

    for key, title, specs in _sections(share, mins_share is not None, player.position):
        metrics = [metric(s, s.getter(player), (s.getter(p) for p in pool)) for s in specs]
        if key == "advanced" and ext_row is not None:
            mine = ext_row(player.id)
            if mine is not None:  # нет Understat у игрока — метрик не рисуем
                rows = [r for r in (ext_row(p.id) for p in pool) if r is not None]
                metrics += [
                    metric(s, s.getter(mine), (s.getter(r) for r in rows))
                    for s in _understat_specs()
                ]
        collapsed = key == "defence" and player.position in (Position.MID, Position.FWD)
        out.append(Section(key, title, tuple(metrics), collapsed))
    return out


def percentile_color(pct: int) -> str:
    """Цвет полоски перцентиля — токен темы (светлая / тёмная) с запасным значением."""
    if pct >= 70:
        return "var(--fpl-good, #16804A)"
    if pct >= 40:
        return "var(--fpl-warn, #A86A12)"
    return "var(--fpl-bad, #C8373E)"


def percentile_bar_html(pct: int | None) -> str:
    """Тонкая полоска + «88 %» под метрикой; пусто, если процентиля нет."""
    if pct is None:
        return ""
    width = max(0, min(100, pct))
    return (
        '<div style="display:flex;align-items:center;gap:8px;margin:-6px 2px 14px">'
        '<div style="flex:1;height:4px;background:var(--fpl-line-strong, rgba(128,128,128,0.25));'
        'border-radius:2px;overflow:hidden">'
        f'<div style="width:{width}%;height:100%;background:{percentile_color(pct)}"></div></div>'
        '<span style="font-size:11.5px;color:var(--fpl-muted, inherit);min-width:2.6em;'
        f'text-align:right;font-variant-numeric:tabular-nums">{pct} %</span></div>'
    )


# ---------- календарь ----------


@dataclass(frozen=True)
class FixtureCell:
    gw: int
    label: str  # «LEE (H)», «LEE (H) · NFO (A)» или «—»
    fsi: float | None  # средняя сложность тура; None — blank
    css: str  # класс фона theme.fsi_class (blank — fpl-fsi-0)
    title: str  # подпись при наведении, без жаргона


def fixture_cells(pred: PlayerPrediction | None, n: int = FIXTURES_SHOWN) -> list[FixtureCell]:
    cells: list[FixtureCell] = []
    for g in (pred.by_gw if pred else [])[:n]:
        if not g.fixtures:
            cells.append(FixtureCell(g.gw, "—", None, BLANK_CLASS, "Нет матча в этом туре"))
            continue
        label = " · ".join(f"{f.opponent} ({'H' if f.is_home else 'A'})" for f in g.fixtures)
        fsi = gw_difficulty([f.fsi for f in g.fixtures])
        level = max(1, min(5, round(fsi)))
        title = f"Сложность {level} из 5 — {fmt.FSI_LABELS.get(level, '?')} соперник"
        cells.append(FixtureCell(g.gw, label, fsi, theme.fsi_class(level), title))
    return cells


FIXTURE_CSS = f"""
<style>
{fmt.TIP_CSS_RULES}
{theme.FSI_CSS_RULES}
.fpl-fixture {{
  border-radius: 10px; padding: 10px 6px; text-align: center; min-height: 62px;
  box-sizing: border-box;
}}
.fpl-fixture .gw {{ font-size: 11px; font-weight: 500; opacity: 0.8; letter-spacing: 0.02em; }}
.fpl-fixture .opp {{ font-size: 15px; font-weight: 700; line-height: 1.35; }}
.fpl-fx-legend {{
  display: flex; align-items: center; flex-wrap: wrap; gap: 4px; margin-top: 8px;
  font-size: 12px; color: var(--fpl-muted, inherit);
}}
.fpl-fx-legend .sw {{ display: inline-block; width: 14px; height: 14px; border-radius: 4px; }}
</style>
"""


def fixture_cell_html(cell: FixtureCell) -> str:
    inner = (
        f'<div class="fpl-fixture {cell.css}"><div class="gw">GW{cell.gw}</div>'
        f'<div class="opp">{fmt._esc(cell.label)}</div></div>'
    )
    return FIXTURE_CSS + fmt.tip(cell.title, mark=inner, css_class="tip-block", escape_mark=False)


def fixture_legend_html() -> str:
    """Легенда: (H) — дома, (A) — в гостях; квадратики цветов 1–5."""
    chips = "".join(f'<span class="sw {theme.fsi_class(i)}"></span>' for i in range(1, 6))
    return (
        FIXTURE_CSS + '<div class="fpl-fx-legend">(H) — дома, (A) — в гостях · сложность '
        f"матча: лёгкий {chips} тяжёлый</div>"
    )


# ---------- FRR: ранг календаря команды ----------


@dataclass(frozen=True)
class RunRank:
    rank: int  # 1 = самый лёгкий календарь
    n_teams: int
    avg_fsi: float


def team_fsi_by_gw(
    preds: Mapping[int, PlayerPrediction], bs: Bootstrap, gws: Sequence[int]
) -> dict[int, list[float]]:
    """team_id -> сложность по турам gws (среднее FSI матчей команды, blank = BLANK_FSI).
    Достаточно одного игрока на команду; расхождений между игроками одного клуба нет."""
    per_team: dict[int, dict[int, float]] = {}
    for pid, pred in preds.items():
        try:
            team_id = bs.player(pid).team
        except KeyError:
            continue
        got = per_team.setdefault(team_id, {})
        for g in pred.by_gw:
            if g.gw in gws and g.gw not in got:
                got[g.gw] = gw_difficulty([f.fsi for f in g.fixtures])
    return {t: [by_gw.get(g, BLANK_FSI) for g in gws] for t, by_gw in per_team.items() if by_gw}


def team_fixture_run_rank(team_fsi: Mapping[int, Sequence[float]], team_id: int) -> RunRank | None:
    """Ранг команды по среднему FSI на ближайшие туры: 1 — самый лёгкий; равные средние
    делят ранг (1 + число команд со строго меньшим средним). None — команды нет в данных."""
    avg = {t: sum(v) / len(v) for t, v in team_fsi.items() if v}
    if team_id not in avg:
        return None
    mine = avg[team_id]
    rank = 1 + sum(1 for v in avg.values() if v < mine - 1e-9)
    return RunRank(rank=rank, n_teams=len(avg), avg_fsi=round(mine, 2))


# ---------- похожие игроки ----------


@dataclass(frozen=True)
class SimilarPlayer:
    id: int
    name: str  # web_name
    club: str
    position: str
    price: float
    status_tone: str
    status_text: str
    xpts: float  # за SIMILAR_GWS туров
    ownership: float
    fsi: float  # сумма сложности за SIMILAR_GWS туров
    compare: str  # строка сравнения с текущим игроком (русская)


def _similar(
    p: Player, bs: Bootstrap, pred: PlayerPrediction | None, compare: str, n_gws: int
) -> SimilarPlayer:
    tone, text = status_badge(p.status, p.chance_of_playing_next_round)
    return SimilarPlayer(
        id=p.id,
        name=p.web_name,
        club=bs.team(p.team).short_name,
        position=p.position.short,
        price=p.price,
        status_tone=tone,
        status_text=text,
        xpts=xpts_over(pred, n_gws),
        ownership=float(p.selected_by_percent or 0),
        fsi=fsi_over(pred, n_gws),
        compare=compare,
    )


def competitors_for_minutes(
    player: Player,
    bs: Bootstrap,
    preds: Mapping[int, PlayerPrediction],
    *,
    n_gws: int = SIMILAR_GWS,
    limit: int = 3,
) -> list[SimilarPlayer]:
    """Тот же клуб, та же позиция; top-limit по xPts за n_gws туров.
    Строка: «£2.2m дешевле · −1.8 xPts за 3 тура»."""
    mine = xpts_over(preds.get(player.id), n_gws)
    cands = [
        p
        for p in bs.elements
        if p.id != player.id and p.team == player.team and p.element_type == player.element_type
    ]
    cands.sort(key=lambda p: (-xpts_over(preds.get(p.id), n_gws), p.id))
    out = []
    for p in cands[:limit]:
        pred = preds.get(p.id)
        delta = xpts_over(pred, n_gws) - mine
        compare = (
            f"{money_delta(p.price - player.price)} · {fmt.signed(delta, 1)} xPts за "
            f"{fmt.plural(n_gws, 'тур', 'тура', 'туров')}"
        )
        out.append(_similar(p, bs, pred, compare, n_gws))
    return out


def differential_bridge(
    player: Player,
    bs: Bootstrap,
    preds: Mapping[int, PlayerPrediction],
    *,
    n_gws: int = SIMILAR_GWS,
    limit: int = 3,
    max_ownership: float = DIFFERENTIAL_OWNERSHIP,
) -> list[SimilarPlayer]:
    """Та же позиция, владение < max_ownership %, цена не выше, сумма сложности матчей за n_gws
    туров не больше, чем у текущего; top-limit по xPts. Строка: «2.2 % владение · календарь
    легче: 3.1 против 6.4»."""
    my_fsi = fsi_over(preds.get(player.id), n_gws)
    cands = []
    for p in bs.elements:
        if p.id == player.id or p.element_type != player.element_type:
            continue
        if float(p.selected_by_percent or 0) >= max_ownership or p.price > player.price:
            continue
        if fsi_over(preds.get(p.id), n_gws) > my_fsi:
            continue
        cands.append(p)
    cands.sort(key=lambda p: (-xpts_over(preds.get(p.id), n_gws), p.id))
    out = []
    for p in cands[:limit]:
        pred = preds.get(p.id)
        fsi = fsi_over(pred, n_gws)
        run = (
            f"календарь легче: {fsi:.1f} против {my_fsi:.1f}"
            if fsi < my_fsi
            else f"календарь такой же: {fsi:.1f}"
        )
        compare = f"{float(p.selected_by_percent or 0):.1f} % владение · {run}"
        out.append(_similar(p, bs, pred, compare, n_gws))
    return out


# ---------- новости об игроке ----------

NEWS_ORIGIN_RU: dict[str, str] = {
    "cached": "сохранённый разбор",
    "extracted": "разобрано сейчас",
    "unavailable": "разбора нет",
}


def _order_text(order: int | None) -> str:
    return "—" if not order else f"№{order}"


def setpiece_section(player: Player) -> Section:
    """Блок «Set pieces»: пенальти / штрафные / угловые / сейвы из bootstrap FPL."""
    saves_text = str(player.saves) if player.position == Position.GKP else "—"
    metrics = (
        Metric(
            "pen_order",
            "Pens",
            _order_text(player.penalties_order),
            None,
            "Очередь пенальти по FPL. №1 — бьёт пенальти команды; слабо повышает xPts "
            f"(+{settings.xpts_setpiece_weight:.2f} за полный матч, XPTS_SETPIECE_WEIGHT).",
            model="бонус к xG за 90 минут первому пенальтисту",
        ),
        Metric(
            "dfk_order",
            "Direct FK",
            _order_text(player.direct_freekicks_order),
            None,
            "Очередь прямых штрафных. №1 — штатник; ещё меньший бонус к xPts "
            "(XPTS_FREEKICK_WEIGHT).",
            model="небольшой бонус к xPts штатному исполнителю прямых штрафных",
        ),
        Metric(
            "corners_order",
            "Corners",
            _order_text(player.corners_and_indirect_freekicks_order),
            None,
            "Очередь угловых и непрямых штрафных. На xPts не влияет — только справка.",
        ),
        Metric(
            "saves",
            "Saves",
            saves_text,
            None,
            "Сейвы за сезон (FPL). У вратаря уже входят в компонент saves модели xPts; "
            "отдельный вес XPTS_SAVES_WEIGHT по умолчанию 0, чтобы не двоить.",
            model=MODEL_SAVES if player.position == Position.GKP else None,
        ),
    )
    return Section("setpieces", "Set pieces", metrics)


def understat_section(player: Player, row: Any | None) -> Section | None:
    """Блок «Understat», если имя сопоставилось. row — UnderstatPlayer или None."""
    if row is None:
        return None
    xg90 = "—" if row.xg90 is None else f"{row.xg90:.2f}"
    xa90 = "—" if row.xa90 is None else f"{row.xa90:.2f}"
    shots = str(row.shots)
    shots90 = "—" if row.shots90 is None else f"{row.shots90:.2f}"
    metrics = (
        Metric(
            "u_xg90",
            "xG90",
            xg90,
            None,
            f"Ожидаемые голы на 90 минут по Understat. В модели — "
            f"{settings.xpts_understat_blend:.0%} этого числа и "
            f"{1.0 - settings.xpts_understat_blend:.0%} нашего xG90, сдвиг не больше "
            f"±{settings.xpts_understat_shift_cap:.0%} (XPTS_UNDERSTAT_BLEND).",
            model="слабый blend с нашим xG за 90 минут",
        ),
        Metric(
            "u_xa90",
            "xA90",
            xa90,
            None,
            "Ожидаемые ассисты на 90 минут по Understat. Тот же слабый blend, что у xG90.",
            model="слабый blend с нашим xA за 90 минут",
        ),
        Metric(
            "u_shots",
            "Shots",
            shots,
            None,
            "Удары за сезон (Understat). На xPts не умножаем — только прозрачный контекст.",
        ),
        Metric(
            "u_shots90",
            "Shots/90",
            shots90,
            None,
            "Удары на 90 минут по Understat.",
        ),
    )
    return Section("understat", "Understat", metrics)


_POINTS_ORIGIN_HELP = (
    "Очки за последние туры по правилам FPL: откуда набрал и похоже ли это на удачу."
)


def points_origin_section(data: Any) -> Section | None:
    """Блок «Откуда очки за N туров»: строка категорий + метка. Нет окна — None."""
    if data is None:
        return None
    payload = data.as_dict() if hasattr(data, "as_dict") else dict(data)
    n = int(payload.get("n_rounds") or 0)
    cats = dict(payload.get("categories") or {})
    if n <= 0 and not cats:
        return None
    from fplcopilot.core.points_form import LABEL_RU, compact_line

    line = compact_line(cats)
    if not line:
        line = "нет очков"
    label = payload.get("label_ru") or LABEL_RU.get(str(payload.get("label") or ""), "—")
    if not payload.get("label"):
        label = "—"
    tours = n if n else int(payload.get("last_n") or 3)
    total = payload.get("total_points")
    total_text = "—" if total is None else str(int(total))
    metrics = (
        Metric("origin_line", "Раскладка", line, None, _POINTS_ORIGIN_HELP),
        Metric("origin_label", "Метка", label, None, _POINTS_ORIGIN_HELP),
        Metric("origin_total", "Всего", total_text, None, _POINTS_ORIGIN_HELP),
    )
    title = f"Откуда очки за {tours} тура" if tours != 1 else "Откуда очки за 1 тур"
    return Section("points_origin", title, metrics)


def news_verdict_rows(risk: PlayerRisk) -> list[dict[str, str]]:
    """Таблица «Поле / Значение» вердикта по новостям — без слова «сигнал»."""
    origin = NEWS_ORIGIN_RU.get(risk.origin, risk.origin)
    if risk.origin == "cached":
        origin = f"{origin} ({fmt.age_text(risk.age_h)})"
    rows = {
        "Доступность": fmt.AVAILABILITY_RU.get(risk.availability, risk.availability),
        "Вероятность выхода в старте": fmt.num(risk.start_probability),
        "Ожидаемые минуты": str(risk.expected_minutes),
        "Риск ротации": fmt.ROTATION_RU.get(risk.rotation_risk, risk.rotation_risk),
        "Уверенность": fmt.num(risk.confidence),
        "Возвращение": f"GW{risk.return_gw}" if risk.return_gw else "—",
        "Источник": origin,
        "Статус FPL": fmt.status_text(risk.fpl_status, risk.fpl_chance)
        + (f" — {risk.fpl_news}" if risk.fpl_news else ""),
    }
    return [{"Поле": k, "Значение": v} for k, v in rows.items()]


# ---------- альтернативы по цене ----------

PRICE_WINDOW = 0.5  # «Alternatives at the price»: цена ±£0.5m


def run_word(fsi: float, mine: float) -> str:
    """«календарь легче: 8.1 против 9.0» / «тяжелее» / «такой же» (сумма сложности за n туров)."""
    if abs(fsi - mine) < 0.05:
        return f"календарь такой же: {fsi:.1f}"
    word = "легче" if fsi < mine else "тяжелее"
    return f"календарь {word}: {fsi:.1f} против {mine:.1f}"


def alternatives_at_price(
    player: Player,
    bs: Bootstrap,
    preds: Mapping[int, PlayerPrediction],
    *,
    n_gws: int = SIMILAR_GWS,
    limit: int = 3,
    window: float = PRICE_WINDOW,
) -> list[SimilarPlayer]:
    """Та же позиция, другой клуб, цена в пределах ±window, доступен сейчас (статус FPL «a»);
    top-limit по xPts за n_gws туров. Строка: «£0.2m дороже · календарь тяжелее: 9.3 против
    8.7» — как «alternatives at the price» у smartplay."""
    my_fsi = fsi_over(preds.get(player.id), n_gws)
    cands = [
        p
        for p in bs.elements
        if p.id != player.id
        and p.element_type == player.element_type
        and p.team != player.team
        and p.status == "a"
        and abs(p.price - player.price) <= window + 1e-9
    ]
    cands.sort(key=lambda p: (-xpts_over(preds.get(p.id), n_gws), p.id))
    out = []
    for p in cands[:limit]:
        pred = preds.get(p.id)
        compare = (
            f"{money_delta(p.price - player.price)} · {run_word(fsi_over(pred, n_gws), my_fsi)}"
        )
        out.append(_similar(p, bs, pred, compare, n_gws))
    return out


# ---------- «Откуда очки» списком ----------


def points_origin_rows(
    data: Any,
) -> tuple[str, list[tuple[str, int]], str | None, int | None] | None:
    """(заголовок, [(категория по-русски, очки)] без нулей, метка удачи, всего) — для списка
    с мини-столбиками вместо одной длинной строки. Нет окна — None."""
    if data is None:
        return None
    payload = data.as_dict() if hasattr(data, "as_dict") else dict(data)
    n = int(payload.get("n_rounds") or 0)
    cats = dict(payload.get("categories") or {})
    if n <= 0 and not cats:
        return None
    from fplcopilot.core.points_form import CATEGORY_ORDER, CATEGORY_RU, LABEL_RU

    rows = [
        (CATEGORY_RU.get(k, "прочее"), int(cats[k]))
        for k in CATEGORY_ORDER
        if k in cats and int(cats[k]) != 0
    ]
    label = None
    if payload.get("label"):
        label = payload.get("label_ru") or LABEL_RU.get(str(payload.get("label")))
    tours = n if n else int(payload.get("last_n") or 3)
    title = f"Откуда очки за {fmt.plural(tours, 'тур', 'тура', 'туров')}"
    total = payload.get("total_points")
    return title, rows, label, None if total is None else int(total)


# ---------- HTML в стиле smartplay (стили — PLAYER_CSS, внедряются глобально) ----------


def _model_dot(model: str | None) -> str:
    if not model:
        return ""
    return fmt.tip(f"Влияет на прогноз: {model}", mark="", css_class="model")


def section_card_html(section: Section) -> str:
    """Секция карточкой: строки «ПОДПИСЬ ? · число · полоска перцентиля · 88%»; синяя точка —
    метрика входит в прогноз xPts. Без перцентилей (стандарты) — без полоски."""
    rows = []
    for m in section.metrics:
        label = (
            f'<span class="lbl"><span class="t">{fmt._esc(m.label)}</span>'
            f"{fmt.tip(m.help, align='left')}{_model_dot(m.model)}</span>"
        )
        value = f'<b class="val">{fmt._esc(m.text)}</b>'
        if m.pct is None:
            bar, pct = '<span class="bar none"></span>', '<span class="pct"></span>'
        else:
            w = max(0, min(100, m.pct))
            bar = (
                f'<span class="bar"><i style="width:{w}%;background:{percentile_color(m.pct)}">'
                "</i></span>"
            )
            pct = f'<span class="pct">{m.pct}%</span>'
        rows.append(f'<div class="fpl-sp-row">{label}{value}{bar}{pct}</div>')
    return (
        f'<div class="fpl-sp-card"><div class="fpl-sp-title">{fmt._esc(section.title)}</div>'
        + "".join(rows)
        + "</div>"
    )


def ownership_strip_html(section: Section) -> str:
    """«Ownership & Transfers» полосой из четырёх ячеек: купили — зелёным, продали — красным."""
    cells = []
    for m in section.metrics:
        tone = ""
        if m.key in ("tin", "tout", "tnet"):
            tone = "good" if m.text.startswith("+") else "bad" if m.text.startswith("−") else ""
        cells.append(
            f"<div><small>{fmt._esc(m.label)}{fmt.tip(m.help, align='left')}</small>"
            f'<b class="{tone}">{fmt._esc(m.text)}</b></div>'
        )
    return (
        f'<div class="fpl-sp-card"><div class="fpl-sp-title">{fmt._esc(section.title)}</div>'
        f'<div class="fpl-sp-strip">{"".join(cells)}</div></div>'
    )


def key_numbers_html(keys: Sequence[KeyNumber]) -> str:
    """Строка ключевых чисел шапки: крупное число, под ним подпись; xPts — цветом эмблемы."""
    cells = []
    for k in keys:
        cls = "accent" if k.label == "xPts" else ""
        delta = f"<em>{fmt._esc(k.delta)}</em>" if k.delta else ""
        cells.append(
            f'<div class="{cls}"><b>{fmt._esc(k.value)}</b>{delta}'
            f"<small>{fmt._esc(k.label)}{fmt.tip(k.help)}</small></div>"
        )
    return f'<div class="fpl-keynums">{"".join(cells)}</div>'


def report_tags_html(head: Header, player: Player) -> str:
    """Верх шапки: подпись «Player report» и метки цена · позиция · клуб."""
    club = head.meta.rsplit("· ", 1)[-1]
    pills = "".join(
        f"<span>{fmt._esc(x)}</span>" for x in (f"£{player.price:.1f}", player.position.short, club)
    )
    return (
        '<div class="fpl-report-top"><span class="lbl">Player report</span>'
        f'<span class="tags">{pills}</span></div>'
    )


def report_xpts_html(xpts: str) -> str:
    """Крупный xPts справа в шапке."""
    return f'<div class="fpl-report-big"><small>xPts</small><b>{fmt._esc(xpts)}</b></div>'


def report_club_html(head: Header) -> str:
    news = f'<span class="news">{fmt._esc(head.news)}</span>' if head.news else ""
    return (
        f'<div class="fpl-report-club"><span>{fmt._esc(head.club)}</span>'
        f'<span class="fpl-status {head.status_tone}">{fmt._esc(head.status_text)}</span>{news}</div>'
    )


def fixtures_strip_html(cells: Sequence[FixtureCell]) -> str:
    """«Fixtures (next 6 GWs)» одной полосой: тур мелко, соперник чипом цвета сложности."""
    items = "".join(
        fmt.tip(
            c.title,
            mark=(
                f'<span class="fx"><small>GW{c.gw}</small>'
                f'<b class="{c.css}">{fmt._esc(c.label)}</b></span>'
            ),
            css_class="tip-block",
            escape_mark=False,
        )
        for c in cells
    )
    return f'<div class="fpl-sp-card"><div class="fpl-sp-fixtures">{items}</div></div>'


def similar_card_html(s: SimilarPlayer) -> str:
    """Карточка похожего игрока: имя, клуб · позиция · цена, статус, 3GW xPts справа, строка
    сравнения."""
    badge = (
        ""
        if s.status_tone == "green"
        else f'<span class="fpl-status {s.status_tone}">{fmt._esc(s.status_text)}</span>'
    )
    return (
        '<div class="fpl-sim"><div class="top"><div><strong>'
        f"{fmt._esc(s.name)}</strong><small>{fmt._esc(s.club)} · {fmt._esc(s.position)} · "
        f'£{s.price:.1f}</small></div><div class="x"><b>{s.xpts:.1f}</b><small>3GW</small></div>'
        f"</div>{badge}<p>{fmt._esc(s.compare)}</p></div>"
    )


def points_origin_html(
    rows: Sequence[tuple[str, int]], label: str | None, total: int | None
) -> str:
    """Список «категория — очки» с мини-столбиками (минусы — красным)."""
    top = max((abs(v) for _, v in rows), default=1) or 1
    items = "".join(
        f'<div class="fpl-sp-row origin"><span class="lbl"><span class="t">{fmt._esc(k)}</span>'
        f'</span><b class="val">{"+" if v > 0 else "−" if v < 0 else ""}{abs(v)}</b>'
        f'<span class="bar"><i style="width:{round(100 * abs(v) / top)}%;background:'
        f'{"var(--fpl-good)" if v > 0 else "var(--fpl-bad)"}"></i></span><span class="pct"></span></div>'
        for k, v in rows
    )
    foot = []
    if total is not None:
        foot.append(f"всего {fmt.points_text(total)}")
    if label:
        foot.append(f"оценка: {label}")
    tail = f'<p class="fpl-sp-foot">{" · ".join(foot)}</p>' if foot else ""
    return f'<div class="fpl-sp-card">{items or "<p>Очков нет.</p>"}{tail}</div>'


PLAYER_CSS = """
.st-key-card_player_report [data-testid="stHeading"] h2 {
  font-size: 2.6rem; font-weight: 800; letter-spacing: -0.035em; line-height: 1.1; padding: 6px 0 8px;
}
.tip.model {
  width: 7px; height: 7px; border-radius: 50%; background: var(--fpl-brand); margin-left: 6px;
  vertical-align: 1px; flex: 0 0 auto;
}
.fpl-model-legend { display: flex; align-items: flex-start; gap: 8px; font-size: 13px; color: var(--fpl-muted); margin: 4px 0 12px; }
.fpl-model-legend i { width: 7px; height: 7px; border-radius: 50%; background: var(--fpl-brand); display: inline-block; flex: 0 0 auto; margin-top: 6px; }
.fpl-sp-card {
  background: var(--fpl-surface); border: 1px solid var(--fpl-line-strong); border-radius: 12px;
  padding: 16px 20px 10px; box-shadow: var(--fpl-shadow); margin: 0 0 14px; overflow: visible;
}
.fpl-sp-title {
  font-size: 12px; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--fpl-accent); padding-bottom: 10px; margin-bottom: 4px; border-bottom: 1px solid var(--fpl-line);
}
.fpl-sp-row {
  display: grid; grid-template-columns: minmax(96px, 1.3fr) minmax(52px, auto) minmax(60px, 2fr) 42px;
  align-items: center; gap: 12px; padding: 9px 0; border-bottom: 1px solid var(--fpl-line);
}
.fpl-sp-row:last-child { border-bottom: 0; }
.fpl-sp-row .lbl {
  display: flex; align-items: center; min-width: 0; font-size: 12px; font-weight: 700;
  letter-spacing: 0.06em; text-transform: uppercase; color: var(--fpl-muted);
}
.fpl-sp-row.origin .lbl { text-transform: none; letter-spacing: 0; font-size: 14px; font-weight: 600; color: var(--fpl-text-2); }
.fpl-sp-row .lbl .t { overflow-wrap: anywhere; }
.fpl-sp-row .val { font-family: var(--fpl-mono); font-size: 15px; font-weight: 600; text-align: right; color: var(--fpl-text); white-space: nowrap; }
.fpl-sp-row .bar { height: 8px; border-radius: 4px; background: var(--fpl-surface-2); border: 1px solid var(--fpl-line); overflow: hidden; }
.fpl-sp-row .bar.none { visibility: hidden; }
.fpl-sp-row .bar i { display: block; height: 100%; border-radius: 4px; }
.fpl-sp-row .pct { font-family: var(--fpl-mono); font-size: 12px; color: var(--fpl-faint); text-align: right; }
.fpl-sp-foot { margin: 8px 0 6px; font-size: 13px; color: var(--fpl-muted); }
.fpl-sp-strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; padding: 10px 0 8px; }
.fpl-sp-strip > div { background: var(--fpl-surface-2); border: 1px solid var(--fpl-line); border-radius: 8px; padding: 12px 14px; display: flex; flex-direction: column; gap: 4px; text-align: center; align-items: center; }
.fpl-sp-strip small { font-size: 12px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--fpl-muted); display: flex; align-items: center; }
.fpl-sp-strip b { font-family: var(--fpl-mono); font-size: 19px; font-weight: 600; }
.fpl-sp-strip b.good { color: var(--fpl-good); } .fpl-sp-strip b.bad { color: var(--fpl-bad); }
.fpl-sp-fixtures { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 10px; padding: 4px 0 8px; }
.fpl-sp-fixtures .fx { display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 8px 4px; border: 1px solid var(--fpl-line); border-radius: 8px; background: var(--fpl-surface-2); }
.fpl-sp-fixtures .fx small { font-family: var(--fpl-mono); font-size: 12px; color: var(--fpl-muted); }
.fpl-sp-fixtures .fx b { font-size: 13.5px; font-weight: 700; padding: 3px 10px; border-radius: 6px; white-space: nowrap; }
.fpl-report-top { display: flex; justify-content: space-between; align-items: center; }
.fpl-report-top .lbl { font-size: 12px; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; color: var(--fpl-accent); }
.fpl-report-top .tags { display: flex; gap: 6px; }
.fpl-report-top .tags span {
  font-family: var(--fpl-mono); font-size: 12px; font-weight: 600; padding: 3px 8px; border-radius: 5px;
  border: 1px solid var(--fpl-line-strong); background: var(--fpl-surface-2);
}
.fpl-report-big { display: flex; flex-direction: column; align-items: flex-end; }
.fpl-report-big small { font-size: 12px; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; color: var(--fpl-muted); }
.fpl-report-big b { font-family: var(--fpl-mono); font-size: 52px; font-weight: 600; line-height: 1; color: var(--fpl-accent); letter-spacing: -0.04em; }
.fpl-report-club { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; font-size: 15px; color: var(--fpl-text-2); margin: -6px 0 6px; }
.fpl-report-club .news { font-size: 13.5px; color: var(--fpl-muted); flex-basis: 100%; }
.fpl-status {
  font-size: 11.5px; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase;
  padding: 2px 8px; border-radius: 4px; background: var(--fpl-good-bg); color: var(--fpl-good);
}
.fpl-status.yellow, .fpl-status.orange { background: var(--fpl-warn-bg); color: var(--fpl-warn); }
.fpl-status.red, .fpl-status.gray { background: var(--fpl-bad-bg); color: var(--fpl-bad); }
.fpl-keynums {
  display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); border-top: 1px solid var(--fpl-line);
  margin-top: 12px; padding-top: 14px;
}
.fpl-keynums > div { display: flex; flex-direction: column; align-items: center; gap: 2px; border-right: 1px solid var(--fpl-line); min-width: 0; }
.fpl-keynums > div:last-child { border-right: 0; }
.fpl-keynums b { font-family: var(--fpl-mono); font-size: 22px; font-weight: 600; white-space: nowrap; }
.fpl-keynums > div.accent b { color: var(--fpl-accent); }
.fpl-keynums em { font-style: normal; font-family: var(--fpl-mono); font-size: 11.5px; color: var(--fpl-faint); }
.fpl-keynums small { font-size: 11.5px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--fpl-muted); display: flex; align-items: center; }
.fpl-sim { display: flex; flex-direction: column; gap: 6px; }
.fpl-sim .top { display: flex; justify-content: space-between; gap: 10px; }
.fpl-sim .top div { display: flex; flex-direction: column; min-width: 0; }
.fpl-sim strong { font-size: 16px; font-weight: 750; }
.fpl-sim small { font-family: var(--fpl-mono); font-size: 12px; color: var(--fpl-muted); }
.fpl-sim .x { align-items: flex-end; }
.fpl-sim .x b { font-family: var(--fpl-mono); font-size: 20px; font-weight: 600; line-height: 1.1; }
.fpl-sim .fpl-status { align-self: flex-start; }
.fpl-sim p { margin: 0; font-size: 13px; color: var(--fpl-text-2); line-height: 1.45; }
@media (max-width: 1100px) { .fpl-keynums { grid-template-columns: repeat(3, minmax(0, 1fr)); row-gap: 12px; } }
"""
