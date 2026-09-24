"""Чистые функции сравнения двух игроков (страница «Сравнение»).

Сбор фактов для ИИ-объяснения — только код (суммы xPts, сложность туров, минуты, цена,
владение, сохранённые новости). LLM получает готовый JSON и не считает числа.
Без streamlit — unit-тесты в tests/test_compare.py.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from fplcopilot.agent.llm import LLMUsage, estimate_cost, load_agent_prompt, price_for
from fplcopilot.agent.tools import GWPrediction, PlayerPrediction, PlayerRisk
from fplcopilot.app import format as fmt
from fplcopilot.app import player_card as card
from fplcopilot.app import theme
from fplcopilot.data.schemas import Bootstrap, Player
from fplcopilot.rag.llm import get_openai_client

FIXTURES_SHOWN = 6
HORIZON_3 = 3
HORIZON_5 = 5
COMPARE_MODEL = "gpt-4o-mini"
COMPARE_TEMPERATURE = 0.2

# Жаргон и символы, запрещённые в русских фразах объяснения / фактов для экрана.
FORBIDDEN_IN_PROSE = re.compile(
    r"(Δ|→|←|↔|\bsd\b|\bxPts\b|\bFSI\b|\broute\b|\bp_start\b|\bexp_minutes\b)",
    re.IGNORECASE,
)

# Цвета сложности — общие классы theme.fsi_class (как на карточке игрока и в составе).
BLANK_CLASS = card.BLANK_CLASS

COMPARE_COLUMN_HELP: dict[str, str] = {
    "Price": "Текущая цена в FPL, миллионы фунтов",
    "TSB%": "Доля менеджеров FPL, у которых игрок в составе",
    "Form": "Средние очки за матч за последние 30 дней",
    "Pts/Game": "Средние очки за сыгранный матч в сезоне",
    "Pts": "Очки FPL за сезон",
    "xMins": "Ожидаемые минуты на ближайший тур по модели",
    "xPts": "Прогноз очков на ближайший тур",
    "3GW xPts": "Сумма прогноза очков на ближайшие 3 тура",
    "5GW xPts": "Сумма прогноза очков на ближайшие 5 туров",
    "Start%": "Шанс выйти в стартовом составе на ближайший тур",
    "FRR#": (
        "Рейтинг календаря команды среди клубов лиги по средней сложности матчей "
        "на 5 ближайших туров: 1 — самый лёгкий"
    ),
    "Pens": "Очередь пенальти по FPL: 1 — бьёт пенальти команды",
    "U xG90": "Ожидаемые голы на 90 минут по Understat (если оба сопоставились)",
    "U Shots/90": "Удары на 90 минут по Understat (если оба сопоставились)",
}

OWNERSHIP_TEMPLATE = 30.0  # выше — «надёжный шаблон»
OWNERSHIP_DIFF = 10.0  # ниже — «редкий»

ROTATION_RU: dict[str, str] = {
    "low": "низкий",
    "medium": "средний",
    "high": "высокий",
    "unknown": "нет данных",
}


def steps_word(n: int) -> str:
    """1 ступень / 2 ступени / 5 ступеней."""
    n = abs(int(n))
    tail, tail2 = n % 10, n % 100
    if tail == 1 and tail2 != 11:
        return f"{n} ступень"
    if 2 <= tail <= 4 and not 12 <= tail2 <= 14:
        return f"{n} ступени"
    return f"{n} ступеней"


# ---------- примитивы текста без жаргона ----------


def plain_points(x: float | None, digits: int = 1) -> str:
    if x is None:
        return "нет данных"
    return f"{x:.{digits}f}"


def plain_delta_points(x: float, digits: int = 1) -> str:
    """«плюс 1.4 очка» / «минус 0.5 очка» / «столько же очков»."""
    if abs(x) < 5 * 10 ** (-(digits + 1)):
        return "столько же очков"
    word = "плюс" if x > 0 else "минус"
    n = abs(x)
    unit = "очко" if abs(n - 1.0) < 0.05 else "очка"
    return f"{word} {n:.{digits}f} {unit}"


def plain_percent(x: float | None, digits: int = 0) -> str:
    if x is None:
        return "нет данных"
    return f"{x:.{digits}f} процентов"


def difficulty_level(fsi: float | None) -> int | None:
    if fsi is None:
        return None
    return max(1, min(5, round(fsi)))


def plain_difficulty(level: int | None) -> str:
    if level is None:
        return "нет матча"
    return f"сложность {level} из 5"


def fixture_side_label(fixtures: Sequence[Any] | None) -> tuple[str, float | None]:
    """«TOT (д) · BUR (в)» и средняя сложность; пусто — («—», None)."""
    if not fixtures:
        return "—", None
    parts = []
    for f in fixtures:
        side = "д" if f.is_home else "в"
        parts.append(f"{f.opponent} ({side})")
    fsi = card.gw_difficulty([int(f.fsi) for f in fixtures])
    return " · ".join(parts), fsi


def xpts_sum(pred: PlayerPrediction | None, n: int) -> float | None:
    if pred is None or not pred.by_gw:
        return None
    gws = pred.by_gw[:n]
    if not gws:
        return None
    return round(sum(g.xpts for g in gws), 2)


def next_gw(pred: PlayerPrediction | None) -> GWPrediction | None:
    return pred.by_gw[0] if pred and pred.by_gw else None


def points_per_million(xpts: float | None, price: float) -> float | None:
    if xpts is None or price <= 0:
        return None
    return round(xpts / price, 2)


def ownership_tag(own: float) -> str:
    if own >= OWNERSHIP_TEMPLATE:
        return "надёжный шаблон"
    if own < OWNERSHIP_DIFF:
        return "редкий"
    return "обычный"


def better_side(
    a: float | None, b: float | None, *, higher_is_better: bool
) -> Literal["a", "b", "tie", "na"]:
    if a is None or b is None:
        return "na"
    if abs(a - b) < 1e-9:
        return "tie"
    if higher_is_better:
        return "a" if a > b else "b"
    return "a" if a < b else "b"


# ---------- строки таблицы сравнения ----------


@dataclass(frozen=True)
class MetricRow:
    key: str
    label: str
    help: str
    a_text: str
    b_text: str
    a_raw: float | None
    b_raw: float | None
    winner: Literal["a", "b", "tie", "na"]  # кто лучше по метрике


def _fmt_price(x: float) -> str:
    return f"£{x:.1f}m"


def _fmt_own(x: float) -> str:
    return f"{x:.1f}%"


def _fmt_num(x: float | None, digits: int = 1) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def _fmt_int(x: float | None) -> str:
    return "—" if x is None else str(round(x))


def _fmt_pct01(x: float | None) -> str:
    """p_start 0..1 -> «65%»."""
    return "—" if x is None else f"{100 * x:.0f}%"


def metric_rows(
    player_a: Player,
    player_b: Player,
    pred_a: PlayerPrediction | None,
    pred_b: PlayerPrediction | None,
    *,
    frr_a: int | None,
    frr_b: int | None,
    gw: int,
    ext_a: Any | None = None,
    ext_b: Any | None = None,
) -> list[MetricRow]:
    na, nb = next_gw(pred_a), next_gw(pred_b)
    x3a, x3b = xpts_sum(pred_a, HORIZON_3), xpts_sum(pred_b, HORIZON_3)
    x5a, x5b = xpts_sum(pred_a, HORIZON_5), xpts_sum(pred_b, HORIZON_5)
    own_a = float(player_a.selected_by_percent or 0)
    own_b = float(player_b.selected_by_percent or 0)
    specs: list[tuple[str, str, str, float | None, float | None, str, str, bool | None]] = [
        (
            "price",
            "Price",
            COMPARE_COLUMN_HELP["Price"],
            player_a.price,
            player_b.price,
            _fmt_price(player_a.price),
            _fmt_price(player_b.price),
            False,
        ),
        (
            "tsb",
            "TSB%",
            COMPARE_COLUMN_HELP["TSB%"],
            own_a,
            own_b,
            _fmt_own(own_a),
            _fmt_own(own_b),
            None,  # без подсветки «лучше»
        ),
        (
            "form",
            "Form",
            COMPARE_COLUMN_HELP["Form"],
            player_a.form,
            player_b.form,
            _fmt_num(player_a.form, 1),
            _fmt_num(player_b.form, 1),
            True,
        ),
        (
            "ppg",
            "Pts/Game",
            COMPARE_COLUMN_HELP["Pts/Game"],
            player_a.points_per_game,
            player_b.points_per_game,
            _fmt_num(player_a.points_per_game, 1),
            _fmt_num(player_b.points_per_game, 1),
            True,
        ),
        (
            "pts",
            "Pts",
            COMPARE_COLUMN_HELP["Pts"],
            float(player_a.total_points),
            float(player_b.total_points),
            str(player_a.total_points),
            str(player_b.total_points),
            True,
        ),
        (
            "xmins",
            f"xMins GW{gw}",
            COMPARE_COLUMN_HELP["xMins"],
            na.exp_minutes if na else None,
            nb.exp_minutes if nb else None,
            _fmt_int(na.exp_minutes if na else None),
            _fmt_int(nb.exp_minutes if nb else None),
            True,
        ),
        (
            "xpts",
            f"xPts GW{gw}",
            COMPARE_COLUMN_HELP["xPts"],
            na.xpts if na else None,
            nb.xpts if nb else None,
            _fmt_num(na.xpts if na else None, 2),
            _fmt_num(nb.xpts if nb else None, 2),
            True,
        ),
        (
            "xpts3",
            "3GW xPts",
            COMPARE_COLUMN_HELP["3GW xPts"],
            x3a,
            x3b,
            _fmt_num(x3a, 2),
            _fmt_num(x3b, 2),
            True,
        ),
        (
            "xpts5",
            "5GW xPts",
            COMPARE_COLUMN_HELP["5GW xPts"],
            x5a,
            x5b,
            _fmt_num(x5a, 2),
            _fmt_num(x5b, 2),
            True,
        ),
        (
            "start",
            "Start%",
            COMPARE_COLUMN_HELP["Start%"],
            na.p_start if na else None,
            nb.p_start if nb else None,
            _fmt_pct01(na.p_start if na else None),
            _fmt_pct01(nb.p_start if nb else None),
            True,
        ),
        (
            "frr",
            "FRR#",
            COMPARE_COLUMN_HELP["FRR#"],
            float(frr_a) if frr_a is not None else None,
            float(frr_b) if frr_b is not None else None,
            "—" if frr_a is None else str(frr_a),
            "—" if frr_b is None else str(frr_b),
            False,
        ),
        (
            "pen",
            "Pens",
            COMPARE_COLUMN_HELP["Pens"],
            float(player_a.penalties_order or 0) or None,
            float(player_b.penalties_order or 0) or None,
            "—" if not player_a.penalties_order else f"№{player_a.penalties_order}",
            "—" if not player_b.penalties_order else f"№{player_b.penalties_order}",
            False,
        ),
    ]
    ua = getattr(ext_a, "understat", None) if ext_a is not None else None
    ub = getattr(ext_b, "understat", None) if ext_b is not None else None
    if ua is not None and ub is not None:
        specs.extend(
            [
                (
                    "u_xg90",
                    "U xG90",
                    COMPARE_COLUMN_HELP["U xG90"],
                    ua.xg90,
                    ub.xg90,
                    _fmt_num(ua.xg90, 2),
                    _fmt_num(ub.xg90, 2),
                    True,
                ),
                (
                    "u_shots90",
                    "U Shots/90",
                    COMPARE_COLUMN_HELP["U Shots/90"],
                    ua.shots90,
                    ub.shots90,
                    _fmt_num(ua.shots90, 2),
                    _fmt_num(ub.shots90, 2),
                    True,
                ),
            ]
        )
    out: list[MetricRow] = []
    for key, label, help_text, ra, rb, ta, tb, hib in specs:
        if hib is None:
            winner: Literal["a", "b", "tie", "na"] = "na"
        else:
            winner = better_side(ra, rb, higher_is_better=hib)
        out.append(MetricRow(key, label, help_text, ta, tb, ra, rb, winner))
    return out


@dataclass(frozen=True)
class FixtureCompareRow:
    gw: int
    a_label: str
    b_label: str
    a_fsi: float | None
    b_fsi: float | None
    a_level: int | None
    b_level: int | None
    delta_steps: int | None  # a − b по сложности: отрицательное = у A легче
    delta_text: str  # «у Saka легче на 2 ступени» / «одинаковая сложность» / «»


def fixture_compare_rows(
    pred_a: PlayerPrediction | None,
    pred_b: PlayerPrediction | None,
    name_a: str,
    name_b: str,
    *,
    n: int = FIXTURES_SHOWN,
) -> list[FixtureCompareRow]:
    by_a = {g.gw: g for g in (pred_a.by_gw if pred_a else [])}
    by_b = {g.gw: g for g in (pred_b.by_gw if pred_b else [])}
    gws = sorted(set(by_a) | set(by_b))[:n]
    if not gws and pred_a and pred_a.by_gw:
        gws = [g.gw for g in pred_a.by_gw[:n]]
    rows: list[FixtureCompareRow] = []
    for gw in gws:
        ga, gb = by_a.get(gw), by_b.get(gw)
        la, fa = fixture_side_label(ga.fixtures if ga else None)
        lb, fb = fixture_side_label(gb.fixtures if gb else None)
        lev_a, lev_b = difficulty_level(fa), difficulty_level(fb)
        delta = None
        delta_text = ""
        if lev_a is not None and lev_b is not None:
            delta = lev_a - lev_b
            if delta == 0:
                delta_text = "одинаковая сложность"
            elif delta < 0:
                delta_text = f"у {name_a} легче на {steps_word(abs(delta))}"
            else:
                delta_text = f"у {name_b} легче на {steps_word(delta)}"
        rows.append(
            FixtureCompareRow(gw, la, lb, fa, fb, lev_a, lev_b, delta, delta_text)
        )
    return rows


# ---------- факты из новостей (без выдумок) ----------


def news_facts(risk: PlayerRisk | None, player_name: str) -> dict[str, Any]:
    """Структура для FACTS.news; при отсутствии разбора — честная пометка."""
    empty = {
        "player": player_name,
        "has_signal": False,
        "note": "разбора новостей нет",
        "evidence": [],
    }
    if risk is None or risk.origin == "unavailable":
        return empty
    # пустой/бесполезный вердикт (нет цитат и нет уверенности) — не выдаём за разбор
    if not risk.evidence and float(risk.confidence or 0) <= 0:
        return empty
    if risk.abstained and not risk.evidence:
        return empty
    from fplcopilot.app import format as fmt

    evidence = [
        {
            "source": e.source,
            "date": e.date,
            "url": e.url,
            "quote": e.quote,
        }
        for e in risk.evidence
    ]
    rot = ROTATION_RU.get(risk.rotation_risk, risk.rotation_risk)
    return {
        "player": player_name,
        "has_signal": True,
        "origin": risk.origin,
        "availability": risk.availability,
        "availability_ru": fmt.AVAILABILITY_RU.get(risk.availability, risk.availability),
        "start_probability": risk.start_probability,
        "start_probability_plain": plain_percent(100 * risk.start_probability, 0),
        "expected_minutes": risk.expected_minutes,
        "rotation_risk": risk.rotation_risk,
        "rotation_risk_ru": rot,
        "return_gw": risk.return_gw,
        "confidence": risk.confidence,
        "summary": risk.summary or "",
        "evidence": evidence,
        "note": risk.note,
        "age_h": risk.age_h,
    }


def news_plain_lines(facts: dict[str, Any]) -> list[str]:
    """Готовые русские строки про новости — для экрана и тестов (без жаргона)."""
    name = facts.get("player", "игрок")
    if not facts.get("has_signal"):
        return [f"По игроку {name} разбора новостей нет."]
    rot = facts.get("rotation_risk_ru") or ROTATION_RU.get(
        str(facts.get("rotation_risk")), str(facts.get("rotation_risk"))
    )
    lines = [
        (
            f"{name}: доступность — {facts.get('availability_ru', facts.get('availability'))}, "
            f"шанс выйти в старте {facts.get('start_probability_plain')}, "
            f"ожидаемые минуты {facts.get('expected_minutes')}, "
            f"риск ротации — {rot}"
            + (
                f", возвращение в туре {facts['return_gw']}"
                if facts.get("return_gw")
                else ""
            )
            + "."
        )
    ]
    for ev in facts.get("evidence") or []:
        lines.append(
            f"«{ev['quote']}» — {ev['source']}, {ev['date']}"
            + (f" ({ev['url']})" if ev.get("url") else "")
        )
    return lines


# ---------- сбор FACTS ----------


def _horizon_block(
    name_a: str,
    name_b: str,
    xa: float | None,
    xb: float | None,
    n: int,
) -> dict[str, Any]:
    winner = better_side(xa, xb, higher_is_better=True)
    margin = None if xa is None or xb is None else round(abs(xa - xb), 2)
    leader = {"a": name_a, "b": name_b, "tie": None, "na": None}[winner]
    text = ""
    if winner == "na":
        text = f"Прогноза на горизонт {n} не хватает для сравнения."
    elif winner == "tie":
        horizon = "ближайший тур" if n == 1 else f"{n} туров"
        text = (
            f"На {horizon} у {name_a} и {name_b} одинаковый прогноз: "
            f"{plain_points(xa, 2)} очков."
        )
    else:
        loser = name_b if winner == "a" else name_a
        horizon = "ближайший тур" if n == 1 else f"{n} туров"
        text = (
            f"На {horizon} лидирует {leader}: {plain_points(xa if winner == 'a' else xb, 2)} "
            f"против {plain_points(xb if winner == 'a' else xa, 2)} у {loser} "
            f"({plain_delta_points(margin or 0, 2)})."
        )
    return {
        "n_gws": n,
        "a_xpts": xa,
        "b_xpts": xb,
        "winner": winner,
        "leader": leader,
        "margin_points": margin,
        "text": text,
    }


def build_compare_facts(
    player_a: Player,
    player_b: Player,
    pred_a: PlayerPrediction | None,
    pred_b: PlayerPrediction | None,
    risk_a: PlayerRisk | None,
    risk_b: PlayerRisk | None,
    *,
    bs: Bootstrap,
    frr_a: int | None,
    frr_b: int | None,
    strategy: str = "balanced",
) -> dict[str, Any]:
    """Детерминированные факты для таблицы и ИИ. Числа уже округлены."""
    name_a, name_b = player_a.web_name, player_b.web_name
    team_a, team_b = bs.team(player_a.team).short_name, bs.team(player_b.team).short_name
    na, nb = next_gw(pred_a), next_gw(pred_b)
    la, fa = fixture_side_label(na.fixtures if na else None)
    lb, fb = fixture_side_label(nb.fixtures if nb else None)
    lev_a, lev_b = difficulty_level(fa), difficulty_level(fb)

    x1a = na.xpts if na else None
    x1b = nb.xpts if nb else None
    next_block = _horizon_block(name_a, name_b, x1a, x1b, 1)
    next_block.update(
        {
            "a_fixture": la,
            "b_fixture": lb,
            "a_difficulty": lev_a,
            "b_difficulty": lev_b,
            "a_difficulty_plain": plain_difficulty(lev_a),
            "b_difficulty_plain": plain_difficulty(lev_b),
            "gw": na.gw if na else (nb.gw if nb else None),
        }
    )
    if lev_a is not None and lev_b is not None and lev_a != lev_b:
        easier = name_a if lev_a < lev_b else name_b
        steps = abs(lev_a - lev_b)
        next_block["fixture_note"] = (
            f"Ближайший матч у {easier} легче на {steps_word(steps)} "
            f"({plain_difficulty(min(lev_a, lev_b))} против {plain_difficulty(max(lev_a, lev_b))})."
        )
    else:
        next_block["fixture_note"] = ""

    fx_rows = fixture_compare_rows(pred_a, pred_b, name_a, name_b)
    # Сводка календаря: где у кого легче
    a_easier, b_easier = [], []
    for r in fx_rows:
        if r.delta_steps is None or r.delta_steps == 0:
            continue
        if r.delta_steps < 0:
            a_easier.append(r.gw)
        else:
            b_easier.append(r.gw)
    calendar_summary = ""
    if a_easier or b_easier:
        parts = []
        if a_easier:
            parts.append(
                f"у {name_a} легче туры "
                + ", ".join(f"GW{g}" for g in a_easier)
            )
        if b_easier:
            parts.append(
                f"у {name_b} легче туры "
                + ", ".join(f"GW{g}" for g in b_easier)
            )
        calendar_summary = "; ".join(parts) + "."

    mins_a = na.exp_minutes if na else None
    mins_b = nb.exp_minutes if nb else None
    ps_a = na.p_start if na else None
    ps_b = nb.p_start if nb else None

    x3a, x3b = xpts_sum(pred_a, HORIZON_3), xpts_sum(pred_b, HORIZON_3)
    ppm_a = points_per_million(x3a, player_a.price)
    ppm_b = points_per_million(x3b, player_b.price)
    value_winner = better_side(ppm_a, ppm_b, higher_is_better=True)

    own_a = float(player_a.selected_by_percent or 0)
    own_b = float(player_b.selected_by_percent or 0)
    news_a = news_facts(risk_a, name_a)
    news_b = news_facts(risk_b, name_b)

    facts = {
        "strategy": strategy,
        "player_a": {
            "id": player_a.id,
            "name": name_a,
            "full_name": player_a.full_name,
            "team": team_a,
            "position": player_a.position.short,
            "price": player_a.price,
            "ownership": own_a,
            "ownership_tag": ownership_tag(own_a),
            "form": player_a.form,
            "points_per_game": player_a.points_per_game,
            "total_points": player_a.total_points,
        },
        "player_b": {
            "id": player_b.id,
            "name": name_b,
            "full_name": player_b.full_name,
            "team": team_b,
            "position": player_b.position.short,
            "price": player_b.price,
            "ownership": own_b,
            "ownership_tag": ownership_tag(own_b),
            "form": player_b.form,
            "points_per_game": player_b.points_per_game,
            "total_points": player_b.total_points,
        },
        "next_gw": next_block,
        "horizon_3": _horizon_block(name_a, name_b, x3a, x3b, HORIZON_3),
        "horizon_5": _horizon_block(
            name_a, name_b, xpts_sum(pred_a, HORIZON_5), xpts_sum(pred_b, HORIZON_5), HORIZON_5
        ),
        "minutes": {
            "a": mins_a,
            "b": mins_b,
            "winner": better_side(mins_a, mins_b, higher_is_better=True),
            "text": _mins_text(name_a, name_b, mins_a, mins_b),
        },
        "start_chance": {
            "a": ps_a,
            "b": ps_b,
            "a_plain": plain_percent(100 * ps_a, 0) if ps_a is not None else None,
            "b_plain": plain_percent(100 * ps_b, 0) if ps_b is not None else None,
            "winner": better_side(ps_a, ps_b, higher_is_better=True),
            "text": _start_text(name_a, name_b, ps_a, ps_b),
        },
        "value": {
            "a_points_per_million_3gw": ppm_a,
            "b_points_per_million_3gw": ppm_b,
            "winner": value_winner,
            "text": _value_text(name_a, name_b, ppm_a, ppm_b, player_a.price, player_b.price),
        },
        "ownership": {
            "a": own_a,
            "b": own_b,
            "a_tag": ownership_tag(own_a),
            "b_tag": ownership_tag(own_b),
            "text": (
                f"{name_a}: владение {own_a:.1f} процентов ({ownership_tag(own_a)}); "
                f"{name_b}: владение {own_b:.1f} процентов ({ownership_tag(own_b)})."
            ),
        },
        "frr": {
            "a": frr_a,
            "b": frr_b,
            "text": _frr_text(name_a, name_b, frr_a, frr_b),
        },
        "fixtures": [
            {
                "gw": r.gw,
                "a_label": r.a_label,
                "b_label": r.b_label,
                "a_difficulty": r.a_level,
                "b_difficulty": r.b_level,
                "delta_steps": r.delta_steps,
                "delta_text": r.delta_text,
            }
            for r in fx_rows
        ],
        "calendar_summary": calendar_summary,
        "news_a": news_a,
        "news_b": news_b,
        "news_lines": news_plain_lines(news_a) + news_plain_lines(news_b),
        "caveats_seed": _caveats_seed(news_a, news_b, pred_a, pred_b),
    }
    return facts


def _mins_text(na: str, nb: str, a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "Ожидаемых минут для сравнения нет."
    w = better_side(a, b, higher_is_better=True)
    if w == "tie":
        return f"Ожидаемые минуты одинаковы: {round(a)}."
    leader, other, la, lo = (na, nb, a, b) if w == "a" else (nb, na, b, a)
    return (
        f"Ожидаемые минуты выше у {leader}: {round(la)} против {round(lo)} у {other}."
    )


def _start_text(na: str, nb: str, a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "Шанса старта для сравнения нет."
    w = better_side(a, b, higher_is_better=True)
    if w == "tie":
        return f"Шанс выйти в старте одинаковый: {plain_percent(100 * a, 0)}."
    leader, other, la, lo = (na, nb, a, b) if w == "a" else (nb, na, b, a)
    return (
        f"Шанс выйти в старте выше у {leader}: {plain_percent(100 * la, 0)} "
        f"против {plain_percent(100 * lo, 0)} у {other}."
    )


def _value_text(
    na: str, nb: str, ppm_a: float | None, ppm_b: float | None, pa: float, pb: float
) -> str:
    if ppm_a is None or ppm_b is None:
        return "Выигрыш на миллион для сравнения недоступен."
    w = better_side(ppm_a, ppm_b, higher_is_better=True)
    price_note = (
        f"{na} стоит {pa:.1f} миллиона, {nb} — {pb:.1f} миллиона."
    )
    if w == "tie":
        return f"{price_note} Отдача на миллион за 3 тура одинаковая: {ppm_a:.2f}."
    leader = na if w == "a" else nb
    return (
        f"{price_note} Выигрыш на миллион за 3 тура выше у {leader}: "
        f"{(ppm_a if w == 'a' else ppm_b):.2f} против {(ppm_b if w == 'a' else ppm_a):.2f}."
    )


def _frr_text(na: str, nb: str, a: int | None, b: int | None) -> str:
    if a is None or b is None:
        return "Рейтинга календаря команд нет."
    if a == b:
        return f"Рейтинг календаря команд одинаковый: {a}."
    better = na if a < b else nb
    return (
        f"Календарь команды легче у {better}: рейтинг {min(a, b)} против {max(a, b)} "
        f"(1 — самый лёгкий среди клубов)."
    )


def _caveats_seed(
    news_a: dict[str, Any],
    news_b: dict[str, Any],
    pred_a: PlayerPrediction | None,
    pred_b: PlayerPrediction | None,
) -> list[str]:
    out = [
        "Прогноз очков — модель, а не гарантия результата матча.",
        "Состав и трансферы в публичном API FPL запаздывают до дедлайна.",
    ]
    if not news_a.get("has_signal") and not news_b.get("has_signal"):
        out.append("Разбора новостей по обоим игрокам нет — выводы только по прогнозу и календарю.")
    elif not news_a.get("has_signal"):
        out.append(f"Разбора новостей по {news_a.get('player')} нет.")
    elif not news_b.get("has_signal"):
        out.append(f"Разбора новостей по {news_b.get('player')} нет.")
    if pred_a is None or pred_b is None or not (pred_a and pred_a.by_gw) or not (pred_b and pred_b.by_gw):
        out.append("По одному из игроков прогноз неполный — сравнение по турам ограничено.")
    return out


def prose_is_clean(text: str) -> bool:
    """True, если в тексте нет запрещённого жаргона/символов."""
    return FORBIDDEN_IN_PROSE.search(text or "") is None


def assert_facts_prose_clean(facts: Mapping[str, Any]) -> list[str]:
    """Список полей с запрещённым жаргоном (пусто — всё чисто)."""
    bad: list[str] = []
    for key in (
        "calendar_summary",
        ("next_gw", "text"),
        ("next_gw", "fixture_note"),
        ("horizon_3", "text"),
        ("horizon_5", "text"),
        ("minutes", "text"),
        ("start_chance", "text"),
        ("value", "text"),
        ("ownership", "text"),
        ("frr", "text"),
    ):
        if isinstance(key, tuple):
            node: Any = facts
            for k in key:
                node = (node or {}).get(k) if isinstance(node, Mapping) else None
            path, val = ".".join(key), node
        else:
            path, val = key, facts.get(key)
        if isinstance(val, str) and val and not prose_is_clean(val):
            bad.append(path)
    for i, line in enumerate(facts.get("news_lines") or []):
        if isinstance(line, str) and line and not prose_is_clean(line):
            bad.append(f"news_lines[{i}]")
    for i, c in enumerate(facts.get("caveats_seed") or []):
        if isinstance(c, str) and c and not prose_is_clean(c):
            bad.append(f"caveats_seed[{i}]")
    return bad


# ---------- HTML ----------


def _esc(s: Any) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


COMPARE_TABLE_CSS = f"""
<style>
{fmt.TIP_CSS_RULES}
{fmt.DATA_TABLE_CSS_RULES}
{theme.FSI_CSS_RULES}
table.fpl-compare {{ font-size: 14px; }}
table.fpl-compare th {{ font-size: 14px; font-weight: 650; color: var(--fpl-text, inherit); }}
table.fpl-compare th .team {{
  font-size: 12px; font-weight: 400; color: var(--fpl-muted, inherit); margin-top: 2px;
}}
table.fpl-compare td {{ padding: 9px 10px; }}
table.fpl-compare td.win {{ font-weight: 700; color: var(--fpl-good, #16804A); }}
table.fpl-compare td.metric {{ color: var(--fpl-muted, inherit); white-space: nowrap; }}
table.fpl-fx th, table.fpl-fx td {{ text-align: center; }}
table.fpl-fx td.delta {{ text-align: left; font-size: 12.5px; color: var(--fpl-muted, inherit); }}
table.fpl-fx .cell {{
  display: inline-block; min-width: 7rem; padding: 7px 10px; border-radius: 8px;
  font-size: 13px; font-weight: 600;
}}
.fpl-fx-legend {{
  display: flex; align-items: center; flex-wrap: wrap; gap: 4px; margin-top: 8px;
  font-size: 12px; color: var(--fpl-muted, inherit);
}}
.fpl-fx-legend .sw {{ display: inline-block; width: 14px; height: 14px; border-radius: 4px; }}
</style>
"""


def compare_table_html(
    rows: Sequence[MetricRow],
    name_a: str,
    name_b: str,
    *,
    team_a: str = "",
    team_b: str = "",
    pos_a: str = "",
    pos_b: str = "",
) -> str:
    head_a = _esc(name_a) + (
        f'<div class="team">{_esc(team_a)} · {_esc(pos_a)}</div>' if team_a else ""
    )
    head_b = _esc(name_b) + (
        f'<div class="team">{_esc(team_b)} · {_esc(pos_b)}</div>' if team_b else ""
    )
    body = []
    for r in rows:
        cls_a = "num win" if r.winner == "a" else "num"
        cls_b = "num win" if r.winner == "b" else "num"
        help_q = fmt.tip(r.help) if r.help else ""
        body.append(
            "<tr>"
            f'<td class="metric">{_esc(r.label)}{help_q}</td>'
            f'<td class="{cls_a}">{_esc(r.a_text)}</td>'
            f'<td class="{cls_b}">{_esc(r.b_text)}</td>'
            "</tr>"
        )
    return (
        COMPARE_TABLE_CSS
        + '<table class="fpl-table fpl-compare"><thead><tr>'
        f'<th></th><th class="num">{head_a}</th><th class="num">{head_b}</th>'
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def fixture_compare_html(rows: Sequence[FixtureCompareRow], name_a: str, name_b: str) -> str:
    def chip(label: str, level: int | None) -> str:
        if level is None:
            css, title = BLANK_CLASS, "Нет матча в этом туре"
        else:
            css, title = theme.fsi_class(level), plain_difficulty(level)
        inner = f'<span class="cell {css}">{_esc(label)}</span>'
        return fmt.tip(title, mark=inner, css_class="tip-block", escape_mark=False)

    head = (
        f"<tr><th>Тур</th><th>{_esc(name_a)}</th><th>{_esc(name_b)}</th>"
        "<th>Сравнение</th></tr>"
    )
    body = []
    for r in rows:
        body.append(
            "<tr>"
            f"<td>GW{r.gw}</td>"
            f"<td>{chip(r.a_label, r.a_level)}</td>"
            f"<td>{chip(r.b_label, r.b_level)}</td>"
            f'<td class="delta">{_esc(r.delta_text)}</td>'
            "</tr>"
        )
    legend_chips = "".join(f'<span class="sw {theme.fsi_class(i)}"></span>' for i in range(1, 6))
    legend = (
        '<div class="fpl-fx-legend">'
        f"(д) — дома, (в) — в гостях · сложность матча: лёгкий {legend_chips} тяжёлый</div>"
    )
    return (
        COMPARE_TABLE_CSS
        + '<table class="fpl-table fpl-fx"><thead>'
        + head
        + "</thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
        + legend
    )


# ---------- LLM объяснение ----------


class CompareExplanation(BaseModel):
    verdict_next_gw: str = Field(description="Один-два предложения: кто лучше на ближайший тур и почему")
    verdict_3gw: str = Field(description="Один-два предложения: кто лучше на 3 тура и почему")
    why: list[str] = Field(description="2–5 пунктов «Почему» по фактам")
    news_points: list[str] = Field(
        description="Пункты из новостей с опорой на цитаты; если разбора нет — одна честная фраза"
    )
    caveats: list[str] = Field(description="Оговорки; включить caveats_seed из фактов")


def explain_compare(
    facts: Mapping[str, Any],
    *,
    model: str = COMPARE_MODEL,
    temperature: float = COMPARE_TEMPERATURE,
    client: Any | None = None,
) -> tuple[CompareExplanation, LLMUsage]:
    """Structured LLM-вызов. client — для тестов (фейк с .chat.completions.parse)."""
    system = load_agent_prompt("compare.system")
    user = (
        "Сравни двух игроков Fantasy Premier League только по FACTS ниже. "
        "Ответ — structured JSON по схеме.\n\nFACTS:\n"
        + json.dumps(facts, ensure_ascii=False, indent=1)
    )
    openai = client or get_openai_client()
    started = time.perf_counter()
    completion = openai.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format=CompareExplanation,
        temperature=temperature,
        max_completion_tokens=900,
    )
    latency = (time.perf_counter() - started) * 1000
    msg = completion.choices[0].message
    usage = completion.usage
    p_tok = usage.prompt_tokens if usage else 0
    c_tok = usage.completion_tokens if usage else 0
    _, known = price_for(model)
    info = LLMUsage(
        model=model,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        latency_ms=round(latency),
        cost_usd=estimate_cost(model, p_tok, c_tok),
        price_known=known,
    )
    if msg.parsed is None:
        raise RuntimeError(f"LLM returned no structured compare answer: {msg.refusal or 'empty'}")
    return msg.parsed, info


def explanation_cache_key(pid_a: int, pid_b: int, strategy: str) -> str:
    lo, hi = sorted((pid_a, pid_b))
    # порядок игроков на экране важен: ключ включает исходный порядок a,b
    return f"compare_explain:{pid_a}:{pid_b}:{strategy}:{lo}-{hi}"
