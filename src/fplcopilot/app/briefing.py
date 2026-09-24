"""Страница «Брифинг» (стартовая): сборка карточек из выходов инструментов — чистые функции.

Главный ход — рекомендуемый маршрут `recommend_transfers` (или «трансфер можно не делать»);
капитан — два первых варианта `optimize_team.captain_options`; «Следить до дедлайна» —
серьёзные проблемы состава (`format.serious_problems`) с сохранённым разбором новостей
(`analyze_player_risk`, cached_only). Ничего не выдумывается: нет данных — нет строки.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from fplcopilot.agent.tools import (
    CaptainOption,
    GameweekContext,
    LineupOut,
    PlanOut,
    PlayerPrediction,
    PlayerRisk,
    RouteOut,
    RoutesOut,
)
from fplcopilot.app import format as fmt
from fplcopilot.app.ui_kit import fixture_short
from fplcopilot.data.schemas import Bootstrap


def _word(n: int, one: str, few: str, many: str) -> str:
    return fmt.plural(n, one, few, many).split(" ", 1)[1]


def greeting(first_name: str | None, hour: int) -> str:
    """«Доброе утро, Жанибек.» по местному часу; без имени — «Добрый вечер.»."""
    if 5 <= hour < 12:
        part = "Доброе утро"
    elif 12 <= hour < 18:
        part = "Добрый день"
    elif 18 <= hour < 23:
        part = "Добрый вечер"
    else:
        part = "Доброй ночи"
    name = (first_name or "").strip()
    return f"{part}, {name}." if name else f"{part}."


def rank_share(rank: int | None, total: int | None) -> str | None:
    """Место в общем зачёте долей от всех менеджеров: «топ 18 %», «топ 0.5 %»; None — нет данных."""
    if not rank or not total or rank <= 0 or total <= 0:
        return None
    share = 100.0 * rank / total
    if share >= 10:
        return f"топ {share:.0f} %"
    if share >= 1:
        return f"топ {share:.1f} %"
    return f"топ {max(share, 0.01):.2f} %"


def recommended_route(rt: RoutesOut | None) -> RouteOut | None:
    if rt is None or rt.recommended_rank is None:
        return None
    return next((r for r in rt.routes if r.rank == rt.recommended_rank), None)


def route_sides(
    route: RouteOut, bs: Bootstrap | None
) -> tuple[list[tuple[str, str, float | None]], list[tuple[str, str, float | None]]]:
    """(продаём, покупаем): [(имя, клуб, цена)] по id маршрута; клуб и цена — из bootstrap."""

    def side(names: Sequence[str], ids: Sequence[int]) -> list[tuple[str, str, float | None]]:
        out = []
        for i, name in enumerate(names):
            club, price = "", None
            if bs is not None and i < len(ids):
                try:
                    p = bs.player(int(ids[i]))
                    club, price = bs.team(p.team).short_name, float(p.price)
                except KeyError:
                    pass
            out.append((name, club, price))
        return out

    return side(route.out, route.out_ids), side(route.in_, route.in_ids)


def route_headline(route: RouteOut) -> str:
    outs = ", ".join(route.out) or "—"
    ins = ", ".join(route.in_) or "—"
    return f"{outs} → {ins}"


def route_pills(route: RouteOut, preds: Mapping[int, PlayerPrediction]) -> list[tuple[str, str]]:
    """Аргументы маршрута таблетками: бесплатный ли ход, выигрыш уже в ближайшем туре, шанс
    выхода в старте у покупаемых (модель минут). Цвет: зелёный — плюс, жёлтый — риск."""
    out: list[tuple[str, str]] = []
    if route.hit_cost:
        out.append((f"платный: −{route.hit_cost} очка", "warn"))
    else:
        out.append(("бесплатный трансфер", "good"))
    nxt = route.gain_next_gw
    out.append(
        (f"{fmt.points_text(nxt, signed=True)} уже в ближайшем туре", "good" if nxt > 0 else "")
    )
    for pid, name in zip(route.in_ids, route.in_, strict=False):
        pred = preds.get(int(pid))
        if pred and pred.by_gw:
            p = pred.by_gw[0].p_start * 100
            out.append((f"{name} выходит в старте с шансом {p:.0f} %", "" if p >= 75 else "warn"))
    return out


def route_card_pills(route: RouteOut, horizon: int) -> list[tuple[str, str]]:
    """Числа карточки маршрута на «К дедлайну»: выигрыш в ближайшем туре и за горизонт
    (зелёным, если в плюс), платный ли ход (жёлтым)."""
    tours = fmt.plural(horizon, "тур", "тура", "туров")

    def tone(x: float) -> str:
        return "good" if x > 0.05 else "bad" if x < -0.05 else ""

    out = [
        (
            f"{fmt.points_text(route.gain_next_gw, signed=True)} в ближайшем туре",
            tone(route.gain_next_gw),
        ),
        (
            f"{fmt.points_text(route.gain_horizon, signed=True)} за {tours}",
            tone(route.gain_horizon),
        ),
    ]
    if route.hit_cost:
        out.append((f"платный: −{route.hit_cost} очка", "warn"))
    else:
        out.append(("бесплатный трансфер", ""))
    return out


def vs_best_text(gap: float) -> str:
    """Подпись к «Ваш старт»: gap = лучший состав − ваш старт."""
    if gap <= 0.005:
        return "как у лучшего"
    return f"на {gap:.2f} меньше"


def buy_risks(route: RouteOut) -> list[str]:
    """Риски покупки из `risk_note` оптимизатора (статус FPL, большой разброс, редкий выбор) —
    по-русски; «надёжный выбор» (template) риском не считается."""
    out = []
    for part in (route.risk_note or "").split(";"):
        part = part.strip()
        if not part.lower().startswith("in ") or "template" in part.lower():
            continue
        line = fmt.risk_note_line_ru(part)
        if line:
            out.append(line)
    return out


def captain_pair(lu: LineupOut | None) -> list[CaptainOption]:
    """Два лучших варианта капитана по очкам с учётом удвоения (captain_points = 2 · xPts)."""
    if lu is None:
        return []
    return sorted(lu.captain_options, key=lambda o: -o.captain_points)[:2]


def captain_sentence(lu: LineupOut) -> str:
    """«Капитан — Haaland: в среднем 12.4 очка с учётом удвоения, на 2.1 больше, чем у Saka.
    Вице-капитан — Saka.»"""
    pair = captain_pair(lu)
    if not pair:
        return f"Капитан — {lu.captain}, вице-капитан — {lu.vice}."
    first = pair[0]
    text = (
        f"Капитан — {first.name}: в среднем {fmt.points_text(first.captain_points)} "
        "с учётом удвоения"
    )
    if len(pair) > 1:
        gap = round(first.captain_points - pair[1].captain_points, 1)
        gap_t = f"{gap:g}" if gap != int(gap) else f"{int(gap)}"
        text += f", на {gap_t} больше, чем у {pair[1].name}"
    return text + f". Вице-капитан — {lu.vice}."


def captain_change(lu: LineupOut) -> str | None:
    """«Капитаном лучше поставить Saka, а не Haaland: на 2.5 очка больше с учётом удвоения.» —
    если капитан лучшего состава не ваш."""
    cur = (lu.current_captain or "").strip()
    if not cur or cur == lu.captain:
        return None
    pts = {o.name: o.captain_points for o in lu.captain_options}
    text = f"Капитаном лучше поставить {lu.captain}, а не {cur}"
    if lu.captain in pts and cur in pts:
        gap = pts[lu.captain] - pts[cur]
        text += f": на {fmt.points_text(gap)} больше с учётом удвоения"
    return text + "."


def lineup_changes(lu: LineupOut, ctx: GameweekContext) -> tuple[list[str], list[str]]:
    """Чем лучший состав тура отличается от вашего старта: (выпустить со скамейки, убрать на
    скамейку) — только игроки из вашего состава (покупки маршрута сюда не попадают)."""
    squad = {p.name: p for p in ctx.squad or []}
    best = {p.name for p in lu.starters}
    mine = {n for n, p in squad.items() if p.is_starting}
    start = [p.name for p in lu.starters if p.name in squad and p.name not in mine]
    bench = [n for n in (p.name for p in ctx.squad or []) if n in mine and n not in best]
    return start, bench


def captain_ranks(lu: LineupOut, n: int = 3) -> list[dict[str, Any]]:
    """Рейтинг капитанов: имя, клуб, соперник и шанс старта, очки с удвоением, тег по владению."""
    out = []
    for o in sorted(lu.captain_options, key=lambda o: -o.captain_points)[:n]:
        p_start = o.p_start * 100 if o.p_start <= 1 else o.p_start
        out.append(
            {
                "name": o.name,
                "club": o.team,
                "sub": f"{fixture_short(o.fixture)} · шанс старта {p_start:.0f} %",
                "value": f"{o.captain_points:.1f}",
                "tag": fmt.CAPTAIN_TAG_RU.get(o.tag, o.tag),
            }
        )
    return out


def xi_outlook(
    ctx: GameweekContext, preds: Mapping[int, PlayerPrediction], gws: Sequence[int]
) -> list[tuple[str, float]]:
    """Прогноз нынешнего стартового состава по турам без изменений: [(«GW6», сумма xPts,
    капитан ×2)]. Тур без прогноза ни у одного игрока — пропускается."""
    out = []
    for g in gws:
        total, seen = 0.0, False
        for p in ctx.squad or []:
            if not p.is_starting:
                continue
            pred = preds.get(p.id)
            gp = next((x for x in (pred.by_gw if pred else []) if x.gw == g), None)
            if gp is None:
                continue
            seen = True
            total += gp.xpts * (2 if p.is_captain else 1)
        if seen:
            out.append((f"GW{g}", round(total, 1)))
    return out


def plan_vs_hold(plan: PlanOut) -> list[tuple[str, float | None, float | None]]:
    """Прогноз по турам двумя сериями — из одного вызова `build_gameweek_plan`:
    («GW6», если ничего не менять, по плану трансферов с учётом −4 за платные ходы).
    «Ничего не менять» — лучший состав из нынешних 15 на каждый тур (оптимизатор без
    трансферов); «по плану» — состав плана на тур минус штраф за хиты этого тура."""
    gws = sorted(int(g) for g in plan.xi_points_by_gw)
    out = []
    for g in gws:
        k = str(g)
        base = plan.baseline_xi_points_by_gw.get(k)
        xp = plan.xi_points_by_gw.get(k)
        planned = None if xp is None else round(xp - float(plan.hits_by_gw.get(k, 0) or 0), 2)
        out.append((f"GW{g}", base, planned))
    return out


INJURY_RU: tuple[tuple[str, str], ...] = (
    ("hamstring", "задняя поверхность бедра"),
    ("knee", "колено"),
    ("ankle", "голеностоп"),
    ("groin", "пах"),
    ("calf", "икра"),
    ("thigh", "бедро"),
    ("hip", "бедро"),
    ("back injury", "спина"),
    ("back problem", "спина"),
    ("shoulder", "плечо"),
    ("foot", "стопа"),
    ("toe", "стопа"),
    ("head injury", "голова"),
    ("concussion", "сотрясение"),
    ("illness", "болезнь"),
    ("muscle", "мышцы"),
    ("muscular", "мышцы"),
    ("knock", "ушиб"),
    ("suspen", "дисквалификация"),
    ("personal", "личные причины"),
)


def injury_ru(news: str | None) -> str | None:
    """Коротко, что случилось, по-русски из новости FPL: «Knee injury - 75% chance» -> «колено»."""
    low = (news or "").lower()
    return next((ru for en, ru in INJURY_RU if re.search(rf"\b{re.escape(en)}", low)), None)


def watch_filter(
    problems: Sequence[Mapping[str, Any]], ctx: GameweekContext, lu: LineupOut | None
) -> list[Mapping[str, Any]]:
    """Серьёзные проблемы для Брифинга — только те, что влияют на состав тура: игрок в вашем
    старте или в лучшем составе на тур. Запасной с риском ротации (например, дешёвый
    бенч-фоддер с шансом старта 0 %) — не проблема: без него 11 не меняются."""
    starting = {p.id for p in ctx.squad or [] if p.is_starting}
    best = {p.id for p in lu.starters} if lu is not None else set()
    return [pr for pr in problems if int(pr["id"]) in starting | best]


def watch_rows(
    problems: Sequence[Mapping[str, Any]], signals: Mapping[int, PlayerRisk]
) -> list[dict[str, Any]]:
    """Одна строка на игрока: тон, имя, короткий статус («под вопросом (75 %), колено»),
    источник · дата (или «по статусу FPL»)."""
    out = []
    for pr in problems:
        pid = int(pr["id"])
        status = str(pr["label"])
        where = injury_ru(str(pr.get("news") or ""))
        if where and where not in status:
            status += f", {where}"
        meta = "по статусу FPL"
        risk = signals.get(pid)
        if risk is not None and risk.origin != "unavailable" and risk.evidence:
            ev = risk.evidence[0]
            meta = f"{ev.source} · {ev.date}"
        out.append(
            {
                "id": pid,
                "name": str(pr["name"]),
                "tone": str(pr["flag"]),
                "status": status,
                "meta": meta,
            }
        )
    return out


IMPORTANT_AVAILABILITY = frozenset({"doubtful", "injured", "suspended", "unavailable"})
NEWS_DAYS = 10  # новости старше — не в ленте


def is_important(risk: PlayerRisk) -> bool:
    """Важная новость по игроку: разбор новостей говорит о сомнении, травме, дисквалификации,
    недоступности, риске ротации или сроке возвращения — не «в строю / нет данных»."""
    if risk.origin == "unavailable" or not risk.evidence:
        return False
    return (
        risk.availability in IMPORTANT_AVAILABILITY
        or risk.rotation_risk in ("medium", "high")
        or risk.return_gw is not None
    )


def _clip(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip(" ,.;:") + "…"


def news_feed(
    signals: Mapping[int, PlayerRisk],
    names: Mapping[int, str],
    now: Any,
    *,
    days: int = NEWS_DAYS,
    limit: int = 12,
    clip: int = 150,
) -> list[dict[str, Any]]:
    """Лента важных новостей по игрокам состава из сохранённых разборов (без LLM): по каждой
    цитате-доказательству не старше `days` дней — игрок, статус, цитата (обрезанная), источник ·
    дата, ссылка; свежие сверху, дубли цитат убраны."""
    from datetime import datetime, timedelta

    cutoff = now - timedelta(days=days)
    seen: set[tuple[int, str]] = set()
    items = []
    for pid, risk in signals.items():
        if not is_important(risk):
            continue
        status = fmt.AVAILABILITY_RU.get(risk.availability, risk.availability)
        if risk.availability == "fit" and risk.rotation_risk in ("medium", "high"):
            status = "риск ротации"
        tone = "bad" if risk.availability in ("injured", "suspended", "unavailable") else "warn"
        for ev in risk.evidence:
            try:
                published = datetime.fromisoformat(str(ev.published_at))
            except ValueError:
                continue
            if published.tzinfo is None:
                published = published.replace(tzinfo=now.tzinfo)
            if published < cutoff or not ev.quote:
                continue
            key = (pid, ev.quote.strip()[:80])
            if key in seen:
                continue
            seen.add(key)
            items.append(
                {
                    "id": pid,
                    "player": names.get(pid, risk.player),
                    "status": status,
                    "tone": tone,
                    "quote": _clip(ev.quote, clip),
                    "meta": f"{ev.source} · {ev.date}",
                    "url": ev.url,
                    "_at": published,
                }
            )
    items.sort(key=lambda i: i["_at"], reverse=True)
    for i in items:
        i.pop("_at")
    return items[:limit]


def lead_text(
    route: RouteOut | None,
    rt: RoutesOut | None,
    lu: LineupOut | None,
    problems: Sequence[Mapping[str, Any]],
) -> str:
    """Подзаголовок: что сделать до дедлайна — из тех же данных, что карточки."""
    parts = []
    if route is not None:
        parts.append("сделать трансфер")
    elif rt is not None:
        parts.append("решить, держать ли трансфер")
    if lu is not None:
        parts.append("выбрать капитана")
    if problems:
        n = len(problems)
        parts.append(f"проверить {n} {_word(n, 'игрока', 'игроков', 'игроков')} с проблемами")
    if not parts:
        return "Данных для брифинга пока нет."
    if len(parts) == 1:
        body = parts[0]
    else:
        body = ", ".join(parts[:-1]) + " и " + parts[-1]
    return f"До дедлайна нужно {body}."
