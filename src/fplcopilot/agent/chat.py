"""Чат v4: история диалога, служебные тексты на языке вопроса, факты и headline новых интентов.

Чистые функции без сети / БД / LLM (граф вызывает их из узлов, тесты — напрямую):

- история: `turn_meta` (что этот ход решил: интент, игроки, тур, фишки, ограничения) уходит
  в следующий ход вместе с репликами; `render_history` — компактный текст для роутера и
  объяснителя (роутер v4 переписывает уточнение «а на gw7?» в самостоятельный вопрос);
- служебные тексты: отказ (off_topic / betting), фолбэк при сбое объяснителя, оговорка валидатора —
  на языке вопроса (ru / en), без технических подробностей (они — в футере «Как получен ответ»);
- новые интенты v4 (squad_review, fixtures, chips, general_fpl) и прогнозный рейтинг: компактные
  FACTS из вывода инструментов и детерминированный headline — числа только из инструментов.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

HISTORY_MESSAGES = 6  # последних реплик (3 хода) в историю роутера / объяснителя
HISTORY_CHARS = 400  # обрезка текста одной реплики ассистента (вердикт + начало)
CHIP_CODES: dict[str, str] = {
    "bboost": "bboost",
    "bench_boost": "bboost",
    "bench boost": "bboost",
    "bb": "bboost",
    "3xc": "3xc",
    "triple_captain": "3xc",
    "triple captain": "3xc",
    "tc": "3xc",
    "wildcard": "wildcard",
    "wc": "wildcard",
    "freehit": "freehit",
    "free_hit": "freehit",
    "free hit": "freehit",
    "fh": "freehit",
}
PLAN_CHIPS = frozenset({"bboost", "3xc"})  # фишки, которые build_gameweek_plan ставит в тур
CHIP_NAMES = {
    "wildcard": "Wildcard",
    "freehit": "Free Hit",
    "bboost": "Bench Boost",
    "3xc": "Triple Captain",
}

# Русские названия клубов (основы, падеж снимается translit.case_stems) -> код FPL.
RU_CLUBS: dict[str, str] = {
    "арсенал": "ARS", "астон": "AVL", "вилла": "AVL", "борнмут": "BOU", "брентфорд": "BRE",
    "брайтон": "BHA", "бернли": "BUR", "челси": "CHE", "ковентри": "COV", "кристал": "CRY",
    "пэлас": "CRY", "эвертон": "EVE", "фулхэм": "FUL", "фулхем": "FUL", "халл": "HUL",
    "ипсвич": "IPS", "лидс": "LEE", "ливерпуль": "LIV", "сити": "MCI", "ман сити": "MCI",
    "манчестер сити": "MCI", "юнайтед": "MUN", "мю": "MUN", "манчестер юнайтед": "MUN",
    "ньюкасл": "NEW", "ноттингем": "NFO", "форест": "NFO", "тоттенхэм": "TOT",
    "тоттенхем": "TOT", "шпоры": "TOT", "сандерленд": "SUN", "вест хэм": "WHU", "вулверхэмптон": "WOL",
    "волки": "WOL", "лестер": "LEI",
}  # fmt: skip


# ---------- история ----------


def chip_code(raw: Any) -> str | None:
    if raw is None:
        return None
    return CHIP_CODES.get(str(raw).strip().lower())


def turn_meta(state: dict[str, Any]) -> dict[str, Any]:
    """Что решил этот ход — передаётся в историю следующего (только JSON-значения)."""
    players = {int(p["id"]): p.get("name") for p in state.get("players") or []}
    sc = state.get("scenario") or {}
    meta: dict[str, Any] = {
        "intent": state.get("intent"),
        "question": state.get("standalone_query") or state.get("query"),
        "players": [p.get("name") for p in state.get("players") or []],
        "gw": state.get("gw"),
        "target_gw": state.get("target_gw"),
        "horizon": state.get("horizon"),
        "chips": list(state.get("chips_asked") or []),
        "keep": [players.get(int(x), x) for x in sc.get("keep") or []],
        "sell": [players.get(int(x), x) for x in sc.get("sell") or []],
        "buy": [players.get(int(x), x) for x in sc.get("buy") or []],
        "allow_hit": sc.get("allow_hit"),
        "use_wildcard": sc.get("use_wildcard"),
        "teams": list(state.get("team_mentions") or []),
    }
    if state.get("ranking"):
        meta["ranking"] = {k: v for k, v in state["ranking"].items() if v is not None}
    headline = (state.get("facts") or {}).get("headline")
    if headline:
        meta["headline"] = str(headline)[:300]
    return {k: v for k, v in meta.items() if v not in (None, [], {}, "")}


def _short(text: str, limit: int = HISTORY_CHARS) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_history(history: Sequence[dict[str, Any]] | None) -> str:
    """Последние HISTORY_MESSAGES реплик: пользователь — текст, ассистент — начало ответа и
    meta хода. Пусто -> ''."""
    lines: list[str] = []
    for turn in list(history or [])[-HISTORY_MESSAGES:]:
        role = turn.get("role")
        if role == "user":
            lines.append(f"user: {_short(str(turn.get('content') or ''), 300)}")
        elif role == "assistant":
            meta = turn.get("meta") or {}
            ctx = "; ".join(
                f"{k}={v}" for k, v in meta.items() if k not in ("question", "headline")
            )
            head = f"assistant [{ctx}]" if ctx else "assistant"
            body = _short(str(turn.get("content") or ""))
            lines.append(f"{head}: {body}")
            if meta.get("headline"):
                lines.append(f"  (computed verdict of that turn: {meta['headline']})")
    return "\n".join(lines)


def history_from_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Сообщения страницы чата (st.session_state) -> история агента: реплики пользователя и
    ответы ассистента с meta их хода (`final_state` после HITL важнее исходного `state`)."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "user":
            out.append({"role": "user", "content": str(m.get("content") or "")})
        elif m.get("role") == "assistant":
            st = m.get("final_state") or m.get("state") or {}
            out.append(
                {
                    "role": "assistant",
                    "content": str(st.get("answer") or ""),
                    "meta": turn_meta(st),
                }
            )
    return out[-HISTORY_MESSAGES:]


# ---------- служебные тексты на языке вопроса ----------


def _ru(language: str | None) -> bool:
    return (language or "en") == "ru"


def refusal_text(intent: str | None, language: str | None) -> str:
    ru = _ru(language)
    if intent == "betting":
        if ru:
            return (
                "Советов по ставкам я не даю — ни коэффициентов, ни букмекеров, ни прогнозов на "
                "исход матча.\n\n"
                "Зато могу помочь с Fantasy Premier League: состояние и прогноз очков игрока, "
                "трансферы (с оценкой хита), капитан, стартовый состав, план на несколько туров и "
                "фишки, календарь клубов, рейтинги игроков. Например: «Кого поставить капитаном в "
                "этом туре?»"
            )
        return (
            "I don't give betting advice — no odds, bookmakers or match-result tips.\n\n"
            "What I can do instead: a player's fitness and expected FPL points, transfers (with hit "
            "verdicts), captaincy, the starting XI, multi-gameweek plans and chips, club fixtures, "
            'player rankings. For example: "Who should I captain this week?"'
        )
    if ru:
        return (
            "Я помогаю только с Fantasy Premier League, а этот вопрос не про неё — спросите, "
            "например, «Кого продать в этом туре?» или «С кем играет Арсенал в ближайшие 3 тура?»"
        )
    return (
        "I only help with Fantasy Premier League, and this question is outside it — try, for "
        'example, "Who should I sell this week?" or "Arsenal fixtures for the next 3 gameweeks?"'
    )


def explain_fallback_text(
    headline: str | None, caveats: Sequence[str], language: str | None
) -> str:
    """Ответ без LLM, когда объяснитель недоступен: вердикт кода и оговорки, без JSON."""
    ru = _ru(language)
    lines = [
        (
            "Модель объяснения сейчас недоступна — вот вывод расчёта без пояснений."
            if ru
            else "The explanation model is unavailable right now — here is the computed result."
        ),
        "",
        f"**{'Расчёт' if ru else 'Result'}:** {headline or ('нет данных' if ru else 'n/a')}",
    ]
    if caveats:
        lines += ["", f"**{'Оговорки' if ru else 'Caveats'}:**"] + [f"- {c}" for c in caveats]
    lines += [
        "",
        "Попробуйте повторить вопрос через минуту." if ru else "Please retry in a minute.",
    ]
    return "\n".join(lines)


def validation_note(language: str | None, flagged: Sequence[str]) -> str:
    """Оговорка к ответу, который после регенерации всё ещё не сошёлся с расчётом."""
    items = ", ".join(flagged)
    if _ru(language):
        return (
            f"\n\n_Часть значений ({items}) не удалось сверить с расчётом — перепроверьте их; "
            "подробности — в «Как получен ответ»._"
        )
    return (
        f"\n\n_Some values ({items}) could not be matched to the computed facts — double-check "
        "them; details are under “How the answer was produced”._"
    )


def not_in_squad_note(name: str, role: str, language: str | None) -> str:
    """Оговорка: условие «не продавать» / «продать» про игрока, которого нет в составе."""
    if _ru(language):
        what = "не продавать" if role == "keep" else "продать"
        return f"{name} нет в вашем составе — условие «{what} {name}» не применено."
    what = "keep" if role == "keep" else "sell"
    return f"{name} is not in your squad — the '{what} {name}' condition was not applied."


# ---------- клубы ----------


def resolve_teams(mentions: Iterable[str], teams: Sequence[Any]) -> tuple[list[int], list[str]]:
    """Упоминания клубов -> id bootstrap: код / имя / прозвище (TEAM_ALIASES), русское имя в
    любом падеже. (ids, не найденные)."""
    from fplcopilot.agent.translit import case_stems
    from fplcopilot.rag.entity_matcher import TEAM_ALIASES, strip_accents

    by_code = {t.short_name.upper(): t.id for t in teams}
    names: dict[str, int] = {}
    for t in teams:
        names[strip_accents(t.name).lower()] = t.id
        names[t.short_name.lower()] = t.id
        for alias in TEAM_ALIASES.get(t.short_name, ()):
            names[strip_accents(alias).lower()] = t.id
    ids: list[int] = []
    missing: list[str] = []
    for raw in mentions:
        m = strip_accents(str(raw)).strip().lower()
        if not m:
            continue
        tid = names.get(m) or by_code.get(m.upper())
        if tid is None:
            words = re.findall(r"[а-яё]+", m)
            for i in range(len(words), 0, -1):
                for j in range(len(words) - i + 1):
                    phrase = words[j : j + i]
                    stems = [case_stems(w) for w in phrase]
                    for combo in _combos(stems):
                        code = RU_CLUBS.get(" ".join(combo))
                        if code and code in by_code:
                            tid = by_code[code]
                            break
                    if tid:
                        break
                if tid:
                    break
        if tid is None:
            for name, i in names.items():
                if len(m) >= 4 and (m in name or name in m):
                    tid = i
                    break
        if tid is None:
            missing.append(str(raw))
        elif tid not in ids:
            ids.append(tid)
    return ids, missing


def _combos(stems: list[list[str]]) -> list[list[str]]:
    out: list[list[str]] = [[]]
    for options in stems:
        out = [prev + [o] for prev in out for o in (options or [""])][:64]
    return out


# ---------- факты новых интентов ----------


def fixtures_facts(out: Any, requested: Sequence[int]) -> dict[str, Any]:
    """FACTS.fixtures: запрошенные клубы — полностью, плюс топ-5 лёгких и 3 тяжёлых календаря."""
    teams = list(out.teams)
    rows = {t.team_id: t for t in teams}
    asked = [rows[i] for i in requested if i in rows]

    def row(t: Any) -> dict[str, Any]:
        return {
            "rank_easiest": t.rank,
            "team": t.team,
            "club": t.team_name,
            "fixtures": t.fixtures,
            "matches": t.matches,
            "mean_fsi": t.mean_fsi,
        }

    return {
        "window": f"GW{out.gw_from}–GW{out.gw_to}"
        if out.gw_to > out.gw_from
        else f"GW{out.gw_from}",
        "ranking_rule": out.ranking_rule,
        "teams_asked": [row(t) for t in asked],
        "easiest_runs": [row(t) for t in teams[:5]],
        "hardest_runs": [row(t) for t in teams[-3:][::-1]],
        "clubs_ranked": len(teams),
    }


def forecast_ranking_facts(out: Any) -> dict[str, Any]:
    return {
        "kind": "forecast (future): expected points from the xPts model, not past points",
        "window": f"GW{out.gw_from}–GW{out.gw_to}"
        if out.gw_to > out.gw_from
        else f"GW{out.gw_from}",
        "metric": out.metric,
        "metric_label": out.metric_label,
        "filters": out.filters,
        "filters_definition": out.filters_definition,
        "candidates_after_filters": out.candidates,
        "rows": [
            {
                "rank": r.rank,
                "player": r.name,
                "full_name": r.full_name,
                "team": r.team,
                "position": r.position,
                "price": r.price,
                "ownership_pct": r.ownership,
                "xpts_total": r.xpts_total,
                "xpts_per_million": r.xpts_per_million,
                "xpts_by_gw": r.xpts_by_gw,
                "p_start_next_gw": r.p_start_next,
                "fixtures": r.fixtures,
                "in_your_squad": r.in_squad,
            }
            for r in out.rows
        ],
    }


def chips_facts(out: Any) -> dict[str, Any]:
    return {
        "gw": out.gw,
        "available_now": out.available_now or [],
        "chips": [
            {
                "chip": c.name,
                "available_in_gw": c.available_now,
                "played_in": [f"GW{g}" for g in c.played_gws],
                "current_window": c.window_now,
                "available_again_from": (
                    f"GW{c.next_available_gw}" if c.next_available_gw else None
                ),
            }
            for c in out.chips
        ],
        "rule": "; ".join(out.notes),
    }


def trends_facts(out: Any) -> dict[str, Any]:
    def row(r: Any) -> dict[str, Any]:
        return {
            "player": r.name,
            "team": r.team,
            "position": r.position,
            "price": r.price,
            "ownership_pct": r.ownership,
            "net_transfers": r.net_transfers_event,
            "price_change_already_happened_this_gw": r.price_change_event,
        }

    return {
        "gw": out.gw,
        "most_transferred_in": [row(r) for r in out.most_transferred_in],
        "most_transferred_out": [row(r) for r in out.most_transferred_out],
        "note": "; ".join(out.notes),
    }


def gameweek_facts(state: dict[str, Any]) -> dict[str, Any]:
    summary = state.get("squad_summary") or {}
    out: dict[str, Any] = {
        "next_gw": state.get("gw"),
        "deadline_utc": state.get("deadline"),
        "last_finished_gw": state.get("current_gw"),
    }
    if summary:
        out.update(
            {
                "free_transfers": summary.get("free_transfers"),
                "bank": summary.get("bank"),
            }
        )
    return out


def gw_review_facts(out: Any) -> dict[str, Any]:
    """FACTS.gw_review: разбор завершённого тура — факт, прогноз до дедлайна, роль в составе."""
    rows = list(out.rows)
    starters = [r for r in rows if r.role != "bench"]
    scored = [r for r in starters if r.diff_vs_forecast is not None]
    below = sorted(scored, key=lambda r: r.diff_vs_forecast)[:3]
    above = sorted(scored, key=lambda r: -r.diff_vs_forecast)[:3]

    def line(r: Any) -> str:
        pts = "no data" if r.points is None else f"{r.points} pts"
        if r.multiplier > 1 and r.counted_points is not None:
            pts += f" (x{r.multiplier} = {r.counted_points})"
        fc = "" if r.xpts_forecast is None else f", forecast {r.xpts_forecast} xPts"
        mins = "" if r.minutes is None else f", {r.minutes} min"
        return f"{r.name} ({r.team}, {r.position}, {r.role}): {pts}{mins}{fc}"

    facts: dict[str, Any] = {
        "gw": f"GW{out.gw}",
        "available": bool(out.finished and out.history_available),
        "manager_points": out.manager_points,
        "average_points_of_all_managers": out.average_points,
        "transfers_cost": out.transfers_cost,
        "points_left_on_bench": out.points_on_bench,
        "active_chip": out.active_chip,
        "captain": out.captain,
        "captain_points_counted": out.captain_points,
        "best_starter": out.best_starter,
        "best_starter_points": out.best_starter_points,
        "best_bench_player": out.best_bench,
        "best_bench_points": out.best_bench_points,
        "players": [line(r) for r in rows],
        "biggest_shortfalls_vs_forecast": [
            f"{r.name}: {r.points} pts vs forecast {r.xpts_forecast} ({r.diff_vs_forecast:+})"
            for r in below
            if r.diff_vs_forecast < 0
        ],
        "biggest_overperformers_vs_forecast": [
            f"{r.name}: {r.points} pts vs forecast {r.xpts_forecast} ({r.diff_vs_forecast:+})"
            for r in above
            if r.diff_vs_forecast > 0
        ],
        "forecast_available": out.forecast_available,
        "notes": list(out.notes),
        "rule": (
            "facts of a finished gameweek; say what happened, never why a player scored (no "
            "match events here) and never what the manager 'should have known'"
        ),
    }
    return facts


def price_facts() -> dict[str, Any]:
    return {
        "price_change_prediction_available": False,
        "note": (
            "price changes are NOT predicted by this system; transfer_trends shows transfer "
            "activity and the price changes that already happened this gameweek"
        ),
    }


# ---------- headline новых интентов ----------


def headline_v4(intent: str, facts: dict[str, Any]) -> str | None:
    gw = facts.get("gw")
    if intent == "squad_review" and facts.get("squad_review"):
        sr = facts["squad_review"]
        issues = sr.get("issues") or []
        line = (
            f"Squad review GW{gw}: {len(issues)} issue(s)"
            + (f" ({'; '.join(issues[:4])})" if issues else " (no flagged problems)")
            + f"; best XI {sr.get('best_xi_xpts')} xPts next GW vs your current XI "
            f"{sr.get('current_xi_xpts')}"
        )
        fix = sr.get("best_fix")
        if fix:
            line += (
                f"; best fix {', '.join(fix['out'])} -> {', '.join(fix['in'])} "
                f"({fix['gain_horizon']:+} xPts over {sr.get('horizon_gws')} GW, hit "
                f"{fix['hit_cost']})"
            )
        else:
            line += "; no transfer beats rolling"
        return line
    if intent == "fixtures" and facts.get("fixtures"):
        fx = facts["fixtures"]
        asked = fx.get("teams_asked") or []
        if asked:
            return "; ".join(
                f"{t['club']} {fx['window']}: "
                + ", ".join(f"{g} {v}" for g, v in t["fixtures"].items())
                + f" (mean FSI {t['mean_fsi']}, rank {t['rank_easiest']} of {fx['clubs_ranked']})"
                for t in asked
            )
        easy = fx.get("easiest_runs") or []
        return f"Easiest fixture runs {fx['window']}: " + "; ".join(
            f"{t['rank_easiest']}. {t['club']} (mean FSI {t['mean_fsi']})" for t in easy[:3]
        )
    if intent == "chips" and facts.get("chips_status"):
        cs = facts["chips_status"]
        avail = cs.get("available_now") or []
        used = [
            f"{c['chip']} played in {', '.join(c['played_in'])}"
            + (
                f", available again from {c['available_again_from']}"
                if c.get("available_again_from")
                else ""
            )
            for c in cs["chips"]
            if not c["available_in_gw"]
        ]
        line = f"Chips for GW{cs['gw']}: available {', '.join(avail) if avail else 'none'}"
        if used:
            line += "; unavailable: " + "; ".join(used)
        if facts.get("chip_plan"):
            line += f"; {facts['chip_plan']['summary']}"
        return line
    if intent == "gw_review" and facts.get("gw_review"):
        rv = facts["gw_review"]
        if not rv.get("available"):
            return (
                f"Review of {rv.get('gw')}: no data ({'; '.join(rv.get('notes') or []) or rv.get('reason')})"
                " — nothing is reviewed, nothing is invented"
            )
        line = (
            f"Review of {rv['gw']} (facts): you scored {rv.get('manager_points')} pts vs average "
            f"{rv.get('average_points_of_all_managers')}; captain {rv.get('captain')} "
            f"{rv.get('captain_points_counted')} pts counted; best starter {rv.get('best_starter')} "
            f"{rv.get('best_starter_points')}; points left on bench {rv.get('points_left_on_bench')}"
        )
        if rv.get("biggest_shortfalls_vs_forecast"):
            line += "; below forecast: " + "; ".join(rv["biggest_shortfalls_vs_forecast"])
        return line
    if intent == "general_fpl":
        gwf = facts.get("gameweek") or {}
        line = f"Gameweek context: next GW{gwf.get('next_gw')}, deadline {gwf.get('deadline_utc')}"
        if gwf.get("free_transfers") is not None:
            line += f", your free transfers {gwf.get('free_transfers')}, bank £{gwf.get('bank')}"
        if facts.get("transfer_trends"):
            top = facts["transfer_trends"]["most_transferred_in"][:3]
            line += "; most transferred in (activity, not a price forecast): " + ", ".join(
                f"{r['player']} ({r['net_transfers']:+})" for r in top
            )
        if facts.get("price_changes"):
            line += "; price changes are NOT predicted"
        return line
    if facts.get("forecast_ranking"):
        fr = facts["forecast_ranking"]
        rows = fr.get("rows") or []
        if not rows:
            return f"No player passes the filters ({fr['filters_definition']}), {fr['window']}"
        return f"Top {len(rows)} by {fr['metric_label']}: " + "; ".join(
            f"{r['rank']}. {r['player']} ({r['team']}, {r['position']}, £{r['price']}, "
            f"{r['xpts_total']} xPts, {r['xpts_per_million']} xPts/£m)"
            for r in rows[:3]
        )
    return None
