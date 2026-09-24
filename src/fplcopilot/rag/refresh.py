"""Пакетное обновление новостных сигналов и дайджестов клубов (перед дедлайном / демо).

    uv run python -m fplcopilot.rag.refresh --manager 895045 --dry-run
    uv run python -m fplcopilot.rag.refresh --manager 895045 --max-players 40 --max-usd 0.10
    uv run python -m fplcopilot.rag.refresh --max-players 10 --no-teams --mode hybrid_rerank

Зачем: сигнал игрока создаётся только по запросу (кнопка на карточке, упоминание в чате), а таблица
состава и карточка читают сохранённый сигнал любого возраста — у большинства игроков прогноз
фактически без новостей. Здесь сигналы обновляются пачкой, в порядке важности:

  1. состав менеджера (--manager): сигнал старше --max-age-h (как AGENT_SIGNAL_MAX_AGE_H, 12 ч);
  2. игроки со статусом FPL d/i/s/u (кроме ушедших из клуба): сигнал старше --max-age-h или
     официальная новость FPL новее сигнала;
  3. игроки, у которых после их последнего сигнала появились статьи с их тегом
     (news_articles.players, по времени загрузки fetched_at, окно --since-days) — по числу статей;

затем дайджесты клубов этих игроков (rag/team_news.py), если сохранённый старше --max-age-h.
Лимиты: --max-players, --max-teams, --max-usd (оценка по токенам; следующий вызов не начинается,
если бюджет его не покрывает). Режим по умолчанию — dense (docs/EVALS.md §3.3 / §4.3: для
пакетного пути точность dense не ниже hybrid_rerank, retrieval в ~3 раза дешевле, без reranker).
Abstention (нет статей про игрока) стоит 0 — LLM не вызывается.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text

from fplcopilot.config import settings
from fplcopilot.data import Bootstrap, FPLClient
from fplcopilot.db import session_scope
from fplcopilot.rag.query import parse_as_of
from fplcopilot.rag.retrieve import MODES, Retriever

log = logging.getLogger(__name__)

EST_CALL_USD = 0.0009  # один вызов gpt-4o-mini ≈ 4.5k входных + 0.2k выходных токенов (docs/rag.md)
PRICE_IN, PRICE_OUT = 0.15, 0.60  # $/1M токенов gpt-4o-mini
REFRESH_STATUSES = ("d", "i", "s", "u")

_LAST_SIGNALS = text(
    "SELECT player_id, max(as_of) FROM player_signals WHERE as_of <= :as_of GROUP BY player_id"
)
_LAST_DIGESTS = text(
    "SELECT team_id, max(as_of) FROM team_news_digests WHERE as_of <= :as_of GROUP BY team_id"
)
_NEW_ARTICLES = text(
    """
    SELECT p.pid, count(*) AS n_new
    FROM news_articles a CROSS JOIN LATERAL unnest(a.players) AS p(pid)
    LEFT JOIN (
        SELECT player_id, max(as_of) AS last_sig FROM player_signals
        WHERE as_of <= :as_of GROUP BY player_id
    ) s ON s.player_id = p.pid
    WHERE a.published_at <= :as_of AND a.fetched_at >= :since AND a.fetched_at <= :as_of
      AND (s.last_sig IS NULL OR a.fetched_at > s.last_sig)
    GROUP BY p.pid
    ORDER BY n_new DESC, p.pid
    """
)


@dataclass
class Target:
    player_id: int
    reason: str


@dataclass
class RefreshReport:
    players: list[dict[str, Any]] = field(default_factory=list)
    teams: list[dict[str, Any]] = field(default_factory=list)
    spent_usd: float = 0.0
    llm_calls: int = 0
    skipped_budget: int = 0
    seconds: float = 0.0


def call_cost(timings: dict[str, Any]) -> float:
    return (
        float(timings.get("prompt_tokens") or 0) * PRICE_IN
        + float(timings.get("completion_tokens") or 0) * PRICE_OUT
    ) / 1_000_000


def select_players(
    bs: Bootstrap,
    *,
    as_of: datetime,
    last_signal: dict[int, datetime],
    new_articles: list[tuple[int, int]],
    squad_ids: list[int],
    max_age: timedelta,
    max_players: int,
) -> list[Target]:
    """Чистая функция отбора (см. докстринг модуля): состав -> статусы FPL -> новые статьи."""
    from fplcopilot.rag.team_news import _DEPARTED

    def stale(pid: int) -> bool:
        last = last_signal.get(pid)
        return last is None or as_of - last > max_age

    out: list[Target] = []
    seen: set[int] = set()

    def add(pid: int, reason: str) -> None:
        if pid not in seen and len(out) < max_players:
            seen.add(pid)
            out.append(Target(pid, reason))

    known = {p.id for p in bs.elements}
    for pid in squad_ids:
        if pid in known and stale(pid):
            add(pid, "squad")
    flagged = sorted(
        (p for p in bs.elements if p.status in REFRESH_STATUSES),
        key=lambda p: -float(p.selected_by_percent or 0),
    )
    for p in flagged:
        if p.status == "u" and _DEPARTED.search(p.news or ""):
            continue
        last = last_signal.get(p.id)
        news_newer = p.news_added is not None and last is not None and p.news_added > last
        if stale(p.id) or news_newer:
            add(p.id, f"fpl status {p.status}")
    for pid, n in new_articles:
        if pid in known:
            add(pid, f"{n} new article(s)")
    return out


def _squad_ids(client: FPLClient, bs: Bootstrap, manager_id: int | None) -> list[int]:
    if not manager_id:
        return []
    gw = bs.current_event.id if bs.current_event else 1
    try:
        return [sp.player.id for sp in client.squad(manager_id, gw).players]
    except Exception as exc:  # noqa: BLE001 — нет публичных picks: только статусы и статьи
        log.warning("squad of manager %s unavailable: %s", manager_id, exc)
        return []


def refresh(
    *,
    as_of: datetime,
    manager_id: int | None = None,
    max_players: int = 25,
    max_teams: int = 6,
    max_usd: float = 0.05,
    max_age_h: float | None = None,
    since_days: float = 7.0,
    mode: str = "dense",
    k: int = 8,
    teams: bool = True,
    dry_run: bool = False,
) -> RefreshReport:
    from fplcopilot.rag.extract import extract_signal
    from fplcopilot.rag.team_news import extract_team_news

    started = time.perf_counter()
    max_age = timedelta(
        hours=max_age_h if max_age_h is not None else settings.agent_signal_max_age_h
    )
    client = FPLClient()
    bs = client.bootstrap()
    with session_scope() as s:
        last_signal = {int(p): d for p, d in s.execute(_LAST_SIGNALS, {"as_of": as_of})}
        last_digest = {int(t): d for t, d in s.execute(_LAST_DIGESTS, {"as_of": as_of})}
        new_articles = [
            (int(p), int(n))
            for p, n in s.execute(
                _NEW_ARTICLES, {"as_of": as_of, "since": as_of - timedelta(days=since_days)}
            )
        ]
    squad = _squad_ids(client, bs, manager_id)
    targets = select_players(
        bs,
        as_of=as_of,
        last_signal=last_signal,
        new_articles=new_articles,
        squad_ids=squad,
        max_age=max_age,
        max_players=max_players,
    )
    club_order: list[int] = []
    for pid in [t.player_id for t in targets] + squad:
        tid = bs.player(pid).team
        last = last_digest.get(tid)
        fresh = last is not None and as_of - last <= max_age
        if tid not in club_order and not fresh:
            club_order.append(tid)
    club_order = club_order[:max_teams] if teams else []

    rep = RefreshReport()
    if dry_run:
        rep.players = [
            {"player": bs.player(t.player_id).web_name, "reason": t.reason, "status": "planned"}
            for t in targets
        ]
        rep.teams = [{"team": bs.team(t).short_name, "status": "planned"} for t in club_order]
        rep.spent_usd = EST_CALL_USD * (len(targets) + len(club_order))  # верхняя оценка
        return rep

    retriever = Retriever()
    for t in targets:
        if rep.spent_usd + EST_CALL_USD > max_usd:
            rep.skipped_budget += 1
            continue
        timings: dict[str, Any] = {}
        name = bs.player(t.player_id).web_name
        try:
            sig = extract_signal(
                t.player_id,
                as_of,
                mode=mode,  # type: ignore[arg-type]
                k=k,
                retriever=retriever,
                bs=bs,
                save=True,
                timings=timings,
            )
        except Exception as exc:  # noqa: BLE001 — один игрок не валит пачку
            log.warning("refresh %s failed: %s", name, exc)
            rep.players.append({"player": name, "reason": t.reason, "status": f"error {exc}"})
            continue
        rep.spent_usd += call_cost(timings)
        rep.llm_calls += int(timings.get("llm_calls") or 0)
        rep.players.append(
            {
                "player": name,
                "reason": t.reason,
                "status": "abstained" if sig.abstained else sig.availability,
                "confidence": sig.confidence,
                "evidence": len(sig.evidence),
                "ms": round(float(timings.get("total_ms") or 0)),
            }
        )
    for tid in club_order:
        if rep.spent_usd + EST_CALL_USD > max_usd:
            rep.skipped_budget += 1
            continue
        timings = {}
        try:
            d = extract_team_news(
                tid,
                as_of,
                mode=mode,
                k=k,
                retriever=retriever,
                bs=bs,
                timings=timings,  # type: ignore[arg-type]
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("refresh club %s failed: %s", tid, exc)
            rep.teams.append({"team": bs.team(tid).short_name, "status": f"error {exc}"})
            continue
        rep.spent_usd += call_cost(timings)
        rep.llm_calls += int(timings.get("llm_calls") or 0)
        rep.teams.append(
            {
                "team": bs.team(tid).short_name,
                "status": "no club news" if not d.items else f"{len(d.items)} item(s)",
                "ms": round(float(timings.get("total_ms") or 0)),
            }
        )
    rep.seconds = time.perf_counter() - started
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FPL Copilot: пакетное обновление новостных сигналов")
    ap.add_argument("--manager", type=int, default=settings.fpl_manager_id, help="состав первым")
    ap.add_argument("--max-players", type=int, default=25)
    ap.add_argument("--max-teams", type=int, default=6)
    ap.add_argument("--max-usd", type=float, default=0.05, help="бюджет LLM на прогон (оценка)")
    ap.add_argument(
        "--max-age-h", type=float, default=None, help="по умолчанию AGENT_SIGNAL_MAX_AGE_H"
    )
    ap.add_argument("--since-days", type=float, default=7.0, help="окно новых статей")
    ap.add_argument("--mode", choices=MODES, default="dense")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--no-teams", action="store_true", help="без дайджестов клубов")
    ap.add_argument("--as-of", dest="as_of", help="ISO-время; по умолчанию сейчас")
    ap.add_argument("--dry-run", action="store_true", help="показать план и верхнюю оценку цены")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    rep = refresh(
        as_of=parse_as_of(args.as_of) if args.as_of else datetime.now(UTC),
        manager_id=args.manager,
        max_players=args.max_players,
        max_teams=args.max_teams,
        max_usd=args.max_usd,
        max_age_h=args.max_age_h,
        since_days=args.since_days,
        mode=args.mode,
        k=args.k,
        teams=not args.no_teams,
        dry_run=args.dry_run,
    )
    head = "PLAN (dry run)" if args.dry_run else "DONE"
    print(f"{head}: {len(rep.players)} player(s), {len(rep.teams)} club(s)")
    for p in rep.players:
        extra = "".join(
            f" {k}={p[k]}" for k in ("confidence", "evidence", "ms") if p.get(k) is not None
        )
        print(f"  {p['player']:<18} [{p['reason']}] -> {p['status']}{extra}")
    for t in rep.teams:
        print(f"  club {t['team']:<5} -> {t['status']}" + (f" ms={t['ms']}" if t.get("ms") else ""))
    cost = "≤ " if args.dry_run else "≈ "
    print(
        f"LLM calls {rep.llm_calls}, cost {cost}${rep.spent_usd:.4f} (budget ${args.max_usd}), "
        f"skipped by budget {rep.skipped_budget}, {rep.seconds:.0f} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
