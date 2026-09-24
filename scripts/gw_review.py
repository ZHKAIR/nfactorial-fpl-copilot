"""Разбор тура после финального свистка: xPts v0 против ep_next FPL против факта.

    uv run python scripts/gw_review.py --gw 5 [--model-version v0] [--sync]

Берёт последний прогноз каждого игрока из xpts_predictions, сделанный ДО дедлайна тура,
подтягивает фактические очки из player_gw_history (при --sync сначала обновляет историю
свежими element-summary) и печатает MAE / RMSE / Spearman / top-20 для модели и для
ep_next_snapshot, а также суммы по компонентам (где именно модель промахнулась).
Пока тур не завершён — сообщает об этом и выходит.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict

from sqlalchemy import text

from fplcopilot.core.backtest import MethodResult, evaluate, format_results
from fplcopilot.core.history import actual_points, load_history, sync_history
from fplcopilot.core.scoring import points_breakdown
from fplcopilot.data import FPLClient
from fplcopilot.db import session_scope

log = logging.getLogger("gw_review")

_LATEST_BEFORE_DEADLINE = text(
    """
    SELECT DISTINCT ON (player_id)
        player_id, xpts, ep_next_snapshot, p_start, exp_minutes, components, as_of
    FROM xpts_predictions
    WHERE gw = :gw AND model_version = :mv AND as_of <= :deadline
    ORDER BY player_id, as_of DESC, id DESC
    """
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FPL Copilot: обзор тура xPts vs ep_next vs факт")
    ap.add_argument("--gw", type=int, required=True)
    ap.add_argument("--model-version", default="v0")
    ap.add_argument("--sync", action="store_true", help="обновить player_gw_history перед разбором")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    client = FPLClient(cache_ttl=0) if args.sync else FPLClient()
    bs = client.bootstrap()
    event = next((e for e in bs.events if e.id == args.gw), None)
    if event is None:
        print(f"GW{args.gw} не найден")
        return 2
    if not event.finished:
        print(
            f"GW{args.gw} not finished (deadline {event.deadline_time:%Y-%m-%d %H:%M}Z, "
            f"finished={event.finished}). Запустите после завершения тура: "
            f"uv run python scripts/gw_review.py --gw {args.gw} --sync"
        )
        return 0

    if args.sync:
        print(f"sync: {sync_history(client)}")
    actual = actual_points(args.gw)
    if not actual:
        print("В player_gw_history нет строк за этот тур — запустите с --sync")
        return 1

    with session_scope() as s:
        rows = s.execute(
            _LATEST_BEFORE_DEADLINE,
            {"gw": args.gw, "mv": args.model_version, "deadline": event.deadline_time},
        ).all()
    if not rows:
        print(f"Нет прогнозов {args.model_version} на GW{args.gw}, сделанных до дедлайна")
        return 1

    xpts = {int(r[0]): float(r[1]) for r in rows}
    ep_next = {int(r[0]): float(r[2]) for r in rows if r[2] is not None}
    components = {int(r[0]): dict(r[5]) for r in rows}
    as_of = max(r[6] for r in rows)
    print(
        f"GW{args.gw}: {len(rows)} прогнозов {args.model_version} (последний as_of "
        f"{as_of:%Y-%m-%d %H:%M}Z), {len(ep_next)} снимков ep_next, факт по {len(actual)} игрокам\n"
    )

    results: list[MethodResult] = [evaluate(args.gw, f"xpts_{args.model_version}", xpts, actual)]
    if ep_next:
        results.append(evaluate(args.gw, "ep_next", ep_next, actual))
    print(format_results(results))

    # Суммы по компонентам: где модель систематически ошибается.
    pred_sum: dict[str, float] = defaultdict(float)
    act_sum: dict[str, float] = defaultdict(float)
    history = [r for r in load_history() if r.round == args.gw]
    for pid, comps in components.items():
        if pid in actual:
            for k, v in comps.items():
                pred_sum[k] += float(v)
    for r in history:
        if r.element in xpts:
            for k, v in points_breakdown(r, bs.player(r.element).position).items():
                act_sum[k] += v
    print("\ncomponent      predicted    actual")
    for k, v in pred_sum.items():
        print(f"{k:<14} {v:>9.1f} {act_sum.get(k, 0.0):>9.1f}")
    print(
        f"{'total':<14} {sum(pred_sum.values()):>9.1f} {sum(a[0] for a in actual.values()):>9.1f}"
    )

    top = sorted(xpts, key=lambda pid: -xpts[pid])[: args.top]
    print(f"\ntop-{args.top} by xPts: player, xPts, ep_next, actual")
    for pid in top:
        p = bs.player(pid)
        ep = ep_next.get(pid)
        print(
            f"  {p.web_name:<16} {bs.team(p.team).short_name:<4} {p.position.short:<3} "
            f"{xpts[pid]:>5.2f} {ep if ep is not None else '-':>5} {actual.get(pid, (0, 0))[0]:>3}"
        )
    misses = sorted(
        (pid for pid in xpts if pid in actual), key=lambda pid: -abs(xpts[pid] - actual[pid][0])
    )[:10]
    print("\nbiggest misses (|xPts - actual|):")
    for pid in misses:
        p = bs.player(pid)
        print(
            f"  {p.web_name:<16} {bs.team(p.team).short_name:<4} {p.position.short:<3} "
            f"xPts={xpts[pid]:.2f} actual={actual[pid][0]} minutes={actual[pid][1]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
