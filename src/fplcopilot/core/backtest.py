"""Walk-forward проверка xPts v0 на сыгранных турах против простых бейзлайнов.

    uv run python -m fplcopilot.core.backtest [--gws 3,4] [--out docs/xpts_backtest.json]

Для каждого тура g прогноз строится ТОЛЬКО по раундам < g (make_context -> rows_before) и
сравнивается с фактическими total_points из player_gw_history. Ограничения честно:
- снимки статусов FPL есть только с 17.09, поэтому для прошлых туров все считаются
  доступными (status='a'); игроки без минут в предыдущих раундах получают низкий p_start
  из модели минут, но травмированный основной игрок «предсказывается» как играющий;
- 2–3 обучающих тура — результаты индикативные; настоящий тест — живой прогноз GW5.

Бейзлайны: form3 — среднее очков за последние 3 матча; ppg — очки / матчи с минутами;
pos_avg — среднее очков по позиции в окне (константа).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fplcopilot.config import PROJECT_ROOT
from fplcopilot.core.history import (
    SeasonTotals,
    group_by_player,
    load_history,
    load_past_seasons,
    rows_before,
)
from fplcopilot.core.stats import mae, rmse, spearman
from fplcopilot.core.xpts import MODEL_VERSION, XPtsBreakdown, make_context, predict_all
from fplcopilot.data import Bootstrap, Fixture, FPLClient
from fplcopilot.data.schemas import PlayerGWHistory

log = logging.getLogger(__name__)

TOP_N = 20
DEFAULT_OUT = PROJECT_ROOT / "docs" / "xpts_backtest.json"


@dataclass
class MethodResult:
    gw: int
    method: str
    n: int
    mae: float
    rmse: float
    spearman: float
    spearman_played: float  # только игроки с минутами > 0 в туре
    top20_actual_mean: float  # средние фактические очки топ-20 по методу
    top20_hits: int  # сколько из топ-20 метода попали в фактический топ-20
    mean_pred: float
    mean_actual: float


def baselines(
    history: dict[int, list[PlayerGWHistory]], bs: Bootstrap
) -> dict[str, dict[int, float]]:
    """form3 / ppg / pos_avg по строкам ДО тура (history уже отфильтрована)."""
    pos_rows: dict[int, list[int]] = defaultdict(list)
    for pid, rows in history.items():
        pos_rows[int(bs.player(pid).position)].extend(r.total_points for r in rows)
    pos_avg = {pos: (sum(v) / len(v) if v else 0.0) for pos, v in pos_rows.items()}

    form3: dict[int, float] = {}
    ppg: dict[int, float] = {}
    pos: dict[int, float] = {}
    for p in bs.elements:
        rows = history.get(p.id, [])
        last3 = rows[-3:]
        form3[p.id] = sum(r.total_points for r in last3) / len(last3) if last3 else 0.0
        played = [r for r in rows if r.minutes > 0]
        ppg[p.id] = sum(r.total_points for r in played) / len(played) if played else 0.0
        pos[p.id] = pos_avg.get(int(p.position), 0.0)
    return {"form3": form3, "ppg": ppg, "pos_avg": pos}


def unavailable_by_history(
    bs: Bootstrap, all_rows: list[PlayerGWHistory], gw: int, deadline: datetime
) -> dict[int, tuple[str, int | None]]:
    """Замена статусов для прошлых туров: ни минуты во всех раундах до gw -> недоступен ('u').

    Единственное, что можно восстановить без снимков статусов. Травмированный после GW2
    основной игрок так не ловится — ограничение бэктеста.
    """
    window = group_by_player(rows_before(all_rows, gw, deadline))
    out: dict[int, tuple[str, int | None]] = {}
    for p in bs.elements:
        rows = window.get(p.id, [])
        if not rows or sum(r.minutes for r in rows) == 0:
            out[p.id] = ("u", None)
    return out


def evaluate(
    gw: int,
    method: str,
    pred: dict[int, float],
    actual: dict[int, tuple[int, int]],
) -> MethodResult:
    ids = [pid for pid in actual if pid in pred]
    p = [pred[i] for i in ids]
    a = [float(actual[i][0]) for i in ids]
    played = [i for i in ids if actual[i][1] > 0]
    top_pred = sorted(ids, key=lambda i: -pred[i])[:TOP_N]
    top_actual = set(sorted(ids, key=lambda i: -actual[i][0])[:TOP_N])
    return MethodResult(
        gw=gw,
        method=method,
        n=len(ids),
        mae=round(mae(p, a), 3),
        rmse=round(rmse(p, a), 3),
        spearman=round(spearman(p, a), 3),
        spearman_played=round(
            spearman([pred[i] for i in played], [float(actual[i][0]) for i in played]), 3
        ),
        top20_actual_mean=round(sum(actual[i][0] for i in top_pred) / max(1, len(top_pred)), 2),
        top20_hits=len(set(top_pred) & top_actual),
        mean_pred=round(sum(p) / max(1, len(p)), 3),
        mean_actual=round(sum(a) / max(1, len(a)), 3),
    )


def backtest_gw(
    gw: int,
    *,
    bs: Bootstrap,
    fixtures: list[Fixture],
    all_rows: list[PlayerGWHistory],
    past: dict[int, SeasonTotals],
) -> tuple[list[MethodResult], list[XPtsBreakdown]]:
    event = next(e for e in bs.events if e.id == gw)
    if not event.finished:
        raise ValueError(f"GW{gw} ещё не завершён")
    ctx = make_context(
        gw,
        bs=bs,
        fixtures=fixtures,
        history_rows=all_rows,
        past=past,
        signals={},
        as_of=event.deadline_time,
        live=False,  # статусов на тот момент нет -> все 'a' (см. докстринг модуля)
        status_overrides=unavailable_by_history(bs, all_rows, gw, event.deadline_time),
    )
    preds = predict_all(gw, ctx=ctx)

    actual: dict[int, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for r in all_rows:
        if r.round == gw:
            pts, mins = actual[r.element]
            actual[r.element] = (pts + r.total_points, mins + r.minutes)
    actual = dict(actual)

    methods: dict[str, dict[int, float]] = {
        f"xpts_{MODEL_VERSION}": {p.player_id: p.xpts for p in preds}
    }
    methods.update(baselines(group_by_player(rows_before(all_rows, gw)), bs))
    results = [evaluate(gw, name, pred, actual) for name, pred in methods.items()]
    return results, preds


def format_results(results: list[MethodResult]) -> str:
    head = (
        f"{'GW':>3} {'method':<9} {'n':>4} {'MAE':>6} {'RMSE':>6} {'ρ all':>6} {'ρ played':>8} "
        f"{'top20 mean':>10} {'hits':>4} {'mean pred':>9} {'mean act':>8}"
    )
    lines = [head, "-" * len(head)]
    for r in results:
        lines.append(
            f"{r.gw:>3} {r.method:<9} {r.n:>4} {r.mae:>6.3f} {r.rmse:>6.3f} {r.spearman:>6.3f} "
            f"{r.spearman_played:>8.3f} {r.top20_actual_mean:>10.2f} {r.top20_hits:>4} "
            f"{r.mean_pred:>9.3f} {r.mean_actual:>8.3f}"
        )
    return "\n".join(lines)


def run(gws: list[int], *, client: FPLClient | None = None) -> dict[str, Any]:
    client = client or FPLClient()
    bs = client.bootstrap()
    fixtures = client.fixtures()
    all_rows = load_history()
    past = load_past_seasons()
    results: list[MethodResult] = []
    top_lists: dict[str, list[dict[str, Any]]] = {}
    for gw in gws:
        res, preds = backtest_gw(gw, bs=bs, fixtures=fixtures, all_rows=all_rows, past=past)
        results.extend(res)
        actual = {r.element: r.total_points for r in all_rows if r.round == gw}
        top_lists[f"gw{gw}"] = [
            {
                "player": p.web_name,
                "team": p.team,
                "pos": p.position,
                "xpts": round(p.xpts, 2),
                "actual": actual.get(p.player_id),
            }
            for p in preds[:10]
        ]
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_version": MODEL_VERSION,
        "gws": gws,
        "results": [asdict(r) for r in results],
        "top10": top_lists,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FPL Copilot: walk-forward проверка xPts")
    ap.add_argument("--gws", default="3,4", help="сыгранные туры через запятую")
    ap.add_argument(
        "--out", default=str(DEFAULT_OUT), help="куда сохранить JSON ('' — не сохранять)"
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    gws = [int(x) for x in args.gws.split(",") if x.strip()]
    report = run(gws)
    results = [MethodResult(**r) for r in report["results"]]
    print(format_results(results))
    for key, rows in report["top10"].items():
        print(f"\n{key} top-10 by xPts (actual points):")
        for i, r in enumerate(rows, start=1):
            print(
                f"  {i:>2} {r['player']:<16} {r['team']:<4} {r['pos']:<3} {r['xpts']:>5.2f} -> {r['actual']}"
            )
    if args.out:
        out = Path(args.out)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
