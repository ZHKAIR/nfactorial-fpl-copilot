"""Снимки планов трансферов (plan_snapshots, 006_plans.sql): сохранить, взять последний, diff.

diff_plans — структурное сравнение двух планов (какие ходы появились/исчезли/сменили цель/тур,
поменялась ли рекомендация). Причины («потому что Palmer получил травму») здесь не выводятся —
это работа агента поверх diff и issues.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import BaseModel
from sqlalchemy import text

from fplcopilot.core.candidates import Candidate
from fplcopilot.core.optimizer import CHIP_TITLES, PlannedMove, TransferPlan
from fplcopilot.db import session_scope

ChangeKind = Literal["added", "removed", "changed_target", "moved_gw", "recommendation", "chips"]


class PlanChange(BaseModel):
    gw: int
    kind: ChangeKind
    player_ids: list[int]
    detail: str


def inputs_hash(
    squad: Iterable[int],
    bank: float,
    free_transfers: int,
    strategy: str,
    horizon: int,
    from_gw: int,
    pool: dict[int, Candidate] | None = None,
    *,
    chips: Mapping[int, str] | None = None,
) -> str:
    """sha1 входов плана; xPts пула округлены до 0.01, чтобы шум не менял хэш. Фишки входят в
    хэш, только если заданы (хэш плана без фишек прежний)."""
    payload: dict[str, object] = {
        "squad": sorted(squad),
        "bank": round(bank, 1),
        "ft": free_transfers,
        "strategy": strategy,
        "horizon": horizon,
        "from_gw": from_gw,
        "xpts": (
            {
                pid: {gw: round(x, 2) for gw, x in sorted(c.xpts_by_gw.items())}
                for pid, c in sorted(pool.items())
            }
            if pool
            else {}
        ),
    }
    if chips:
        payload["chips"] = {str(gw): chip for gw, chip in sorted(chips.items())}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


_INSERT = text(
    """
    INSERT INTO plan_snapshots (manager_id, gw, strategy, horizon, plan, inputs_hash)
    VALUES (:manager_id, :gw, :strategy, :horizon, CAST(:plan AS jsonb), :inputs_hash)
    RETURNING id
    """
)

_LATEST = text(
    """
    SELECT plan FROM plan_snapshots
    WHERE manager_id = :manager_id AND gw = :gw
      AND (CAST(:strategy AS text) IS NULL OR strategy = CAST(:strategy AS text))
    ORDER BY created_at DESC, id DESC
    LIMIT 1
    """
)


def save_plan(plan: TransferPlan, *, inputs_hash: str = "") -> int:
    with session_scope() as s:
        row = s.execute(
            _INSERT,
            {
                "manager_id": plan.manager_id,
                "gw": plan.from_gw,
                "strategy": plan.strategy,
                "horizon": plan.horizon,
                "plan": plan.model_dump_json(by_alias=True),
                "inputs_hash": inputs_hash,
            },
        ).one()
    return int(row[0])


def latest_plan(manager_id: int, gw: int, strategy: str | None = None) -> TransferPlan | None:
    with session_scope() as s:
        row = s.execute(_LATEST, {"manager_id": manager_id, "gw": gw, "strategy": strategy}).first()
    if row is None:
        return None
    data = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    return TransferPlan.model_validate(data)


def _moves(plan: TransferPlan) -> list[PlannedMove]:
    return [m for gw in sorted(plan.moves_by_gw) for m in plan.moves_by_gw[gw]]


def _chips_text(chips: Mapping[int, str]) -> str:
    return ", ".join(f"GW{gw} {CHIP_TITLES.get(c, c)}" for gw, c in sorted(chips.items())) or "—"


def diff_plans(prev: TransferPlan, new: TransferPlan) -> list[PlanChange]:
    """Структурный diff ходов: added / removed / changed_target (тот же out, другой in) /
    moved_gw (та же пара в другом туре) + смена рекомендации и запланированных фишек."""
    changes: list[PlanChange] = []
    prev_pairs = {(m.out, m.in_): m for m in _moves(prev)}
    new_pairs = {(m.out, m.in_): m for m in _moves(new)}
    prev_by_out = {m.out: m for m in _moves(prev)}
    new_by_out = {m.out: m for m in _moves(new)}

    for pair, m in new_pairs.items():
        if pair in prev_pairs:
            if prev_pairs[pair].gw != m.gw:
                changes.append(
                    PlanChange(
                        gw=m.gw,
                        kind="moved_gw",
                        player_ids=[m.out, m.in_],
                        detail=f"{m.out_name} -> {m.in_name}: GW{prev_pairs[pair].gw} -> GW{m.gw}",
                    )
                )
            continue
        old = prev_by_out.get(m.out)
        if old is not None:
            changes.append(
                PlanChange(
                    gw=m.gw,
                    kind="changed_target",
                    player_ids=[m.out, old.in_, m.in_],
                    detail=f"{m.out_name}: {old.in_name} -> {m.in_name}",
                )
            )
        else:
            changes.append(
                PlanChange(
                    gw=m.gw,
                    kind="added",
                    player_ids=[m.out, m.in_],
                    detail=f"{m.out_name} -> {m.in_name}",
                )
            )
    for pair, m in prev_pairs.items():
        if pair in new_pairs or m.out in new_by_out:
            continue  # учтено выше как moved_gw / changed_target
        changes.append(
            PlanChange(
                gw=m.gw,
                kind="removed",
                player_ids=[m.out, m.in_],
                detail=f"{m.out_name} -> {m.in_name}",
            )
        )
    if prev.recommendation != new.recommendation:
        changes.append(
            PlanChange(
                gw=new.from_gw,
                kind="recommendation",
                player_ids=[],
                detail=f"{prev.recommendation} -> {new.recommendation}",
            )
        )
    if prev.chips_by_gw != new.chips_by_gw:
        changes.append(
            PlanChange(
                gw=new.from_gw,
                kind="chips",
                player_ids=[],
                detail=f"{_chips_text(prev.chips_by_gw)} -> {_chips_text(new.chips_by_gw)}",
            )
        )
    changes.sort(key=lambda c: (c.gw, c.kind, c.player_ids))
    return changes
