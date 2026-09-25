"""Вход оптимизатора: прогнозы xPts по горизонту, кандидаты, обрезка пула, диагностика состава.

Всё детерминировано: прогнозы — core/xpts.predict_all на каждый тур горизонта (кэш в памяти на
время вызова; ~0.15 с на тур), кандидат — игрок с xpts/дисперсией по турам, ценой (продажа ≈
покупка, см. docs/optimizer.md) и владением. diagnose_squad размечает проблемы состава по
статусу FPL, новостному сигналу, ожидаемым минутам, силе фикстур и xPts против медианы позиции.
"""

from __future__ import annotations

import logging
import re
import statistics
from collections.abc import Iterable, Sequence
from datetime import date, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import text

from fplcopilot.core.strategy import POSITION_QUOTA, SQUAD_SIZE
from fplcopilot.core.xpts import (
    MODEL_VERSION,
    XPtsBreakdown,
    build_context,
    predict_all,
    save_predictions,
)
from fplcopilot.data import Bootstrap, FPLClient, Squad
from fplcopilot.db import session_scope

log = logging.getLogger(__name__)

UNAVAILABLE = frozenset({"i", "s", "u", "n"})
# Дешёвые «энейблеры» на позицию, которых почти нет в топе по xPts, но без них не собрать бюджет.
CHEAP_PRICE = {1: 4.5, 2: 4.5, 3: 5.0, 4: 5.5}
CHEAP_PER_POSITION = 6
NOT_PLAYING_P_START = 0.5
NOT_PLAYING_MINUTES = 45.0
HARD_FIXTURES_MEAN_FSI = 4.0
REGULAR_STARTER_P_START = 0.6  # медиана позиции считается по таким игрокам

IssueKind = Literal[
    "injured",
    "suspended",
    "unavailable",
    "doubtful",
    "not_playing",
    "fixtures",
    "low_xpts",
    "blank",
]
STATUS_KIND: dict[str, IssueKind] = {
    "i": "injured",
    "s": "suspended",
    "u": "unavailable",
    "n": "unavailable",
}
SIGNAL_KIND: dict[str, IssueKind] = {
    "injured": "injured",
    "suspended": "suspended",
    "unavailable": "unavailable",
}


class Candidate(BaseModel):
    player_id: int
    name: str
    position: int  # Position: 1 GKP, 2 DEF, 3 MID, 4 FWD
    team_id: int
    team: str = ""
    price: float  # млн; цена продажи приближена ценой покупки
    xpts_by_gw: dict[int, float]
    variance_by_gw: dict[int, float] = Field(default_factory=dict)
    ownership: float = 0.0  # selected_by_percent
    status: str = "a"
    in_squad: bool = False
    # для правила «ценный актив» (core/assets.py): очки «если здоров», шанс сыграть и тур
    # возвращения, старт в ближайшем туре, динамика цены и трансферов
    xpts_fit_by_gw: dict[int, float] = Field(default_factory=dict)
    chance: int | None = None
    news: str = ""
    return_gw: int | None = None
    p_start: float = 1.0
    price_change_start: float = 0.0  # £m с начала сезона (минус — подешевел)
    transfers_net: int = 0  # transfers_in_event − transfers_out_event

    def xpts(self, gw: int) -> float:
        return self.xpts_by_gw.get(gw, 0.0)

    def xpts_fit(self, gw: int) -> float:
        return self.xpts_fit_by_gw.get(gw, self.xpts(gw))

    def variance(self, gw: int) -> float:
        return self.variance_by_gw.get(gw, 0.0)

    def horizon_xpts(self, gws: Iterable[int]) -> float:
        return sum(self.xpts(g) for g in gws)


class SquadIssue(BaseModel):
    player_id: int
    name: str = ""
    kind: IssueKind
    severity: int  # 3 — не сыграет, 2 — серьёзно, 1 — мягко
    detail: str
    gw: int


# ---------- прогнозы по горизонту ----------


class PredictionStore:
    """predict_all по турам с кэшем в памяти; persist пишет только туры без строк model_version."""

    def __init__(
        self,
        client: FPLClient | None = None,
        *,
        as_of: datetime | None = None,
        provider_name: str | None = None,
    ) -> None:
        self.client = client or FPLClient()
        self.as_of = as_of
        self.provider_name = provider_name
        self._cache: dict[int, dict[int, XPtsBreakdown]] = {}

    @property
    def bootstrap(self) -> Bootstrap:
        return self.client.bootstrap()

    def get(self, gw: int) -> dict[int, XPtsBreakdown]:
        if gw not in self._cache:
            ctx = build_context(
                gw, self.as_of, client=self.client, provider_name=self.provider_name
            )
            self._cache[gw] = {p.player_id: p for p in predict_all(gw, ctx=ctx)}
        return self._cache[gw]

    def horizon(self, gws: Sequence[int]) -> dict[int, dict[int, XPtsBreakdown]]:
        return {gw: self.get(gw) for gw in gws}

    def persist(
        self, gws: Sequence[int], *, model_version: str = MODEL_VERSION, only_missing: bool = True
    ) -> int:
        """Пишет прогнозы в xpts_predictions. only_missing защищает уже сохранённые туры
        (живой эксперимент GW5): новая строка с поздним as_of стала бы «последней до дедлайна»."""
        targets = list(gws)
        if only_missing and targets:
            with session_scope() as s:
                have = {
                    int(r[0])
                    for r in s.execute(
                        text(
                            "SELECT DISTINCT gw FROM xpts_predictions "
                            "WHERE model_version = :mv AND gw = ANY(:gws)"
                        ),
                        {"mv": model_version, "gws": targets},
                    )
                }
            skipped = [g for g in targets if g in have]
            if skipped:
                log.info("xpts_predictions: туры %s уже сохранены, пропускаем", skipped)
            targets = [g for g in targets if g not in have]
        n = 0
        for gw in targets:
            n += save_predictions(list(self.get(gw).values()))
        return n


# ---------- кандидаты ----------


def build_candidates(
    bs: Bootstrap,
    preds_by_gw: dict[int, dict[int, XPtsBreakdown]],
    squad_ids: Iterable[int] = (),
) -> dict[int, Candidate]:
    squad = set(squad_ids)
    first = min(preds_by_gw) if preds_by_gw else None
    out: dict[int, Candidate] = {}
    for p in bs.elements:
        xpts = {
            gw: round(preds[p.id].xpts, 4) for gw, preds in preds_by_gw.items() if p.id in preds
        }
        var = {
            gw: round(preds[p.id].variance, 4) for gw, preds in preds_by_gw.items() if p.id in preds
        }
        fit = {
            gw: preds[p.id].xpts_fit
            for gw, preds in preds_by_gw.items()
            if p.id in preds and preds[p.id].xpts_fit is not None
        }
        head = preds_by_gw[first].get(p.id) if first is not None else None
        sig = (head.signal or {}) if head is not None else {}
        kickoffs = sorted(
            (f.kickoff_time.date(), gw)
            for gw, preds in preds_by_gw.items()
            if p.id in preds
            for f in preds[p.id].fixtures
            if f.kickoff_time is not None
        )
        out[p.id] = Candidate(
            player_id=p.id,
            name=p.web_name,
            position=int(p.position),
            team_id=p.team,
            team=bs.team(p.team).short_name,
            price=p.price,
            xpts_by_gw=xpts,
            variance_by_gw=var,
            ownership=float(p.selected_by_percent or 0.0),
            status=p.status,
            in_squad=p.id in squad,
            xpts_fit_by_gw=fit,
            chance=p.chance_of_playing_next_round,
            news=p.news,
            return_gw=sig.get("return_gw")
            or return_gw_from_news(p.news, kickoffs, max(preds_by_gw, default=0)),
            p_start=head.p_start if head is not None else 1.0,
            price_change_start=p.cost_change_start / 10,
            transfers_net=p.transfers_in_event - p.transfers_out_event,
        )
    return out


_RETURN_DATE = re.compile(
    r"\b(?:Expected back|Suspended until|Unavailable until|Out until)\s+"
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+(?P<mon>[A-Za-z]{3})",
    re.IGNORECASE,
)
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")


def return_gw_from_news(
    news: str, kickoffs: Sequence[tuple[date, int]], last_gw: int
) -> int | None:
    """Тур возвращения из новости FPL («Knee injury - Expected back 18 Oct»): первый матч клуба
    в горизонте с датой не раньше даты возврата (как GWCalendar в rag/extract.py; «until» у FPL
    включительно). Возврат позже всех матчей горизонта — last_gw + 1. Даты в новости нет — None."""
    m = _RETURN_DATE.search(news or "")
    if not m or not kickoffs:
        return None
    mon = m.group("mon")[:3].lower()
    if mon not in _MONTHS:
        return None
    first = kickoffs[0][0]
    try:
        when = date(first.year, _MONTHS.index(mon) + 1, int(m.group("day")))
    except ValueError:
        return None
    if when < first - timedelta(days=180):
        when = when.replace(year=when.year + 1)
    return next((gw for day, gw in kickoffs if day >= when), last_gw + 1)


def squad_candidates(cands: dict[int, Candidate], squad_ids: Iterable[int]) -> list[Candidate]:
    return [cands[pid] for pid in squad_ids]


def build_pool(
    cands: dict[int, Candidate],
    *,
    squad_ids: Iterable[int],
    gws: Sequence[int],
    size: int = 120,
    bank: float = 0.0,
    exclude: Iterable[int] = (),
    cheap_per_position: int = CHEAP_PER_POSITION,
    include: Iterable[int] = (),
) -> dict[int, Candidate]:
    """Текущий состав + top-`size` по сумме xPts за горизонт (квоты по позициям 2/5/5/3 от 15)
    + по `cheap_per_position` лучших дешёвых на позицию + `include` (покупки «на спаде» из
    core/assets.py — их статус i/s не отсекается). Вне пула: статусы i/s/u/n, `exclude`,
    цена выше максимально доступной (банк + самый дорогой игрок состава), нулевой xPts."""
    squad = list(dict.fromkeys(squad_ids))
    excluded = set(exclude)
    pool: dict[int, Candidate] = {pid: cands[pid] for pid in squad}
    max_price = bank + max((cands[pid].price for pid in squad), default=15.0)
    for pid in include:
        if pid in cands and pid not in excluded and cands[pid].price <= max_price + 1e-9:
            pool[pid] = cands[pid]
    eligible = [
        c
        for pid, c in cands.items()
        if pid not in pool
        and pid not in excluded
        and c.status not in UNAVAILABLE
        and c.price <= max_price + 1e-9
        and c.horizon_xpts(gws) > 0
    ]
    eligible.sort(key=lambda c: (-c.horizon_xpts(gws), c.price, c.player_id))
    for pos, quota in POSITION_QUOTA.items():
        n_pos = max(1, round(size * quota / SQUAD_SIZE))
        by_pos = [c for c in eligible if c.position == pos]
        for c in by_pos[:n_pos]:
            pool[c.player_id] = c
        cheap = [c for c in by_pos if c.price <= CHEAP_PRICE[pos] and c.player_id not in pool]
        for c in cheap[:cheap_per_position]:
            pool[c.player_id] = c
    return pool


# ---------- диагностика ----------


def position_medians(
    preds: dict[int, XPtsBreakdown],
    cands: dict[int, Candidate],
    gws: Sequence[int],
    *,
    min_p_start: float = REGULAR_STARTER_P_START,
) -> dict[int, float]:
    """Медиана xPts за горизонт по позиции среди регулярных стартеров (p_start >= 0.6)."""
    by_pos: dict[int, list[float]] = {pos: [] for pos in POSITION_QUOTA}
    for pid, pred in preds.items():
        c = cands.get(pid)
        if c is None or pred.p_start < min_p_start:
            continue
        by_pos[c.position].append(c.horizon_xpts(gws))
    return {pos: (statistics.median(v) if v else 0.0) for pos, v in by_pos.items()}


def diagnose_squad(
    squad_ids: Iterable[int],
    preds_by_gw: dict[int, dict[int, XPtsBreakdown]],
    cands: dict[int, Candidate],
    from_gw: int,
    *,
    starters: Iterable[int] | None = None,
) -> list[SquadIssue]:
    """Проблемы состава на дедлайн from_gw. starters — кто попадает в лучшие 11 (low_xpts только
    для них: запасной за 4.0 с низким xPts — норма, а не проблема)."""
    gws = sorted(preds_by_gw)
    preds = preds_by_gw[from_gw]
    medians = position_medians(preds, cands, gws)
    xi = set(starters) if starters is not None else set(squad_ids)
    issues: list[SquadIssue] = []
    for pid in squad_ids:
        c = cands[pid]
        pred = preds.get(pid)
        if pred is None:
            continue

        def add(kind: IssueKind, severity: int, detail: str, *, _pid: int = pid, _c=c) -> None:
            issues.append(
                SquadIssue(
                    player_id=_pid,
                    name=_c.name,
                    kind=kind,
                    severity=severity,
                    detail=detail,
                    gw=from_gw,
                )
            )

        sig = pred.signal or {}
        ret = f", return GW{sig['return_gw']}" if sig.get("return_gw") is not None else ""
        hard = STATUS_KIND.get(pred.status)
        if hard:
            add(hard, 3, f"FPL status {pred.status} (chance {pred.chance}){ret}")
        elif sig.get("used") and sig.get("availability") in SIGNAL_KIND:
            add(
                SIGNAL_KIND[sig["availability"]],
                3,
                f"news: {sig['availability']} conf {sig['confidence']:.2f}{ret} (FPL status a)",
            )
        elif pred.status == "d":
            chance = pred.chance if pred.chance is not None else 50
            add("doubtful", 2 if chance <= 50 else 1, f"FPL doubtful {chance}%{ret}")
        elif (pred.p_start < NOT_PLAYING_P_START or pred.exp_minutes < NOT_PLAYING_MINUTES) and (
            c.position != 1 or pid in xi
        ):  # запасной вратарь и не должен играть
            add(
                "not_playing",
                2,
                f"p_start {pred.p_start:.2f}, exp_minutes {pred.exp_minutes:.0f}",
            )
        if not pred.fixtures:
            add("blank", 1, f"no fixture in GW{from_gw}")
        fsi = [
            f.fixture_strength_index
            for g in gws
            if (pg := preds_by_gw[g].get(pid)) is not None
            for f in pg.fixtures
        ]
        if fsi and statistics.fmean(fsi) >= HARD_FIXTURES_MEAN_FSI:
            add(
                "fixtures",
                1,
                f"mean fixture strength {statistics.fmean(fsi):.1f}/5 over GW{gws[0]}–{gws[-1]}",
            )
        if pid in xi and pred.status == "a" and not hard:
            median = medians.get(c.position, 0.0)
            total = c.horizon_xpts(gws)
            if median > 0 and total < median:
                add(
                    "low_xpts",
                    2,
                    f"{total:.1f} xPts over horizon vs position median {median:.1f}",
                )
    issues.sort(key=lambda i: (-i.severity, i.player_id))
    return issues


def problem_players(issues: Iterable[SquadIssue], *, min_severity: int = 2) -> set[int]:
    return {i.player_id for i in issues if i.severity >= min_severity}


# ---------- состав менеджера ----------


CHIP_NAMES = ("wildcard", "freehit", "bboost", "3xc")  # имена чипов FPL API (bootstrap.chips)


def chip_available(bs: Bootstrap, chips_used: Sequence[str], gw: int, chip: str) -> bool:
    """Есть ли чип `chip` на дедлайн gw: окно чипа из bootstrap минус сыгранные (по именам).

    В Squad.chips_used только имена, без туров, поэтому: в первой половине (до GW19) — доступен,
    если ни одного такого не сыграно; во второй — если сыграно меньше двух. Чип первой половины
    сгорает после GW19 (правила 2026/27: каждый чип дважды, по половинам сезона).
    """
    windows = sorted((c.start_event, c.stop_event) for c in bs.chips if c.name == chip)
    if not windows:
        return False
    current = [i for i, (a, b) in enumerate(windows) if a <= gw <= b]
    if not current:
        return False
    used = sum(1 for name in chips_used if name == chip)
    return used <= current[0]


def wildcard_available(bs: Bootstrap, chips_used: Sequence[str], gw: int) -> bool:
    return chip_available(bs, chips_used, gw, "wildcard")


def chips_available(bs: Bootstrap, chips_used: Sequence[str], gw: int) -> list[str]:
    """Чипы, доступные на дедлайн gw, в порядке CHIP_NAMES."""
    return [c for c in CHIP_NAMES if chip_available(bs, chips_used, gw, c)]


def squad_state(squad: Squad) -> tuple[list[int], float, int]:
    """(id игроков, банк в млн, оценка бесплатных трансферов) для старта модели."""
    ids = [p.player.id for p in sorted(squad.players, key=lambda p: p.pick.position)]
    return ids, float(squad.bank), int(squad.free_transfers or 1)
