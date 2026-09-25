"""Клиент публичного FPL API с дисковым кэшем.

Только чтение, без авторизации. Все ответы приводятся к схемам из schemas.py.
Кэш — JSON-файлы в settings.cache_dir/fpl с TTL; в час дедлайна API нестабилен,
поэтому кэш — часть дизайна, а не оптимизация.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self

import httpx

from fplcopilot.config import settings
from fplcopilot.data.prices import purchase_prices, selling_price
from fplcopilot.data.schemas import (
    Bootstrap,
    ElementSummary,
    Entry,
    EntryHistoryRow,
    Fixture,
    LeagueStandings,
    ManagerHistory,
    PicksResponse,
    SalePrice,
    Squad,
    SquadPlayer,
    TransferRow,
)

log = logging.getLogger(__name__)

BASE_URL = "https://fantasy.premierleague.com/api/"
HEADERS = {
    # Без User-Agent FPL иногда отдаёт 403.
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) fpl-copilot/0.1",
    "Accept": "application/json",
}

MAX_FREE_TRANSFERS = 5
CHIPS_WITHOUT_TRANSFER_COST = {"wildcard", "freehit"}


class FPLClient:
    def __init__(
        self,
        cache_dir: Path | None = None,
        cache_ttl: int | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.cache_dir = (cache_dir or settings.cache_dir) / "fpl"
        self.cache_ttl = settings.fpl_cache_ttl if cache_ttl is None else cache_ttl
        self._http = httpx.Client(base_url=BASE_URL, headers=HEADERS, timeout=timeout)
        self._bootstrap: Bootstrap | None = None

    # ---------- низкоуровневый слой ----------

    def _cache_path(self, path: str) -> Path:
        key = hashlib.sha1(path.encode()).hexdigest()[:16]
        safe = path.strip("/").replace("/", "_").replace("?", "_").replace("=", "-")
        return self.cache_dir / f"{safe}_{key}.json"

    def _get(self, path: str, *, ttl: int | None = None) -> Any:
        ttl = self.cache_ttl if ttl is None else ttl
        cache_file = self._cache_path(path)
        if ttl > 0 and cache_file.exists():
            age = time.time() - cache_file.stat().st_mtime
            if age < ttl:
                log.debug("cache hit %s (age %.0fs)", path, age)
                return json.loads(cache_file.read_text())

        resp = self._http.get(path)
        resp.raise_for_status()
        data = resp.json()

        if ttl > 0:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(data))
        return data

    def clear_cache(self) -> int:
        n = 0
        if self.cache_dir.exists():
            for f in self.cache_dir.glob("*.json"):
                f.unlink()
                n += 1
        self._bootstrap = None
        return n

    # ---------- эндпоинты ----------

    def bootstrap(self, *, refresh: bool = False) -> Bootstrap:
        if self._bootstrap is None or refresh:
            data = self._get("bootstrap-static/", ttl=0 if refresh else None)
            self._bootstrap = Bootstrap.model_validate(data)
        return self._bootstrap

    def fixtures(self, event: int | None = None) -> list[Fixture]:
        path = "fixtures/" if event is None else f"fixtures/?event={event}"
        return [Fixture.model_validate(f) for f in self._get(path)]

    def entry(self, manager_id: int) -> Entry:
        return Entry.model_validate(self._get(f"entry/{manager_id}/"))

    def picks(self, manager_id: int, gw: int) -> PicksResponse:
        return PicksResponse.model_validate(self._get(f"entry/{manager_id}/event/{gw}/picks/"))

    def history(self, manager_id: int) -> ManagerHistory:
        return ManagerHistory.model_validate(self._get(f"entry/{manager_id}/history/"))

    def element_summary(self, player_id: int) -> ElementSummary:
        return ElementSummary.model_validate(self._get(f"element-summary/{player_id}/"))

    def transfers(self, manager_id: int) -> list[TransferRow]:
        return [TransferRow.model_validate(t) for t in self._get(f"entry/{manager_id}/transfers/")]

    def sale_prices(self, manager_id: int, player_ids: Iterable[int]) -> dict[int, SalePrice]:
        """Цены покупки / продажи игроков состава (правила FPL, prices.py). Стартовый состав
        команды, созданной с GW1, куплен по цене начала сезона; созданной позже — по цене в
        туре создания (element-summary)."""
        bs = self.bootstrap()
        ids = list(player_ids)
        entry = self.entry(manager_id)
        hist = self.history(manager_id)
        started = entry.started_event or 1
        start_cost: dict[int, int | None] = {}
        for pid in ids:
            p = bs.player(pid)
            if started <= 1:
                start_cost[pid] = p.now_cost - p.cost_change_start
            else:
                rows = [h for h in self.element_summary(pid).history if h.round >= started]
                start_cost[pid] = rows[0].value if rows and rows[0].value else None
        bought = purchase_prices(
            ids,
            self.transfers(manager_id),
            start_cost,
            freehit_events=[c.event for c in hist.chips if c.name == "freehit"],
        )
        out: dict[int, SalePrice] = {}
        for pid, cost in bought.items():
            now = bs.player(pid).now_cost
            out[pid] = SalePrice(
                purchase=cost / 10, selling=selling_price(cost, now) / 10, now=now / 10
            )
        return out

    def league_standings(self, league_id: int, page: int = 1) -> LeagueStandings:
        data = self._get(f"leagues-classic/{league_id}/standings/?page_standings={page}")
        return LeagueStandings(
            league_id=data["league"]["id"],
            league_name=data["league"]["name"],
            results=data["standings"]["results"],
            has_next=data["standings"].get("has_next", False),
        )

    # ---------- производные ----------

    def current_gw(self) -> int | None:
        ev = self.bootstrap().current_event
        return ev.id if ev else None

    def next_gw(self) -> int | None:
        ev = self.bootstrap().next_event
        return ev.id if ev else None

    def squad(self, manager_id: int, gw: int | None = None) -> Squad:
        """Состав менеджера на тур gw (по умолчанию — последний сыгранный/текущий).

        До дедлайна публичный API отдаёт picks только за прошедшие туры,
        поэтому "текущий состав" = picks последнего завершённого тура.
        """
        bs = self.bootstrap()
        if gw is None:
            gw = self.current_gw() or 1
        picks = self.picks(manager_id, gw)
        hist = self.history(manager_id)

        players = [
            SquadPlayer(
                pick=p,
                player=bs.player(p.element),
                team=bs.team(bs.player(p.element).team),
            )
            for p in picks.picks
        ]
        return Squad(
            manager_id=manager_id,
            gw=gw,
            players=players,
            bank=picks.entry_history.bank / 10,
            team_value=picks.entry_history.value / 10,
            active_chip=picks.active_chip,
            free_transfers=estimate_free_transfers(hist, upto_gw=gw),
            chips_used=[c.name for c in hist.chips],
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def estimate_free_transfers(history: ManagerHistory, upto_gw: int | None = None) -> int:
    """Оценка бесплатных трансферов, доступных на СЛЕДУЮЩИЙ после upto_gw тур.

    Правила 2026/27: после GW1 даётся 1 FT в тур; неиспользованные копятся до 5;
    на Wildcard/Free Hit трансферы бесплатны и банк FT сохраняется.
    Публичный API не отдаёт FT напрямую, поэтому считаем по истории.
    """
    chips_by_gw = {c.event: c.name for c in history.chips}
    rows: list[EntryHistoryRow] = sorted(history.current, key=lambda r: r.event)
    if upto_gw is not None:
        rows = [r for r in rows if r.event <= upto_gw]

    ft = 0  # перед GW1 трансферы неограниченные, FT не начисляется
    for row in rows:
        if row.event == 1:
            ft = 1  # к дедлайну GW2 доступен 1 FT
            continue
        chip = chips_by_gw.get(row.event)
        if chip in CHIPS_WITHOUT_TRANSFER_COST:
            used = 0
        else:
            used = min(row.event_transfers, ft)
        ft = min(MAX_FREE_TRANSFERS, ft - used + 1)
    return max(ft, 1) if rows else 1
