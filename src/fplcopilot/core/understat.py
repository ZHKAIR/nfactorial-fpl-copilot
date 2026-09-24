"""Публичная статистика Understat (EPL) без ключа: xG/xA/удары игрока и xG/xGA команды.

С осени 2026 страница лиги больше не кладёт `playersData` в HTML — те же числа отдаёт
`GET /getLeagueData/{league}/{season}` (это и есть JSON, который раньше парсили из
`JSON.parse('...')` / `getLeaguePlayers`). HTML-парсер оставлен: тесты и fallback, если
эндпоинт снова встроят в страницу. Сеть никогда не роняет приложение: кэш 12 ч, затем
протухший кэш, затем пустой снимок.
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from fplcopilot.config import settings
from fplcopilot.core.odds import canonical_team, team_index
from fplcopilot.data.schemas import Player, Team

log = logging.getLogger(__name__)

LEAGUE = "EPL"
DATA_URL = "https://understat.com/getLeagueData/{league}/{season}"
PAGE_URL = "https://understat.com/league/{league}/{season}"
TIMEOUT = 12.0
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) fpl-copilot/0.1",
    "Accept": "application/json,text/html;q=0.9",
    "X-Requested-With": "XMLHttpRequest",
}
# Understat пишет имена без диакритики / иначе, чем FPL web_name.
PLAYER_ALIASES: dict[str, tuple[str, ...]] = {
    "joao pedro": ("joao pedro",),
    "martin odegaard": ("odegaard",),
    "josko gvardiol": ("gvardiol",),
    "bruno fernandes": ("b fernandes",),
    "bruno guimaraes": ("bruno g",),
    "alisson": ("a becker", "alisson becker"),
    "yeremi pino": ("yeremy",),
    "yehor yarmolyuk": ("yarmoliuk",),
    "ferdi kadioglu": ("f kadioglu",),
    "jair": ("jair cunha",),
    "riccardo calafiori": ("calafiori",),
}
_EMBEDDED_JSON = re.compile(
    r"(?:var\s+)?(?P<name>playersData|teamsData|datesData)\s*=\s*"
    r"JSON\.parse\(\s*(['\"])(?P<body>.*?)\2\s*\)",
    re.DOTALL,
)
_TRANS = str.maketrans(
    {
        "ø": "o",
        "æ": "ae",
        "ß": "ss",
        "ð": "d",
        "ł": "l",
        "đ": "d",
        "þ": "th",
        "ı": "i",  # турецкая i без точки (Kadıoğlu)
    }
)


def normalize_person(name: str) -> str:
    """Нижний регистр, без диакритики (Ø→o, ã→a), только [a-z0-9] и пробелы."""
    s = unicodedata.normalize("NFKD", name).lower().translate(_TRANS)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _i(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True)
class UnderstatPlayer:
    understat_id: str
    player_name: str
    team_title: str
    minutes: int
    games: int
    goals: int
    assists: int
    xg: float
    xa: float
    npxg: float
    shots: int

    @property
    def xg90(self) -> float | None:
        return None if self.minutes <= 0 else self.xg * 90.0 / self.minutes

    @property
    def xa90(self) -> float | None:
        return None if self.minutes <= 0 else self.xa * 90.0 / self.minutes

    @property
    def npxg90(self) -> float | None:
        return None if self.minutes <= 0 else self.npxg * 90.0 / self.minutes

    @property
    def shots90(self) -> float | None:
        return None if self.minutes <= 0 else self.shots * 90.0 / self.minutes


@dataclass(frozen=True)
class UnderstatTeam:
    understat_id: str
    title: str
    matches: int
    xg: float
    xga: float
    pens_won: int = 0
    pens_conceded: int = 0
    # (date_iso, won, conceded) по сыгранным матчам, по дате
    match_pens: tuple[tuple[str, int, int], ...] = ()

    @property
    def xg_per_game(self) -> float | None:
        return None if self.matches <= 0 else self.xg / self.matches

    @property
    def xga_per_game(self) -> float | None:
        return None if self.matches <= 0 else self.xga / self.matches


@dataclass
class UnderstatSnapshot:
    season: int
    league: str = LEAGUE
    players: list[UnderstatPlayer] = field(default_factory=list)
    teams: list[UnderstatTeam] = field(default_factory=list)
    fetched_at: float | None = None
    source: str = "none"  # api | html | cache | stale-cache | empty
    note: str = ""

    @property
    def empty(self) -> bool:
        return not self.players and not self.teams


def decode_understat_json(escaped: str) -> Any:
    """JSON из `JSON.parse('...')`: обычный JSON или `\x7B`-экранирование Understat."""
    for candidate in (escaped, _unicode_escape(escaped)):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            continue
    raise ValueError("не удалось разобрать JSON.parse Understat")


def _unicode_escape(escaped: str) -> str:
    return escaped.encode("utf-8").decode("unicode_escape")


def extract_embedded_json(html: str) -> dict[str, Any]:
    """Достаёт playersData / teamsData / datesData из HTML (старый формат страницы)."""
    out: dict[str, Any] = {}
    for m in _EMBEDDED_JSON.finditer(html):
        key = {"playersData": "players", "teamsData": "teams", "datesData": "dates"}[m.group("name")]
        out[key] = decode_understat_json(m.group("body"))
    return out


def _player_from_row(row: dict[str, Any]) -> UnderstatPlayer:
    return UnderstatPlayer(
        understat_id=str(row.get("id") or ""),
        player_name=str(row.get("player_name") or row.get("player") or ""),
        team_title=str(row.get("team_title") or row.get("team") or ""),
        minutes=_i(row.get("time")),
        games=_i(row.get("games")),
        goals=_i(row.get("goals")),
        assists=_i(row.get("assists")),
        xg=_f(row.get("xG")),
        xa=_f(row.get("xA")),
        npxg=_f(row.get("npxG")),
        shots=_i(row.get("shots")),
    )


PENALTY_XG = 0.76  # один пенальти в Understat ≈ 0.76 xG


def pens_from_xg_gap(xg: float, npxg: float, *, pen_xg: float = PENALTY_XG) -> int:
    """Пенальти в матче: round((xG − npxG) / 0.76). Отрицательный зазор не бывает — режем в 0."""
    if pen_xg <= 0:
        return 0
    return max(0, round((_f(xg) - _f(npxg)) / pen_xg))


def _team_from_row(row: dict[str, Any]) -> UnderstatTeam:
    history = row.get("history") or []
    played = [h for h in history if isinstance(h, dict)]
    match_pens: list[tuple[str, int, int]] = []
    for h in played:
        won = pens_from_xg_gap(h.get("xG"), h.get("npxG"))
        conc = pens_from_xg_gap(h.get("xGA"), h.get("npxGA"))
        match_pens.append((str(h.get("date") or ""), won, conc))
    match_pens.sort(key=lambda r: r[0])
    return UnderstatTeam(
        understat_id=str(row.get("id") or ""),
        title=str(row.get("title") or row.get("name") or ""),
        matches=len(played),
        xg=sum(_f(h.get("xG")) for h in played),
        xga=sum(_f(h.get("xGA")) for h in played),
        pens_won=sum(p[1] for p in match_pens),
        pens_conceded=sum(p[2] for p in match_pens),
        match_pens=tuple(match_pens),
    )


def parse_league_payload(data: dict[str, Any]) -> tuple[list[UnderstatPlayer], list[UnderstatTeam]]:
    raw_players = data.get("players") or data.get("playersData") or []
    raw_teams = data.get("teams") or data.get("teamsData") or {}
    players = [_player_from_row(r) for r in raw_players if isinstance(r, dict)]
    if isinstance(raw_teams, dict):
        team_rows = [v for v in raw_teams.values() if isinstance(v, dict)]
    else:
        team_rows = [r for r in raw_teams if isinstance(r, dict)]
    teams = [_team_from_row(r) for r in team_rows]
    return players, teams


def parse_html(html: str) -> tuple[list[UnderstatPlayer], list[UnderstatTeam]]:
    return parse_league_payload(extract_embedded_json(html))


def _name_keys(name: str) -> set[str]:
    n = normalize_person(name)
    if not n:
        return set()
    keys = {n}
    parts = n.split()
    keys.add(parts[-1])
    if len(parts) >= 2:
        keys.add(f"{parts[0]} {parts[-1]}")
        if len(parts[0]) == 1:
            keys.add(parts[-1])
    return keys


def _alias_keys(name: str) -> set[str]:
    n = normalize_person(name)
    keys = _name_keys(n)
    extra = PLAYER_ALIASES.get(n, ())
    for item in extra:
        keys |= _name_keys(item)
    # обратный индекс: если FPL web_name — алиас, подтянуть канон
    for canon, variants in PLAYER_ALIASES.items():
        if n == canon or n in variants or n in _name_keys(canon):
            keys |= _name_keys(canon)
            for v in variants:
                keys |= _name_keys(v)
    return keys


def _us_team_ids(team_title: str, index: dict[str, int]) -> set[int]:
    ids: set[int] = set()
    for part in (team_title or "").split(","):
        tid = index.get(canonical_team(part.strip()))
        if tid is not None:
            ids.add(tid)
    return ids


def match_players(
    fpl_players: Sequence[Player],
    us_players: Sequence[UnderstatPlayer],
    teams: Iterable[Team],
) -> tuple[dict[int, UnderstatPlayer], list[UnderstatPlayer]]:
    """FPL player_id -> Understat. Несопоставленные Understat (с минутами) возвращаются списком."""
    index = team_index(teams)
    by_full: dict[str, list[Player]] = {}
    by_web: dict[str, list[Player]] = {}
    by_last_team: dict[tuple[str, int], list[Player]] = {}
    by_last: dict[str, list[Player]] = {}
    for p in fpl_players:
        for key in _alias_keys(p.full_name) | _alias_keys(p.web_name) | _alias_keys(p.second_name):
            by_full.setdefault(key, []).append(p)
        web = normalize_person(p.web_name)
        if web:
            by_web.setdefault(web, []).append(p)
        tokens = [
            t
            for t in (
                *normalize_person(p.second_name).split(),
                *normalize_person(p.web_name).split(),
            )
            if len(t) >= 4
        ]
        for last in tokens:
            by_last.setdefault(last, []).append(p)
            by_last_team.setdefault((last, p.team), []).append(p)

    used: set[int] = set()
    matched: dict[int, UnderstatPlayer] = {}
    unmatched: list[UnderstatPlayer] = []

    def pick(cands: Sequence[Player], team_ids: set[int]) -> Player | None:
        alive = [c for c in cands if c.id not in used]
        if team_ids:
            same = [c for c in alive if c.team in team_ids]
            if same:
                alive = same
        if not alive:
            return None
        if len(alive) > 1:
            alive = sorted(alive, key=lambda c: (-c.minutes, c.id))
        return alive[0]

    for u in us_players:
        team_ids = _us_team_ids(u.team_title, index)
        keys = _alias_keys(u.player_name)
        last = normalize_person(u.player_name).split()[-1:] or [""]
        last_key = last[0]
        found: Player | None = None
        for key in keys:
            found = pick(by_full.get(key, []), team_ids) or pick(by_web.get(key, []), team_ids)
            if found:
                break
        if found is None and last_key:
            if team_ids:
                pooled: list[Player] = []
                for tid in team_ids:
                    pooled.extend(by_last_team.get((last_key, tid), []))
                found = pick(pooled, team_ids)
            if found is None:
                last_hits = by_last.get(last_key, [])
                if len({p.id for p in last_hits}) == 1:
                    found = pick(last_hits, team_ids)
        if found is None:
            if u.minutes >= 90:
                unmatched.append(u)
            continue
        used.add(found.id)
        matched[found.id] = u

    if unmatched:
        sample = ", ".join(f"{u.player_name} ({u.team_title})" for u in unmatched[:8])
        log.warning(
            "understat: не сопоставлено с FPL %d игроков (≥90 мин), например: %s",
            len(unmatched),
            sample,
        )
    return matched, unmatched


class UnderstatClient:
    """GET getLeagueData + fallback HTML; дисковый кэш `.cache/understat/epl_{season}.json`."""

    def __init__(
        self,
        *,
        season: int | None = None,
        cache_dir: Path | None = None,
        cache_ttl: int | None = None,
        timeout: float = TIMEOUT,
        clock: Callable[[], float] = time.time,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.season = settings.understat_season if season is None else season
        self.cache_path = (cache_dir or settings.cache_dir) / "understat" / f"epl_{self.season}.json"
        self.cache_ttl = settings.understat_cache_ttl if cache_ttl is None else cache_ttl
        self.timeout = timeout
        self.clock = clock
        self._transport = transport
        self.last_source: str = "none"

    def read_cache(self) -> tuple[dict[str, Any], float] | None:
        try:
            payload = json.loads(self.cache_path.read_text())
            return dict(payload["data"]), float(payload["fetched_at"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _write_cache(self, data: dict[str, Any]) -> None:
        now = self.clock()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(
                {
                    "fetched_at": now,
                    "fetched_at_iso": datetime.fromtimestamp(now, UTC).isoformat(),
                    "season": self.season,
                    "league": LEAGUE,
                    "data": data,
                }
            )
        )

    def _http(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout, headers=HEADERS, transport=self._transport)

    def _request_json(self) -> dict[str, Any]:
        url = DATA_URL.format(league=LEAGUE, season=self.season)
        with self._http() as http:
            resp = http.get(url)
            resp.raise_for_status()
            data = resp.json()
        if not isinstance(data, dict):
            raise TypeError("Understat getLeagueData: ожидался объект")
        return data

    def _request_html(self) -> dict[str, Any]:
        url = PAGE_URL.format(league=LEAGUE, season=self.season)
        with self._http() as http:
            resp = http.get(url, headers={**HEADERS, "Accept": "text/html"})
            resp.raise_for_status()
        extracted = extract_embedded_json(resp.text)
        if not extracted:
            raise ValueError("в HTML нет JSON.parse playersData/teamsData")
        return extracted

    def fetch_raw(self, *, force: bool = False) -> dict[str, Any]:
        cached = self.read_cache()
        if cached is not None and not force and self.cache_ttl > 0:
            age = self.clock() - cached[1]
            if age < self.cache_ttl:
                self.last_source = "cache"
                return cached[0]

        def stale(reason: str) -> dict[str, Any]:
            assert cached is not None
            age_h = (self.clock() - cached[1]) / 3600
            log.warning("%s — отдаём протухший кэш Understat (возраст %.1f ч)", reason, age_h)
            self.last_source = "stale-cache"
            return cached[0]

        try:
            data = self._request_json()
            source = "api"
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            log.info("understat getLeagueData недоступен (%s), пробуем HTML", type(exc).__name__)
            try:
                data = self._request_html()
                source = "html"
            except (httpx.HTTPError, ValueError, TypeError) as exc2:
                reason = f"Understat недоступен ({type(exc2).__name__})"
                if cached is not None:
                    return stale(reason)
                raise

        if self.cache_ttl > 0:
            self._write_cache(data)
        self.last_source = source
        return data

    def snapshot(self, *, force: bool = False) -> UnderstatSnapshot:
        """Снимок или пустой объект. Исключений наружу нет."""
        if not settings.understat_enabled:
            return UnderstatSnapshot(season=self.season, source="empty", note="understat_enabled=false")
        try:
            data = self.fetch_raw(force=force)
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            log.warning("understat: нет сети и нет кэша (%s) — внешняя стата выключена", type(exc).__name__)
            return UnderstatSnapshot(
                season=self.season,
                source="empty",
                note="Understat недоступен; используем только поля FPL",
            )
        players, teams = parse_league_payload(data)
        note = ""
        if not players:
            note = f"Understat EPL {self.season}/{self.season + 1} пуст — только поля FPL"
            log.warning("understat: %s", note)
        return UnderstatSnapshot(
            season=self.season,
            players=players,
            teams=teams,
            fetched_at=self.clock(),
            source=self.last_source,
            note=note,
        )
