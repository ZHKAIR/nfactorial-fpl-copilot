"""Букмекерские коэффициенты -> ожидаемые голы (The Odds API v4, рынок h2h = 1X2).

Слои:
- математика (чистые функции): снятие маржи, подбор (λ_home, λ_away) по рынку 1X2 (+ тотал)
  через Пуассон-сетку, вероятность сухого матча;
- OddsClient: GET /v4/sports/soccer_epl/odds (regions=eu, markets=h2h, decimal), таймаут 10 с,
  дисковый кэш .cache/odds/epl_h2h.json с TTL ODDS_CACHE_TTL (6 ч) — бесплатный тариф даёт
  500 запросов/месяц, заголовки x-requests-used/remaining пишутся в лог; при ошибке сети
  отдаётся протухший кэш с предупреждением;
- parse_snapshot / match_fixtures: медиана коэффициентов 1/X/2 по букмекерам и сопоставление
  матчей с фикстурами FPL по именам команд (словарь алиасов + нормализация) и дате;
- OddsProvider: гибрид — матчи, покрытые рынком (1–2 ближайших тура), считаются по odds,
  остальные прозрачно уходят в fallback (TeamRatingProvider); без ключа/сети/кэша — целиком
  fallback с одним warning, без исключений наружу.

ВАЖНО: коэффициенты — только внутренний вход модели. Пользователю показываются
xg_for/xg_against/clean_sheet_prob/fixture_strength_index (1–5) и источник (odds|team_rating),
но никогда не сами odds и не вероятности исходов. Ключ API не попадает в логи и сообщения ошибок.
"""

from __future__ import annotations

import json
import logging
import math
import re
import statistics
import time
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np

from fplcopilot.config import settings
from fplcopilot.data import Fixture, Team

if TYPE_CHECKING:
    from fplcopilot.core.fixtures import FixtureStrengthProvider

log = logging.getLogger(__name__)

LAMBDA_MIN = 0.1
LAMBDA_MAX = 4.5
COARSE_STEP = 0.02
FINE_STEP = 0.002
MAX_GOALS = 12  # усечение матрицы счёта; хвост P(>12) при λ<=4.5 < 1e-3


def remove_margin(decimal_odds: list[float]) -> list[float]:
    """Пропорциональное снятие маржи: p_i = (1/o_i) / Σ(1/o_j). Сумма = 1, порядок сохранён."""
    if not decimal_odds:
        return []
    if any(o <= 1.0 for o in decimal_odds):
        raise ValueError("десятичные коэффициенты должны быть > 1.0")
    implied = [1.0 / o for o in decimal_odds]
    total = sum(implied)
    return [p / total for p in implied]


def clean_sheet_prob(lambda_opponent: float) -> float:
    """P(соперник не забил) при Poisson(λ) = exp(-λ)."""
    return math.exp(-max(0.0, lambda_opponent))


def _pmf_table(lams: np.ndarray, max_goals: int = MAX_GOALS) -> np.ndarray:
    """(n, max_goals+1): P(X = g) для каждого λ в lams."""
    g = np.arange(max_goals + 1)
    log_fact = np.array([math.lgamma(i + 1) for i in g])
    return np.exp(-lams[:, None] + g[None, :] * np.log(lams[:, None]) - log_fact[None, :])


def poisson_match_probs(
    lam_home: float, lam_away: float, max_goals: int = MAX_GOALS
) -> tuple[float, float, float]:
    """(P(1), P(X), P(2)) для независимых Пуассонов."""
    ph = _pmf_table(np.array([lam_home]), max_goals)[0]
    pa = _pmf_table(np.array([lam_away]), max_goals)[0]
    m = np.outer(ph, pa)
    home = float(np.tril(m, -1).sum())
    draw = float(np.trace(m))
    away = float(np.triu(m, 1).sum())
    return home, draw, away


def prob_total_over(lam_home: float, lam_away: float, line: float) -> float:
    """P(X + Y > line); сумма Пуассонов — Пуассон(λh + λa)."""
    lam = lam_home + lam_away
    k = math.floor(line)
    cdf = sum(math.exp(-lam + i * math.log(lam) - math.lgamma(i + 1)) for i in range(k + 1))
    return max(0.0, 1.0 - cdf)


def _grid_probs(lh: np.ndarray, la: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Для сетки λ_home (n) × λ_away (m) — матрицы P(1), P(X), P(2) размера (n, m)."""
    ph = _pmf_table(lh)  # (n, G)
    pa = _pmf_table(la)  # (m, G)
    cum_a = np.cumsum(pa, axis=1)  # P(Y <= g)
    cum_h = np.cumsum(ph, axis=1)
    # P(home win) = Σ_g P(X=g) P(Y<=g-1)
    prev_a = np.concatenate([np.zeros((pa.shape[0], 1)), cum_a[:, :-1]], axis=1)
    prev_h = np.concatenate([np.zeros((ph.shape[0], 1)), cum_h[:, :-1]], axis=1)
    p_home = ph @ prev_a.T
    p_away = (pa @ prev_h.T).T
    p_draw = ph @ pa.T
    return p_home, p_draw, p_away


def xg_from_1x2_and_total(
    p_home: float,
    p_draw: float,
    p_away: float,
    total_line: float | None = 2.5,
    p_over: float | None = None,
    *,
    total_weight: float = 1.0,
) -> tuple[float, float]:
    """Подбор (λ_home, λ_away) под безмаржинальные вероятности рынка.

    Минимизируем сумму квадратов отклонений Пуассон-модели от (p_home, p_draw, p_away) и, если
    заданы total_line и p_over, от P(total > line). Сначала грубая сетка шагом 0.02, затем
    уточнение шагом 0.002 вокруг лучшей точки. Без поправки Диксона-Коулза (v0).
    """
    probs = np.array([p_home, p_draw, p_away], dtype=float)
    if np.any(probs < 0) or not math.isclose(float(probs.sum()), 1.0, abs_tol=0.02):
        raise ValueError("p_home + p_draw + p_away должны давать ~1 (снимите маржу remove_margin)")
    use_total = total_line is not None and p_over is not None

    def objective(lh: np.ndarray, la: np.ndarray) -> np.ndarray:
        ph, pd, pa = _grid_probs(lh, la)
        loss = (ph - p_home) ** 2 + (pd - p_draw) ** 2 + (pa - p_away) ** 2
        if use_total:
            k = math.floor(total_line)  # type: ignore[arg-type]
            lam = lh[:, None] + la[None, :]
            cdf = np.zeros_like(lam)
            for i in range(k + 1):
                cdf += np.exp(-lam + i * np.log(lam) - math.lgamma(i + 1))
            loss = loss + total_weight * ((1.0 - cdf) - p_over) ** 2
        return loss

    coarse = np.arange(LAMBDA_MIN, LAMBDA_MAX + 1e-9, COARSE_STEP)
    loss = objective(coarse, coarse)
    i, j = np.unravel_index(int(np.argmin(loss)), loss.shape)
    lh0, la0 = float(coarse[i]), float(coarse[j])

    span = COARSE_STEP
    fine_h = np.clip(np.arange(lh0 - span, lh0 + span + 1e-9, FINE_STEP), LAMBDA_MIN, LAMBDA_MAX)
    fine_a = np.clip(np.arange(la0 - span, la0 + span + 1e-9, FINE_STEP), LAMBDA_MIN, LAMBDA_MAX)
    loss = objective(fine_h, fine_a)
    i, j = np.unravel_index(int(np.argmin(loss)), loss.shape)
    return round(float(fine_h[i]), 3), round(float(fine_a[j]), 3)


# ---------- The Odds API: клиент и кэш ----------

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
ODDS_SPORT = "soccer_epl"
ODDS_REGIONS = "eu"
ODDS_MARKETS = "h2h"
ODDS_TIMEOUT = 10.0  # сек на запрос
ODDS_CACHE_FILE = "epl_h2h.json"
SOURCE_ODDS = "odds"
SOURCE_TEAM_RATING = "team_rating"
# Матч из API и фикстура FPL считаются одним событием, если совпала пара команд и кикофф
# отличается не больше этого (переносы внутри одной недели допустимы, разные круги — нет).
MATCH_TIME_TOLERANCE = timedelta(hours=72)


class OddsUnavailable(RuntimeError):
    """Снимок коэффициентов получить нельзя (нет ключа/сети и нет кэша). Ключ в текст не попадает."""


def _describe_error(exc: BaseException) -> str:
    """Короткое описание ошибки httpx без URL (в query string лежит apiKey)."""
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return type(exc).__name__


class OddsClient:
    """GET odds с дисковым кэшем по образцу FPLClient: JSON-файл + TTL.

    В файле хранится обёртка {fetched_at, requests_used, requests_remaining, data}, чтобы
    протухший кэш можно было отдать при ошибке сети и показать возраст снимка.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        cache_dir: Path | None = None,
        cache_ttl: int | None = None,
        timeout: float = ODDS_TIMEOUT,
        clock: Callable[[], float] = time.time,
        transport: httpx.BaseTransport | None = None,
        sport: str = ODDS_SPORT,
    ) -> None:
        self.api_key = api_key or None
        self.cache_path = (cache_dir or settings.cache_dir) / "odds" / ODDS_CACHE_FILE
        self.cache_ttl = settings.odds_cache_ttl if cache_ttl is None else cache_ttl
        self.timeout = timeout
        self.clock = clock
        self._transport = transport  # httpx.MockTransport в тестах
        self.sport = sport
        self.requests_used: str | None = None
        self.requests_remaining: str | None = None
        self.last_source: str | None = None  # api | cache | stale-cache

    # --- кэш ---

    def read_cache(self) -> tuple[list[dict[str, Any]], float] | None:
        """(data, fetched_at) или None, если файла нет / он битый."""
        try:
            payload = json.loads(self.cache_path.read_text())
            return list(payload["data"]), float(payload["fetched_at"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _write_cache(self, data: list[dict[str, Any]]) -> None:
        now = self.clock()
        payload = {
            "fetched_at": now,
            "fetched_at_iso": datetime.fromtimestamp(now, UTC).isoformat(),
            "sport": self.sport,
            "regions": ODDS_REGIONS,
            "markets": ODDS_MARKETS,
            "requests_used": self.requests_used,
            "requests_remaining": self.requests_remaining,
            "data": data,
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(payload))

    # --- сеть ---

    def _request(self) -> list[dict[str, Any]]:
        params = {
            "regions": ODDS_REGIONS,
            "markets": ODDS_MARKETS,
            "oddsFormat": "decimal",
            "apiKey": self.api_key,
        }
        # httpx пишет полный URL (с apiKey) в лог уровня INFO — на время запроса глушим.
        httpx_log = logging.getLogger("httpx")
        prev_level = httpx_log.level
        httpx_log.setLevel(max(prev_level, logging.WARNING))
        try:
            with httpx.Client(timeout=self.timeout, transport=self._transport) as http:
                resp = http.get(ODDS_API_URL.format(sport=self.sport), params=params)
        finally:
            httpx_log.setLevel(prev_level)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            raise TypeError("The Odds API: ожидался список матчей")
        self.requests_used = resp.headers.get("x-requests-used")
        self.requests_remaining = resp.headers.get("x-requests-remaining")
        log.info(
            "The Odds API: %d матчей %s/%s; запросов использовано %s, осталось %s",
            len(data),
            self.sport,
            ODDS_MARKETS,
            self.requests_used,
            self.requests_remaining,
        )
        return data

    def fetch_raw(self, *, force: bool = False) -> list[dict[str, Any]]:
        """Сырой ответ API: свежий кэш -> сеть -> протухший кэш; иначе OddsUnavailable."""
        cached = self.read_cache()
        if cached is not None and not force and self.cache_ttl > 0:
            age = self.clock() - cached[1]
            if age < self.cache_ttl:
                log.debug("odds cache hit (age %.0fs)", age)
                self.last_source = "cache"
                return cached[0]

        def stale(reason: str) -> list[dict[str, Any]]:
            assert cached is not None
            age_h = (self.clock() - cached[1]) / 3600
            log.warning("%s — отдаём протухший кэш коэффициентов (возраст %.1f ч)", reason, age_h)
            self.last_source = "stale-cache"
            return cached[0]

        if not self.api_key:
            if cached is not None:
                return stale("ODDS_API_KEY пуст")
            raise OddsUnavailable("ODDS_API_KEY не задан и кэша коэффициентов нет")
        try:
            data = self._request()
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            reason = f"The Odds API недоступен ({_describe_error(exc)})"
            if cached is not None:
                return stale(reason)
            raise OddsUnavailable(reason) from exc
        if self.cache_ttl > 0:
            self._write_cache(data)
        self.last_source = "api"
        return data


# ---------- снимок: парсинг и агрегация по букмекерам ----------


@dataclass(frozen=True)
class MatchOdds:
    """Матч из снимка с агрегированными по букмекерам десятичными коэффициентами 1/X/2.

    Внутренняя структура провайдера: наружу (FixtureView/FixtureInput) уходят только λ.
    """

    home_team: str
    away_team: str
    commence_time: datetime
    home: float
    draw: float
    away: float
    bookmakers: int

    def probabilities(self) -> tuple[float, float, float]:
        p = remove_margin([self.home, self.draw, self.away])
        return p[0], p[1], p[2]

    def expected_goals(self) -> tuple[float, float]:
        """(λ_home, λ_away) по безмаржинальному 1X2 через Пуассон-сетку."""
        ph, pd, pa = self.probabilities()
        return xg_from_1x2_and_total(ph, pd, pa, None, None)


def aggregate_prices(
    prices: Sequence[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Медиана по букмекерам для каждого исхода: устойчива к выбросам одиночных линий."""
    if not prices:
        raise ValueError("нет коэффициентов для агрегации")
    return (
        statistics.median(p[0] for p in prices),
        statistics.median(p[1] for p in prices),
        statistics.median(p[2] for p in prices),
    )


def _parse_time(value: str) -> datetime:
    dt = datetime.fromisoformat(value)  # '2026-10-10T11:30:00Z' -> aware UTC
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def parse_snapshot(raw: Iterable[dict[str, Any]]) -> list[MatchOdds]:
    """Ответ /v4/sports/{sport}/odds -> MatchOdds по матчу (только полные линии 1/X/2 > 1.0)."""
    out: list[MatchOdds] = []
    for event in raw:
        home, away = event.get("home_team"), event.get("away_team")
        if not home or not away or not event.get("commence_time"):
            log.warning("odds: событие без команд/времени пропущено: %s", event.get("id"))
            continue
        prices: list[tuple[float, float, float]] = []
        for book in event.get("bookmakers") or []:
            market = next(
                (m for m in book.get("markets") or [] if m.get("key") == ODDS_MARKETS), None
            )
            if market is None:
                continue
            by_name = {o.get("name"): o.get("price") for o in market.get("outcomes") or []}
            h, d, a = by_name.get(home), by_name.get("Draw"), by_name.get(away)
            if all(isinstance(x, int | float) and x > 1.0 for x in (h, d, a)):
                prices.append((float(h), float(d), float(a)))  # type: ignore[arg-type]
        if not prices:
            log.warning("odds: %s v %s — ни одной полной линии h2h, пропускаем", home, away)
            continue
        h, d, a = aggregate_prices(prices)
        out.append(MatchOdds(home, away, _parse_time(event["commence_time"]), h, d, a, len(prices)))
    return out


# ---------- сопоставление команд и фикстур ----------

# canonical -> варианты написания (The Odds API, FPL name/short_name, обиходные). Все 20 клубов
# 2026/27 + клубы, которые ходят между АПЛ и Чемпионшипом, чтобы словарь пережил смену сезона.
TEAM_ALIASES: dict[str, tuple[str, ...]] = {
    "arsenal": ("ars",),
    "aston villa": ("villa", "avl"),
    "bournemouth": ("afc bournemouth", "bou"),
    "brentford": ("bre",),
    "brighton": ("brighton and hove albion", "brighton hove albion", "bha"),
    "chelsea": ("che",),
    "coventry city": ("coventry", "cov"),
    "crystal palace": ("palace", "cry"),
    "everton": ("eve",),
    "fulham": ("ful",),
    "hull city": ("hull", "hul"),
    "ipswich town": ("ipswich", "ips"),
    "leeds united": ("leeds", "lee"),
    "liverpool": ("liv",),
    "manchester city": ("man city", "mci"),
    "manchester united": ("man utd", "man united", "mun"),
    "newcastle united": ("newcastle", "new"),
    "nottingham forest": ("nottm forest", "nott m forest", "forest", "nfo"),
    "tottenham hotspur": ("tottenham", "spurs", "tot"),
    "sunderland": ("sun",),
    "west ham united": ("west ham", "whu"),
    "wolverhampton wanderers": ("wolves", "wolverhampton", "wol"),
    "leicester city": ("leicester", "lei"),
    "southampton": ("sou",),
    "burnley": ("bur",),
    "luton town": ("luton", "lut"),
    "sheffield united": ("sheffield utd", "shu"),
    "norwich city": ("norwich", "nor"),
    "watford": ("wat",),
    "west bromwich albion": ("west brom", "wba"),
    "middlesbrough": ("boro", "mid"),
    "derby county": ("derby", "der"),
    "stoke city": ("stoke", "stk"),
    "swansea city": ("swansea", "swa"),
    "cardiff city": ("cardiff", "car"),
    "huddersfield town": ("huddersfield", "hud"),
    "blackburn rovers": ("blackburn", "bla"),
    "queens park rangers": ("qpr",),
    "wigan athletic": ("wigan", "wig"),
    "bolton wanderers": ("bolton", "bol"),
}


def normalize_team_name(name: str) -> str:
    """Нижний регистр, без диакритики/апострофов/пунктуации, '&' -> 'and', без FC/AFC."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("&", " and ").replace("'", "").replace("’", "").replace(".", "")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\b(fc|afc)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_ALIAS_INDEX: dict[str, str] = {
    normalize_team_name(v): canon
    for canon, variants in TEAM_ALIASES.items()
    for v in (canon, *variants)
}


def canonical_team(name: str) -> str:
    """Каноническое имя клуба; незнакомое имя возвращается нормализованным как есть."""
    n = normalize_team_name(name)
    return _ALIAS_INDEX.get(n, n)


def team_index(teams: Iterable[Team]) -> dict[str, int]:
    """canonical -> team_id по FPL name, затем short_name (short_name не перекрывает name)."""
    index: dict[str, int] = {}
    for t in teams:
        index.setdefault(canonical_team(t.name), t.id)
    for t in teams:
        index.setdefault(canonical_team(t.short_name), t.id)
    return index


def _kickoff_gap(fixture: Fixture, when: datetime, tolerance: timedelta) -> timedelta:
    """|kickoff_time − commence_time|; фикстура без даты (перенос) — заведомо вне допуска."""
    if fixture.kickoff_time is None:
        return tolerance * 2
    return abs(fixture.kickoff_time - when)


def match_fixtures(
    matches: Sequence[MatchOdds],
    fixtures: Sequence[Fixture],
    teams: Iterable[Team],
    *,
    tolerance: timedelta = MATCH_TIME_TOLERANCE,
) -> tuple[dict[int, MatchOdds], list[MatchOdds]]:
    """fixture_id -> MatchOdds по паре (хозяева, гости) и близости кикоффа; остальное — unmatched.

    Для несопоставленного матча пишется warning: он останется на fallback-провайдере.
    """
    index = team_index(teams)
    by_pair: dict[tuple[int, int], list[Fixture]] = {}
    for f in fixtures:
        by_pair.setdefault((f.team_h, f.team_a), []).append(f)

    matched: dict[int, MatchOdds] = {}
    unmatched: list[MatchOdds] = []
    for m in matches:
        h, a = index.get(canonical_team(m.home_team)), index.get(canonical_team(m.away_team))
        if h is None or a is None:
            missing = m.home_team if h is None else m.away_team
            log.warning(
                "odds: клуб %r не сопоставлен с FPL — %s v %s на fallback",
                missing,
                m.home_team,
                m.away_team,
            )
            unmatched.append(m)
            continue
        candidates = by_pair.get((h, a), [])
        best = min(
            candidates, key=lambda f: _kickoff_gap(f, m.commence_time, tolerance), default=None
        )
        if best is None or _kickoff_gap(best, m.commence_time, tolerance) > tolerance:
            log.warning(
                "odds: %s v %s (%s) не совпал с фикстурой FPL по паре/дате — на fallback",
                m.home_team,
                m.away_team,
                m.commence_time.date(),
            )
            unmatched.append(m)
            continue
        matched[best.id] = m
    return matched, unmatched


# ---------- провайдер силы фикстур ----------


class OddsProvider:
    """Сила фикстур по букмекерам с прозрачным fallback (XPTS_FIXTURE_PROVIDER=odds).

    Снимок загружается лениво при первом обращении и сопоставляется с фикстурами сезона.
    Матч с рынком -> remove_margin -> xg_from_1x2_and_total -> (xg_home, xg_away); матч без
    рынка (дальние туры, несопоставленные имена) -> fallback (TeamRatingProvider). Если снимка
    нет вовсе (ключ пуст, сеть и кэш недоступны) — один warning, всё считает fallback.

    league_avg: по одному рынку 1X2 уровень голов идентифицируется плохо — Пуассон занижает
    ничьи, и подгонка под рыночную P(X) даёт тотал ≈ 2.55 вместо ≈ 2.9 в АПЛ. Поэтому, как и
    TeamRatingProvider, все λ снимка масштабируются к среднему league_avg (относительные силы и
    доля хозяев остаются рыночными). None — сырые λ рынка (для тестов и рынка totals).
    """

    name = SOURCE_ODDS

    def __init__(
        self,
        api_key: str | None = None,
        *,
        teams: Iterable[Team] = (),
        fixtures: Iterable[Fixture] = (),
        fallback: FixtureStrengthProvider | None = None,
        client: OddsClient | None = None,
        league_avg: float | None = None,
    ) -> None:
        self.api_key = api_key
        self.teams = list(teams)
        self.fixtures = list(fixtures)
        self.fallback = fallback
        self.client = client or OddsClient(api_key)
        self.league_avg = league_avg
        self.scale = 1.0
        self._matched: dict[int, MatchOdds] | None = None
        self._xg: dict[int, tuple[float, float]] = {}
        self.unmatched: list[MatchOdds] = []

    # --- снимок ---

    def load(self) -> dict[int, MatchOdds]:
        """fixture_id -> MatchOdds; при недоступности API — пустой словарь и один warning."""
        if self._matched is not None:
            return self._matched
        try:
            raw = self.client.fetch_raw()
        except OddsUnavailable as exc:
            fb = self.fallback.name if self.fallback is not None else "нет"
            log.warning("odds: %s — сила всех фикстур по fallback-провайдеру (%s)", exc, fb)
            self._matched = {}
            return self._matched
        matches = parse_snapshot(raw)
        self._matched, self.unmatched = match_fixtures(matches, self.fixtures, self.teams)
        self._xg = {fid: m.expected_goals() for fid, m in self._matched.items()}
        if self.league_avg and self._xg:
            lams = [x for pair in self._xg.values() for x in pair]
            self.scale = self.league_avg / (sum(lams) / len(lams))
            self._xg = {
                fid: (round(h * self.scale, 3), round(a * self.scale, 3))
                for fid, (h, a) in self._xg.items()
            }
        gws = sorted(
            {f.event for f in self.fixtures if f.id in self._matched and f.event is not None}
        )
        log.info(
            "odds: сопоставлено %d/%d матчей с фикстурами FPL (туры %s, источник %s, "
            "масштаб λ %.3f)",
            len(self._matched),
            len(matches),
            gws or "-",
            self.client.last_source,
            self.scale,
        )
        return self._matched

    def fetch_snapshot(self) -> dict[int, tuple[float, float]]:
        """fixture_id -> (λ_home, λ_away) для всех сопоставленных матчей снимка."""
        self.load()
        return dict(self._xg)

    def covered_gws(self) -> list[int]:
        """Туры, хотя бы один матч которых покрыт рынком."""
        ids = self.load()
        return sorted({f.event for f in self.fixtures if f.id in ids and f.event is not None})

    # --- протокол FixtureStrengthProvider ---

    def source(self, fixture: Fixture) -> str:
        """odds, если матч покрыт рынком; иначе имя fallback-провайдера."""
        if fixture.id in self.load():
            return SOURCE_ODDS
        return self.fallback.name if self.fallback is not None else SOURCE_TEAM_RATING

    def expected_goals(self, fixture: Fixture) -> tuple[float, float]:
        self.load()
        if fixture.id in self._xg:
            return self._xg[fixture.id]
        if self.fallback is None:
            raise KeyError(f"нет коэффициентов для фикстуры {fixture.id} и нет fallback-провайдера")
        return self.fallback.expected_goals(fixture)
