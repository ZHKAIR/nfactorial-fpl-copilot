"""OddsProvider (The Odds API v4): парсинг снимка, агрегация по букмекерам, маппинг имён клубов,
сопоставление с фикстурами FPL, кэш/TTL, fallback без ключа/сети, гибрид odds+team_rating и
guardrail «коэффициенты наружу не выходят». Без сети; живой запрос — под маркером network."""

import json
import logging
import math
import statistics
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from fplcopilot.config import settings
from fplcopilot.core.fixtures import (
    LEAGUE_AVG_XG,
    FixtureView,
    TeamRatingProvider,
    fixture_views,
    get_provider,
    provider_source,
)
from fplcopilot.core.odds import (
    MatchOdds,
    OddsClient,
    OddsProvider,
    OddsUnavailable,
    aggregate_prices,
    canonical_team,
    match_fixtures,
    normalize_team_name,
    parse_snapshot,
    team_index,
)
from fplcopilot.core.xpts import FixtureInput, make_context, predict_all
from fplcopilot.data.schemas import Bootstrap, Fixture, Team

SAMPLE_PATH = Path(__file__).parent / "fixtures" / "odds_epl_sample.json"
SECRET = "k-secret-odds-key"

# 20 клубов 2026/27 так, как их отдаёт bootstrap FPL (id, name, short_name)
TEAMS_2026 = [
    (1, "Arsenal", "ARS"),
    (2, "Aston Villa", "AVL"),
    (3, "Bournemouth", "BOU"),
    (4, "Brentford", "BRE"),
    (5, "Brighton", "BHA"),
    (6, "Chelsea", "CHE"),
    (7, "Coventry City", "COV"),
    (8, "Crystal Palace", "CRY"),
    (9, "Everton", "EVE"),
    (10, "Fulham", "FUL"),
    (11, "Hull City", "HUL"),
    (12, "Ipswich Town", "IPS"),
    (13, "Leeds", "LEE"),
    (14, "Liverpool", "LIV"),
    (15, "Man City", "MCI"),
    (16, "Man Utd", "MUN"),
    (17, "Newcastle", "NEW"),
    (18, "Nott'm Forest", "NFO"),
    (19, "Spurs", "TOT"),
    (20, "Sunderland", "SUN"),
]
# те же клубы в The Odds API (живой ответ 22.09.2026)
ODDS_API_NAMES = {
    "Arsenal": 1,
    "Aston Villa": 2,
    "Bournemouth": 3,
    "Brentford": 4,
    "Brighton and Hove Albion": 5,
    "Chelsea": 6,
    "Coventry City": 7,
    "Crystal Palace": 8,
    "Everton": 9,
    "Fulham": 10,
    "Hull City": 11,
    "Ipswich Town": 12,
    "Leeds United": 13,
    "Liverpool": 14,
    "Manchester City": 15,
    "Manchester United": 16,
    "Newcastle United": 17,
    "Nottingham Forest": 18,
    "Tottenham Hotspur": 19,
    "Sunderland": 20,
}


def _teams() -> list[Team]:
    return [
        Team(id=i, name=n, short_name=s, strength_overall_home=3, strength_overall_away=3)
        for i, n, s in TEAMS_2026
    ]


def _fixtures() -> list[Fixture]:
    """Фикстуры FPL под матчи сэмпла (GW6–7), одна GW8 без рынка и «двойник» пары ARS–LEE
    без даты (перенос) — сопоставление должно выбрать матч с ближайшим кикоффом."""
    rows = [
        {"id": 51, "event": 6, "team_h": 1, "team_a": 13, "kickoff_time": "2026-10-10T11:30:00Z"},
        {"id": 60, "event": 6, "team_h": 20, "team_a": 5, "kickoff_time": "2026-10-10T14:00:00Z"},
        {"id": 59, "event": 6, "team_h": 16, "team_a": 19, "kickoff_time": "2026-10-10T16:30:00Z"},
        {"id": 55, "event": 6, "team_h": 8, "team_a": 18, "kickoff_time": "2026-10-11T13:00:00Z"},
        {"id": 69, "event": 7, "team_h": 18, "team_a": 1, "kickoff_time": "2026-10-18T15:30:00Z"},
        {"id": 71, "event": 8, "team_h": 1, "team_a": 9, "kickoff_time": "2026-10-24T14:00:00Z"},
        {"id": 300, "event": None, "team_h": 1, "team_a": 13, "kickoff_time": None},
    ]
    return [Fixture.model_validate(r) for r in rows]


def _sample_raw() -> list[dict]:
    return json.loads(SAMPLE_PATH.read_text())


def _mock_client(
    tmp_path: Path,
    handler,
    *,
    api_key: str | None = SECRET,
    ttl: int = 3600,
    clock=None,
) -> OddsClient:
    return OddsClient(
        api_key,
        cache_dir=tmp_path,
        cache_ttl=ttl,
        clock=clock or (lambda: 1_790_000_000.0),
        transport=httpx.MockTransport(handler),
    )


def _ok_handler(payload, headers=None):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, headers=headers or {})

    return handler


# ---------- парсинг и агрегация ----------


def test_sample_fixture_parses_with_median_over_bookmakers():
    raw = _sample_raw()
    assert len(raw) == 5 and all(len(ev["bookmakers"]) == 3 for ev in raw)
    matches = parse_snapshot(raw)
    assert len(matches) == 5
    assert all(m.bookmakers == 3 and m.commence_time.tzinfo is not None for m in matches)

    ars = next(m for m in matches if m.home_team == "Arsenal" and m.away_team == "Leeds United")
    prices = {o["name"]: [] for b in raw[0]["bookmakers"] for o in b["markets"][0]["outcomes"]}
    for b in raw[0]["bookmakers"]:
        for o in b["markets"][0]["outcomes"]:
            prices[o["name"]].append(o["price"])
    assert ars.home == statistics.median(prices["Arsenal"])
    assert ars.draw == statistics.median(prices["Draw"])
    assert ars.away == statistics.median(prices["Leeds United"])
    assert ars.commence_time == datetime(2026, 10, 10, 11, 30, tzinfo=UTC)

    ph, pd, pa = ars.probabilities()
    assert math.isclose(ph + pd + pa, 1.0, abs_tol=1e-9) and ph > pa
    lam_h, lam_a = ars.expected_goals()
    assert lam_h > 1.7 and 0.4 < lam_a < 1.0  # явный фаворит дома


def test_aggregate_prices_is_median_per_outcome_and_ignores_outlier():
    prices = [(1.5, 4.0, 6.0), (1.52, 4.2, 6.5), (9.0, 4.1, 6.2)]  # третья линия — ошибка
    assert aggregate_prices(prices) == (1.52, 4.1, 6.2)
    assert aggregate_prices([(2.0, 3.0, 4.0), (2.2, 3.2, 4.4)]) == (2.1, 3.1, 4.2)
    with pytest.raises(ValueError):
        aggregate_prices([])


def test_parse_skips_incomplete_lines_and_events_without_market(caplog):
    raw = _sample_raw()
    # у одного букмекера нет цены на гостей -> линия не учитывается
    raw[0]["bookmakers"][0]["markets"][0]["outcomes"] = [
        o for o in raw[0]["bookmakers"][0]["markets"][0]["outcomes"] if o["name"] != "Draw"
    ]
    # у второго события вообще нет h2h -> событие пропущено
    for b in raw[1]["bookmakers"]:
        b["markets"] = [{"key": "totals", "outcomes": []}]
    # цена <= 1.0 невалидна
    raw[2]["bookmakers"][0]["markets"][0]["outcomes"][0]["price"] = 1.0
    with caplog.at_level(logging.WARNING, logger="fplcopilot.core.odds"):
        matches = parse_snapshot(raw)
    assert len(matches) == 4
    by_home = {m.home_team: m for m in matches}
    assert by_home["Arsenal"].bookmakers == 2
    assert by_home[raw[2]["home_team"]].bookmakers == 2
    assert raw[1]["home_team"] not in by_home
    assert any("ни одной полной линии" in r.message for r in caplog.records)


# ---------- имена клубов ----------


def test_all_20_clubs_map_from_odds_api_names_to_fpl_teams():
    index = team_index(_teams())
    for name, team_id in ODDS_API_NAMES.items():
        assert index[canonical_team(name)] == team_id, name
    # обратное направление: FPL name и short_name ведут к тому же клубу
    for team_id, name, short in TEAMS_2026:
        assert index[canonical_team(name)] == team_id
        assert index[canonical_team(short)] == team_id


@pytest.mark.parametrize(
    "a,b",
    [
        ("Nott'm Forest", "Nottingham Forest"),
        ("Brighton & Hove Albion", "Brighton and Hove Albion"),
        ("Brighton", "Brighton and Hove Albion"),
        ("AFC Bournemouth", "Bournemouth"),
        ("Tottenham Hotspur FC", "Spurs"),
        ("Man Utd", "Manchester United"),
        ("Wolves", "Wolverhampton Wanderers"),
        ("West Ham", "West Ham United"),
        ("Leicester", "Leicester City"),
        ("  Leeds  united ", "Leeds"),
    ],
)
def test_aliases_and_normalisation(a, b):
    assert canonical_team(a) == canonical_team(b)


def test_normalize_strips_diacritics_and_punctuation():
    assert normalize_team_name("Atlético de Madrid F.C.") == "atletico de madrid"
    assert normalize_team_name("Nott'm Forest") == "nottm forest"
    # незнакомый клуб не ломает маппинг: возвращается нормализованным как есть
    assert canonical_team("Real Sociedad") == "real sociedad"


# ---------- сопоставление с фикстурами ----------


def test_match_fixtures_by_team_pair_and_kickoff(caplog):
    matches = parse_snapshot(_sample_raw())
    with caplog.at_level(logging.WARNING, logger="fplcopilot.core.odds"):
        matched, unmatched = match_fixtures(matches, _fixtures(), _teams())
    assert set(matched) == {51, 60, 59, 55, 69}  # не 300 (перенос без даты) и не 71 (GW8)
    assert unmatched == [] and not caplog.records
    assert matched[51].home_team == "Arsenal" and matched[69].home_team == "Nottingham Forest"


def test_unknown_club_and_wrong_date_go_to_unmatched_with_warning(caplog):
    good = parse_snapshot(_sample_raw())[0]
    stranger = MatchOdds("Arsenal", "Real Sociedad", good.commence_time, 1.5, 4.0, 6.0, 3)
    far = MatchOdds(
        "Arsenal", "Leeds United", good.commence_time + timedelta(days=30), 1.5, 4.0, 6.0, 3
    )
    with caplog.at_level(logging.WARNING, logger="fplcopilot.core.odds"):
        matched, unmatched = match_fixtures([good, stranger, far], _fixtures(), _teams())
    assert set(matched) == {51} and unmatched == [stranger, far]
    messages = [r.message for r in caplog.records]
    assert any("Real Sociedad" in m and "не сопоставлен" in m for m in messages)
    assert any("не совпал с фикстурой FPL по паре/дате" in m for m in messages)


# ---------- клиент: кэш, TTL, протухший кэш, лимиты ----------


def test_cache_ttl_stale_fallback_and_quota_logging(tmp_path, caplog):
    calls = {"n": 0, "fail": False}
    payload = _sample_raw()

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        assert request.url.params["apiKey"] == SECRET
        assert request.url.params["markets"] == "h2h" and request.url.params["regions"] == "eu"
        if calls["fail"]:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(
            200,
            json=payload,
            headers={"x-requests-used": "7", "x-requests-remaining": "493"},
        )

    now = [1_790_000_000.0]
    client = _mock_client(tmp_path, handler, ttl=3600, clock=lambda: now[0])
    with caplog.at_level(logging.INFO, logger="fplcopilot.core.odds"):
        assert client.fetch_raw() == payload and client.last_source == "api"
        assert calls["n"] == 1
        assert client.cache_path == tmp_path / "odds" / "epl_h2h.json"
        assert client.cache_path.exists()
        assert (client.requests_used, client.requests_remaining) == ("7", "493")
        assert any("использовано 7, осталось 493" in r.message for r in caplog.records)

        # в пределах TTL — кэш, сеть не трогаем
        assert client.fetch_raw() == payload and client.last_source == "cache"
        assert calls["n"] == 1

        # TTL истёк — новый запрос
        now[0] += 3601
        assert client.fetch_raw() == payload and client.last_source == "api"
        assert calls["n"] == 2

        # TTL истёк, сеть упала — протухший кэш с предупреждением
        now[0] += 3601
        calls["fail"] = True
        assert client.fetch_raw() == payload and client.last_source == "stale-cache"
        assert calls["n"] == 3
    stale = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(stale) == 1 and "протухший кэш" in stale[0].message and "1.0 ч" in stale[0].message
    assert SECRET not in caplog.text  # ключ не попадает в логи

    # без ключа, но с кэшем — тоже протухший кэш, без исключения
    no_key = OddsClient(None, cache_dir=tmp_path, cache_ttl=1, clock=lambda: now[0] + 10)
    assert no_key.fetch_raw() == payload and no_key.last_source == "stale-cache"


def test_http_error_without_cache_raises_unavailable_and_hides_key(tmp_path, caplog):
    def unauthorized(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "invalid key"})

    client = _mock_client(tmp_path, unauthorized)
    with caplog.at_level(logging.DEBUG), pytest.raises(OddsUnavailable) as exc:
        client.fetch_raw()
    assert "HTTP 401" in str(exc.value) and SECRET not in str(exc.value)
    assert SECRET not in caplog.text
    assert not client.cache_path.exists()

    with pytest.raises(OddsUnavailable, match="ODDS_API_KEY не задан"):
        OddsClient(None, cache_dir=tmp_path, cache_ttl=3600).fetch_raw()


def test_zero_ttl_disables_cache(tmp_path):
    client = _mock_client(tmp_path, _ok_handler(_sample_raw()), ttl=0)
    assert len(client.fetch_raw()) == 5 and not client.cache_path.exists()


# ---------- провайдер: гибрид и fallback ----------


def _provider(tmp_path, handler=None, **kw) -> OddsProvider:
    teams, fixtures = _teams(), _fixtures()
    fallback = TeamRatingProvider(teams, None, calibration_fixtures=fixtures)
    client = _mock_client(tmp_path, handler or _ok_handler(_sample_raw()), **kw)
    return OddsProvider(SECRET, teams=teams, fixtures=fixtures, fallback=fallback, client=client)


def test_provider_uses_odds_for_covered_fixtures_and_team_rating_for_the_rest(tmp_path):
    provider = _provider(tmp_path)
    fixtures = {f.id: f for f in _fixtures()}
    assert provider.name == "odds" and provider.covered_gws() == [6, 7]

    snapshot = provider.fetch_snapshot()
    assert set(snapshot) == {51, 60, 59, 55, 69}
    lam_h, lam_a = provider.expected_goals(fixtures[51])
    assert (lam_h, lam_a) == snapshot[51] and lam_h > lam_a
    assert provider.source(fixtures[51]) == "odds"

    # GW8 рынком не покрыт -> ровно то, что даёт TeamRatingProvider
    assert provider.expected_goals(fixtures[71]) == provider.fallback.expected_goals(fixtures[71])
    assert provider.source(fixtures[71]) == "team_rating"
    assert provider_source(provider, fixtures[71]) == "team_rating"

    gw6 = fixture_views(provider, list(fixtures.values()), 6)
    assert {fv.fsi_source for views in gw6.values() for fv in views} == {"odds"}
    gw8 = fixture_views(provider, list(fixtures.values()), 8)
    assert {fv.fsi_source for views in gw8.values() for fv in views} == {"team_rating"}
    ars_home = next(fv for fv in gw6[1] if fv.is_home)
    assert ars_home.fixture_strength_index <= 2 and ars_home.xg_for == lam_h
    assert math.isclose(ars_home.clean_sheet_prob, math.exp(-lam_a))


def test_league_avg_scales_snapshot_to_target_level_keeping_ratios(tmp_path):
    raw = _provider(tmp_path)
    scaled = OddsProvider(
        SECRET,
        teams=_teams(),
        fixtures=_fixtures(),
        fallback=raw.fallback,
        client=_mock_client(tmp_path, _ok_handler(_sample_raw())),
        league_avg=LEAGUE_AVG_XG,
    )
    lams = [x for pair in scaled.fetch_snapshot().values() for x in pair]
    assert math.isclose(sum(lams) / len(lams), LEAGUE_AVG_XG, rel_tol=0.01)
    assert raw.scale == 1.0 and scaled.scale != 1.0
    h0, a0 = raw.fetch_snapshot()[51]
    h1, a1 = scaled.fetch_snapshot()[51]
    assert math.isclose(h1 / a1, h0 / a0, rel_tol=0.02)  # доля хозяев осталась рыночной


def test_without_key_and_cache_provider_falls_back_entirely_with_one_warning(tmp_path, caplog):
    teams, fixtures = _teams(), _fixtures()
    fallback = TeamRatingProvider(teams, None, calibration_fixtures=fixtures)
    client = OddsClient(None, cache_dir=tmp_path, cache_ttl=3600)
    provider = OddsProvider(None, teams=teams, fixtures=fixtures, fallback=fallback, client=client)
    with caplog.at_level(logging.WARNING, logger="fplcopilot.core.odds"):
        for f in fixtures:
            assert provider.expected_goals(f) == fallback.expected_goals(f)
            assert provider.source(f) == "team_rating"
        assert provider.fetch_snapshot() == {} and provider.covered_gws() == []
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "ODDS_API_KEY не задан" in warnings[0].message


def test_network_failure_without_cache_falls_back_and_never_raises(tmp_path, caplog):
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    provider = _provider(tmp_path, down)
    with caplog.at_level(logging.WARNING, logger="fplcopilot.core.odds"):
        fx = _fixtures()[0]
        assert provider.expected_goals(fx) == provider.fallback.expected_goals(fx)
    assert [r.message for r in caplog.records if "ReadTimeout" in r.message]
    assert SECRET not in caplog.text


def test_get_provider_builds_hybrid_from_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "odds_api_key", None)
    monkeypatch.setattr(settings, "cache_dir", tmp_path)
    monkeypatch.setattr(settings, "xpts_fixture_provider", "odds")
    provider = get_provider(teams=_teams(), calibration_fixtures=_fixtures())
    assert isinstance(provider, OddsProvider)
    assert isinstance(provider.fallback, TeamRatingProvider)
    assert provider.league_avg == LEAGUE_AVG_XG
    assert provider.client.cache_path == tmp_path / "odds" / "epl_h2h.json"
    fx = _fixtures()[0]
    assert provider.expected_goals(fx) == provider.fallback.expected_goals(fx)  # ключа нет


# ---------- guardrail: коэффициенты не выходят наружу ----------

# в описании матча не должно быть ни цен, ни исходов, ни букмекеров
FORBIDDEN_KEYS = {"odds", "price", "prices", "home", "draw", "away", "bookmaker", "bookmakers"}
# на уровне всего прогноза 'price' — цена игрока в £, поэтому проверяем без неё
FORBIDDEN_TOP_KEYS = FORBIDDEN_KEYS - {"price"}


def _keys(obj, acc: set[str]) -> set[str]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            acc.add(str(k).lower())
            _keys(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _keys(v, acc)
    return acc


def test_user_facing_schemas_carry_only_fsi_xg_cs_and_source():
    from fplcopilot.agent.tools import FixtureBrief

    view_fields = set(FixtureView.__dataclass_fields__)
    assert view_fields == {
        "fixture_id",
        "gw",
        "team_id",
        "opponent_id",
        "is_home",
        "kickoff_time",
        "xg_for",
        "xg_against",
        "fsi_source",
    }
    assert FixtureInput.model_fields["fsi_source"].default == "team_rating"
    assert FixtureBrief.model_fields["fsi_source"].default == "team_rating"
    for fields in (FixtureInput.model_fields, FixtureBrief.model_fields):
        assert not (set(fields) & FORBIDDEN_KEYS)


def test_prediction_json_has_source_label_but_no_odds(tmp_path):
    """Синтетический bootstrap (как в test_core_xpts) + снимок с ценами: наружу только FSI/xG/CS."""
    bs = Bootstrap.model_validate(
        {
            "events": [
                {"id": 1, "name": "GW1", "deadline_time": "2026-08-21T17:30:00Z", "finished": True},
                {"id": 2, "name": "GW2", "deadline_time": "2026-08-28T17:30:00Z", "is_next": True},
            ],
            "teams": [
                {"id": 1, "name": "Alpha", "short_name": "ALP", "strength_overall_home": 4},
                {"id": 2, "name": "Beta", "short_name": "BET", "strength_overall_home": 3},
            ],
            "elements": [
                {"id": 1, "web_name": "P1", "team": 1, "element_type": 3, "now_cost": 50},
                {"id": 2, "web_name": "P2", "team": 2, "element_type": 2, "now_cost": 45},
            ],
        }
    )
    fixtures = [
        Fixture.model_validate(
            {"id": 3, "event": 2, "team_h": 2, "team_a": 1, "kickoff_time": "2026-08-29T14:00:00Z"}
        )
    ]
    prices = (2.35, 3.4, 3.05)
    raw = [
        {
            "id": "x",
            "home_team": "Beta",
            "away_team": "Alpha",
            "commence_time": "2026-08-29T14:00:00Z",
            "bookmakers": [
                {
                    "key": "pinnacle",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Beta", "price": prices[0]},
                                {"name": "Draw", "price": prices[1]},
                                {"name": "Alpha", "price": prices[2]},
                            ],
                        }
                    ],
                }
            ],
        }
    ]
    fallback = TeamRatingProvider(bs.teams, None, calibration_fixtures=fixtures)
    provider = OddsProvider(
        SECRET,
        teams=bs.teams,
        fixtures=fixtures,
        fallback=fallback,
        client=_mock_client(tmp_path, _ok_handler(raw)),
    )
    ctx = make_context(
        2,
        bs=bs,
        fixtures=fixtures,
        history_rows=[],
        as_of=datetime(2026, 8, 28, 17, 30, tzinfo=UTC),
        live=False,
        now=datetime(2026, 9, 1, tzinfo=UTC),
        provider=provider,
    )
    preds = predict_all(2, ctx=ctx)
    assert len(preds) == 2
    dumped = [p.model_dump(mode="json") for p in preds]
    keys = _keys(dumped, set())
    assert not (keys & FORBIDDEN_TOP_KEYS), keys & FORBIDDEN_TOP_KEYS
    fixture_keys = _keys([d["fixtures"] for d in dumped], set())
    assert not (fixture_keys & FORBIDDEN_KEYS), fixture_keys & FORBIDDEN_KEYS
    for p in preds:
        assert len(p.fixtures) == 1 and p.fixtures[0].fsi_source == "odds"
        assert 1 <= p.fixtures[0].fixture_strength_index <= 5
        assert not any(str(x) in " ".join(p.notes) for x in prices)
    text = json.dumps(dumped)
    for price in prices:
        assert f": {price}" not in text  # сами цены как значения полей не появляются
    assert '"fsi_source": "odds"' in text


# ---------- живой запрос (тратит 1 запрос из месячного лимита) ----------


@pytest.mark.network
def test_live_snapshot_matches_all_fpl_fixtures(tmp_path):
    from fplcopilot.data import FPLClient

    if not settings.odds_api_key:
        pytest.skip("ODDS_API_KEY не задан")
    client = OddsClient(settings.odds_api_key, cache_dir=tmp_path, cache_ttl=0)
    raw = client.fetch_raw()
    assert client.last_source == "api" and client.requests_used is not None
    matches = parse_snapshot(raw)
    assert matches
    fpl = FPLClient()
    matched, unmatched = match_fixtures(matches, fpl.fixtures(), fpl.bootstrap().teams)
    assert unmatched == [] and len(matched) == len(matches)
