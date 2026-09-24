"""RAG v2 (rag/extract.py): календарь return_date -> return_gw, разбор дат FPL, формула
expected_minutes, abstention до LLM, нормализация v2-черновика, рендер промпта v2. Без БД и сети."""

from datetime import UTC, date, datetime, timedelta

import pytest

from fplcopilot.data.schemas import Bootstrap, Fixture
from fplcopilot.prompts import available_versions, load_prompt
from fplcopilot.rag import extract as ex
from fplcopilot.rag.extract import (
    ABSTAIN_SUMMARY,
    UNKNOWN_EXPECTED_MINUTES,
    EvidenceDraft,
    FPLPrior,
    GWCalendar,
    SignalDraft,
    SignalDraftV2,
    build_messages,
    draft_schema,
    expected_minutes_for,
    next_event_as_of,
    normalise_draft,
    parse_fpl_return_date,
    parse_return_date,
    should_abstain,
)
from fplcopilot.rag.retrieve import RetrievedChunk

AS_OF = datetime(2026, 9, 17, 9, 0, tzinfo=UTC)
CHE, MCI, COV, BRE, TOT, IPS, SUN, BHA = 6, 15, 7, 5, 18, 10, 17, 4


def _fixture(fid: int, event: int, kickoff: str, home: int, away: int) -> dict:
    return {"id": fid, "event": event, "kickoff_time": kickoff, "team_h": home, "team_a": away}


def _bs() -> Bootstrap:
    return Bootstrap.model_validate(
        {
            "events": [
                {"id": 4, "name": "Gameweek 4", "deadline_time": "2026-09-12T12:30:00Z"},
                {"id": 5, "name": "Gameweek 5", "deadline_time": "2026-09-18T17:30:00Z", "is_next": True},
                {"id": 6, "name": "Gameweek 6", "deadline_time": "2026-10-10T10:00:00Z"},
                {"id": 7, "name": "Gameweek 7", "deadline_time": "2026-10-17T10:00:00Z"},
                {"id": 8, "name": "Gameweek 8", "deadline_time": "2026-10-23T17:30:00Z"},
            ],
            "teams": [
                {"id": CHE, "name": "Chelsea", "short_name": "CHE"},
                {"id": MCI, "name": "Man City", "short_name": "MCI"},
                {"id": COV, "name": "Coventry", "short_name": "COV"},
                {"id": BRE, "name": "Brentford", "short_name": "BRE"},
                {"id": TOT, "name": "Spurs", "short_name": "TOT"},
                {"id": IPS, "name": "Ipswich", "short_name": "IPS"},
                {"id": SUN, "name": "Sunderland", "short_name": "SUN"},
                {"id": BHA, "name": "Brighton", "short_name": "BHA"},
            ],
            "elements": [
                {"id": 159, "web_name": "Caicedo", "first_name": "Moisés", "second_name": "Caicedo Corozo", "team": CHE, "element_type": 3, "now_cost": 65, "status": "i", "chance_of_playing_next_round": 0, "news": "Calf injury - Expected back 18 Sep"},
                {"id": 398, "web_name": "Foden", "first_name": "Phil", "second_name": "Foden", "team": MCI, "element_type": 3, "now_cost": 90, "status": "s", "news": "Suspended until 17 Oct"},
                {"id": 492, "web_name": "Awoniyi", "first_name": "Taiwo", "second_name": "Awoniyi", "team": COV, "element_type": 4, "now_cost": 55, "status": "s", "news": "Suspended until 19 Oct"},
                {"id": 411, "web_name": "Haaland", "first_name": "Erling", "second_name": "Haaland", "team": MCI, "element_type": 4, "now_cost": 155},
                {"id": 1, "web_name": "Raya", "first_name": "David", "second_name": "Raya Martin", "team": MCI, "element_type": 1, "now_cost": 55},
                {"id": 496, "web_name": "Kinský", "first_name": "Antonín", "second_name": "Kinský", "team": TOT, "element_type": 1, "now_cost": 45},
                {"id": 10, "web_name": "White", "first_name": "Benjamin", "second_name": "White", "team": MCI, "element_type": 2, "now_cost": 55},
            ],
        }
    )  # fmt: skip


def _fixtures() -> list[Fixture]:
    rows = [
        _fixture(41, 4, "2026-09-12T14:00:00Z", CHE, MCI),
        _fixture(42, 4, "2026-09-14T19:00:00Z", COV, TOT),
        _fixture(51, 5, "2026-09-18T19:00:00Z", BRE, CHE),  # GW5 first kickoff, Chelsea play
        _fixture(52, 5, "2026-09-19T16:30:00Z", TOT, COV),
        _fixture(53, 5, "2026-09-20T13:00:00Z", MCI, SUN),
        _fixture(61, 6, "2026-10-10T14:00:00Z", CHE, BHA),
        _fixture(62, 6, "2026-10-11T15:30:00Z", TOT, MCI),
        _fixture(63, 6, "2026-10-12T19:00:00Z", COV, SUN),
        _fixture(71, 7, "2026-10-17T14:00:00Z", MCI, IPS),  # Foden: "until 17 Oct" = this match
        _fixture(72, 7, "2026-10-18T13:00:00Z", BHA, CHE),
        _fixture(73, 7, "2026-10-19T19:00:00Z", TOT, COV),  # Awoniyi: "until 19 Oct" = this match
        _fixture(81, 8, "2026-10-24T14:00:00Z", COV, BRE),
        _fixture(82, 8, "2026-10-24T16:30:00Z", CHE, TOT),
        _fixture(83, 8, "2026-10-25T14:00:00Z", MCI, BHA),
    ]
    return [Fixture.model_validate(r) for r in rows]


@pytest.fixture(scope="module")
def cal() -> GWCalendar:
    return GWCalendar.from_fpl(_bs(), _fixtures())


# ---------- календарь: дата -> тур ----------


def test_calendar_entries_have_windows(cal):
    gw5 = next(e for e in cal.entries if e.gw == 5)
    assert gw5.deadline == datetime(2026, 9, 18, 17, 30, tzinfo=UTC)
    assert gw5.first_kickoff == datetime(2026, 9, 18, 19, 0, tzinfo=UTC)
    assert gw5.last_kickoff == datetime(2026, 9, 20, 13, 0, tzinfo=UTC)
    assert [e.gw for e in cal.upcoming(AS_OF, n=3)] == [5, 6, 7]


def test_return_18_sep_with_gw5_deadline_and_first_kickoff_same_day_is_gw5(cal):
    # "Expected back 18 Sep": дедлайн GW5 18.09 17:30, первый матч 18.09 19:00 -> GW5
    assert cal.gw_for_date(date(2026, 9, 18)) == 5
    assert cal.gw_for_date(date(2026, 9, 18), team_id=CHE) == 5  # Chelsea играют 18.09
    assert cal.gw_for_date(date(2026, 9, 17)) == 5  # день до тура -> тот же тур
    assert cal.gw_for_date(date(2026, 9, 14)) == 4  # GW4 ещё не закончился 14.09


def test_return_17_oct_is_first_gw_with_kickoff_on_or_after(cal):
    assert cal.gw_for_date(date(2026, 10, 17)) == 7  # GW6 заканчивается 12.10, GW7 начинается 17.10
    assert cal.gw_for_date(date(2026, 10, 13)) == 7  # между турами -> следующий
    assert cal.gw_for_date(date(2026, 10, 10)) == 6


def test_until_17_oct_exclusive_means_available_from_18_oct(cal):
    # «until 17 Oct» как «доступен с 18.10»: первый тур с матчем >= 18.10 — всё ещё GW7 (18–19.10)
    assert cal.gw_for_date(date(2026, 10, 17), exclusive=True) == 7
    # у клуба, чей единственный матч GW7 — 17.10, «с 18.10» означает уже GW8
    assert cal.gw_for_date(date(2026, 10, 17), team_id=MCI, exclusive=True) == 8
    # FPL-конвенция (включительно): «Suspended until 17 Oct» у Фодена = матч MCI 17.10 -> GW7
    assert cal.gw_for_date(date(2026, 10, 17), team_id=MCI) == 7
    # Awoniyi «Suspended until 19 Oct» = COV играют 19.10 (понедельник GW7) -> GW7
    assert cal.gw_for_date(date(2026, 10, 19), team_id=COV) == 7
    assert cal.gw_for_date(date(2026, 10, 19)) == 7


def test_calendar_without_fixtures_falls_back_to_deadlines(cal):
    bare = GWCalendar.from_fpl(_bs(), [])
    assert bare.gw_for_date(date(2026, 9, 18)) == 5  # дедлайн GW5 18.09 >= 18.09
    assert bare.gw_for_date(date(2026, 9, 19)) == 6  # без окна матчей — по дедлайну
    assert cal.gw_for_date(date(2027, 6, 1)) is None  # за пределами календаря
    # фикстуры клуба кончились раньше даты -> по окнам туров (GW8 есть в календаре)
    assert cal.gw_for_date(date(2026, 10, 26), team_id=BRE) is None
    assert cal.gw_for_date(date(2026, 10, 22), team_id=BRE) == 8


def test_calendar_render_shows_deadline_window_and_team_fixture(cal):
    bs = _bs()
    block = cal.render(AS_OF, team=bs.team(CHE), n=3)
    lines = block.splitlines()
    assert lines[0].startswith("GW5: deadline 2026-09-18 17:30Z; matches 2026-09-18 to 2026-09-20")
    assert "Chelsea plays 2026-09-18" in lines[0]
    assert "GW7" in lines[2] and "Chelsea plays 2026-10-18" in lines[2]


# ---------- разбор дат из текста FPL ----------


@pytest.mark.parametrize(
    "news, expected",
    [
        ("Calf injury - Expected back 18 Sep", (date(2026, 9, 18), "expected_back")),
        ("Suspended until 17 Oct", (date(2026, 10, 17), "suspended_until")),
        ("Hamstring injury - Expected back 11 Oct", (date(2026, 10, 11), "expected_back")),
        ("Knee injury - Expected back 3 Jan", (date(2027, 1, 3), "expected_back")),  # перенос года
        ("Knock - 75% chance of playing", None),
        ("Ankle injury - Unknown return date", None),
        ("", None),
    ],
)
def test_parse_fpl_return_date(news, expected):
    assert parse_fpl_return_date(news, AS_OF) == expected


def test_parse_return_date_accepts_iso_and_day_month():
    assert parse_return_date("2026-09-18", AS_OF) == date(2026, 9, 18)
    assert parse_return_date("2026-10-17T00:00:00", AS_OF) == date(2026, 10, 17)
    assert parse_return_date("18 Sep", AS_OF) == date(2026, 9, 18)
    assert parse_return_date("17 October 2026", AS_OF) == date(2026, 10, 17)
    assert parse_return_date("after the international break", AS_OF) is None
    assert parse_return_date(None, AS_OF) is None


# ---------- expected_minutes: формула ----------


def test_expected_minutes_formula():
    assert expected_minutes_for("injured", 0.7, "FWD") == 0
    assert expected_minutes_for("suspended", 0.0, "MID") == 0
    assert expected_minutes_for("unavailable", 0.0, "DEF") == 0
    assert expected_minutes_for("unknown", 0.5, "MID") == UNKNOWN_EXPECTED_MINUTES
    assert expected_minutes_for("fit", 1.0, "GKP") == 90
    assert expected_minutes_for("fit", 0.9, "GKP") == 81  # вратарь на замену не выходит
    assert expected_minutes_for("fit", 0.9, "FWD") == round(0.9 * 79 + 0.1 * 9)  # 72
    assert expected_minutes_for("fit", 0.9, "DEF") == round(0.9 * 85 + 0.1 * 9)  # 77
    assert expected_minutes_for("doubtful", 0.5, "MID") == round(0.5 * 78 + 0.5 * 9)  # 44
    assert expected_minutes_for("fit", 0.0, "MID") == 9  # только шанс выйти на замену
    assert expected_minutes_for("fit", 1.7, "MID") == 78  # кламп p в [0, 1]


# ---------- abstention до LLM ----------


def chunk(cid: int, text: str, *, source="google_news", players=(), rerank=None, dense=None):
    return RetrievedChunk(
        chunk_id=cid,
        article_id=cid,
        text=text,
        source=source,
        url=f"https://x/{cid}",
        title=text.split("\n")[0],
        published_at=AS_OF - timedelta(days=1),
        players=list(players),
        rerank_score=rerank,
        dense_score=dense,
    )


def test_abstain_when_no_official_chunk_and_no_mention():
    bs = _bs()
    kinsky, spurs = bs.player(496), bs.team(TOT)
    noise = [
        chunk(
            1, "Spurs injury update: Tonali, Porro latest return dates", players=[455], rerank=0.74
        ),
        chunk(2, "Tottenham press conference: De Zerbi on the Villa game", rerank=0.5),
    ]
    d = should_abstain(noise, noise, kinsky, spurs, mode="hybrid_rerank", rerank_threshold=0.2)
    assert d.abstain and "mentions" in d.reason
    assert not d.has_official and not d.any_mention and d.top_score == pytest.approx(0.74)


def test_abstain_on_low_rerank_score_even_if_name_appears():
    bs = _bs()
    kinsky, spurs = bs.player(496), bs.team(TOT)
    weak = [chunk(1, "Kinsky signs new boot deal", rerank=0.05)]
    d = should_abstain(weak, weak, kinsky, spurs, mode="hybrid_rerank", rerank_threshold=0.2)
    assert d.abstain and d.any_mention and "score 0.050 < 0.2" in d.reason


def test_no_abstain_with_official_chunk_or_player_specific_candidates():
    bs = _bs()
    caicedo, chelsea = bs.player(159), bs.team(CHE)
    official = chunk(
        1,
        "Caicedo: Calf injury - Expected back 18 Sep",
        source="fpl_api",
        players=[159],
        rerank=0.99,
    )
    d = should_abstain([official], [official], caicedo, chelsea, mode="hybrid_rerank")
    assert not d.abstain and d.has_official
    # официальный чанк среди кандидатов, но не в top-k — всё равно не отказываемся
    d = should_abstain(
        [chunk(2, "Chelsea team news", rerank=0.1)],
        [official],
        caicedo,
        chelsea,
        mode="hybrid_rerank",
    )
    assert not d.abstain and d.has_official
    # без официального, но чанк про игрока с высоким score
    press = chunk(3, "Moises Caicedo (calf) has sat out the last two league matches", rerank=0.9)
    d = should_abstain([press], [press], caicedo, chelsea, mode="hybrid_rerank")
    assert not d.abstain and d.any_mention
    # dense: косинус ниже порога -> отказ; выше -> нет
    low = chunk(4, "Moises Caicedo transfer talk", dense=0.30)
    assert should_abstain(
        [low], [low], caicedo, chelsea, mode="dense", dense_threshold=0.40
    ).abstain
    high = chunk(5, "Moises Caicedo transfer talk", dense=0.61)
    assert not should_abstain(
        [high], [high], caicedo, chelsea, mode="dense", dense_threshold=0.40
    ).abstain
    # режимы без калиброванного порога (bm25/hybrid): решает только проверка упоминаний
    assert not should_abstain([press], [press], caicedo, chelsea, mode="hybrid").abstain


def test_extract_signal_abstains_without_calling_llm(monkeypatch):
    """Полный extract_signal с фейковым retriever'ом: нет чанков про игрока -> LLM не вызывается."""
    bs = _bs()
    noise = [
        chunk(
            1, "Spurs injury update: Tonali, Porro latest return dates", players=[455], rerank=0.1
        ),
        chunk(2, "Tottenham press conference: De Zerbi on the Villa game", rerank=0.08),
    ]

    class FakeRetriever:
        def __init__(self):
            self.last_timings = {"total_ms": 1.0}
            self.last_candidates = noise

        def search(self, *a, **kw):
            return noise

    calls = []
    monkeypatch.setattr(ex, "call_llm", lambda *a, **kw: calls.append(1) or (None, {}))
    monkeypatch.setattr(ex, "fpl_prior", lambda player, as_of, **kw: FPLPrior("a", None, "", None))
    timings: dict = {}
    sig = ex.extract_signal(
        496,
        AS_OF,
        mode="hybrid_rerank",
        retriever=FakeRetriever(),
        bs=bs,  # type: ignore[arg-type]
        prompt_version="v2",
        abstain=True,
        calendar=GWCalendar.from_fpl(bs, []),
        save=False,
        timings=timings,
    )
    assert calls == []  # LLM не вызывался
    assert sig.abstained and sig.availability == "unknown" and sig.confidence == 0
    assert sig.evidence == [] and sig.return_gw is None
    assert sig.summary.startswith(ABSTAIN_SUMMARY) and "status a (available)" in sig.summary
    assert sig.expected_minutes == UNKNOWN_EXPECTED_MINUTES and sig.start_probability == 0.5
    assert timings["llm_calls"] == 0 and timings["prompt_tokens"] == 0
    assert timings["abstain_reason"] == "no retrieved chunk mentions the player"
    assert sig.retrieved_chunk_ids == [1, 2] and sig.prompt_version == "v2"


# ---------- нормализация v2-черновика ----------


def test_normalise_v2_draft_official_date_wins_and_minutes_by_formula(cal):
    bs = _bs()
    caicedo = bs.player(159)
    prior = FPLPrior("i", 0, "Calf injury - Expected back 18 Sep", AS_OF - timedelta(days=18))
    raw = SignalDraftV2(
        availability="injured", start_probability=0.0, rotation_risk="unknown",
        return_date=None, return_gw=18,  # модель спутала день месяца с туром — код игнорирует
        confidence=0.9, summary="Out with a calf injury.", evidence=[],
    )  # fmt: skip
    draft, source = normalise_draft(raw, player=caicedo, prior=prior, calendar=cal, as_of=AS_OF)
    assert isinstance(draft, SignalDraft)
    assert draft.return_gw == 5 and source == "fpl_news"
    assert draft.expected_minutes == 0


def test_normalise_v2_draft_uses_llm_date_when_fpl_news_has_none(cal):
    bs = _bs()
    foden = bs.player(398)
    prior = FPLPrior("s", 0, "Suspended", None)  # без даты
    raw = SignalDraftV2(
        availability="suspended", start_probability=0.0, rotation_risk="low",
        return_date="2026-10-17", return_gw=None, confidence=0.9, summary="Banned.", evidence=[],
    )  # fmt: skip
    draft, source = normalise_draft(raw, player=foden, prior=prior, calendar=cal, as_of=AS_OF)
    assert draft.return_gw == 7 and source == "llm_date"  # MCI играют 17.10 -> GW7
    # только номер тура от модели — принимается как есть (источник llm_gw)
    raw2 = raw.model_copy(update={"return_date": None, "return_gw": 6})
    draft2, source2 = normalise_draft(raw2, player=foden, prior=prior, calendar=cal, as_of=AS_OF)
    assert draft2.return_gw == 6 and source2 == "llm_gw"
    # fit-игрок: минуты по формуле от p_start и позиции
    haaland = bs.player(411)
    raw3 = SignalDraftV2(
        availability="fit", start_probability=0.9, rotation_risk="low",
        return_date=None, return_gw=None, confidence=0.8, summary="Fit.", evidence=[],
    )  # fmt: skip
    draft3, _ = normalise_draft(
        raw3, player=haaland, prior=FPLPrior("a", None, "", None), calendar=cal, as_of=AS_OF
    )
    assert draft3.expected_minutes == expected_minutes_for("fit", 0.9, "FWD") == 72


def test_v1_draft_passes_through_normalise_unchanged():
    bs = _bs()
    d = SignalDraft(
        availability="doubtful", start_probability=0.7, expected_minutes=60, rotation_risk="low",
        return_gw=None, confidence=0.8, summary="x", evidence=[EvidenceDraft(chunk_id=1, quote="q")],
    )  # fmt: skip
    out, source = normalise_draft(
        d, player=bs.player(159), prior=FPLPrior("d", 75, "", None), calendar=None, as_of=AS_OF
    )
    assert out is d and source is None


def test_draft_schema_per_prompt_version():
    assert draft_schema("v1") is SignalDraft
    assert draft_schema("v2") is SignalDraftV2
    assert "expected_minutes" not in SignalDraftV2.model_fields
    assert "return_date" in SignalDraftV2.model_fields


# ---------- промпт v2 ----------


def test_prompt_v2_renders_calendar_and_rules(cal):
    bs = _bs()
    assert available_versions("signal_extraction") == ["v1", "v2", "v3", "v4"]
    prompt = load_prompt("signal_extraction", "v2")
    prior = FPLPrior("i", 0, "Calf injury - Expected back 18 Sep", AS_OF - timedelta(days=18))
    doc = chunk(7, "Caicedo: Calf injury - Expected back 18 Sep", source="fpl_api", players=[159])
    msgs = build_messages(
        prompt, player=bs.player(159), team=bs.team(CHE), prior=prior, chunks=[doc],
        as_of=AS_OF, next_event=next_event_as_of(bs, AS_OF), calendar=cal,
    )  # fmt: skip
    system, user = msgs[0]["content"], msgs[1]["content"]
    assert (
        "GW5: deadline 2026-09-18 17:30Z; matches 2026-09-18 to 2026-09-20; Chelsea plays 2026-09-18"
        in user
    )
    assert "GW6:" in user and "GW7:" in user
    assert "Next gameweek: GW5 (deadline 2026-09-18 17:30Z)" in user
    assert "do not restate the gameweek number" in user
    assert 'news="Calf injury - Expected back 18 Sep"' in user
    for phrase in (
        "return_date",
        "No player-specific news found.",
        "Teammates, the manager, the club as a whole",
        "Form, goals, assists",
        "keep typos, prices",
        "Do not output expected minutes",
        "Never put a day of the month into return_gw",
        'never write "for Gameweek 5"',
    ):
        assert phrase in system, phrase
    # v1-шаблон рендерится с тем же набором kwargs (лишний gw_calendar игнорируется)
    v1 = load_prompt("signal_extraction", "v1")
    msgs_v1 = build_messages(
        v1, player=bs.player(159), team=bs.team(CHE), prior=prior, chunks=[doc],
        as_of=AS_OF, next_event=next_event_as_of(bs, AS_OF), calendar=cal,
    )  # fmt: skip
    assert "GW5: deadline" not in msgs_v1[1]["content"]
    assert "expected_minutes" in v1.system and "return_date" not in v1.system
