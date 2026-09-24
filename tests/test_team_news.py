"""Новости клуба (rag/team_news.py) и проверка summary (rag/summary_check.py): без БД, сети и LLM.

- retrieval клуба: HARD RULE as_of (store, игнорирующий as_of, всё равно не пропускает будущее),
  только документы про клуб, не старше двух недель;
- валидация дайджеста: verbatim-цитаты, чужой клуб, проверка имён игроков, claim / summary;
- отсутствующие игроки — из статусов FPL и календаря (Foden «Suspended until 17 Oct» -> GW7), а не
  из текста статьи («next two Premier League games» от 18.09);
- summary сигнала: неподтверждённые даты / соперники вырезаются (живой пример Cherki 23.09).
"""

from datetime import UTC, datetime, timedelta

import pytest

from fplcopilot.data.schemas import Bootstrap, Fixture
from fplcopilot.rag import extract as ex
from fplcopilot.rag import team_news as tn
from fplcopilot.rag.extract import Evidence, FPLPrior, GWCalendar, SignalDraftV2
from fplcopilot.rag.retrieve import (
    CorpusDoc,
    NoopReranker,
    RetrievedChunk,
    Retriever,
    matches_entities,
    tokenize_bm25,
)
from fplcopilot.rag.summary_check import build_support, check_summary

AS_OF = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
MCI, SUN, CHE = 15, 17, 6


def _bs() -> Bootstrap:
    return Bootstrap.model_validate(
        {
            "events": [
                {"id": 5, "name": "Gameweek 5", "deadline_time": "2026-09-18T17:30:00Z", "finished": True},
                {"id": 6, "name": "Gameweek 6", "deadline_time": "2026-10-10T10:00:00Z", "is_next": True},
                {"id": 7, "name": "Gameweek 7", "deadline_time": "2026-10-17T10:00:00Z"},
            ],
            "teams": [
                {"id": MCI, "name": "Man City", "short_name": "MCI"},
                {"id": SUN, "name": "Sunderland", "short_name": "SUN"},
                {"id": CHE, "name": "Chelsea", "short_name": "CHE"},
            ],
            "elements": [
                {"id": 398, "web_name": "Foden", "first_name": "Phil", "second_name": "Foden", "team": MCI, "element_type": 3, "now_cost": 70, "status": "s", "news": "Suspended until 17 Oct", "selected_by_percent": "9.0"},
                {"id": 417, "web_name": "Cherki", "first_name": "Rayan", "second_name": "Cherki", "team": MCI, "element_type": 3, "now_cost": 78, "selected_by_percent": "20.0"},
                {"id": 411, "web_name": "Haaland", "first_name": "Erling", "second_name": "Haaland", "team": MCI, "element_type": 4, "now_cost": 155, "status": "d", "chance_of_playing_next_round": 75, "news": "Knock - 75% chance of playing"},
                {"id": 430, "web_name": "Phillips", "first_name": "Kalvin", "second_name": "Phillips", "team": MCI, "element_type": 3, "now_cost": 45, "status": "u", "news": "Has joined Sheffield United on loan for the rest of the season"},
                {"id": 159, "web_name": "Caicedo", "first_name": "Moisés", "second_name": "Caicedo Corozo", "team": CHE, "element_type": 3, "now_cost": 65},
            ],
        }
    )  # fmt: skip


def _fixtures() -> list[Fixture]:
    rows = [
        (61, 6, "2026-10-11T14:00:00Z", MCI, SUN),
        (71, 7, "2026-10-17T14:00:00Z", CHE, MCI),
    ]
    return [
        Fixture.model_validate({"id": i, "event": e, "kickoff_time": k, "team_h": h, "team_a": a})
        for i, e, k, h, a in rows
    ]


def chunk(cid, text, *, days_ago=1.0, teams=(), players=(), source="bbc_football"):
    return RetrievedChunk(
        chunk_id=cid,
        article_id=cid,
        text=text,
        source=source,
        url=f"https://x/{cid}",
        title=text.split("\n")[0],
        published_at=AS_OF - timedelta(days=days_ago),
        players=list(players),
        teams=list(teams),
    )


# ---------- retrieval клуба: as_of, «про клуб», свежесть ----------

CORPUS = [
    chunk(1, "City notes\nMaresca said he will rotate the squad in the cup.", teams=[MCI]),
    chunk(2, "FUTURE\nMan City news: Maresca resigns.", days_ago=-1, teams=[MCI]),
    chunk(3, "Chelsea news\nChelsea boss said the squad is tired.", teams=[CHE]),
    chunk(4, "Old City presser\nMaresca on the squad in August.", days_ago=30, teams=[MCI]),
    chunk(5, "Cherki update\nRayan Cherki trained with the squad.", players=[417]),
    chunk(6, "Referees\nOfficials named for the weekend fixtures.", days_ago=2),
]


class LazyStore:
    """Store, который НЕ фильтрует as_of (проверяем защиту в глубину в Retriever и team_news)."""

    def __init__(self, docs):
        self.docs = docs

    def dense(self, qvec, *, as_of, player_ids, team_ids, n):
        out = []
        for d in self.docs:
            if (player_ids or team_ids) and not matches_entities(d, player_ids, team_ids):
                continue
            c = RetrievedChunk(**d.__dict__)
            c.dense_score = 0.5
            out.append(c)
        return out[:n]

    def corpus(self):
        return (len(self.docs), len(self.docs)), [
            CorpusDoc(chunk=d, tokens=tokenize_bm25(d.text)) for d in self.docs
        ]


def _retriever() -> Retriever:
    return Retriever(
        store=LazyStore(CORPUS),  # type: ignore[arg-type]
        embed=lambda q: tokenize_bm25(q),
        reranker=NoopReranker(),
        candidates=10,
    )


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid_rerank"])
def test_team_retrieval_respects_as_of_and_keeps_only_club_recent_chunks(mode):
    bs = _bs()
    city = bs.team(MCI)
    got = tn.retrieve_for_team(
        _retriever(), city, as_of=AS_OF, k=8, mode=mode, squad=tn.squad_of(bs, MCI)
    )
    ids = {c.chunk_id for c in got}
    assert 2 not in ids  # из будущего — ни в одном режиме
    assert 3 not in ids and 6 not in ids  # другой клуб / ни о ком
    assert 4 not in ids  # старше 14 дней
    assert all(c.published_at <= AS_OF for c in got)
    assert 1 in ids and ids <= {1, 5}


def test_team_retrieval_rejects_naive_as_of():
    bs = _bs()
    with pytest.raises(ValueError):
        tn.retrieve_for_team(
            _retriever(),
            bs.team(MCI),
            as_of=AS_OF.replace(tzinfo=None),
            k=4,
            mode="dense",
            squad=tn.squad_of(bs, MCI),
        )


# ---------- валидация дайджеста ----------


def test_validate_team_draft_keeps_verbatim_club_items_and_drops_the_rest():
    bs = _bs()
    city, squad = bs.team(MCI), tn.squad_of(bs, MCI)
    docs = [
        chunk(1, "City notes\nMaresca said he will rotate the squad in the cup.", teams=[MCI]),
        chunk(3, "Round-up\nChelsea boss Rosenior said the squad is tired.", teams=[MCI, CHE]),
    ]
    draft = tn.TeamNewsDraft(
        items=[
            tn.TeamNewsItemDraft(
                kind="rotation",
                players=["Cherki"],  # в документе его нет -> имя снимается
                claim="Maresca will rotate in the cup.",
                chunk_id=1,
                quote="Maresca said he will rotate the squad in the cup.",
            ),
            tn.TeamNewsItemDraft(
                kind="manager_quote",
                players=[],
                claim="City are exhausted.",
                chunk_id=1,
                quote="Maresca says City are exhausted",  # пересказ -> долой
            ),
            tn.TeamNewsItemDraft(
                kind="form_context",
                players=[],
                claim="The squad is tired.",
                chunk_id=3,
                quote="Chelsea boss Rosenior said the squad is tired.",  # чужой клуб -> долой
            ),
        ],
        summary="Maresca will rotate in the cup against Sunderland on 30 September.",
    )
    items, summary, fixes = tn.validate_team_draft(
        draft, docs, city, squad=squad, matcher=ex.get_matcher(bs)
    )
    assert [i.chunk_id for i in items] == [1] and items[0].players == []
    assert items[0].quote == "Maresca said he will rotate the squad in the cup."
    # summary с неподтверждённым соперником и датой -> шаблон из пунктов
    assert "Sunderland" not in summary and summary.startswith("Man City club news: rotation")
    assert fixes >= 4


def test_no_items_gives_fixed_summary():
    bs = _bs()
    draft = tn.TeamNewsDraft(items=[], summary="City look strong.")
    items, summary, fixes = tn.validate_team_draft(
        draft, [], bs.team(MCI), squad=tn.squad_of(bs, MCI)
    )
    assert items == [] and summary == tn.NO_NEWS_SUMMARY and fixes == 1


# ---------- отсутствующие игроки клуба: из статусов FPL и календаря ----------


def test_club_absences_come_from_fpl_status_and_calendar_not_article_text():
    bs = _bs()
    cal = GWCalendar.from_fpl(bs, _fixtures())

    def prior(p, as_of):
        return FPLPrior(p.status, p.chance_of_playing_next_round, p.news or "", None)

    rows = tn.club_absences(
        bs,
        bs.team(MCI),
        AS_OF,
        calendar=cal,
        signals={398: (AS_OF - timedelta(days=5), "suspended")},
        prior=prior,
    )
    by = {r["player"]: r for r in rows}
    assert set(by) == {"Foden", "Haaland"}  # Cherki доступен, Phillips ушёл в аренду
    foden = by["Foden"]
    assert foden["status_label"] == "suspended" and foden["return_date"] == "2026-10-17"
    assert foden["return_gw"] == 7  # первый матч City 17.10 = GW7, а не «next two games» от 18.09
    assert foden["news_availability"] == "suspended"
    assert by["Haaland"]["chance_next"] == 75 and by["Haaland"]["return_gw"] is None


# ---------- summary сигнала: проверка кодом ----------

BBC = chunk(
    10,
    "FPL tips\nRayan Cherki (£7.8m) has disappointed recently but with Phil Foden (£7.0m) "
    "suspended, he should be more likely to start the next two Premier League games.",
    days_ago=5,
    players=[417, 398],
)
SKY = chunk(
    11,
    "Man City 2-0 Norwich\nRayan Cherki scored in the Carabao Cup win over Norwich.",
    days_ago=6,
    players=[417],
    source="sky_football",
)


def _evidence(c: RetrievedChunk, quote: str) -> Evidence:
    return Evidence(
        chunk_id=c.chunk_id, source=c.source, url=c.url, published_at=c.published_at, quote=quote
    )


def test_summary_check_drops_unsupported_opponent_and_date_keeps_the_rest():
    bs = _bs()
    ev = [_evidence(BBC, "with Phil Foden (£7.0m) suspended, he should be more likely to start")]
    res = ex.check_signal_summary(
        "Rayan Cherki is expected to start as Phil Foden is suspended. He scored in the last "
        "match against Sunderland (BBC, 2026-09-20).",
        ev,
        [BBC, SKY],
        player=bs.player(417),
        team=bs.team(MCI),
        prior=FPLPrior("a", None, "", None),
        availability="fit",
    )
    assert res.summary == "Rayan Cherki is expected to start as Phil Foden is suspended."
    assert res.fixes == 1 and "Sunderland" in res.reasons[0] and "2026-09-20" in res.reasons[0]


def test_summary_check_allows_cited_dates_sources_and_plain_words():
    sup = build_support(
        [BBC.text], names=["Rayan Cherki", "Man City", "bbc football"], dates=[BBC.published_at]
    )
    ok = "He trained fully on Thursday and should start, per the BBC (18 Sep)."
    assert check_summary(ok, sup, fallback="x").summary == ok
    bad = check_summary("Out for 6 weeks with a hamstring injury.", sup, fallback="neutral")
    assert bad.replaced and bad.summary == "neutral"  # «6» нет в документах


def test_extract_signal_fixes_unsupported_summary_and_counts_it(monkeypatch):
    bs = _bs()

    class FakeRetriever:
        def __init__(self) -> None:
            self.last_timings = {"total_ms": 1.0}
            self.last_candidates = [BBC, SKY]

        def search(self, *a, **kw):
            return [BBC, SKY]

    draft = SignalDraftV2(
        availability="fit",
        start_probability=0.85,
        rotation_risk="medium",
        return_date=None,
        return_gw=None,
        confidence=0.8,
        summary="Cherki should start with Foden suspended. He scored against Sunderland on 20 Sep.",
        evidence=[
            ex.EvidenceDraft(
                chunk_id=10,
                quote="with Phil Foden (£7.0m) suspended, he should be more likely to start",
            )
        ],
    )
    monkeypatch.setattr(ex, "call_llm", lambda *a, **kw: (draft, {"prompt_tokens": 1}))
    monkeypatch.setattr(ex, "fpl_prior", lambda p, as_of, **kw: FPLPrior("a", None, "", None))
    timings: dict = {}
    sig = ex.extract_signal(
        417,
        AS_OF,
        retriever=FakeRetriever(),  # type: ignore[arg-type]
        bs=bs,
        prompt_version="v2",
        calendar=GWCalendar.from_fpl(bs, _fixtures()),
        save=False,
        timings=timings,
    )
    assert sig.summary == "Cherki should start with Foden suspended."
    assert sig.validation_fixes >= 1 and "Sunderland" in timings["summary_fixes"][0]
    assert [e.chunk_id for e in sig.evidence] == [10]  # цитаты не тронуты
