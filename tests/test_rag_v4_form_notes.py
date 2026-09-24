"""Промпт сигнала v4 (rag/extract.py): form_notes отдельно от доказательств доступности —
схема по версии, правила (d) verbatim и (f) «цитата про этого игрока», правило (c) при одной
только форме, модель минут form_notes не читает. Без БД и сети."""

from datetime import UTC, datetime, timedelta

import pytest

from fplcopilot.core import signals as core_signals
from fplcopilot.data.schemas import Bootstrap
from fplcopilot.prompts import available_versions, load_prompt
from fplcopilot.rag import extract as ex
from fplcopilot.rag.extract import (
    Evidence,
    EvidenceDraft,
    EvidenceDraftV4,
    FormNoteDraft,
    FPLPrior,
    GWCalendar,
    PlayerSignal,
    SignalDraftV2,
    SignalDraftV4,
    draft_schema,
    get_matcher,
    validate_form_notes,
)
from fplcopilot.rag.retrieve import RetrievedChunk

AS_OF = datetime(2026, 9, 23, 18, 0, tzinfo=UTC)
ARS, MCI = 1, 13


def _bs() -> Bootstrap:
    return Bootstrap.model_validate(
        {
            "events": [
                {"id": 5, "name": "Gameweek 5", "deadline_time": "2026-09-18T17:30:00Z"},
                {"id": 6, "name": "Gameweek 6", "deadline_time": "2026-10-03T10:00:00Z", "is_next": True},
            ],
            "teams": [
                {"id": ARS, "name": "Arsenal", "short_name": "ARS"},
                {"id": MCI, "name": "Man City", "short_name": "MCI"},
            ],
            "elements": [
                {"id": 12, "web_name": "Saka", "first_name": "Bukayo", "second_name": "Saka", "team": ARS, "element_type": 3, "now_cost": 95},
                {"id": 14, "web_name": "Ødegaard", "first_name": "Martin", "second_name": "Ødegaard", "team": ARS, "element_type": 3, "now_cost": 80},
                {"id": 420, "web_name": "Cherki", "first_name": "Rayan", "second_name": "Cherki", "team": MCI, "element_type": 3, "now_cost": 65},
                {"id": 411, "web_name": "Haaland", "first_name": "Erling", "second_name": "Haaland", "team": MCI, "element_type": 4, "now_cost": 155},
            ],
        }
    )  # fmt: skip


def chunk(cid: int, text: str, *, players=(), source="sky_football", days_ago=2):
    return RetrievedChunk(
        chunk_id=cid,
        article_id=cid,
        text=text,
        source=source,
        url=f"https://x/{cid}",
        title=text.split("\n")[0],
        published_at=AS_OF - timedelta(days=days_ago),
        players=list(players),
    )


ROUNDUP = chunk(
    1,
    "Premier League talking points\nBukayo Saka, who looks back to his dangerous best with "
    "the ball. Martin Odegaard scored twice against Leeds.",
    players=[12, 14],
)
REPORT = chunk(2, "Man City 3-0 Norwich\nRayan Cherki scored the third.", players=[420, 411])
TRAINING = chunk(3, "Arsenal team news\nBukayo Saka trained fully on Thursday.", players=[12])


def note(cid, quote, kind="form", text=None):
    return FormNoteDraft(kind=kind, text=text or quote, chunk_id=cid, quote=quote)


# ---------- схема и промпт ----------


def test_draft_schema_v4_has_form_notes_v1_to_v3_unchanged():
    assert draft_schema("v4") is SignalDraftV4
    assert "form_notes" in SignalDraftV4.model_fields
    assert draft_schema("v3") is SignalDraftV2 and "form_notes" not in SignalDraftV2.model_fields
    assert {"v1", "v2", "v3", "v4"} <= set(available_versions("signal_extraction"))


def test_prompt_v4_separates_availability_evidence_from_form():
    p = load_prompt("signal_extraction", "v4")
    assert "form_notes" in p.system and "Rayan Cherki scored" in p.system
    assert "form_notes never raise confidence" in p.system
    # правила 1–13 — как в v3 (A/B #4: длинные новые инструкции роняли точность 25 -> 21–23)
    v3 = load_prompt("signal_extraction", "v3")
    assert v3.system.splitlines()[4].split(".")[0] in p.system  # rule 1 unchanged
    user = p.render_user(
        player_name="Bukayo Saka", player_id=12, team_name="Arsenal", position="MID",
        as_of="2026-09-23T18:00Z", next_gw=6, deadline="2026-10-03 10:00Z", fpl_status="a",
        fpl_status_label="available", fpl_chance_next="null", fpl_news="", fpl_news_added="null",
        n_docs=0, documents="(none)", gw_calendar="(none)",
    )  # fmt: skip
    assert "form_notes" in user and "{" not in user


# ---------- validate_form_notes: (d) verbatim, (f) цитата про игрока ----------


def test_form_note_kept_with_source_and_link():
    bs = _bs()
    saka = bs.player(12)
    notes, fixes = validate_form_notes(
        [note(1, "Bukayo Saka, who looks back to his dangerous best with the ball.")],
        [ROUNDUP],
        player=saka,
        team=bs.team(ARS),
        matcher=get_matcher(bs),
    )
    assert fixes == 0 and len(notes) == 1
    n = notes[0]
    assert (n.kind, n.chunk_id, n.source, n.url) == ("form", 1, "sky_football", "https://x/1")
    assert n.published_at == ROUNDUP.published_at


def test_form_note_about_teammate_in_same_chunk_is_dropped():
    bs = _bs()
    notes, fixes = validate_form_notes(
        [note(1, "Martin Odegaard scored twice against Leeds.")],
        [ROUNDUP],
        player=bs.player(12),
        team=bs.team(ARS),
        matcher=get_matcher(bs),
    )
    assert notes == [] and fixes == 1


def test_form_note_not_verbatim_or_unknown_chunk_is_dropped():
    bs = _bs()
    saka = bs.player(12)
    notes, fixes = validate_form_notes(
        [
            note(1, "Saka is the best winger in the league"),  # нет в документе
            note(9, "Bukayo Saka trained fully on Thursday."),  # документа нет среди выданных
            note(2, "Rayan Cherki scored the third."),  # чанк не про Saka
        ],
        [ROUNDUP, REPORT],
        player=saka,
        team=bs.team(ARS),
        matcher=get_matcher(bs),
    )
    assert notes == [] and fixes == 3


def test_form_note_near_verbatim_is_aligned():
    bs = _bs()
    notes, fixes = validate_form_notes(
        [note(2, "Rayan Cherki scored the third", kind="form")],  # без точки — это ок
        [REPORT],
        player=bs.player(420),
        team=bs.team(MCI),
        matcher=get_matcher(bs),
    )
    assert len(notes) == 1 and fixes == 0 and notes[0].quote == "Rayan Cherki scored the third"


def test_form_note_text_with_unsupported_number_falls_back_to_quote():
    bs = _bs()
    notes, fixes = validate_form_notes(
        [note(2, "Rayan Cherki scored the third.", text="Cherki has 4 goals since 20 Sep.")],
        [REPORT],
        player=bs.player(420),
        team=bs.team(MCI),
        matcher=get_matcher(bs),
    )
    assert len(notes) == 1 and fixes == 1
    assert notes[0].text == "Rayan Cherki scored the third."


def test_form_note_text_must_restate_its_quote():
    """Живой golden-прогон v4: «started the season strongly, scoring in the first four matches»
    к цитате про прогноз очков — не пересказ цитаты, текст заменяется цитатой."""
    bs = _bs()
    projection = chunk(
        5,
        "Attacking returns\nErling Haaland (£15.5m) is the only player to exceed 1.0 projected "
        "attacking returns this week",
        players=[411],
    )
    q = "Erling Haaland (£15.5m) is the only player to exceed 1.0 projected attacking returns"
    notes, fixes = validate_form_notes(
        [note(5, q, text="Haaland has started the season strongly, scoring in the first matches.")],
        [projection],
        player=bs.player(411),
        team=bs.team(MCI),
        matcher=get_matcher(bs),
    )
    assert fixes == 1 and notes[0].text == q
    ok, fixes_ok = validate_form_notes(
        [note(5, q, text="Haaland is the only player projected to exceed 1.0 attacking returns.")],
        [projection],
        player=bs.player(411),
        team=bs.team(MCI),
        matcher=get_matcher(bs),
    )
    assert fixes_ok == 0 and ok[0].text.startswith("Haaland is the only player")


def test_quote_already_in_evidence_is_not_duplicated():
    bs = _bs()
    ev = Evidence(
        chunk_id=3,
        source="sky_football",
        url="https://x/3",
        published_at=TRAINING.published_at,
        quote="Bukayo Saka trained fully on Thursday.",
    )
    notes, fixes = validate_form_notes(
        [note(3, "Bukayo Saka trained fully on Thursday.")],
        [TRAINING],
        player=bs.player(12),
        team=bs.team(ARS),
        matcher=get_matcher(bs),
        evidence=[ev],
    )
    assert notes == [] and fixes == 0


# ---------- extract_signal v4: правило (c) и отдельное поле ----------


class _Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.last_timings = {"total_ms": 1.0}
        self.last_candidates = chunks

    def search(self, *a, **kw):
        return self.chunks


def _run(monkeypatch, draft, chunks, pid=12):
    bs = _bs()
    monkeypatch.setattr(ex, "call_llm", lambda *a, **kw: (draft, {"prompt_tokens": 1}))
    monkeypatch.setattr(ex, "fpl_prior", lambda player, as_of, **kw: FPLPrior("a", None, "", None))
    timings: dict = {}
    sig = ex.extract_signal(
        pid,
        AS_OF,
        retriever=_Retriever(chunks),
        bs=bs,  # type: ignore[arg-type]
        prompt_version="v4",
        calendar=GWCalendar.from_fpl(bs, []),
        save=False,
        timings=timings,
    )
    return sig, timings


def _v4(**kw):
    base = {
        "availability": "fit", "start_probability": 0.85, "rotation_risk": "low",
        "return_date": None, "return_gw": None, "confidence": 0.7,
        "summary": "Bukayo Saka trained fully on Thursday.", "evidence": [], "form_notes": [],
    }  # fmt: skip
    return SignalDraftV4(**{**base, **kw})


def test_only_form_news_gives_unknown_with_form_notes(monkeypatch):
    draft = _v4(
        confidence=1.0,
        summary="Saka is in great form.",
        form_notes=[note(1, "Bukayo Saka, who looks back to his dangerous best with the ball.")],
    )
    sig, timings = _run(monkeypatch, draft, [ROUNDUP])
    assert sig.availability == "unknown" and sig.confidence == 0 and sig.evidence == []
    assert [n.kind for n in sig.form_notes] == ["form"]
    assert "No fitness or selection news" in sig.summary and "status a" in sig.summary
    assert timings["form_note_fixes"] == 0


def test_availability_evidence_and_form_notes_stay_separate(monkeypatch):
    draft = _v4(
        evidence=[
            EvidenceDraftV4(
                about="training", chunk_id=3, quote="Bukayo Saka trained fully on Thursday."
            )
        ],
        form_notes=[
            note(1, "Bukayo Saka, who looks back to his dangerous best with the ball."),
            note(1, "Martin Odegaard scored twice against Leeds."),  # одноклубник -> выброшено
        ],
    )
    sig, timings = _run(monkeypatch, draft, [ROUNDUP, TRAINING])
    assert sig.availability == "fit" and sig.confidence == 0.7
    assert [e.chunk_id for e in sig.evidence] == [3]
    assert [n.chunk_id for n in sig.form_notes] == [1]
    assert timings["form_note_fixes"] == 1 and sig.validation_fixes >= 1


def test_form_only_quote_moves_from_evidence_to_form_notes(monkeypatch):
    """Живой Saka 23.09: модель клала «dangerous best» в evidence (и в form_notes)."""
    q = "Bukayo Saka, who looks back to his dangerous best with the ball."
    draft = _v4(
        confidence=1.0,
        evidence=[EvidenceDraftV4(about="selection", chunk_id=1, quote=q)],
        form_notes=[note(1, q.rstrip("."))],
    )
    sig, timings = _run(monkeypatch, draft, [ROUNDUP])
    assert timings["form_quotes_moved"] == 1
    assert sig.evidence == [] and sig.availability == "unknown" and sig.confidence == 0
    assert [n.chunk_id for n in sig.form_notes] == [1]


@pytest.mark.parametrize(
    ("about", "status", "expected"),
    [("selection", "a", 0.8), ("rotation", "a", 0.8), ("training", "a", 1.0), ("selection", "d", 1.0)],
)  # fmt: skip
def test_indirect_selection_evidence_caps_fit_confidence(monkeypatch, about, status, expected):
    draft = _v4(
        confidence=1.0,
        evidence=[
            EvidenceDraftV4(about=about, chunk_id=3, quote="Bukayo Saka trained fully on Thursday.")
        ],
    )
    bs = _bs()
    monkeypatch.setattr(ex, "call_llm", lambda *a, **kw: (draft, {}))
    monkeypatch.setattr(
        ex, "fpl_prior", lambda player, as_of, **kw: FPLPrior(status, None, "", None)
    )
    sig = ex.extract_signal(
        12, AS_OF, retriever=_Retriever([TRAINING]), bs=bs, prompt_version="v4",  # type: ignore[arg-type]
        calendar=GWCalendar.from_fpl(bs, []), save=False,
    )  # fmt: skip
    assert sig.availability == "fit" and sig.confidence == expected


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_older_prompts_never_return_form_notes(monkeypatch, version):
    bs = _bs()
    draft = SignalDraftV2(
        availability="fit", start_probability=0.85, rotation_risk="low", return_date=None,
        return_gw=None, confidence=0.7, summary="Bukayo Saka trained fully on Thursday.",
        evidence=[EvidenceDraft(chunk_id=3, quote="Bukayo Saka trained fully on Thursday.")],
    )  # fmt: skip
    monkeypatch.setattr(ex, "call_llm", lambda *a, **kw: (draft, {}))
    monkeypatch.setattr(ex, "fpl_prior", lambda player, as_of, **kw: FPLPrior("a", None, "", None))
    sig = ex.extract_signal(
        12, AS_OF, retriever=_Retriever([TRAINING]), bs=bs, prompt_version=version,  # type: ignore[arg-type]
        calendar=GWCalendar.from_fpl(bs, []), save=False,
    )  # fmt: skip
    assert sig.form_notes == []


@pytest.mark.parametrize(
    ("quote", "kind"),
    [
        ("Rayan Cherki scored", "form"),
        ("Bukayo Saka, who looks back to his dangerous best with the ball.", "form"),
        ("Erling Haaland (£15.5m) is the only player to exceed 1.0 projected attacking returns", "form"),
        ("David Raya (£6.0m) and Declan Rice (£7.4m) had the night off altogether.", "availability"),
        ("“Yeah, that was planned.” – Mikel Arteta on Gabriel Magalhaes’ half-time substitution", "availability"),
        ("Calf injury - Expected back 18 Sep", "availability"),
        ("City will continue to revel in that signing.", "other"),
    ],
)  # fmt: skip
def test_eval_quote_kind_dictionary(quote, kind):
    from evals.metrics import quote_kind

    assert quote_kind(quote) == kind


def test_player_signal_defaults_and_minutes_model_ignores_form_notes():
    fields = PlayerSignal.model_fields
    assert fields["form_notes"].default_factory is list
    # core/signals (модель минут / xPts) не читает form_notes: текст не превращается в числа
    assert "form_notes" not in str(core_signals._LATEST.text)
    assert not hasattr(core_signals.SignalLite, "form_notes")
