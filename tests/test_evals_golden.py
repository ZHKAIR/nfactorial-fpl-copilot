"""Golden dataset (evals/golden/*.jsonl) validates against the schemas and the labeling protocol.

Unit part: structure, ranges, as_of, tag coverage, schema rejections. `db`/`network` parts check
that article ids / player ids still exist (skipped without Postgres / FPL API).
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from sqlalchemy import text

from evals import GOLDEN_DIR
from evals.schemas import (
    AVAILABILITY_CLASSES,
    RetrievalExample,
    SignalExample,
    load_retrieval,
    load_signals,
)

AS_OF = datetime(2026, 9, 17, 9, 0, tzinfo=UTC)
SIGNALS = load_signals()
RETRIEVAL = load_retrieval()


# ---------- structure ----------


def test_golden_files_exist_and_sizes():
    assert (GOLDEN_DIR / "README.md").exists()
    assert len(SIGNALS) >= 20
    assert len(RETRIEVAL) >= 15
    assert len(SIGNALS) + len(RETRIEVAL) >= 35


def test_ids_unique_and_as_of_fixed():
    ids = [s.id for s in SIGNALS] + [r.id for r in RETRIEVAL]
    assert len(ids) == len(set(ids))
    assert {s.as_of for s in SIGNALS} == {AS_OF}
    assert {r.as_of for r in RETRIEVAL} == {AS_OF}
    assert all(s.as_of.tzinfo is not None for s in SIGNALS)


def test_signal_labels_are_sane():
    for s in SIGNALS:
        e = s.expected
        assert e.availability in AVAILABILITY_CLASSES
        assert 0 <= e.start_probability_min <= e.start_probability_max <= 1, s.id
        if e.availability in {"injured", "suspended", "unavailable"}:
            assert e.start_probability_max == 0, s.id
            assert e.expected_minutes_max == 0, s.id
        if e.availability == "unknown":
            assert not s.must_have_evidence and not s.source_urls, s.id
        if s.must_have_evidence:
            assert s.source_urls, s.id
        assert len(s.rationale) >= 40 and s.tags, s.id
        assert all(t == t.lower() and " " not in t for t in s.tags), s.id


def test_signal_player_ids_unique():
    pids = [s.player_id for s in SIGNALS]
    assert len(pids) == len(set(pids)), "one example per player keeps the set independent"


def test_required_coverage_of_hard_cases():
    tags = {t for s in SIGNALS for t in s.tags}
    for required in (
        "no_coverage", "ambiguous_name", "conflict", "goalkeeper", "promoted_club", "fpl_api_only",
        "return_date", "misleading_news",
    ):  # fmt: skip
        assert required in tags, required
    classes = {s.expected.availability for s in SIGNALS}
    assert {"fit", "doubtful", "injured", "suspended", "unavailable", "unknown"} <= classes
    assert sum(1 for s in SIGNALS if "no_coverage" in s.tags) >= 3
    assert sum(1 for s in SIGNALS if "conflict" in s.tags) >= 1
    assert all(s.expected.availability_alt for s in SIGNALS if "conflict" in s.tags), (
        "conflict cases must list the alternate acceptable label"
    )
    assert all(s.expected.return_gw is not None for s in SIGNALS if "return_date" in s.tags)


def test_retrieval_labels_are_sane():
    no_rel = [r for r in RETRIEVAL if not r.has_relevant]
    assert len(no_rel) >= 2
    assert all(r.player_id or r.team_id for r in RETRIEVAL), "every query has an entity filter"
    for r in RETRIEVAL:
        assert len(r.relevant_article_ids) == len(r.relevant_urls), r.id
        assert len(set(r.relevant_article_ids)) == len(r.relevant_article_ids), r.id
        assert all(u.startswith(("http", "fpl://")) for u in r.relevant_urls), r.id
    # topical coverage demanded by the protocol
    text_ = " ".join(r.query.lower() for r in RETRIEVAL)
    for word in ("injury", "suspend", "return", "rotation", "press conference"):
        assert word in text_, word


# ---------- schema rejections ----------


def _sig(**over):
    base = {
        "id": "sig_x",
        "player_name": "Xavi",
        "player_id": 1,
        "as_of": "2026-09-17T09:00:00Z",
        "expected": {
            "availability": "fit",
            "start_probability_min": 0.5,
            "start_probability_max": 0.9,
        },
        "must_have_evidence": True,
        "tags": ["fit"],
        "rationale": "x" * 40,
        "source_urls": ["https://example.org/a"],
    }
    base.update(over)
    return base


def test_schema_rejects_bad_ranges_and_inconsistent_rows():
    with pytest.raises(ValidationError, match="start_probability_min > start_probability_max"):
        SignalExample.model_validate(
            _sig(
                expected={
                    "availability": "fit",
                    "start_probability_min": 0.9,
                    "start_probability_max": 0.5,
                }
            )
        )
    with pytest.raises(ValidationError, match="timezone-aware"):
        SignalExample.model_validate(_sig(as_of="2026-09-17T09:00:00"))
    with pytest.raises(ValidationError, match="no_coverage examples cannot require evidence"):
        SignalExample.model_validate(_sig(tags=["no_coverage"], must_have_evidence=True))
    with pytest.raises(ValidationError, match="must expect availability=unknown"):
        SignalExample.model_validate(
            _sig(tags=["no_coverage"], must_have_evidence=False, source_urls=[])
        )
    with pytest.raises(ValidationError, match="expect start_probability 0"):
        SignalExample.model_validate(
            _sig(
                expected={
                    "availability": "injured",
                    "start_probability_min": 0,
                    "start_probability_max": 0.3,
                }
            )
        )
    with pytest.raises(ValidationError, match="must not repeat"):
        SignalExample.model_validate(
            _sig(
                expected={
                    "availability": "fit",
                    "availability_alt": ["fit"],
                    "start_probability_min": 0,
                    "start_probability_max": 1,
                }
            )
        )
    with pytest.raises(ValidationError, match="given together"):
        SignalExample.model_validate(
            _sig(
                expected={
                    "availability": "fit",
                    "start_probability_min": 0,
                    "start_probability_max": 1,
                    "expected_minutes_min": 10,
                }
            )
        )
    with pytest.raises(ValidationError):
        SignalExample.model_validate(_sig(id="bad id"))


def test_retrieval_schema_rejects_misaligned_urls():
    row = {
        "id": "ret_x",
        "query": "who is injured",
        "player_id": 1,
        "as_of": "2026-09-17T09:00:00Z",
        "relevant_article_ids": [1, 2],
        "relevant_urls": ["https://a"],
        "rationale": "x" * 40,
    }
    with pytest.raises(ValidationError, match="align 1:1"):
        RetrievalExample.model_validate(row)
    with pytest.raises(ValidationError, match="duplicates"):
        RetrievalExample.model_validate(
            {**row, "relevant_article_ids": [1, 1], "relevant_urls": ["a", "b"]}
        )
    ok = RetrievalExample.model_validate({**row, "relevant_article_ids": [], "relevant_urls": []})
    assert not ok.has_relevant


def test_loader_rejects_duplicate_ids(tmp_path):
    line = SIGNALS[0].model_dump_json()
    p = tmp_path / "dup.jsonl"
    p.write_text(line + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate ids"):
        load_signals(p)
    p.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_signals(p)


# ---------- existence checks against live data (skipped without DB / network) ----------


@pytest.mark.db
def test_relevant_articles_exist_in_db_with_matching_urls():
    from fplcopilot.db import ping, session_scope

    if not ping():
        pytest.skip("Postgres недоступен")
    wanted = {aid for r in RETRIEVAL for aid in r.relevant_article_ids}
    with session_scope() as s:
        rows = dict(
            s.execute(
                text("SELECT id, url FROM news_articles WHERE id = ANY(:ids)"),
                {"ids": list(wanted)},
            ).all()
        )
    assert wanted <= set(rows), f"missing articles: {sorted(wanted - set(rows))}"
    for r in RETRIEVAL:
        for aid, url in zip(r.relevant_article_ids, r.relevant_urls, strict=True):
            assert rows[aid] == url, (r.id, aid)


@pytest.mark.network
def test_player_ids_match_names_in_live_bootstrap():
    from fplcopilot.data import FPLClient

    bs = FPLClient().bootstrap()
    for s in SIGNALS:
        p = bs.player(s.player_id)
        assert s.player_name.lower() in {p.web_name.lower(), p.full_name.lower()}, s.id
