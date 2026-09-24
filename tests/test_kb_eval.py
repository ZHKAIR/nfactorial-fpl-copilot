"""Golden strategy.jsonl проходит схему; чистые метрики evals/run_kb.py (без сети/БД)."""

import json

import pytest

from evals.run_kb import (
    domain_hit,
    domain_of,
    forbidden_hits,
    keyword_recall,
    load_strategy,
    quotes_verbatim,
)
from fplcopilot.rag.kb.answer import NOT_COVERED
from fplcopilot.rag.kb.registry import KNOWN_TAGS


def test_golden_strategy_is_valid_and_balanced():
    rows = load_strategy()
    assert 8 <= len(rows) <= 12
    assert len({r.id for r in rows}) == len(rows)
    not_covered = [r for r in rows if r.expect_not_covered is True]
    assert len(not_covered) >= 1 and all(not r.must_cite for r in not_covered)
    assert all(not r.expected_keywords for r in not_covered)
    cited = [r for r in rows if r.must_cite]
    assert len(cited) >= 6 and all(r.expected_source_domains for r in cited)
    assert all(r.expected_keywords for r in cited)
    for r in rows:
        if r.filter_tags:
            assert set(r.filter_tags) <= KNOWN_TAGS
        for group in r.expected_keywords:
            assert all(alt.strip() for alt in group.split("|"))
    assert any(r.forbidden_patterns for r in rows)  # правило «никаких выдуманных чисел» измеряется


def test_keyword_recall_with_alternatives():
    assert keyword_recall("You can bank up to 5 free transfers.", ["5|five", "hit"]) == 0.5
    assert keyword_recall("Five transfers.", ["5|five"]) == 1.0
    assert keyword_recall("anything", []) is None


def test_domain_hit_ignores_www_and_handles_no_expectation():
    urls = ["https://www.premierleague.com/en/news/1", "https://fplwatch.com/blog/x"]
    assert domain_hit(urls, ["premierleague.com"]) == 1.0
    assert domain_hit(urls, ["www.livefpl.com"]) == 0.0
    assert domain_hit(urls, []) is None
    assert domain_of("https://WWW.LiveFPL.com/blog") == "livefpl.com"


def test_forbidden_patterns_catch_invented_points_for_a_player():
    pattern = r"Haaland[^.]{0,80}\b\d+(\.\d+)?\s*(points|pts)\b"
    assert forbidden_hits("Haaland should score about 12 points this week.", [pattern]) == [pattern]
    assert (
        forbidden_hits("Haaland is a popular captain; the captain scores double.", [pattern]) == []
    )
    assert forbidden_hits(NOT_COVERED, [pattern]) == []


def test_quotes_verbatim_recheck_is_whitespace_tolerant():
    retrieved = [{"chunk_id": 1, "text": "The maximum number of free transfers   is 5."}]
    assert quotes_verbatim([{"chunk_id": 1, "quote": "free transfers is 5."}], retrieved)
    assert not quotes_verbatim([{"chunk_id": 1, "quote": "is 6."}], retrieved)
    assert not quotes_verbatim([{"chunk_id": 2, "quote": "is 5."}], retrieved)
    assert quotes_verbatim([], retrieved)


@pytest.mark.parametrize("bad_id", ["ret_x", "kb-x", "KB_X"])
def test_golden_ids_must_be_kb_prefixed(bad_id, tmp_path):
    path = tmp_path / "strategy.jsonl"
    row = {"id": bad_id, "question": "How many free transfers?", "rationale": "x" * 45}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_strategy(path)
