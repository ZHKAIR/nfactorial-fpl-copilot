"""Faithfulness judge: prompt rendering, verdict -> score mapping, response parsing (no network)."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from evals.judge import (
    JUDGE_MODEL,
    JudgeVerdict,
    build_judge_messages,
    format_fpl_status,
    judge_faithfulness,
    load_judge_prompt,
    parse_judge_completion,
    verdict_score,
)

EV = [
    SimpleNamespace(
        source="ffscout",
        published_at=datetime(2026, 9, 16, 18, 30, tzinfo=UTC),
        quote="Joao Pedro (£7.8m) has pulled out of the Brazil squad,\n casting doubt over Gameweek 5.",
    ),
    SimpleNamespace(
        source="fpl_api",
        published_at=datetime(2026, 9, 16, 19, 0, tzinfo=UTC),
        quote="Unspecified injury - 75% chance of playing",
    ),
]


def test_prompt_file_loads_and_defines_verdicts():
    prompt = load_judge_prompt()
    for word in (
        "supported",
        "partially_supported",
        "unsupported",
        "EVIDENCE",
        "OFFICIAL FPL STATUS",
    ):
        assert word in prompt


def test_build_messages_puts_data_in_user_turn_only():
    msgs = build_judge_messages(
        "Doubtful for GW5 after pulling out of the Brazil squad.",
        EV,
        fpl_status=format_fpl_status("d", 75, "Unspecified injury - 75% chance of playing"),
    )
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == load_judge_prompt()
    user = msgs[1]["content"]
    assert "SUMMARY:\nDoubtful for GW5" in user
    assert 'status=d; chance_next=75; news="Unspecified injury - 75% chance of playing"' in user
    assert (
        '1. [ffscout 2026-09-16] "Joao Pedro (£7.8m) has pulled out of the Brazil squad, casting doubt'
        in user
    )
    assert '2. [fpl_api 2026-09-16] "Unspecified injury - 75% chance of playing"' in user
    assert format_fpl_status("a", None, "") == 'status=a; chance_next=null; news=""'
    assert "EVIDENCE:\n(none)" in build_judge_messages("x", [], fpl_status="s")[1]["content"]


def test_verdict_score_mapping():
    assert verdict_score(JudgeVerdict(verdict="supported", reason="ok")) == 1.0
    assert verdict_score(JudgeVerdict(verdict="partially_supported", reason="date")) == 0.5
    assert verdict_score(JudgeVerdict(verdict="unsupported", reason="invented")) == 0.0
    assert verdict_score("Supported ") == 1.0  # raw string tolerated
    assert verdict_score("maybe") is None
    assert verdict_score(None) is None
    with pytest.raises(ValueError):
        JudgeVerdict(verdict="kinda", reason="x")  # type: ignore[arg-type]


def _completion(parsed, refusal=None, prompt_tokens=120, completion_tokens=15):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, refusal=refusal))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def test_parse_completion_handles_parsed_refusal_and_empty():
    verdict, tokens = parse_judge_completion(
        _completion(JudgeVerdict(verdict="partially_supported", reason="return GW not in quotes"))
    )
    assert verdict is not None and verdict.verdict == "partially_supported"
    assert tokens == {"prompt_tokens": 120, "completion_tokens": 15}
    verdict, tokens = parse_judge_completion(_completion(None, refusal="I can't"))
    assert verdict is None and tokens["prompt_tokens"] == 120
    verdict, tokens = parse_judge_completion(SimpleNamespace(choices=[], usage=None))
    assert verdict is None and tokens == {"prompt_tokens": 0, "completion_tokens": 0}
    # dict payload (e.g. replayed from JSON) is coerced into the model
    verdict, _ = parse_judge_completion(_completion({"verdict": "supported", "reason": "all good"}))
    assert verdict is not None and verdict.verdict == "supported"


class FakeClient:
    def __init__(self, parsed):
        self.calls = []
        self._parsed = parsed
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self._parse))

    def _parse(self, **kw):
        self.calls.append(kw)
        return _completion(self._parsed)


def test_judge_faithfulness_with_fake_client():
    client = FakeClient(JudgeVerdict(verdict="unsupported", reason="knee not in evidence"))
    res = judge_faithfulness("He has a knee injury.", EV, fpl_status="status=d", client=client)
    assert res.score == 0.0 and res.verdict == "unsupported" and "knee" in res.reason
    assert res.prompt_tokens == 120 and res.completion_tokens == 15 and res.model == JUDGE_MODEL
    call = client.calls[0]
    assert call["model"] == JUDGE_MODEL and call["temperature"] == 0
    assert call["response_format"] is JudgeVerdict
    assert call["messages"][0]["role"] == "system"

    refusing = FakeClient(None)
    res = judge_faithfulness("x", EV, fpl_status="status=a", client=refusing)
    assert res.score is None and res.verdict is None and "no structured verdict" in res.reason
