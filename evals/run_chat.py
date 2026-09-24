"""Chat regression eval: evals/golden/chat.jsonl through the live LangGraph agent, called the way the
Streamlit chat page (app/views/5_chat.py `run_agent`) calls it.

    uv run python -m evals.run_chat --manager 895045 --label baseline          # all rows
    uv run python -m evals.run_chat --only c01,c52 --label after-router-v4     # a subset
    uv run python -m evals.run_chat --tags multi_turn,squad_review             # rows with any tag
    uv run python -m evals.run_chat --retry-errors evals/results/<ts>_chat_baseline.json
    uv run python -m evals.run_chat --metrics evals/results/<ts>_chat_baseline.json \
        --review evals/results/<ts>_chat_baseline_review.json                  # recompute + quality
    uv run python -m evals.run_chat --compare evals/results/A.json evals/results/B.json

What a row measures (evals/golden/chat.jsonl): `expect.intents` — intents acceptable under the
current taxonomy (empty = no fitting intent exists today, the row is excluded from intent
accuracy and counted as a capability gap); `expect.behavior` — answer | partial | clarify | refuse
(answer and partial are the same automatic class "answered"; the manual review separates them);
`expect.answer_lang`; `history` — previous turns; `checks.must_any` / `checks.must_not` —
case-insensitive substrings the final answer must / must not contain.

Calling convention = the chat page: `agent.stream(prompt, manager_id=..., strategy=...,
history=[...])`, the thread id from the "start" event, then `agent.snapshot(thread_id)`. History =
previous messages as the chat builds them (`agent.chat.history_from_messages`: user text;
assistant answer + `meta` of its turn). Canned history rows carry no meta (text only); `--history
replay` runs the previous user turns live and passes their real answers and meta. The baseline
agent (before chat v4) had no `history` keyword — recorded per row as `history_passed=false`.
HITL: a pending paid action (hit / Wildcard) is confirmed automatically (the user would click
"Подтвердить"), a pending name clarification is recorded as behavior "clarify" and not resumed.

Side effects are those of a normal chat question: stale news signals may be re-extracted and cached,
plans are snapshotted by build_gameweek_plan. Checkpoints go to an in-memory saver, not to the UI's
SQLite file. Results: evals/results/<UTC ts>_chat_<label>.json, rewritten after every row.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import os
import re
import sys
import time
import traceback
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals import GOLDEN_DIR, RESULTS_DIR
from evals.metrics import percentile

log = logging.getLogger("evals.chat")

ROWS_PATH = GOLDEN_DIR / "chat.jsonl"
PROJECT_DIR = Path(__file__).resolve().parent.parent
FINGERPRINT_GLOBS = (
    "src/fplcopilot/agent/*.py",
    "src/fplcopilot/agent/prompts/*/*.md",
    "src/fplcopilot/config.py",
    "src/fplcopilot/rag/*.py",
    "src/fplcopilot/app/views/5_chat.py",
)
REFUSE_INTENTS = frozenset({"off_topic", "betting"})
ANSWERED = frozenset({"answer", "partial"})
BEHAVIORS = ("answer", "partial", "clarify", "refuse")
VALIDATION_MARK = "⚠ Validation"
# v4: оговорка валидатора на языке вопроса (agent.chat.validation_note)
VALIDATION_NOTES = (
    VALIDATION_MARK,
    "не удалось сверить с расчётом",
    "could not be matched to the computed",
)
# Exceptions that usually mean a neighbouring edit left the code half-written mid-run.
CODE_ERRORS = (
    "ImportError",
    "ModuleNotFoundError",
    "SyntaxError",
    "NameError",
    "AttributeError",
    "TypeError",
    "IndentationError",
    "KeyError",
)
DEFAULT_BUDGET_USD = 0.6
JUDGE_MODEL = "gpt-4o-mini"

_URL = re.compile(r"https?://\S+|fpl://\S+")
_BRACKET = re.compile(r"\[[^\]\n]*\]")
_CYR = re.compile(r"[А-Яа-яЁё]")
_LAT = re.compile(r"[A-Za-z]")
_CAP_LATIN_WORD = re.compile(r"\b[A-Z][\w'’.-]*")
_DOMAIN_TOKENS = re.compile(r"\b(xPts|GW\d*|FSI|FPL|XI|FT|BB|TC|WC|FH|p\(start\)|n/a)\b")
_EN_WORDS = re.compile(r"\b[a-z]{2,}\b")


# ---------- dataset ----------


def load_rows(path: Path = ROWS_PATH) -> list[dict[str, Any]]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        exp = row.get("expect") or {}
        bad = [b for b in exp.get("behavior") or [] if b not in BEHAVIORS]
        if not row.get("id") or not row.get("query") or not exp.get("behavior") or bad:
            raise ValueError(f"{path}:{n}: id, query, expect.behavior ({BEHAVIORS}) required")
        row.setdefault("history", [])
        row.setdefault("tags", [])
        row.setdefault("checks", {})
        rows.append(row)
    ids = [r["id"] for r in rows]
    dup = [i for i, c in Counter(ids).items() if c > 1]
    if dup:
        raise ValueError(f"duplicate ids in {path}: {dup}")
    return rows


def select_rows(
    rows: list[dict[str, Any]], only: str | None, tags: str | None
) -> list[dict[str, Any]]:
    if only:
        wanted = {x.strip() for x in only.split(",") if x.strip()}
        rows = [r for r in rows if r["id"] in wanted]
    if tags:
        wanted_tags = {x.strip() for x in tags.split(",") if x.strip()}
        rows = [r for r in rows if wanted_tags & set(r["tags"])]
    return rows


# ---------- code fingerprint (neighbouring edits during a run) ----------


def fingerprint() -> dict[str, str]:
    out: dict[str, str] = {}
    for pattern in FINGERPRINT_GLOBS:
        for p in sorted(PROJECT_DIR.glob(pattern)):
            out[str(p.relative_to(PROJECT_DIR))] = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
    return out


def changed_files(before: dict[str, str], after: dict[str, str]) -> list[str]:
    keys = set(before) | set(after)
    return sorted(k for k in keys if before.get(k) != after.get(k))


def git_head() -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


# ---------- agent adapter (mirrors app/views/5_chat.py) ----------


def build_agent(*, grader: bool, prompt_version: str | None) -> Any:
    from fplcopilot.agent.graph import Agent, live_deps
    from fplcopilot.agent.tools import LiveTools

    tools = LiveTools()  # the UI shares one LiveTools per process (common.get_tools)
    return Agent(live_deps(grader=grader, tools=tools, prompt_version=prompt_version))


def history_kwarg(agent: Any) -> str | None:
    params = inspect.signature(agent.stream).parameters
    for name in ("history", "messages"):
        if name in params:
            return name
    return None


def call_agent(
    agent: Any,
    prompt: str,
    *,
    manager_id: int | None,
    strategy: str,
    history: list[dict[str, str]],
) -> tuple[dict[str, Any], list[tuple[str, str]], bool]:
    """(final state, (node, status) events, history_passed) — one chat message."""
    kwargs: dict[str, Any] = {"manager_id": manager_id, "strategy": strategy}
    key = history_kwarg(agent)
    passed = bool(key and history)
    if passed:
        kwargs[key] = history  # type: ignore[index]
    events: list[tuple[str, str]] = []
    thread_id: str | None = None
    for node, line in agent.stream(prompt, **kwargs):
        if node == "start":
            thread_id = line.split()[-1]
        events.append((node, line))
    assert thread_id is not None, "agent.stream yielded no start event"
    return agent.snapshot(thread_id), events, passed


def resume_confirm(
    agent: Any, state: dict[str, Any]
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    events = list(agent.stream_resume(state["thread_id"], "confirm"))
    return agent.snapshot(state["thread_id"]), events


# ---------- per-row classification ----------


def behavior_of(state: dict[str, Any]) -> str:
    intent = state.get("intent")
    if intent == "error":
        return "error"
    if intent in REFUSE_INTENTS:
        return "refuse"
    if state.get("interrupted") and state.get("pending_clarification"):
        return "clarify"
    if not (state.get("answer") or "").strip():
        return "no_answer"
    return "answer"


def detect_lang(text: str | None) -> str | None:
    """ru / en / mixed by the prose: tables, URLs, [citations], capitalised Latin words (names,
    clubs) and domain tokens (xPts, GW6, FSI) are ignored."""
    if not text or not text.strip():
        return None
    prose = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("|"))
    prose = _BRACKET.sub(" ", _URL.sub(" ", prose))
    prose = _DOMAIN_TOKENS.sub(" ", prose)
    prose = _CAP_LATIN_WORD.sub(" ", prose)
    cyr = len(_CYR.findall(prose))
    lat = len(_LAT.findall(prose))
    if cyr + lat < 20:
        return None
    share = cyr / (cyr + lat)
    if share >= 0.6:
        return "ru"
    if share <= 0.1:
        return "en"
    return "mixed"


def english_lines(text: str | None) -> list[str]:
    """Lines of a Russian answer that are English prose (>= 6 lowercase English words, no
    Cyrillic) — e.g. the validator footer."""
    out = []
    for ln in (text or "").splitlines():
        if ln.lstrip().startswith("|") or _CYR.search(ln):
            continue
        if len(_EN_WORDS.findall(_URL.sub(" ", ln))) >= 6:
            out.append(ln.strip()[:160])
    return out


def run_checks(answer: str | None, checks: dict[str, Any]) -> dict[str, Any]:
    low = (answer or "").lower()
    res: dict[str, Any] = {}
    if checks.get("must_any"):
        res["must_any"] = any(s.lower() in low for s in checks["must_any"])
    if checks.get("must_not"):
        hits = [s for s in checks["must_not"] if s.lower() in low]
        res["must_not"] = not hits
        if hits:
            res["must_not_hits"] = hits
    return res


def validation_summary(state: dict[str, Any]) -> dict[str, Any] | None:
    v = state.get("validation")
    if not v:
        return None
    first = (v.get("history") or [None])[0] or {}
    return {
        "passed": v.get("passed"),
        "attempts": v.get("attempts"),
        "unknown_names": v.get("unknown_names"),
        "unknown_numbers": v.get("unknown_numbers"),
        "bad_citations": v.get("bad_citations"),
        "hit_mismatch": v.get("hit_mismatch"),
        "first_attempt_passed": first.get("passed"),
        "first_attempt": {
            k: v for k, v in first.items() if not k.startswith("checked_") and k != "passed" and v
        },
        "final_violations": {
            k: val
            for k, val in v.items()
            if k
            in (
                "unknown_names",
                "unknown_numbers",
                "bad_citations",
                "hit_mismatch",
                "cyrillic_names",
                "unknown_gws",
                "gw_uncovered",
                "captain_mismatch",
                "missing_refs",
                "misattributed_numbers",
            )
            and val
        },
        "pruned_sources": len(v.get("pruned_sources") or []),
    }


def record_from_state(
    row: dict[str, Any],
    state: dict[str, Any],
    *,
    elapsed_s: float,
    events: Sequence[tuple[str, str]],
    history_passed: bool,
    hitl: str | None,
) -> dict[str, Any]:
    answer = state.get("answer")
    behavior = behavior_of(state)
    exp = row["expect"]
    expected_intents = exp.get("intents") or []
    intent = state.get("intent")
    lang = detect_lang(answer)
    calls = state.get("llm_calls") or []
    tools = [
        {
            "node": t.get("node"),
            "tool": t.get("tool"),
            "ok": t.get("ok", True),
            "latency_ms": t.get("latency_ms"),
            "note": str(t.get("note") or "")[:200],
        }
        for t in state.get("tool_log") or []
    ]
    facts = state.get("facts") or {}
    failures = tool_failures(tools)
    return {
        "id": row["id"],
        "query": row["query"],
        "lang": row.get("lang"),
        "tags": row["tags"],
        "history_turns": len(row["history"]),
        "history_passed": history_passed,
        "expected_intents": expected_intents,
        "ideal": exp.get("ideal"),
        "expected_behavior": exp["behavior"],
        "expected_answer_lang": exp.get("answer_lang"),
        "intent": intent,
        "intent_ok": (intent in expected_intents) if expected_intents else None,
        "behavior": behavior,
        "behavior_ok": behavior_ok(behavior, exp["behavior"]),
        "router": {
            "reason": (state.get("router_raw") or {}).get("reason"),
            "language": state.get("language"),
            "players": [p.get("name") for p in state.get("players") or []],
            "unresolved": state.get("unresolved") or [],
            "horizon": state.get("horizon"),
            "scenario": state.get("scenario"),
            "ranking": state.get("ranking"),
            "needs_squad": state.get("needs_squad"),
            "raw": state.get("router_raw"),
        },
        "gw": state.get("gw"),
        "hitl": hitl,
        "tools": tools,
        "tools_called": [t["tool"] for t in tools if t["tool"] not in ("-", None)],
        "facts_keys": sorted(facts.keys()),
        "facts": facts,  # для разбора валидатора офлайн (числа уже округлены инструментами)
        "headline": facts.get("headline"),
        "caveats": state.get("caveats") or [],
        "evidence": [
            {
                "player": e.get("player") or e.get("club"),
                "source": e.get("source"),
                "date": e.get("date"),
            }
            for e in state.get("evidence") or []
        ],
        "answer": answer,
        "answer_lang": lang,
        "lang_ok": (lang == exp.get("answer_lang")) if lang and exp.get("answer_lang") else None,
        "english_lines": english_lines(answer) if exp.get("answer_lang") == "ru" else [],
        "validation": validation_summary(state),
        "validation_warning": any(m in (answer or "") for m in VALIDATION_NOTES),
        "checks": run_checks(answer, row.get("checks") or {}),
        "llm_calls": [
            {
                "purpose": c.get("purpose"),
                "model": c.get("model"),
                "prompt_tokens": c.get("prompt_tokens"),
                "completion_tokens": c.get("completion_tokens"),
                "latency_ms": c.get("latency_ms"),
                "cost_usd": c.get("cost_usd"),
            }
            for c in calls
        ],
        "cost_usd": round(sum(float(c.get("cost_usd") or 0.0) for c in calls), 6),
        "elapsed_s": round(elapsed_s, 2),
        "events": [f"{n}: {s}"[:240] for n, s in events],
        "thread_id": state.get("thread_id"),
        "tool_failures": failures,
        "infra_failure": is_infra_failure(failures),
        "error": None,
    }


def tool_failures(tools: Iterable[dict[str, Any]]) -> list[str]:
    """Failed tool / LLM calls the graph survived (e.g. OpenAI 429 -> explain fallback JSON)."""
    return [
        f"{t['tool']}: {str(t.get('note') or '')[:120]}"
        for t in tools
        if not t.get("ok", True) and t.get("tool") not in (None, "-")
    ]


def is_infra_failure(failures: Sequence[str]) -> bool:
    """Infra noise, not product behaviour — `--retry-errors` re-runs these rows too."""
    return any(
        "RateLimit" in f
        or "APIConnection" in f
        or "Timeout" in f
        or f.startswith(("explain_llm", "router_llm"))
        for f in failures
    )


def refresh_expectations(records: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]) -> None:
    """Re-score stored records against the current rows file (expectations changed, e.g. new
    intents added to `expect.intents`): expected_*, intent_ok, behavior_ok, lang_ok, checks."""
    for r in records:
        row = by_id.get(r["id"])
        if row is None or r.get("behavior") == "error":
            continue
        exp = row["expect"]
        r["expected_intents"] = exp.get("intents") or []
        r["expected_behavior"] = exp["behavior"]
        r["expected_answer_lang"] = exp.get("answer_lang")
        r["ideal"] = exp.get("ideal")
        r["intent_ok"] = (
            (r.get("intent") in r["expected_intents"]) if r["expected_intents"] else None
        )
        r["behavior_ok"] = behavior_ok(r["behavior"], exp["behavior"])
        lang = r.get("answer_lang")
        r["lang_ok"] = (lang == exp.get("answer_lang")) if lang and exp.get("answer_lang") else None
        r["checks"] = run_checks(r.get("answer"), row.get("checks") or {})


def annotate_failures(records: list[dict[str, Any]]) -> None:
    """Result files written before these fields existed: derive them from the stored tool log."""
    for r in records:
        if "infra_failure" not in r and r.get("tools") is not None:
            r["tool_failures"] = tool_failures(r["tools"])
            r["infra_failure"] = is_infra_failure(r["tool_failures"])


def behavior_ok(actual: str, expected: Sequence[str]) -> bool:
    if actual == "answer":
        return bool(ANSWERED & set(expected))
    return actual in expected


def error_record(row: dict[str, Any], exc: BaseException, elapsed_s: float) -> dict[str, Any]:
    tb = traceback.format_exc()
    kind = type(exc).__name__
    return {
        "id": row["id"],
        "query": row["query"],
        "lang": row.get("lang"),
        "tags": row["tags"],
        "history_turns": len(row["history"]),
        "expected_intents": row["expect"].get("intents") or [],
        "ideal": row["expect"].get("ideal"),
        "expected_behavior": row["expect"]["behavior"],
        "expected_answer_lang": row["expect"].get("answer_lang"),
        "behavior": "error",
        "error": {
            "type": kind,
            "message": str(exc)[:500],
            "traceback_tail": tb[-2000:],
            "likely_concurrent_edit": kind in CODE_ERRORS,
        },
        "cost_usd": 0.0,
        "elapsed_s": round(elapsed_s, 2),
    }


def run_row(agent: Any, row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    history = [dict(h) for h in row["history"]]
    replay_cost = 0.0
    try:
        if args.history == "replay" and history:
            # Previous user turns go through the agent like real chat messages; their real
            # answers replace the canned assistant turns.
            replayed: list[dict[str, str]] = []
            for turn in history:
                if turn["role"] != "user":
                    continue
                st, _, _ = call_agent(
                    agent,
                    turn["content"],
                    manager_id=args.manager,
                    strategy=args.strategy,
                    history=replayed,
                )
                replay_cost += float(st.get("cost_estimate") or 0.0)
                replayed += [
                    turn,
                    {
                        "role": "assistant",
                        "content": st.get("answer") or "",
                        "meta": st.get("turn_meta") or {},
                    },
                ]
            history = replayed
        elif args.history == "none":
            history = []
        turn_started = time.perf_counter()
        state, events, passed = call_agent(
            agent, row["query"], manager_id=args.manager, strategy=args.strategy, history=history
        )
        hitl = None
        if state.get("interrupted") and state.get("pending_action"):
            hitl = f"action:{state['pending_action'].get('kind')}"
            if args.hitl == "confirm":
                state, more = resume_confirm(agent, state)
                events += more
        elif state.get("interrupted") and state.get("pending_clarification"):
            hitl = "clarification"
        rec = record_from_state(
            row,
            state,
            elapsed_s=time.perf_counter() - turn_started,
            events=events,
            history_passed=passed,
            hitl=hitl,
        )
        rec["replay_cost_usd"] = round(replay_cost, 6)
        return rec
    except Exception as exc:  # one broken row must not stop the suite
        log.exception("%s failed", row["id"])
        return error_record(row, exc, time.perf_counter() - started)


# ---------- metrics ----------


def _rate(num: int, den: int) -> dict[str, Any]:
    return {"num": num, "den": den, "rate": round(num / den, 3) if den else None}


def compute_metrics(
    records: list[dict[str, Any]], review: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    ok = [r for r in records if r.get("behavior") != "error"]
    scored = [r for r in ok if r.get("expected_intents")]
    expect_answer = [r for r in ok if not set(r["expected_behavior"]) <= {"refuse"}]
    expect_refuse = [r for r in ok if set(r["expected_behavior"]) == {"refuse"}]
    refused = [r for r in ok if r["behavior"] == "refuse"]
    with_text = [r for r in ok if r.get("answer_lang")]
    explained = [r for r in ok if r.get("validation")]
    checked = [r for r in ok if r.get("checks")]
    lat = [float(r["elapsed_s"]) for r in ok]
    lat_answered = [float(r["elapsed_s"]) for r in ok if r["behavior"] == "answer"]
    costs = [float(r.get("cost_usd") or 0.0) for r in records]
    gap_rows = [r for r in ok if not r.get("expected_intents")]
    m: dict[str, Any] = {
        "rows": len(records),
        "errors": len(records) - len(ok),
        "errors_concurrent_edit": sum(
            1 for r in records if (r.get("error") or {}).get("likely_concurrent_edit")
        ),
        "infra_failures": sum(1 for r in records if r.get("infra_failure")),
        "retried_rows": sum(1 for r in records if r.get("retried_after_error")),
        "intent_accuracy": _rate(sum(1 for r in scored if r["intent_ok"]), len(scored)),
        "behavior_accuracy": _rate(sum(1 for r in ok if r["behavior_ok"]), len(ok)),
        "false_refusal": _rate(
            sum(1 for r in expect_answer if r["behavior"] == "refuse"), len(expect_answer)
        ),
        "false_acceptance": _rate(
            sum(1 for r in expect_refuse if r["behavior"] != "refuse"), len(expect_refuse)
        ),
        "capability_gap_rows_refused": _rate(
            sum(1 for r in gap_rows if r["behavior"] == "refuse"), len(gap_rows)
        ),
        "answer_lang_match": _rate(sum(1 for r in with_text if r["lang_ok"]), len(with_text)),
        "ru_answers_with_english_lines": _rate(
            sum(1 for r in with_text if r.get("english_lines")),
            sum(1 for r in with_text if r.get("expected_answer_lang") == "ru"),
        ),
        "refusals_in_query_language": _rate(
            sum(1 for r in refused if r.get("lang_ok")), len(refused)
        ),
        "validator_warning_shown": _rate(
            sum(1 for r in explained if r.get("validation_warning")), len(explained)
        ),
        "validator_failed_first_attempt": _rate(
            sum(1 for r in explained if r["validation"].get("first_attempt_passed") is False),
            len(explained),
        ),
        "checks_passed": _rate(
            sum(
                1 for r in checked if all(v for k, v in r["checks"].items() if k != "must_not_hits")
            ),
            len(checked),
        ),
        "clarifications": sum(1 for r in ok if r["behavior"] == "clarify"),
        "hitl_actions": sum(1 for r in ok if str(r.get("hitl") or "").startswith("action")),
        "history_rows": sum(1 for r in records if r.get("history_turns")),
        "history_passed": sum(1 for r in records if r.get("history_passed")),
        "latency_s": {
            "p50": _pct(lat, 50),
            "p95": _pct(lat, 95),
            "max": max(lat) if lat else None,
            "answered_p50": _pct(lat_answered, 50),
            "answered_p95": _pct(lat_answered, 95),
        },
        "cost_usd": {
            "total": round(sum(costs), 4),
            "replay": round(sum(float(r.get("replay_cost_usd") or 0) for r in records), 4),
            "failed_attempts": round(
                sum(float(r.get("cost_usd_failed_attempt") or 0) for r in records), 4
            ),
            "mean": round(sum(costs) / len(costs), 5) if costs else None,
            "max": round(max(costs), 5) if costs else None,
        },
        "intent_counts": dict(Counter(str(r.get("intent")) for r in ok)),
        "behavior_counts": dict(Counter(r["behavior"] for r in records)),
    }
    by_tag: dict[str, dict[str, int]] = {}
    for r in ok:
        for t in r["tags"]:
            d = by_tag.setdefault(t, {"n": 0, "behavior_ok": 0, "refused": 0})
            d["n"] += 1
            d["behavior_ok"] += int(bool(r["behavior_ok"]))
            d["refused"] += int(r["behavior"] == "refuse")
    m["by_tag"] = dict(sorted(by_tag.items()))
    if review:
        graded = [r for r in records if r["id"] in review]
        grades = Counter(review[r["id"]]["grade"] for r in graded)
        m["quality_review"] = {
            "graded": len(graded),
            **{g: _rate(grades.get(g, 0), len(graded)) for g in ("helpful", "partial", "bad")},
        }
        both = [r for r in graded if r.get("judge")]
        if both:
            pairs = Counter((review[r["id"]]["grade"], r["judge"]["grade"]) for r in both)
            manual_bad = [r for r in both if review[r["id"]]["grade"] == "bad"]
            m["judge_vs_manual"] = {
                "agreement": _rate(sum(v for (a, b), v in pairs.items() if a == b), len(both)),
                "bad_recall": _rate(
                    sum(1 for r in manual_bad if r["judge"]["grade"] == "bad"), len(manual_bad)
                ),
                "confusion_manual_to_judge": {
                    f"{a}->{b}": v for (a, b), v in sorted(pairs.items())
                },
            }
    return m


def _pct(xs: list[float], q: float) -> float | None:
    return round(percentile(xs, q), 2) if xs else None


# ---------- optional LLM judge (approximates the manual review for before/after) ----------

JUDGE_SYSTEM = """You grade answers of FPL Copilot, a Fantasy Premier League assistant used by
Russian-speaking managers. Given the conversation so far, the user's question, what the ideal
system would do (expected behaviour and capability), and the assistant's answer, grade the answer:
- helpful: answers what was asked (or refuses a truly off-topic request) in the user's language,
  numbers and reasoning are on topic, nothing misleading;
- partial: on topic but incomplete, a generic answer where a personal one was expected, the wrong
  gameweek, awkward format, technical noise, or an honest "cannot do this" that still helps;
- bad: wrong refusal, wrong language, answers a different question, invented or nonsensical
  reasoning, or misleading claims.
Reply with the grade and one short reason in Russian (<= 20 words)."""


def judge(records: list[dict[str, Any]], rows: dict[str, dict[str, Any]], budget: float) -> float:
    from pydantic import BaseModel

    from fplcopilot.agent.llm import estimate_cost
    from fplcopilot.rag.llm import get_openai_client

    class Grade(BaseModel):
        grade: str
        reason: str

    client = get_openai_client()
    spent = 0.0
    for r in records:
        if r.get("behavior") == "error" or spent > budget:
            continue
        row = rows.get(r["id"]) or {}
        hist = "\n".join(f"{h['role']}: {h['content']}" for h in row.get("history") or [])
        user = (
            f"HISTORY:\n{hist or '(none)'}\n\nQUESTION: {r['query']}\n"
            f"EXPECTED: behaviour {r['expected_behavior']}, capability {r.get('ideal')}, "
            f"answer language {r.get('expected_answer_lang')}; note: {row.get('note', '')}\n"
            f"TOOLS' VERDICT (ground truth the answer must follow; empty if none): "
            f"{r.get('headline') or ''}\n"
            f"ROUTED AS: {r.get('intent')}\n\n"
            f"ANSWER:\n{(r.get('answer') or '(no answer: ' + r['behavior'] + ')')[:6000]}"
        )
        out = client.chat.completions.parse(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            response_format=Grade,
            temperature=0,
            max_completion_tokens=120,
        )
        usage = out.usage
        spent += estimate_cost(
            JUDGE_MODEL,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )
        parsed = out.choices[0].message.parsed
        if parsed is not None:
            g = parsed.grade.strip().lower()
            r["judge"] = {
                "grade": g if g in ("helpful", "partial", "bad") else "partial",
                "reason": parsed.reason,
                "model": JUDGE_MODEL,
            }
    return spent


# ---------- router stability probe (router LLM only, no tools) ----------


def router_probe(
    rows: list[dict[str, Any]], n: int, *, prompt_version: str | None, strategy: str, budget: float
) -> dict[str, Any]:
    """The router call exactly as agent.graph.router builds it (manager squad available), `n`
    times per row: intent distribution, refusal share, flips. Queries only — the router has no
    history input today."""
    from fplcopilot.agent.chat import render_history
    from fplcopilot.agent.graph import _fmt_dt, _iso
    from fplcopilot.agent.llm import RouterRequest, router_llm
    from fplcopilot.agent.tools import GameweekContextInput, LiveTools
    from fplcopilot.config import settings

    tools = LiveTools()
    ctx = tools.get_gameweek_context(GameweekContextInput(manager_id=None, strategy=strategy))
    spent = 0.0
    out: list[dict[str, Any]] = []
    for row in rows:
        if spent >= budget:
            break
        req = RouterRequest(
            query=row["query"],
            as_of=_fmt_dt(_iso(ctx.as_of)),
            gw=ctx.gw,
            deadline=_fmt_dt(_iso(ctx.deadline)),
            has_manager=True,
            strategy=strategy,
            history=render_history(row.get("history") or []),
        )
        intents: list[str] = []
        reasons: list[str] = []
        for _ in range(n):
            res, usage = router_llm(req, model=settings.agent_router_model, version=prompt_version)
            spent += usage.cost_usd
            intents.append(res.intent)
            reasons.append(res.reason)
        c = Counter(intents)
        refused = sum(v for k, v in c.items() if k in REFUSE_INTENTS)
        expected = row["expect"].get("intents") or []
        out.append(
            {
                "id": row["id"],
                "query": row["query"],
                "expected_intents": expected,
                "expected_behavior": row["expect"]["behavior"],
                "intents": dict(c),
                "refused_share": round(refused / n, 2),
                "stable": len(c) == 1,
                "ok_share": round(sum(v for k, v in c.items() if k in expected) / n, 2)
                if expected
                else None,
                "reasons": list(dict.fromkeys(reasons))[:3],
                "standalone": getattr(res, "standalone_query", None),
            }
        )
        print(f"  {row['id']} {dict(c)} refused {refused}/{n} | {row['query'][:70]}", flush=True)
    close = getattr(tools, "close", None)
    if callable(close):
        close()
    answerable = [r for r in out if not set(r["expected_behavior"]) <= {"refuse"}]
    return {
        "n": n,
        "rows": out,
        "cost_usd": round(spent, 5),
        "summary": {
            "rows": len(out),
            "unstable_rows": sum(1 for r in out if not r["stable"]),
            "answerable_rows_refused_at_least_once": sum(
                1 for r in answerable if r["refused_share"] > 0
            ),
            "answerable_rows_refused_always": sum(1 for r in answerable if r["refused_share"] == 1),
            "answerable_rows": len(answerable),
        },
    }


# ---------- output ----------


def save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


def load_review(path: Path | None) -> dict[str, dict[str, Any]] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("reviews", data) if isinstance(data, dict) else data
    if isinstance(items, dict):
        return items
    return {x["id"]: x for x in items}


def fmt_rate(x: Any) -> str:
    if isinstance(x, dict) and "rate" in x:
        return "n/a" if x["rate"] is None else f"{x['num']}/{x['den']} ({x['rate']:.0%})"
    return str(x)


HEADLINE_METRICS = (
    "intent_accuracy",
    "behavior_accuracy",
    "false_refusal",
    "false_acceptance",
    "capability_gap_rows_refused",
    "answer_lang_match",
    "ru_answers_with_english_lines",
    "refusals_in_query_language",
    "validator_warning_shown",
    "validator_failed_first_attempt",
    "checks_passed",
)


def print_metrics(m: dict[str, Any]) -> None:
    print(
        f"rows {m['rows']}, errors {m['errors']} (concurrent-edit-like {m['errors_concurrent_edit']})"
    )
    for k in HEADLINE_METRICS:
        print(f"  {k:<32} {fmt_rate(m[k])}")
    print(f"  {'latency_s':<32} {m['latency_s']}")
    print(f"  {'cost_usd':<32} {m['cost_usd']}")
    print(
        f"  history rows {m['history_rows']}, history passed to agent {m['history_passed']}; "
        f"clarifications {m['clarifications']}, HITL actions {m['hitl_actions']}"
    )
    if m.get("quality_review"):
        q = m["quality_review"]
        print(
            f"  quality (manual) graded {q['graded']}: helpful {fmt_rate(q['helpful'])}, "
            f"partial {fmt_rate(q['partial'])}, bad {fmt_rate(q['bad'])}"
        )
    if m.get("quality_judge"):
        q = m["quality_judge"]
        print(
            f"  quality (LLM judge) graded {q['graded']}: helpful {fmt_rate(q['helpful'])}, "
            f"partial {fmt_rate(q['partial'])}, bad {fmt_rate(q['bad'])}"
        )
    if m.get("judge_vs_manual"):
        j = m["judge_vs_manual"]
        print(
            f"  judge vs manual: agreement {fmt_rate(j['agreement'])}, "
            f"bad recall {fmt_rate(j['bad_recall'])}, {j['confusion_manual_to_judge']}"
        )


def judge_metrics(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    graded = [r for r in records if r.get("judge")]
    if not graded:
        return None
    c = Counter(r["judge"]["grade"] for r in graded)
    return {
        "graded": len(graded),
        **{g: _rate(c.get(g, 0), len(graded)) for g in ("helpful", "partial", "bad")},
    }


def compare(a_path: Path, b_path: Path) -> None:
    a = json.loads(a_path.read_text(encoding="utf-8"))
    b = json.loads(b_path.read_text(encoding="utf-8"))
    print(
        f"A = {a_path.name} ({a['meta'].get('label')})\nB = {b_path.name} ({b['meta'].get('label')})"
    )
    for k in (*HEADLINE_METRICS, "quality_review", "quality_judge"):
        va, vb = a["metrics"].get(k), b["metrics"].get(k)
        if isinstance(va, dict) and "helpful" in va:
            va, vb = va.get("helpful"), (vb or {}).get("helpful")
            k += ".helpful"
        print(f"  {k:<40} {fmt_rate(va):>16}  ->  {fmt_rate(vb)}")
    for key in ("p50", "p95"):
        print(
            f"  latency {key:<32} {a['metrics']['latency_s'][key]:>16}  ->  "
            f"{b['metrics']['latency_s'][key]}"
        )
    print(
        f"  cost total {'':<29} {a['metrics']['cost_usd']['total']:>16}  ->  "
        f"{b['metrics']['cost_usd']['total']}"
    )
    rb = {r["id"]: r for r in b["records"]}
    print("\nchanged rows (intent / behavior / lang / warning):")
    for ra in a["records"]:
        r2 = rb.get(ra["id"])
        if r2 is None:
            continue
        fa = (
            ra.get("intent"),
            ra.get("behavior"),
            ra.get("answer_lang"),
            ra.get("validation_warning"),
        )
        fb = (
            r2.get("intent"),
            r2.get("behavior"),
            r2.get("answer_lang"),
            r2.get("validation_warning"),
        )
        if fa != fb:
            print(f"  {ra['id']} {ra['query'][:50]!r}: {fa} -> {fb}")


def print_row(rec: dict[str, Any], spent: float) -> None:
    if rec.get("error"):
        e = rec["error"]
        print(
            f"  {rec['id']} ERROR {e['type']}: {e['message'][:120]}"
            + (" [likely concurrent edit]" if e["likely_concurrent_edit"] else ""),
            flush=True,
        )
        return
    mark = "ok " if rec["behavior_ok"] and rec["intent_ok"] is not False else "BAD"
    print(
        f"  {mark} {rec['id']} intent={rec['intent']}{'' if rec['intent_ok'] is not False else '(!)'} "
        f"beh={rec['behavior']} lang={rec['answer_lang']} warn={int(rec['validation_warning'])} "
        f"{rec['elapsed_s']:.1f}s ${rec['cost_usd']:.4f} | total ${spent:.3f} | {rec['query'][:60]}",
        flush=True,
    )


# ---------- main ----------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Chat regression eval through the live agent")
    ap.add_argument("--rows", type=Path, default=ROWS_PATH)
    ap.add_argument("--manager", type=int, default=895045)
    ap.add_argument(
        "--strategy", default="balanced", choices=["conservative", "balanced", "aggressive"]
    )
    ap.add_argument("--prompt-version", default=None, help="AGENT_PROMPT_VERSION by default")
    ap.add_argument("--no-grader", dest="grader", action="store_false")
    ap.add_argument("--only", help="comma-separated row ids")
    ap.add_argument("--tags", help="comma-separated tags (rows with any of them)")
    ap.add_argument(
        "--history",
        choices=["canned", "replay", "none"],
        default="canned",
        help="canned: dataset turns; replay: run previous user turns live; none",
    )
    ap.add_argument("--hitl", choices=["confirm", "stop"], default="confirm")
    ap.add_argument("--label", default="run")
    ap.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET_USD)
    ap.add_argument("--judge", action="store_true", help="LLM judge grade per row (gpt-4o-mini)")
    ap.add_argument("--judge-only", type=Path, help="add judge grades to an existing result file")
    ap.add_argument("--retry-errors", type=Path, help="re-run error rows of a result file in place")
    ap.add_argument("--metrics", type=Path, help="recompute metrics of a result file")
    ap.add_argument("--review", type=Path, help="manual review JSON {id: {grade, reason}}")
    ap.add_argument(
        "--refresh-expect",
        action="store_true",
        help="with --metrics: re-score records against the current rows file",
    )
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("A", "B"))
    ap.add_argument(
        "--router-probe",
        type=int,
        metavar="N",
        help="router LLM only, N calls per selected row (stability of the intent)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("httpx", "httpcore", "openai", "langsmith"):
        logging.getLogger(name).setLevel(logging.WARNING)

    if args.compare:
        compare(*args.compare)
        return 0
    review = load_review(args.review)
    all_rows = load_rows(args.rows)
    by_id = {r["id"]: r for r in all_rows}

    if args.router_probe:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        probe = router_probe(
            select_rows(all_rows, args.only, args.tags),
            args.router_probe,
            prompt_version=args.prompt_version,
            strategy=args.strategy,
            budget=args.budget_usd,
        )
        probe["meta"] = {
            "label": args.label,
            "git_head": git_head(),
            "fingerprint": fingerprint(),
            "prompt_version": args.prompt_version or "default",
        }
        path = RESULTS_DIR / f"{stamp}_chat_router_probe_{args.label}.json"
        save(probe, path)
        print(probe["summary"], f"cost ${probe['cost_usd']}", f"saved {path}", sep="\n")
        return 0

    if args.metrics or args.judge_only:
        path = args.metrics or args.judge_only
        payload = json.loads(path.read_text(encoding="utf-8"))
        annotate_failures(payload["records"])
        if args.refresh_expect:
            refresh_expectations(payload["records"], by_id)
            payload["meta"]["expectations_refreshed_from"] = str(args.rows)
        if args.judge_only:
            payload["meta"]["judge_cost_usd"] = round(
                judge(payload["records"], by_id, args.budget_usd), 5
            )
        payload["metrics"] = compute_metrics(payload["records"], review)
        jm = judge_metrics(payload["records"])
        if jm:
            payload["metrics"]["quality_judge"] = jm
        if review:
            payload["meta"]["review_file"] = str(args.review)
        save(payload, path)
        print_metrics(payload["metrics"])
        return 0

    if args.retry_errors:
        path = args.retry_errors
        payload = json.loads(path.read_text(encoding="utf-8"))
        annotate_failures(payload["records"])
        todo = [
            by_id[r["id"]]
            for r in payload["records"]
            if r.get("behavior") == "error" or r.get("infra_failure")
        ]
        meta = payload["meta"]
    else:
        todo = select_rows(all_rows, args.only, args.tags)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = RESULTS_DIR / f"{stamp}_chat_{args.label}.json"
        meta = {
            "label": args.label,
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "rows_file": str(
                args.rows.relative_to(PROJECT_DIR) if args.rows.is_absolute() else args.rows
            ),
            "manager_id": args.manager,
            "strategy": args.strategy,
            "history_mode": args.history,
            "hitl": args.hitl,
            "git_head": git_head(),
        }
        payload = {"meta": meta, "records": [], "metrics": {}}
    print(f"{len(todo)} row(s) -> {path}")

    from fplcopilot.config import settings

    before = fingerprint()
    agent = build_agent(grader=args.grader, prompt_version=args.prompt_version)
    meta.update(
        {
            "prompt_version": agent.deps.prompt_version,
            "router_model": settings.agent_router_model,
            "explain_model": settings.agent_explain_model,
            "explain_temperature": settings.agent_explain_temperature,
            "grader": args.grader,
            "history_kwarg": history_kwarg(agent),
            "fingerprint_start": meta.get("fingerprint_start") or before,
        }
    )
    records: list[dict[str, Any]] = payload["records"]
    index = {r["id"]: i for i, r in enumerate(records)}
    spent = sum(
        float(r.get("cost_usd") or 0.0) + float(r.get("replay_cost_usd") or 0.0) for r in records
    )
    try:
        for row in todo:
            if spent >= args.budget_usd:
                print(f"budget ${args.budget_usd} reached — stopping", flush=True)
                break
            rec = run_row(agent, row, args)
            if rec["id"] in index:
                old = records[index[rec["id"]]]
                rec["retried_after_error"] = (old.get("error") or {}).get("type") or (
                    "; ".join(old.get("tool_failures") or [])[:300] or None
                )
                rec["cost_usd_failed_attempt"] = old.get("cost_usd")
                records[index[rec["id"]]] = rec
            else:
                index[rec["id"]] = len(records)
                records.append(rec)
            spent += float(rec.get("cost_usd") or 0.0) + float(rec.get("replay_cost_usd") or 0.0)
            print_row(rec, spent)
            payload["metrics"] = compute_metrics(records, review)
            save(payload, path)
    finally:
        close = getattr(agent, "close", None)
        if callable(close):
            close()
    after = fingerprint()
    meta["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    meta["fingerprint_end"] = after
    meta["files_changed_during_run"] = changed_files(before, after)
    if args.judge:
        meta["judge_cost_usd"] = round(judge(records, by_id, max(0.0, args.budget_usd - spent)), 5)
    payload["metrics"] = compute_metrics(records, review)
    jm = judge_metrics(records)
    if jm:
        payload["metrics"]["quality_judge"] = jm
    save(payload, path)
    print()
    print_metrics(payload["metrics"])
    if meta["files_changed_during_run"]:
        print("files changed during the run:", ", ".join(meta["files_changed_during_run"]))
    print(f"saved {path}")
    return 0


def hard_exit(rc: int) -> None:
    """As agent/cli.hard_exit: skip interpreter teardown when the flashrank ONNX ranker was
    loaded (its thread pool crashes on macOS after finalisation)."""
    import atexit

    sys.stdout.flush()
    sys.stderr.flush()
    if "flashrank" in sys.modules:
        atexit._run_exitfuncs()
        os._exit(rc)
    sys.exit(rc)


if __name__ == "__main__":
    hard_exit(main())
