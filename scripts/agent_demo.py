"""Демо LangGraph-агента (v2) и мини-A/B промптов объяснителя.

    uv run python scripts/agent_demo.py [--manager 895045] [--strategy balanced] [--out docs/agent_demo_output_v2.md]
    uv run python scripts/agent_demo.py --ab docs/agent_prompt_ab.md      # + A/B v1 vs v2 на тех же фактах

Сценарии покрывают все ветки графа: статус игрока (сигнал из БД), трансфер с принудительной
продажей, капитан, план на 5 туров, what-if с хитом (HITL: прерывание, затем resume reject и
confirm на двух тредах), отказ по ставкам, уточнение неоднозначного имени (v2: HITL-прерывание ->
resume с выбранным игроком -> полный ответ), off-topic; v2 добавляет два русских вопроса (ответ на
русском; стратегический вопрос с цитатами KB) и вопрос про хит, где объяснитель цитирует правило из
KB (`rules_context`). Транскрипт — материал для защиты: прогресс по узлам, ответ, футер с
инструментами/латентностью/стоимостью.

Мини-A/B (`--ab`): для 8 исходных сценариев факты считаются один раз (граф v2), а объяснитель
запускается дважды — с промптом v1 и v2 — на одном и том же запросе; сравниваются нарушения
валидатора до регенерации, регенерации, токены/стоимость, заглушки-цитаты, соблюдение правила
«Sources только по обсуждаемым игрокам» и подпись колонки Δ.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fplcopilot.agent import llm as agent_llm
from fplcopilot.agent.cli import format_footer, hard_exit
from fplcopilot.agent.graph import Agent, _fmt_dt, live_agent
from fplcopilot.agent.llm import ExplainRequest
from fplcopilot.agent.validate import validate_answer
from fplcopilot.config import settings
from fplcopilot.rag.entity_matcher import strip_accents

# (метка, вопрос). Первые 8 — сценарии демо v1 (для A/B), дальше — новые сценарии v2.
SCENARIOS: list[tuple[str, str]] = [
    ("player_status (signal from DB / extraction)", "Is João Pedro fit for GW5?"),
    ("transfer (forced sale, constrained MILP)", "Should I sell Palmer?"),
    ("captain (best XI + captain options)", "Who should I captain this week?"),
    (
        "plan (5 GWs, wildcard compare, diff vs previous plan)",
        "Plan my transfers for the next 5 gameweeks",
    ),
    (
        "what_if (HITL: hit -> interrupt -> reject / confirm; rules_context cited)",
        "What if I take a -4 to bring in Saka and Guéhi?",
    ),
    ("betting refusal (domain guardrail)", "Best odds for Arsenal to win?"),
    (
        "clarification (ambiguous name -> HITL interrupt -> resume with player_id)",
        "Should I sell Gabriel?",
    ),
    ("off_topic", "Give me a recipe for plov"),
    (
        "transfer in Russian (answer in Russian, names/numbers unchanged)",
        "Стоит ли продавать Palmer?",
    ),
    (
        "strategy_question in Russian (KB answer with [n] citations)",
        "Когда лучше сыграть Wildcard?",
    ),
    (
        "transfer, hit asked explicitly (rules_context from KB cited as [source])",
        "Should I take a -4 this week to fix my squad?",
    ),
]
N_ORIGINAL = 8
PLACEHOLDER = re.compile(r"\[\s*(facts?|source|src|evidence|dd\.mm)[^\]\n]{0,40}\]", re.IGNORECASE)
URL = re.compile(r"https?://\S+|fpl://\S+")
SOURCES_HEAD = re.compile(r"^\W*\**\s*(Sources|Источники)\**\W*$", re.IGNORECASE | re.MULTILINE)
NEXT_HEAD = re.compile(
    r"^\W*\**\s*(Caveats|Оговорки|Why|Почему|Verdict|Вердикт)", re.IGNORECASE | re.MULTILINE
)


# ---------- прогон ----------


def run_streamed(
    agent: Agent, query: str, *, manager_id: int, strategy: str
) -> tuple[dict, list[str], float]:
    started = time.perf_counter()
    lines: list[str] = []
    thread_id = None
    for node, status in agent.stream(query, manager_id=manager_id, strategy=strategy):
        if node == "start":
            thread_id = status.split()[-1]
        lines.append(f"[{node}] {status}")
    assert thread_id
    return agent.snapshot(thread_id), lines, time.perf_counter() - started


def resume_streamed(
    agent: Agent, thread_id: str, decision: str | None = None, *, player_id: int | None = None
) -> tuple[dict, list[str], float]:
    started = time.perf_counter()
    lines = [
        f"[{node}] {status}"
        for node, status in agent.stream_resume(thread_id, decision, player_id=player_id)
    ]
    return agent.snapshot(thread_id), lines, time.perf_counter() - started


def block(text: str, lang: str = "text") -> str:
    return f"```{lang}\n{text.rstrip()}\n```"


def interrupt_banner(state: dict) -> str | None:
    if not state.get("interrupted"):
        return None
    if state.get("pending_action"):
        pa = state["pending_action"]
        return (
            f"⏸ **Interrupted before `confirm_action`** — pending {pa['kind']}: {pa['detail']} "
            f"(cost {pa['cost']} pts). Thread `{state['thread_id']}` saved in SQLite checkpoints; "
            "resume with `--decision confirm|reject`."
        )
    if state.get("pending_clarification"):
        pc = state["pending_clarification"]
        cands = ", ".join(f"{c['player_id']} = {c.get('full_name')}" for c in pc["candidates"])
        return (
            f"⏸ **Interrupted before `resolve_clarification`** — ambiguous `{pc['mention']}`: "
            f"{cands}. Resume with `--choose <player_id>` (or `--decision cancel`).\n\n"
            + (state.get("answer") or "")
        )
    return None


def section(title: str, query: str, state: dict, progress: list[str], elapsed: float) -> str:
    answer = interrupt_banner(state) or state.get("answer") or "(no answer)"
    parts = [
        f"### {title}",
        "",
        f"**Query:** `{query}`",
        "",
        "<details><summary>Node progress (stderr)</summary>",
        "",
        block("\n".join(progress)),
        "",
        "</details>",
        "",
        answer,
        "",
        block(format_footer(state, elapsed_s=elapsed)),
        "",
    ]
    return "\n".join(parts)


def summary_row(label: str, state: dict, elapsed: float) -> dict[str, Any]:
    tools = [t for t in state.get("tool_log") or [] if t.get("tool") not in ("-", None)]
    calls = state.get("llm_calls") or []
    v = state.get("validation") or {}
    facts = state.get("facts") or {}
    kb = []
    if facts.get("strategy_answer"):
        kb.append(f"kb answer ({len(facts['strategy_answer'].get('citations') or [])} cit.)")
    if facts.get("rules_context"):
        kb.append(f"rules_context:{facts['rules_context'].get('about')}")
    return {
        "scenario": label,
        "intent": state.get("intent"),
        "lang": state.get("language") or "-",
        "kb": ", ".join(kb) or "-",
        "tools": len(tools),
        "llm_calls": len(calls),
        "tokens": sum(
            int(c.get("prompt_tokens") or 0) + int(c.get("completion_tokens") or 0) for c in calls
        ),
        "cost": float(state.get("cost_estimate") or 0.0),
        "retries": int(state.get("retries") or 0),
        "validation": ("n/a" if not v else ("passed" if v.get("passed") else "failed"))
        + (f" ({v.get('attempts')} attempt(s))" if v else ""),
        "wall_s": elapsed,
    }


def pick_candidate(state: dict) -> int:
    """Кандидат для resume: тот, что в составе менеджера, иначе первый (детерминированно)."""
    cands = (state.get("pending_clarification") or {}).get("candidates") or []
    for c in cands:
        if c.get("in_squad"):
            return int(c["player_id"])
    return int(cands[0]["player_id"])


# ---------- мини-A/B промптов объяснителя ----------


def _fold(s: str) -> str:
    return strip_accents(s or "").lower()


def sources_lines(answer: str) -> tuple[str, list[str]]:
    """(текст до раздела Sources, строки раздела Sources)."""
    m = SOURCES_HEAD.search(answer)
    if not m:
        return answer, []
    body = answer[: m.start()]
    rest = answer[m.end() :]
    nxt = NEXT_HEAD.search(rest)
    section_text = rest[: nxt.start()] if nxt else rest
    lines = [ln.strip() for ln in section_text.splitlines() if URL.search(ln)]
    return body + (rest[nxt.start() :] if nxt else ""), lines


def sources_compliance(answer: str, facts: dict, evidence: list[dict]) -> dict[str, Any]:
    """Каждая строка Sources должна (а) ссылаться на URL из EVIDENCE или rules_context и
    (б) относиться к игроку, который упомянут в тексте ответа (вне раздела Sources)."""
    body, lines = sources_lines(answer)
    body_f = _fold(body)
    by_url: dict[str, set[str]] = {}
    for e in evidence:
        by_url.setdefault(str(e.get("url")), set()).add(str(e.get("player")))
    rule_urls = {
        str(x.get("url")) for x in (facts.get("rules_context") or {}).get("excerpts") or []
    }
    rule_urls |= {
        str(c.get("url")) for c in (facts.get("strategy_answer") or {}).get("citations") or []
    }
    ok = 0
    undiscussed: list[str] = []
    unknown_urls: list[str] = []
    for ln in lines:
        urls = [u.rstrip(").,;") for u in URL.findall(ln)]
        good = True
        for u in urls:
            if u in rule_urls:
                continue
            players = by_url.get(u)
            if players is None:
                good = False
                unknown_urls.append(u)
                continue
            mentioned = any(
                all(tok in body_f for tok in _fold(p).split() if len(tok) > 2) for p in players
            )
            if not mentioned:
                good = False
                undiscussed.extend(sorted(players))
        ok += int(good)
    return {
        "lines": len(lines),
        "compliant_lines": ok,
        "compliant": len(lines) == ok,
        "undiscussed_players": sorted(set(undiscussed)),
        "unknown_urls": unknown_urls,
    }


def delta_labelled(answer: str) -> bool | None:
    """Колонка Δ подписана (например «Δ next GW», «Δ horizon»), а не голая «Δ». None — таблицы нет."""
    headers = [ln for ln in answer.splitlines() if ln.strip().startswith("|") and "Δ" in ln]
    if not headers:
        return None
    cells = [c.strip() for c in headers[0].strip().strip("|").split("|")]
    delta_cells = [c for c in cells if "Δ" in c]
    return all(len(c.replace("Δ", "").split()) >= 1 for c in delta_cells)


def explain_ab(
    label: str, state: dict, versions: tuple[str, ...] = ("v1", "v2")
) -> list[dict[str, Any]]:
    facts = state.get("facts") or {}
    if state.get("user_decision"):
        history = state.get("action_history") or []
        sc = state.get("scenario") or {}
        facts = {
            **facts,
            "user_decision": {
                "decision": state["user_decision"],
                "on": history[-1] if history else None,
                "scenario_after_decision": {
                    "allow_hit": sc.get("allow_hit"),
                    "use_wildcard": sc.get("use_wildcard"),
                },
            },
        }
    evidence = state.get("evidence") or []
    names = [p["name"] for p in state.get("players") or []] + [
        p.get("full_name", "") for p in state.get("players") or []
    ]
    names += [p["name"] for p in (state.get("squad_summary") or {}).get("squad") or []]
    rows = []
    for version in versions:
        req = ExplainRequest(
            query=state["query"],
            intent=state.get("intent") or "unknown",
            strategy=state["strategy"],
            as_of=_fmt_dt(state.get("as_of")),
            facts=facts,
            evidence=evidence,
            caveats=state.get("caveats") or [],
            language=state.get("language") or "en",
        )
        attempts: list[dict[str, Any]] = []
        answer = ""
        for attempt in range(2):
            out, usage = agent_llm.explain_llm(
                req, model=settings.agent_explain_model, version=version
            )
            answer = out.answer_markdown.strip()
            res = validate_answer(answer, facts, evidence, extra_names=names + EXTRA_NAMES)
            attempts.append(
                {
                    "usage": usage,
                    "result": res,
                    "answer": answer,
                }
            )
            if res.passed:
                break
            req = req.model_copy(update={"feedback": res.feedback})
        first = attempts[0]["result"]
        compliance = sources_compliance(answer, facts, evidence)
        rows.append(
            {
                "scenario": label,
                "version": version,
                "violations_first": len(first.unknown_names)
                + len(first.unknown_numbers)
                + len(first.bad_citations)
                + int(first.missing_refs),
                "violation_detail": {
                    "names": first.unknown_names,
                    "numbers": first.unknown_numbers,
                    "citations": first.bad_citations,
                },
                "regenerations": len(attempts) - 1,
                "passed_final": attempts[-1]["result"].passed,
                "answer_tokens": sum(a["usage"].completion_tokens for a in attempts),
                "prompt_tokens": sum(a["usage"].prompt_tokens for a in attempts),
                "cost": sum(a["usage"].cost_usd for a in attempts),
                "placeholders": bool(PLACEHOLDER.search(answer)),
                "sources": compliance,
                "delta_labelled": delta_labelled(answer),
                "rules_cited": bool(
                    facts.get("rules_context")
                    and any(
                        f"[{x['source']}]" in answer
                        for x in facts["rules_context"].get("excerpts") or []
                    )
                ),
                "answer": answer,
            }
        )
    return rows


EXTRA_NAMES: list[str] = []


def write_ab_report(rows: list[dict[str, Any]], out: Path, *, model: str) -> None:
    def agg(version: str) -> dict[str, Any]:
        vs = [r for r in rows if r["version"] == version]
        n = len(vs) or 1
        return {
            "n": len(vs),
            "violations": sum(r["violations_first"] for r in vs),
            "regen": sum(r["regenerations"] for r in vs),
            "passed": sum(r["passed_final"] for r in vs),
            "tokens": sum(r["answer_tokens"] for r in vs) / n,
            "cost": sum(r["cost"] for r in vs),
            "placeholders": sum(r["placeholders"] for r in vs),
            "sources_ok": sum(r["sources"]["compliant"] for r in vs),
            "sources_lines": sum(r["sources"]["lines"] for r in vs),
            "sources_bad_lines": sum(
                r["sources"]["lines"] - r["sources"]["compliant_lines"] for r in vs
            ),
            "delta_ok": sum(1 for r in vs if r["delta_labelled"] is True),
            "delta_tables": sum(1 for r in vs if r["delta_labelled"] is not None),
            "rules_cited": sum(r["rules_cited"] for r in vs),
            "rules_applicable": sum(1 for r in vs if "rules_context" in (r.get("facts") or {})),
        }

    a, b = agg("v1"), agg("v2")
    md = [
        "# Mini A/B: agent explain prompt v1 vs v2",
        "",
        (
            f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M}Z by `scripts/agent_demo.py --ab`. "
            f"Facts computed ONCE per scenario by the v2 graph (router v2, tools, KB context); the "
            f"explainer ({model}, T=0.2) then ran twice on identical FACTS / EVIDENCE / CAVEATS — "
            "with `prompts/v1/explain.system.md` and `prompts/v2/explain.system.md`. Validation = "
            "the deterministic `agent/validate.py` (names, decimals, placeholder citations, KB "
            "markers); one regeneration with feedback on failure, as in the graph."
        ),
        "",
        "## Per scenario",
        "",
        "| scenario | v | violations (1st) | regen | final | answer tok | cost $ | placeholders | Sources lines ok | Δ labelled | rule cited |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        s = r["sources"]
        det = r["violation_detail"]
        viol = str(r["violations_first"])
        flagged = det["names"] + det["numbers"] + det["citations"]
        if flagged:
            viol += " (" + ", ".join(flagged[:4]) + ")"
        dl = {True: "yes", False: "no", None: "n/a"}[r["delta_labelled"]]
        md.append(
            f"| {r['scenario']} | {r['version']} | {viol} | {r['regenerations']} | "
            f"{'pass' if r['passed_final'] else 'FAIL'} | {r['answer_tokens']} | {r['cost']:.4f} | "
            f"{'yes' if r['placeholders'] else 'no'} | {s['compliant_lines']}/{s['lines']}"
            + (
                f" (undiscussed: {', '.join(s['undiscussed_players'])})"
                if s["undiscussed_players"]
                else ""
            )
            + (" (unknown url)" if s["unknown_urls"] else "")
            + f" | {dl} | {'yes' if r['rules_cited'] else '-'} |"
        )
    md += [
        "",
        "## Totals",
        "",
        "| metric | v1 | v2 |",
        "|---|---|---|",
        f"| scenarios | {a['n']} | {b['n']} |",
        f"| validator violations before regeneration | {a['violations']} | {b['violations']} |",
        f"| regenerations | {a['regen']} | {b['regen']} |",
        f"| passed after ≤ 1 regeneration | {a['passed']}/{a['n']} | {b['passed']}/{b['n']} |",
        f"| answer tokens (completion, mean per scenario incl. regenerations) | {a['tokens']:.0f} | {b['tokens']:.0f} |",
        f"| explain cost, $ (all attempts) | {a['cost']:.4f} | {b['cost']:.4f} |",
        f"| answers with placeholder citations ([FACTS], [source, dd.mm]) | {a['placeholders']} | {b['placeholders']} |",
        f"| answers whose Sources list only discussed players / cited URLs | {a['sources_ok']}/{a['n']} | {b['sources_ok']}/{b['n']} |",
        f"| non-compliant Sources lines (undiscussed player or unknown URL) | {a['sources_bad_lines']}/{a['sources_lines']} | {b['sources_bad_lines']}/{b['sources_lines']} |",
        f"| tables with an explicitly labelled Δ column | {a['delta_ok']}/{a['delta_tables']} | {b['delta_ok']}/{b['delta_tables']} |",
        f"| hit/chip answers citing a rules_context excerpt as [source] | {a['rules_cited']} | {b['rules_cited']} |",
        "",
    ]
    out.write_text("\n".join(md), encoding="utf-8")


# ---------- main ----------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FPL Copilot: agent demo transcript (+ prompt A/B)")
    ap.add_argument("--manager", type=int, default=895045)
    ap.add_argument("--strategy", default="balanced")
    ap.add_argument("--out", default="docs/agent_demo_output_v2.md")
    ap.add_argument("--ab", default=None, help="путь отчёта A/B (например docs/agent_prompt_ab.md)")
    ap.add_argument(
        "--prompt-version", default=None, help="версия промптов графа (по умолчанию из настроек)"
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for name in ("httpx", "httpcore", "openai", "langsmith"):
        logging.getLogger(name).setLevel(logging.WARNING)

    agent = live_agent(prompt_version=args.prompt_version)
    version = agent.deps.prompt_version
    EXTRA_NAMES.extend(agent.deps.extra_known_names)
    started_at = datetime.now(UTC)
    md: list[str] = [
        "# Agent demo transcript (prompts v2)",
        "",
        (
            f"Generated {started_at:%Y-%m-%d %H:%M}Z by `scripts/agent_demo.py` — manager "
            f"{args.manager}, strategy `{args.strategy}`, prompt version `{version}`, models "
            f"router/explain `{settings.agent_router_model}`/`{settings.agent_explain_model}` "
            "(see docs/agent.md). Every number in the answers comes from deterministic tools "
            "(xPts v0, MILP optimizer, news signals); strategy rules come from the cited knowledge "
            "base (docs/strategy_kb.md); the LLM only routes and explains, and its answer is "
            "validated against the facts. v2 additions: answers in the user's language, "
            "`rules_context` citations for hit / chip decisions, `strategy_question` answered by "
            "the KB with `[n]` citations, ambiguous names resolved through a second HITL interrupt."
        ),
        "",
    ]
    rows: list[dict[str, Any]] = []
    ab_states: list[tuple[str, dict]] = []
    total_cost = 0.0
    for i, (label, query) in enumerate(SCENARIOS, start=1):
        print(f"[{i}/{len(SCENARIOS)}] {query}", file=sys.stderr, flush=True)
        state, progress, elapsed = run_streamed(
            agent, query, manager_id=args.manager, strategy=args.strategy
        )
        md.append(section(f"{i}. {label}", query, state, progress, elapsed))
        rows.append(summary_row(f"{i}. {label}", state, elapsed))
        total_cost += float(state.get("cost_estimate") or 0.0)
        if state.get("interrupted") and state.get("pending_clarification"):
            pid = pick_candidate(state)
            thread = state["thread_id"]
            print(f"    resume {thread} -> choose {pid}", file=sys.stderr, flush=True)
            st_c, prog_c, el_c = resume_streamed(agent, thread, player_id=pid)
            md.append(
                section(
                    f"{i}a. resume `--choose {pid}` (thread {thread})", query, st_c, prog_c, el_c
                )
            )
            rows.append(summary_row(f"{i}a. clarified -> {pid}", st_c, el_c))
            total_cost += float(st_c.get("cost_estimate") or 0.0) - float(
                state.get("cost_estimate") or 0.0
            )
            state = st_c
        if state.get("interrupted") and state.get("pending_action"):
            # ветка A: reject на том же треде (второй вызов = отдельный процесс в CLI; здесь — тот же объект)
            thread_a = state["thread_id"]
            print(f"    resume {thread_a} -> reject", file=sys.stderr, flush=True)
            st_a, prog_a, el_a = resume_streamed(agent, thread_a, "reject")
            md.append(
                section(f"{i}a. resume `reject` (thread {thread_a})", query, st_a, prog_a, el_a)
            )
            rows.append(summary_row(f"{i}a. reject branch", st_a, el_a))
            total_cost += float(st_a.get("cost_estimate") or 0.0) - float(
                state.get("cost_estimate") or 0.0
            )
            if i <= N_ORIGINAL and st_a.get("answer") and st_a.get("facts"):
                ab_states.append((f"{i}a. {label.split(' (')[0]} reject", st_a))
            # ветка B: тот же вопрос на новом треде, resume confirm
            st_b0, prog_b0, el_b0 = run_streamed(
                agent, query, manager_id=args.manager, strategy=args.strategy
            )
            thread_b = st_b0["thread_id"]
            print(f"    new thread {thread_b} -> confirm", file=sys.stderr, flush=True)
            st_b, prog_b, el_b = resume_streamed(agent, thread_b, "confirm")
            md.append(
                section(
                    f"{i}b. resume `confirm` (new thread {thread_b})",
                    query,
                    st_b,
                    prog_b0 + prog_b,
                    el_b0 + el_b,
                )
            )
            rows.append(summary_row(f"{i}b. confirm branch", st_b, el_b0 + el_b))
            total_cost += float(st_b.get("cost_estimate") or 0.0)
            if i <= N_ORIGINAL and st_b.get("answer") and st_b.get("facts"):
                ab_states.append((f"{i}b. {label.split(' (')[0]} confirm", st_b))
        elif (
            i <= N_ORIGINAL
            and state.get("answer")
            and state.get("facts")
            and state.get("validation")
        ):
            ab_states.append((f"{i}. {label.split(' (')[0]}", state))

    md.append("## Summary")
    md.append("")
    md.append(
        "| scenario | intent | lang | KB | tools | LLM calls | tokens | cost $ | signal retries | validation | wall s |"
    )
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(
            f"| {r['scenario']} | {r['intent']} | {r['lang']} | {r['kb']} | {r['tools']} | {r['llm_calls']} | "
            f"{r['tokens']} | {r['cost']:.4f} | {r['retries']} | {r['validation']} | {r['wall_s']:.1f} |"
        )
    md.append("")
    md.append(
        f"**Total LLM cost for the demo: ≈ ${total_cost:.4f}** ({settings.agent_explain_model} "
        "prices; embeddings for retrieval queries add < $0.0001). Wall time is dominated by LLM "
        "latency and, on first use, by loading the flashrank reranker; optimizer solves take "
        "0.4–1.3 s, the strategy-KB answer 3–5 s (retrieval + one gpt-4o-mini call)."
    )
    md.append("")
    out = Path(args.out)
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out} ({len(rows)} runs, total cost ≈ ${total_cost:.4f})", file=sys.stderr)

    if args.ab:
        ab_rows: list[dict[str, Any]] = []
        for label, st in ab_states:
            print(f"[A/B] {label}", file=sys.stderr, flush=True)
            for r in explain_ab(label, st):
                r["facts"] = st.get("facts") or {}
                ab_rows.append(r)
        ab_cost = sum(r["cost"] for r in ab_rows)
        write_ab_report(ab_rows, Path(args.ab), model=settings.agent_explain_model)
        print(
            f"wrote {args.ab} ({len(ab_states)} scenarios x 2 versions, explain cost ≈ ${ab_cost:.4f})",
            file=sys.stderr,
        )
    agent.close()
    return 0


if __name__ == "__main__":
    hard_exit(main())
