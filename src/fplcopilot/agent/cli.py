"""CLI LangGraph-агента.

    uv run python -m fplcopilot.agent.cli --manager 895045 --strategy balanced "Should I sell Palmer?"
    uv run python -m fplcopilot.agent.cli --thread <id> --decision confirm|reject   # resume после HITL (хит / WC)
    uv run python -m fplcopilot.agent.cli --thread <id> --choose <player_id>        # resume после уточнения имени
    uv run python -m fplcopilot.agent.cli --thread <id> --decision cancel           # отменить уточнение
    uv run python -m fplcopilot.agent.cli --manager 895045 "Who should I captain?" --json

Прогресс по узлам — в stderr, ответ — в stdout, затем компактный футер: интент, инструменты с
латентностью, LLM-вызовы, оценка стоимости, предупреждения валидатора, thread_id. Чекпоинты —
SQLite (AGENT_CHECKPOINT_PATH), поэтому прерванный на подтверждении прогон возобновляется из
другого запуска CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Any

from fplcopilot.agent.graph import Agent, live_agent
from fplcopilot.config import settings

BIG_KEYS = ("predictions", "lineup", "plan", "scenario_result", "router_raw", "signals")


def format_footer(state: dict[str, Any], *, elapsed_s: float | None = None) -> str:
    tools = [t for t in state.get("tool_log") or [] if t.get("tool") not in ("-", None)]
    tool_parts = []
    for t in tools:
        flag = "" if t.get("ok", True) else " FAILED"
        origin = ""
        if t.get("tool") == "analyze_player_risk" and t.get("note"):
            origin = f" [{str(t['note']).split(':')[0]}"
            if t.get("args", {}).get("strategy"):
                origin += f", {t['args']['strategy']}"
            origin += "]"
        tool_parts.append(f"{t['tool']}{origin} {t.get('latency_ms', 0)}ms{flag}")
    calls = state.get("llm_calls") or []
    p_tok = sum(int(c.get("prompt_tokens") or 0) for c in calls)
    c_tok = sum(int(c.get("completion_tokens") or 0) for c in calls)
    call_parts = [
        f"{c.get('purpose', c.get('node'))} {c.get('model')} {c.get('latency_ms', 0)}ms"
        for c in calls
    ]
    v = state.get("validation") or {}
    if state.get("answer") is None:
        validation = "n/a (no answer yet)"
    elif not v:
        validation = "n/a (deterministic answer)"
    elif v.get("passed"):
        validation = (
            f"passed ({v.get('checked_names', 0)} names, {v.get('checked_numbers', 0)} numbers "
            f"checked, attempts {v.get('attempts')})"
        )
    else:
        validation = (
            f"FAILED after {v.get('attempts')} attempts: names {v.get('unknown_names')} "
            f"numbers {v.get('unknown_numbers')}"
            + (
                f" cyrillic {list(v['cyrillic_names'])} (restored to Latin)"
                if v.get("cyrillic_names")
                else ""
            )
        )
    lines = [
        f"— intent: {state.get('intent')} | strategy: {state.get('strategy')} | GW{state.get('gw')} "
        f"| data as of {state.get('as_of')}"
        + (f" | wall {elapsed_s:.1f}s" if elapsed_s is not None else ""),
        "— tools: " + ("; ".join(tool_parts) if tool_parts else "none"),
        f"— LLM calls: {len(calls)}"
        + (f" ({'; '.join(call_parts)})" if call_parts else "")
        + f" | tokens in/out {p_tok}/{c_tok} | cost ≈ ${state.get('cost_estimate', 0.0):.4f}",
        f"— signal retries: {state.get('retries', 0)} | validation: {validation}",
    ]
    if state.get("pending_action") and state.get("interrupted"):
        pa = state["pending_action"]
        lines.append(
            f"— PENDING CONFIRMATION: {pa['kind']} — {pa['detail']} (cost {pa['cost']}). "
            f"Resume: --thread {state.get('thread_id')} --decision confirm|reject"
        )
    elif state.get("pending_clarification") and state.get("interrupted"):
        pc = state["pending_clarification"]
        ids = ", ".join(f"{c['player_id']}={c.get('name')}" for c in pc.get("candidates") or [])
        lines.append(
            f"— PENDING CLARIFICATION: '{pc.get('mention')}' — candidates {ids}. "
            f"Resume: --thread {state.get('thread_id')} --choose <player_id> (or --decision cancel)"
        )
    else:
        lines.append(f"— thread: {state.get('thread_id')}")
    return "\n".join(lines)


def compact_state(state: dict[str, Any]) -> dict[str, Any]:
    """Состояние для --json без гигантских полей (оставляем размеры/ключи)."""
    out: dict[str, Any] = {}
    for k, v in state.items():
        if k in BIG_KEYS and v:
            if isinstance(v, dict):
                out[k] = {"_omitted": True, "keys": sorted(map(str, v.keys()))[:40]}
            else:
                out[k] = {"_omitted": True, "type": type(v).__name__}
        else:
            out[k] = v
    return out


def print_progress(agent: Agent, events, *, quiet: bool) -> None:
    for node, status in events:
        if not quiet:
            print(f"[{node}] {status}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FPL Copilot: LangGraph agent")
    ap.add_argument("query", nargs="?", help="вопрос к агенту")
    ap.add_argument("--manager", type=int, default=settings.fpl_manager_id)
    ap.add_argument(
        "--strategy", default="balanced", choices=["conservative", "balanced", "aggressive"]
    )
    ap.add_argument("--thread", help="thread_id: возобновить прерванный прогон или задать свой id")
    ap.add_argument(
        "--decision",
        choices=["confirm", "reject", "cancel"],
        help="решение для HITL resume: confirm|reject для хита/WC, cancel — отменить уточнение имени",
    )
    ap.add_argument(
        "--choose",
        type=int,
        metavar="PLAYER_ID",
        help="HITL resume после уточнения имени: id выбранного кандидата",
    )
    ap.add_argument(
        "--prompt-version",
        default=None,
        help="версия промптов агента (v1 | v2), по умолчанию AGENT_PROMPT_VERSION",
    )
    ap.add_argument("--checkpoint", help="путь к SQLite-чекпоинтам (AGENT_CHECKPOINT_PATH)")
    ap.add_argument("--no-grader", dest="grader", action="store_false", help="без LLM-грейдера")
    ap.add_argument("--json", action="store_true", help="выдать итоговое состояние JSON")
    ap.add_argument("--quiet", action="store_true", help="без прогресса в stderr")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("httpx", "httpcore", "openai", "langsmith"):
        logging.getLogger(name).setLevel(logging.WARNING)

    resuming = bool(args.decision) or args.choose is not None
    if resuming and not args.thread:
        ap.error("--decision / --choose требуют --thread <id>")
    if not resuming and not args.query:
        ap.error(
            "нужен вопрос (или --thread <id> --decision confirm|reject | --choose <player_id>)"
        )

    agent = live_agent(
        checkpoint_path=args.checkpoint, grader=args.grader, prompt_version=args.prompt_version
    )
    started = time.perf_counter()
    if resuming:
        try:
            print_progress(
                agent,
                agent.stream_resume(args.thread, args.decision, player_id=args.choose),
                quiet=args.quiet,
            )
        except ValueError as exc:
            print(f"ошибка: {exc}", file=sys.stderr)
            return 2
        state = agent.snapshot(args.thread)
    else:
        events = agent.stream(
            args.query, manager_id=args.manager, strategy=args.strategy, thread_id=args.thread
        )
        thread_id = args.thread
        for node, status in events:
            if node == "start":
                thread_id = status.split()[-1]
            if not args.quiet:
                print(f"[{node}] {status}", file=sys.stderr, flush=True)
        assert thread_id is not None
        state = agent.snapshot(thread_id)
    elapsed = time.perf_counter() - started

    agent.close()
    if args.json:
        print(json.dumps(compact_state(state), ensure_ascii=False, indent=1, default=str))
        return 0
    if state.get("interrupted") and state.get("pending_action"):
        pa = state["pending_action"]
        print(
            f"The recommendation includes a {pa['kind']} ({pa['detail']}, cost {pa['cost']} pts).\n"
            f"Confirm to keep it or reject to recompute without it:\n"
            f"  uv run python -m fplcopilot.agent.cli --thread {state['thread_id']} "
            f"--decision confirm|reject"
        )
    elif state.get("interrupted") and state.get("pending_clarification"):
        print(state.get("answer") or "(clarification needed)")
        print(
            f"\n  uv run python -m fplcopilot.agent.cli --thread {state['thread_id']} "
            f"--choose <player_id>   # or --decision cancel"
        )
    else:
        print(state.get("answer") or "(no answer)")
    print()
    print(format_footer(state, elapsed_s=elapsed))
    close = getattr(agent.deps.tools, "close", None)
    if close is not None:
        close()  # ONNX-ранкер flashrank лучше отпустить до teardown интерпретатора
    return 0


def hard_exit(rc: int) -> None:
    """Штатные atexit-обработчики (flush LangSmith и т.п.) выполняем сами, затем выходим без
    teardown интерпретатора: если в процессе загружался ONNX-ранкер flashrank, разрушение его
    пула потоков после финализации Python падает на macOS с `recursive_mutex lock failed`
    (вывод к этому моменту уже напечатан, код возврата сохраняется)."""
    import atexit
    import os

    sys.stdout.flush()
    sys.stderr.flush()
    if "flashrank" in sys.modules:
        atexit._run_exitfuncs()
        os._exit(rc)
    sys.exit(rc)


if __name__ == "__main__":
    hard_exit(main())
