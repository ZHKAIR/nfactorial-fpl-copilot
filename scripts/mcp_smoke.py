"""Smoke-тест MCP-сервера fpl-intelligence через клиент из SDK `mcp` (stdio).

    uv run python scripts/mcp_smoke.py [--manager 895045] [--player Haaland] [--risk "João Pedro"]
    uv run python scripts/mcp_smoke.py --quiet-server        # лог сервера не показывать

Спавнит `uv run python -m fplcopilot.mcp_server`, делает initialize, list_tools / list_resources /
list_resource_templates / list_prompts, затем вызывает get_gameweek_context, predict_player,
recommend_transfers, analyze_player_risk (единственный инструмент с возможным LLM-вызовом),
search_strategy_kb (стратегическая KB, теги hits) и читает fpl://manager/{id}/squad и
fpl://kb/stats. Печатает компактные результаты и латентности (клиентские, с учётом транспорта).
Нужны Postgres и OPENAI_API_KEY из .env (эмбеддинг запроса KB).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


def structured(result: Any) -> Any:
    """CallToolResult -> dict (structured_content, иначе JSON из первого текстового блока)."""
    if getattr(result, "structured_content", None) is not None:
        sc = result.structured_content
        return sc.get("result", sc) if isinstance(sc, dict) and set(sc) == {"result"} else sc
    for block in result.content:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"text": text}
    return None


def short_routes(payload: dict[str, Any]) -> list[str]:
    lines = []
    for r in payload.get("routes") or []:
        lines.append(
            f"  #{r['rank']} {', '.join(r['out'])} -> {', '.join(r['in'])} | hit {r['hit_cost']} "
            f"| Δnext {r['gain_next_gw']:+} | Δhor {r['gain_horizon']:+} | {r['verdict']}"
        )
    return lines


async def main(args: argparse.Namespace, errlog: Any) -> int:
    env = dict(os.environ)
    server = StdioServerParameters(
        command="uv",
        args=["run", "--directory", str(ROOT), "python", "-m", "fplcopilot.mcp_server"],
        env=env,
        cwd=str(ROOT),
    )
    timings: dict[str, int] = {}
    t_all = time.perf_counter()
    async with (
        stdio_client(server, errlog=errlog) as (read, write),
        ClientSession(read, write) as session,
    ):
        t = time.perf_counter()
        init = await session.initialize()
        timings["initialize"] = round((time.perf_counter() - t) * 1000)
        info = init.server_info
        print(
            f"server: {info.name} v{info.version or '-'} protocol {init.protocol_version} "
            f"| instructions {len(init.instructions or '')} chars | {timings['initialize']} ms"
        )

        t = time.perf_counter()
        tools = (await session.list_tools()).tools
        timings["list_tools"] = round((time.perf_counter() - t) * 1000)
        print(f"\ntools ({len(tools)}, {timings['list_tools']} ms):")
        for tool in tools:
            req = tool.input_schema.get("required", [])
            props = list(tool.input_schema.get("properties", {}))
            first = (tool.description or "").strip().split("\n")[0]
            print(f"  - {tool.name}({', '.join(props)}) required={req}\n      {first[:110]}")

        resources = (await session.list_resources()).resources
        templates = (await session.list_resource_templates()).resource_templates
        prompts = (await session.list_prompts()).prompts
        print(f"\nresources: {[str(r.uri) for r in resources]}")
        print(f"resource templates: {[t.uri_template for t in templates]}")
        print("prompts: " + str([(p.name, [a.name for a in (p.arguments or [])]) for p in prompts]))

        async def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            t0 = time.perf_counter()
            res = await session.call_tool(name, arguments, read_timeout_seconds=180)
            ms = round((time.perf_counter() - t0) * 1000)
            timings[name] = ms
            payload = structured(res) or {}
            flag = " ERROR" if payload.get("error") or getattr(res, "is_error", False) else ""
            print(f"\n== {name}({json.dumps(arguments, ensure_ascii=False)}) {ms} ms{flag}")
            return payload

        ctx = await call("get_gameweek_context", {})
        if "error" in ctx:
            print("  ", ctx)
        else:
            print(f"  GW{ctx['gw']} deadline {ctx['deadline']} | as_of {ctx['as_of']}")
            print(f"  note: {ctx.get('squad_note')}")

        pred = await call("predict_player", {"player": args.player, "horizon": 3})
        if "error" in pred:
            print("  ", pred)
        else:
            p = pred["player"]
            print(
                f"  {p['name']} ({p['team']}, {p['position']}, £{p['price']}, status {p['status']}) "
                f"total_xpts {pred['total_xpts']} over {len(pred['by_gw'])} GWs"
            )
            for g in pred["by_gw"]:
                fx = " ".join(
                    f"{'v' if f['is_home'] else '@'}{f['opponent']}(FSI {f['fsi']})"
                    for f in g["fixtures"]
                )
                print(
                    f"    GW{g['gw']}: xPts {g['xpts']} ±{g['sd']} p_start {g['p_start']} "
                    f"min {g['exp_minutes']} | {fx or 'blank'}"
                )

        rec = await call(
            "recommend_transfers", {"manager_id": args.manager, "strategy": "balanced"}
        )
        if "error" in rec:
            print("  ", rec)
        else:
            print(
                f"  GW{rec['gw']} FT {rec['free_transfers']} bank £{rec['bank']} | "
                f"baseline XI {rec['baseline_xi_points']} | recommendation: {rec['recommendation']}"
            )
            print("\n".join(short_routes(rec)))
            if rec.get("notes"):
                print(f"  notes: {rec['notes']}")

        risk = await call("analyze_player_risk", {"player": args.risk})
        if "error" in risk:
            print("  ", risk)
        else:
            print(
                f"  {risk['player']}: FPL status {risk['fpl_status']} ({risk.get('fpl_chance')}%) | "
                f"signal {risk['availability']} conf {risk['confidence']} p_start "
                f"{risk['start_probability']} | origin {risk['origin']} age {risk.get('age_h')} h "
                f"| llm_calls {risk['llm_calls']} | extraction {risk['latency_ms']} ms"
            )
            print(f"  summary: {risk['summary'][:200]}")
            for e in risk["evidence"][:3]:
                print(f"    [{e['source']}, {e['date']}] {e['quote'][:140]}")
            if risk.get("note"):
                print(f"  note: {risk['note']}")

        kb = await call(
            "search_strategy_kb", {"query": args.kb_query, "tags": args.kb_tags, "k": 3}
        )
        if "error" in kb:
            print("  ", kb)
        else:
            print(f"  {len(kb.get('chunks') or [])} chunks (tags {kb.get('tags')}):")
            for c in kb.get("chunks") or []:
                snippet = " ".join(c["text"].split("\n", 1)[-1].split())[:120]
                print(
                    f"    [{c['source']}] {c['title'][:60]} {c['tags']} score {c.get('score')}\n"
                    f"      {snippet}…\n      {c['url']}"
                )
            if kb.get("note"):
                print(f"  note: {kb['note']}")

        uri = f"fpl://manager/{args.manager}/squad"
        t = time.perf_counter()
        res = await session.read_resource(uri)
        timings["read_resource"] = round((time.perf_counter() - t) * 1000)
        text = res.contents[0].text if res.contents else "null"
        squad = json.loads(text)
        print(f"\n== read_resource({uri}) {timings['read_resource']} ms")
        if squad.get("error"):
            print("  ", squad)
        else:
            names = [
                f"{p['name']}{'(C)' if p['is_captain'] else ''}{'*' if not p['is_starting'] else ''}"
                for p in squad.get("squad") or []
            ]
            print(
                f"  squad GW{squad.get('squad_gw')} picks: {', '.join(names)}\n"
                f"  bank £{squad.get('bank')} FT {squad.get('free_transfers')} chips "
                f"{squad.get('chips_available')} issues {len(squad.get('issues') or [])}"
            )

        t = time.perf_counter()
        res = await session.read_resource("fpl://kb/stats")
        timings["read_kb_stats"] = round((time.perf_counter() - t) * 1000)
        stats = json.loads(res.contents[0].text if res.contents else "null") or {}
        print(f"\n== read_resource(fpl://kb/stats) {timings['read_kb_stats']} ms")
        if stats.get("error"):
            print("  ", stats)
        else:
            tags = stats.get("tags") or {}
            print(
                f"  docs {stats.get('docs')} chunks {stats.get('chunks')} "
                f"(embedded {stats.get('chunks_embedded')}) tokens {stats.get('tokens')} | "
                f"sources {len(stats.get('docs_by_source') or {})} | tags "
                + ", ".join(f"{k}:{v['chunks']}" for k, v in list(tags.items())[:6])
            )

        t = time.perf_counter()
        prompt = await session.get_prompt("pre_deadline_review", {"manager_id": str(args.manager)})
        timings["get_prompt"] = round((time.perf_counter() - t) * 1000)
        msg = prompt.messages[0]
        print(
            f"\n== get_prompt(pre_deadline_review) {timings['get_prompt']} ms: role {msg.role}, "
            f"{len(msg.content.text)} chars, first line: {msg.content.text.splitlines()[0][:90]}"
        )

    total = round((time.perf_counter() - t_all) * 1000)
    print(f"\nlatencies (ms): {json.dumps(timings)} | wall {total} ms")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="fpl-intelligence MCP smoke test (stdio client)")
    ap.add_argument("--manager", type=int, default=895045)
    ap.add_argument("--player", default="Haaland")
    ap.add_argument("--risk", default="João Pedro", help="игрок для analyze_player_risk")
    ap.add_argument(
        "--kb-query", default="when is a -4 points hit worth it", help="запрос к search_strategy_kb"
    )
    ap.add_argument(
        "--kb-tags", nargs="*", default=["hits"], help="теги фильтра KB (пусто — весь корпус)"
    )
    ap.add_argument("--quiet-server", action="store_true", help="stderr сервера в /dev/null")
    parsed = ap.parse_args()
    if parsed.quiet_server:
        with open(os.devnull, "w") as devnull:
            sys.exit(asyncio.run(main(parsed, devnull)))
    sys.exit(asyncio.run(main(parsed, sys.stderr)))
