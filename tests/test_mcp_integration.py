"""Интеграция MCP-сервера: спавним `python -m fplcopilot.mcp_server` через stdio-клиент SDK и
вызываем инструменты на живых данных (FPL API с дисковым кэшем + Postgres). Скип без БД."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from fplcopilot.db import ping
from tests.test_mcp_server import MCP_TOOLS

pytestmark = pytest.mark.db

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def _need_db() -> None:
    if not ping():
        pytest.skip("Postgres недоступен (docker compose up -d db)")
    try:
        from fplcopilot.data import FPLClient

        FPLClient().bootstrap()
    except (httpx.HTTPError, OSError, ValueError) as exc:  # сеть/кэш FPL API
        pytest.skip(f"FPL bootstrap недоступен: {exc}")


def _structured(result) -> dict:
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


async def _session_run() -> dict:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "fplcopilot.mcp_server"],
        cwd=str(ROOT),
    )
    out: dict = {}
    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as session,
    ):
        init = await session.initialize()
        out["server"] = init.server_info.name
        out["tools"] = [t.name for t in (await session.list_tools()).tools]
        out["resources"] = [str(r.uri) for r in (await session.list_resources()).resources]
        out["templates"] = [
            t.uri_template for t in (await session.list_resource_templates()).resource_templates
        ]
        out["prompts"] = [p.name for p in (await session.list_prompts()).prompts]
        out["ctx"] = _structured(
            await session.call_tool("get_gameweek_context", {}, read_timeout_seconds=120)
        )
        out["pred"] = _structured(
            await session.call_tool(
                "predict_player", {"player": "Haaland", "horizon": 2}, read_timeout_seconds=120
            )
        )
        out["breakdown"] = _structured(
            await session.call_tool(
                "get_player_points_breakdown",
                {"player_id": out["pred"].get("player", {}).get("id", 1)},
                read_timeout_seconds=60,
            )
        )
        res = await session.read_resource("fpl://gameweek/current")
        out["resource"] = json.loads(res.contents[0].text)
    return out


def test_stdio_server_answers_context_and_prediction():
    out = asyncio.run(_session_run())
    assert out["server"] == "fpl-intelligence"
    assert out["tools"] == list(MCP_TOOLS) and len(out["tools"]) == 14
    assert out["resources"] == ["fpl://gameweek/current", "fpl://kb/stats"]
    assert len(out["templates"]) == 3 and out["prompts"] == ["pre_deadline_review"]

    ctx = out["ctx"]
    assert "error" not in ctx, ctx
    assert isinstance(ctx["gw"], int) and ctx["gw"] >= 1
    assert ctx["deadline"].endswith("Z") or "+" in ctx["deadline"]
    assert "no manager id" in ctx["squad_note"]

    pred = out["pred"]
    assert "error" not in pred, pred
    assert pred["player"]["name"] == "Haaland" and pred["player"]["team"] == "MCI"
    assert 1 <= len(pred["by_gw"]) <= 2
    assert pred["by_gw"][0]["gw"] == ctx["gw"]
    assert isinstance(pred["total_xpts"], float) and pred["total_xpts"] > 0
    assert pred["total_xpts"] == round(pred["total_xpts"], 2)

    brk = out["breakdown"]
    assert "error" not in brk, brk
    assert brk["player_id"] == pred["player"]["id"] and brk["sums_to_total"] is True

    assert out["resource"]["gw"] == ctx["gw"]
