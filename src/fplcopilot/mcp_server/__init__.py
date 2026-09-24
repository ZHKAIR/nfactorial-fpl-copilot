"""MCP-сервер `fpl-intelligence`: 14 tools — 11 инструментов агента (agent/tools.py, включая
`search_strategy_kb` — стратегическая KB с цитатами, и `rank_players`) плюс 3 справочных
(разбор очков, Understat xG/xA игрока, xGA команды), 5 ресурсов `fpl://...` (в т.ч.
`fpl://kb/stats`) и промпт `pre_deadline_review` — для Cursor, Claude Desktop, Inspector.

    uv run python -m fplcopilot.mcp_server                                   # stdio (по умолчанию)
    uv run python -m fplcopilot.mcp_server --transport streamable-http --port 8765
    uv run python scripts/mcp_smoke.py                                       # клиент stdio: список + вызовы

Логика не дублируется: каждый tool — тонкая обёртка над `LiveTools` (один экземпляр на процесс,
ленивый, закрывается в lifespan) или, для справочных, над `core/ext_stats` / `core/points_form`. Имена игроков разрешает детерминированный `PlayerResolver`;
неоднозначность — структурированная ошибка `{"error": "ambiguous", "candidates": [...]}`, а не
догадка. Ошибки инструментов — `{"error": ..., "hint": ...}` вместо stack trace; тайминги — в
stderr (stdout в stdio-транспорте занят протоколом). См. docs/mcp.md.
"""

from fplcopilot.mcp_server.server import SERVER_NAME, mcp

__all__ = ["SERVER_NAME", "mcp"]
