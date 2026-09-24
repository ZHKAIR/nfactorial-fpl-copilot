"""Запуск MCP-сервера fpl-intelligence.

    uv run python -m fplcopilot.mcp_server                                   # stdio
    uv run python -m fplcopilot.mcp_server --transport streamable-http --port 8765
    uv run python -m fplcopilot.mcp_server -v                                # DEBUG-лог в stderr

stdout в stdio-транспорте — канал протокола: весь лог идёт в stderr.
"""

from __future__ import annotations

import argparse
import logging
import sys

from fplcopilot.config import settings
from fplcopilot.mcp_server.server import SERVER_NAME, mcp


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m fplcopilot.mcp_server", description=f"MCP server {SERVER_NAME}"
    )
    ap.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="stdio (Cursor / Claude Desktop, по умолчанию) или streamable-http (Inspector, remote)",
    )
    ap.add_argument("--host", default=settings.mcp_host, help="только для streamable-http")
    ap.add_argument(
        "--port", type=int, default=settings.mcp_port, help="только для streamable-http"
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="DEBUG-лог в stderr")
    args = ap.parse_args(argv)

    # force=True: SDK при создании MCPServer ставит RichHandler с переносами строк; для логов
    # Cursor/Inspector удобнее плоские однострочные записи в stderr.
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    for noisy in ("httpx", "httpcore", "openai", "langsmith", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.transport == "stdio":
        logging.getLogger("fplcopilot.mcp").info("%s: stdio transport", SERVER_NAME)
        mcp.run(transport="stdio")
    else:
        logging.getLogger("fplcopilot.mcp").info(
            "%s: streamable-http on http://%s:%d/mcp", SERVER_NAME, args.host, args.port
        )
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
