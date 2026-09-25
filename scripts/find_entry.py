"""Поиск ID команды FPL по названию внутри классической лиги (публичный API поиска не имеет).

Листает таблицу лиги по 50 команд и ищет подстроку в названии команды или имени менеджера
(без учёта регистра). Страницы кэшируются клиентом, повторный поиск — мгновенный.

Запуск:
    uv run python scripts/find_entry.py "название команды"            # лига Kazakhstan (130)
    uv run python scripts/find_entry.py "название" --league 1541165   # любая другая лига
"""

from __future__ import annotations

import argparse
import time

from fplcopilot.data import FPLClient

KAZAKHSTAN_LEAGUE_ID = 130
REQUEST_PAUSE_S = 0.15  # не долбить FPL API: 9 000 команд ≈ 183 страницы


def main() -> None:
    ap = argparse.ArgumentParser(description="ID команды FPL по названию в лиге")
    ap.add_argument("query", help="часть названия команды или имени менеджера")
    ap.add_argument("--league", type=int, default=KAZAKHSTAN_LEAGUE_ID, help="ID классической лиги")
    args = ap.parse_args()

    client = FPLClient(cache_ttl=6 * 3600)
    needle = args.query.casefold().strip()
    found = 0
    page = 1
    while True:
        st = client.league_standings(args.league, page)
        if page == 1:
            print(f"Лига «{st.league_name}» (ID {st.league_id}), ищу «{args.query}»…")
        for e in st.results:
            if needle in e.entry_name.casefold() or needle in e.player_name.casefold():
                found += 1
                print(
                    f"  ID {e.entry:<10} «{e.entry_name}» — {e.player_name}, "
                    f"место {e.rank}, очки {e.total}"
                )
        if not st.has_next:
            break
        page += 1
        time.sleep(REQUEST_PAUSE_S)
    print(f"Просмотрено страниц: {page}, совпадений: {found}")


if __name__ == "__main__":
    main()
