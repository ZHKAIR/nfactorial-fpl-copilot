"""Smoke-проверка FPL-клиента: тур, дедлайн, состав менеджера, проблемные игроки.

Запуск:
    uv run python scripts/smoke_fpl.py [manager_id]
Без аргумента берётся FPL_MANAGER_ID из .env, а если его нет — лидер общей лиги.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime

from fplcopilot.config import settings
from fplcopilot.data import FPLClient

OVERALL_LEAGUE_ID = 314


def main() -> None:
    client = FPLClient()
    bs = client.bootstrap()

    cur, nxt = bs.current_event, bs.next_event
    now = datetime.now(UTC)
    print(f"Игроков: {len(bs.elements)}, команд: {len(bs.teams)}")
    if cur:
        print(f"Текущий тур: GW{cur.id} ({'завершён' if cur.finished else 'идёт'})")
    if nxt:
        left = nxt.deadline_time - now
        print(
            f"Следующий: GW{nxt.id}, дедлайн {nxt.deadline_time:%d.%m %H:%M} UTC, "
            f"осталось {left.days}д {left.seconds // 3600}ч"
        )

    if len(sys.argv) > 1:
        manager_id = int(sys.argv[1])
    elif settings.fpl_manager_id:
        manager_id = settings.fpl_manager_id
    else:
        manager_id = client.league_standings(OVERALL_LEAGUE_ID).results[0].entry
        print(f"(FPL_MANAGER_ID не задан — беру лидера общей лиги: {manager_id})")

    entry = client.entry(manager_id)
    print(
        f"\nМенеджер: {entry.player_first_name} {entry.player_last_name} — «{entry.name}», "
        f"очки {entry.summary_overall_points}, ранг {entry.summary_overall_rank}"
        + (f", регион {entry.player_region_name}" if entry.player_region_name else "")
    )
    if entry.started_event and cur and entry.started_event > cur.id:
        # Новая команда: публичный API отдаёт picks только за прошедшие туры.
        joined = f"{entry.joined_time:%d.%m %H:%M} UTC" if entry.joined_time else "—"
        print(
            f"Команда создана {joined}, первый тур — GW{entry.started_event}. "
            f"Состав появится в публичном API после дедлайна GW{entry.started_event}; "
            "пока показать нечего."
        )
        return
    squad = client.squad(manager_id)
    print(
        f"Состав за GW{squad.gw}: банк {squad.bank:.1f}, стоимость {squad.team_value:.1f}, "
        f"FT к следующему туру ≈ {squad.free_transfers}, чипы: {squad.chips_used or '—'}"
    )

    print(
        f"\n{'':3} {'Игрок':<16}{'Поз':<5}{'Клуб':<5}{'Цена':>5}{'ep_next':>8}{'Влад%':>7}  Статус"
    )
    for sp in squad.starting_xi + squad.bench:
        p = sp.player
        flag = ""
        if p.status != "a":
            flag = f"[{p.status}] {p.news[:60]}"
        elif p.chance_of_playing_next_round not in (None, 100):
            flag = f"шанс играть {p.chance_of_playing_next_round}%"
        print(
            f"{sp.role:<3} {p.web_name:<16}{p.position.short:<5}{sp.team.short_name:<5}"
            f"{p.price:>5.1f}{(p.ep_next or 0):>8.1f}{(p.selected_by_percent or 0):>7.1f}  {flag}"
        )

    by_team = squad.count_by_team()
    over = {bs.team(t).short_name: n for t, n in by_team.items() if n > 3}
    print(
        f"\nПроверка правил: игроков {len(squad.players)}/15, "
        f"старт {len(squad.starting_xi)}/11, клубов >3: {over or 'нет'}"
    )

    problems = [sp for sp in squad.players if sp.player.status != "a"]
    print(f"Проблемных игроков (status != a): {len(problems)}")


if __name__ == "__main__":
    main()
