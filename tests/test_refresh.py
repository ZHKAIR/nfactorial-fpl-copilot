"""Отбор игроков для пакетного обновления сигналов (rag/refresh.select_players) — без БД."""

from datetime import UTC, datetime, timedelta

from test_agent_fakes import BS

from fplcopilot.rag.refresh import call_cost, select_players

AS_OF = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
AGE = timedelta(hours=12)


def test_priority_squad_then_fpl_flags_then_new_articles_with_cap():
    last = {154: AS_OF - timedelta(hours=2), 4: AS_OF - timedelta(days=2)}  # Palmer свежий
    got = select_players(
        BS,
        as_of=AS_OF,
        last_signal=last,
        new_articles=[(12, 3), (154, 1), (999, 5)],  # 999 нет в bootstrap
        squad_ids=[154, 4, 411],
        max_age=AGE,
        max_players=5,
    )
    assert [(t.player_id, t.reason) for t in got] == [
        (4, "squad"),  # Palmer (154) свежий — пропущен
        (411, "squad"),
        (165, "fpl status d"),  # João Pedro — по владению раньше «u»
        (27, "fpl status u"),  # G.Jesus 0.2 % > Martinelli 0.1 %
        (18, "fpl status u"),
    ]


def test_new_articles_fill_after_flags_and_fresh_signal_is_skipped():
    last = {p.id: AS_OF - timedelta(hours=1) for p in BS.elements}
    got = select_players(
        BS,
        as_of=AS_OF,
        last_signal={**last, 12: AS_OF - timedelta(days=3)},
        new_articles=[(12, 2)],
        squad_ids=[],
        max_age=AGE,
        max_players=10,
    )
    assert [(t.player_id, t.reason) for t in got] == [(12, "2 new article(s)")]


def test_call_cost_from_tokens():
    assert round(call_cost({"prompt_tokens": 4540, "completion_tokens": 202}), 6) == 0.000802
