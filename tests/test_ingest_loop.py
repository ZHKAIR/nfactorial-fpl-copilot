"""--loop ингеста переживает сетевые ошибки: исключение одного цикла не останавливает цикл."""

import httpx

from fplcopilot.rag import ingest


def test_loop_survives_connect_error_in_bootstrap(monkeypatch):
    calls: list[str] = []
    sleeps: list[float] = []

    def flaky_run_once(**_kw):
        calls.append("run")
        if len(calls) == 1:
            raise httpx.ConnectError("[Errno 8] nodename nor servname provided, or not known")
        return {}

    monkeypatch.setattr(ingest, "run_once", flaky_run_once)
    monkeypatch.setattr(ingest, "index_new_articles", lambda: calls.append("index"))
    monkeypatch.setattr(ingest.time, "sleep", sleeps.append)

    assert ingest.run_loop(30, index=True, cycles=3) == 0
    assert calls == ["run", "index", "run", "index", "run", "index"]
    assert len(sleeps) == 2  # между циклами, после последнего не спим
    assert all(0 < s <= 30 * 60 for s in sleeps)


def test_run_cycle_reports_failure_and_still_indexes(monkeypatch):
    indexed: list[bool] = []

    def boom(**_kw):
        raise RuntimeError("db down")

    monkeypatch.setattr(ingest, "run_once", boom)
    monkeypatch.setattr(ingest, "index_new_articles", lambda: indexed.append(True))
    monkeypatch.setattr(ingest, "refresh_signals", lambda: indexed.append(False))

    assert ingest.run_cycle(index=True) is False
    assert indexed == [True]


def test_loop_stops_on_keyboard_interrupt_during_sleep(monkeypatch):
    runs: list[int] = []
    monkeypatch.setattr(ingest, "run_once", lambda **_kw: runs.append(1))

    def interrupt(_s):
        raise KeyboardInterrupt

    monkeypatch.setattr(ingest.time, "sleep", interrupt)
    assert ingest.run_loop(30) == 0
    assert runs == [1]


def test_main_loop_passes_flags(monkeypatch):
    seen: dict = {}

    def fake_loop(every, **kw):
        seen.update(kw, every=every)
        return 0

    monkeypatch.setattr(ingest, "run_loop", fake_loop)
    assert ingest.main(["--loop", "--every", "15", "--index"]) == 0
    assert seen == {
        "every": 15.0,
        "backfill": False,
        "limit": None,
        "index": True,
        "refresh": False,
    }
