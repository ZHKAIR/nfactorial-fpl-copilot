"""CLI извлечения сигнала о доступности игрока.

    uv run python -m fplcopilot.rag.signal --player "João Pedro" [--as-of 2026-09-17T12:00Z]
    uv run python -m fplcopilot.rag.signal --players "João Pedro,Haaland,Saka" --mode dense
    uv run python -m fplcopilot.rag.signal --player Haaland --no-save --json-only

Печатает JSON сигнала и человекочитаемый блок с цитатами и ссылками; по умолчанию
сохраняет строку в player_signals.
"""

from __future__ import annotations

import argparse
import logging
import sys

from fplcopilot.data import Bootstrap, FPLClient
from fplcopilot.rag.extract import PlayerSignal, extract_signal
from fplcopilot.rag.query import parse_as_of
from fplcopilot.rag.retrieve import MODES, Retriever, find_player


def format_signal(sig: PlayerSignal, bs: Bootstrap) -> str:
    p = bs.player(sig.player_id)
    team = bs.team(p.team)
    ret = sig.return_gw if sig.return_gw is not None else "-"
    chance = sig.fpl_chance_next if sig.fpl_chance_next is not None else "-"
    verdict = (
        f"  availability={sig.availability}  start_probability={sig.start_probability:.2f}  "
        f"expected_minutes={sig.expected_minutes}  rotation_risk={sig.rotation_risk}  "
        f"return_gw={ret}  confidence={sig.confidence:.2f}"
    )
    header = f"{sig.player_name} ({team.short_name}, {p.position.short})"
    lines = [
        f"{header} — as of {sig.as_of:%Y-%m-%d %H:%M}Z",
        verdict,
        f"  FPL prior: status={sig.fpl_status} chance_next={chance}",
        f"  summary: {sig.summary}",
        f"  evidence ({len(sig.evidence)}):",
    ]
    for i, ev in enumerate(sig.evidence, start=1):
        quote = " ".join(ev.quote.split())
        lines.append(f'    {i}. [{ev.source} {ev.published_at:%Y-%m-%d}] "{quote}"')
        lines.append(f"       {ev.url}")
    if sig.form_notes:
        lines.append(f"  form & context — not availability evidence ({len(sig.form_notes)}):")
        for i, n in enumerate(sig.form_notes, start=1):
            quote = " ".join(n.quote.split())
            lines.append(f"    {i}. {n.kind}: {n.text}")
            lines.append(f'       [{n.source} {n.published_at:%Y-%m-%d}] "{quote}"')
            lines.append(f"       {n.url}")
    lines.append(
        f"  model={sig.model} prompt={sig.prompt_version} mode={sig.mode} "
        f"fixes={sig.validation_fixes} abstained={'yes' if sig.abstained else 'no'} "
        f"chunks={sig.retrieved_chunk_ids}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FPL Copilot: сигнал о доступности игрока")
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--player", help="имя игрока")
    who.add_argument("--players", help="несколько имён через запятую")
    ap.add_argument("--as-of", dest="as_of", help="ISO-время; по умолчанию сейчас")
    ap.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    ap.add_argument("--prompt", help="версия промпта сигнала (по умолчанию RAG_PROMPT_VERSION)")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--no-save", action="store_true", help="не писать в player_signals")
    ap.add_argument("--json-only", action="store_true", help="только JSON")
    ap.add_argument("--timings", action="store_true", help="латентность по стадиям")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("httpx", "httpx2", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)

    bs = FPLClient().bootstrap()
    names = [n.strip() for n in (args.players or args.player).split(",") if n.strip()]
    as_of = parse_as_of(args.as_of)
    retriever = Retriever()
    rc = 0
    for name in names:
        try:
            player = find_player(bs, name)
        except LookupError as exc:
            print(f"ошибка: {exc}", file=sys.stderr)
            rc = 2
            continue
        timings: dict[str, float] = {}
        sig = extract_signal(
            player.id,
            as_of,
            mode=args.mode,
            k=args.k,
            retriever=retriever,
            bs=bs,
            prompt_version=args.prompt,
            save=not args.no_save,
            timings=timings,
        )
        print(sig.model_dump_json(indent=2))
        if not args.json_only:
            print("\n" + format_signal(sig, bs))
        if args.timings:
            print(
                "  timings: "
                + "  ".join(
                    f"{k}={v:.0f}" for k, v in timings.items() if isinstance(v, int | float)
                )
            )
            for reason in timings.get("summary_fixes") or []:
                print(f"  summary fix: {reason}")
        print()
    return rc


if __name__ == "__main__":
    sys.exit(main())
