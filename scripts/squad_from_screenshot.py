"""Состав со скриншота «Pick Team»/«Transfers»: vision LLM -> резолюция по bootstrap -> правила FPL.

Запуск:
    uv run python scripts/squad_from_screenshot.py samples/pick_team_valid.png
    uv run python scripts/squad_from_screenshot.py shot.png --json            # полный ParsedSquad (raw, usage)
    uv run python scripts/squad_from_screenshot.py shot.png --detail low      # дешевле, менее точно
    uv run python scripts/squad_from_screenshot.py shot.png --squad           # + конвертация в Squad
    uv run python scripts/squad_from_screenshot.py --render-samples           # перегенерировать samples/*.png

Нужен OPENAI_API_KEY в .env (кроме --render-samples). Модель — VISION_MODEL (gpt-4o-mini).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fplcopilot.config import PROJECT_ROOT, settings
from fplcopilot.data import FPLClient
from fplcopilot.vision.extract import squad_from_image
from fplcopilot.vision.schemas import ParsedSquad, ResolvedPlayer
from fplcopilot.vision.to_squad import to_squad

SAMPLES_DIR = PROJECT_ROOT / "samples"


def role(p: ResolvedPlayer) -> str:
    if p.is_captain:
        return "C"
    if p.is_vice_captain:
        return "VC"
    if p.is_bench:
        return f"B{p.bench_order}" if p.bench_order else "B"
    return "XI"


def print_table(parsed: ParsedSquad) -> None:
    print(
        f"{'':4}{'На экране':<16}{'-> FPL':<16}{'Поз':<5}{'Клуб':<5}{'Цена':>6}{'Now':>6}"
        f"{'Conf':>6}  {'Метод':<12}Флаг"
    )
    for p in parsed.players:
        shown = f"£{p.price_as_shown:.1f}" if p.price_as_shown is not None else "—"
        now = f"£{p.price:.1f}" if p.price is not None else "—"
        print(
            f"{role(p):<4}{p.name_as_shown[:15]:<16}{(p.web_name or '?')[:15]:<16}"
            f"{p.position or '?':<5}{p.team_short or '?':<5}{shown:>6}{now:>6}"
            f"{p.match_confidence:>6.2f}  {p.match_method:<12}{'!' if p.is_flagged else ''}"
        )
        if p.match_method == "ambiguous":
            print(f"{'':4}  кандидаты: {', '.join(p.candidates)}")
    bank = f"£{parsed.bank:.1f}m" if parsed.bank is not None else "—"
    ft = parsed.free_transfers if parsed.free_transfers is not None else "—"
    print(
        f"\nЭкран: {parsed.screen_type}; распознано {parsed.resolved_count}/{len(parsed.players)} карточек; "
        f"банк {bank}; FT {ft}; GW{parsed.gw or '?'}; valid={parsed.is_valid}"
    )
    if parsed.issues:
        print("\nНарушения:")
        for i in parsed.issues:
            print(f"  [{'BLOCK' if i.blocking else 'warn '}] {i.code}: {i.message}")
    else:
        print("\nНарушений нет.")
    if parsed.usage:
        u = parsed.usage
        print(
            f"\nМодель {u.model} (detail={u.detail}): {u.prompt_tokens}+{u.completion_tokens} токенов "
            f"≈ ${u.cost_usd:.4f}, {u.latency_ms:.0f} мс"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("image", nargs="?", help="PNG/JPEG/WEBP скриншот")
    ap.add_argument(
        "--json", action="store_true", help="вывести ParsedSquad как JSON (с raw и usage)"
    )
    ap.add_argument(
        "--detail", choices=["low", "high", "auto"], default=None, help="detail изображения"
    )
    ap.add_argument(
        "--model", default=None, help=f"модель (по умолчанию VISION_MODEL={settings.vision_model})"
    )
    ap.add_argument(
        "--squad", action="store_true", help="дополнительно сконвертировать в Squad и показать"
    )
    ap.add_argument("--manager-id", type=int, default=0)
    ap.add_argument("--render-samples", action="store_true", help="перегенерировать samples/*.png")
    ap.add_argument("--samples-dir", type=Path, default=SAMPLES_DIR)
    args = ap.parse_args(argv)

    bs = FPLClient().bootstrap()

    if args.render_samples:
        from fplcopilot.vision.render import render_samples

        for path in render_samples(bs, args.samples_dir):
            print(f"{path} ({path.stat().st_size / 1024:.0f} KB)")
        if not args.image:
            return 0

    if not args.image:
        ap.error("укажите путь к скриншоту или --render-samples")

    parsed = squad_from_image(args.image, bs, model=args.model, detail=args.detail)
    if args.json:
        print(json.dumps(parsed.model_dump(mode="json"), ensure_ascii=False, indent=2))
    else:
        print_table(parsed)

    if args.squad:
        if not parsed.is_valid:
            print("\nSquad не собран: есть blocking-нарушения.")
            return 1
        squad = to_squad(parsed, bs, manager_id=args.manager_id)
        cap = squad.captain
        print(
            f"\nSquad: GW{squad.gw}, банк £{squad.bank:.1f}m, стоимость £{squad.team_value:.1f}m, "
            f"FT {squad.free_transfers}, капитан {cap.player.web_name if cap else '—'}"
        )
        print("XI:   " + ", ".join(p.player.web_name for p in squad.starting_xi))
        print("Bench: " + ", ".join(p.player.web_name for p in squad.bench))
    return 0 if parsed.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
