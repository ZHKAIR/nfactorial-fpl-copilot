"""Синтетические скриншоты «Pick Team» (Pillow) для unit-тестов и smoke-прогонов vision-модели.

Не пиксельная копия FPL, но те же элементы: зелёное поле с разметкой, ряды по позициям
(GKP/DEF/MID/FWD), карточка = «футболка» цвета клуба + белая плашка с фамилией + фиолетовая плашка
с ценой, круглые бейджи C/V, значок травмы, полоса скамейки с подписями, шапка «Pick Team —
Gameweek N», «In the bank £0.5m», «Free Transfers 1». Шрифт — системный TrueType с полной латиницей
(Arial/Helvetica/DejaVu, см. FONT_CANDIDATES): встроенный в Pillow Aileron не содержит «ã» и «£».

Два эталонных состава (sample_specs) собираются из живого bootstrap: валидный и «сломанный»
(14 игроков, два капитана, 4 игрока Arsenal) — для проверки валидации.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from fplcopilot.data.schemas import Bootstrap, Player

log = logging.getLogger(__name__)

WIDTH, HEIGHT = 900, 1280
HEADER_H = 110
PITCH_BOTTOM = 940
CARD_W, CARD_H, CARD_GAP = 120, 160, 28
KIT_H, NAME_H, PRICE_H = 86, 36, 30
ROW_Y = {"GKP": 175, "DEF": 365, "MID": 555, "FWD": 745}
BENCH_TOP = 960
BENCH_CARD_Y = 1005

FPL_PURPLE = "#37003c"
PITCH_GREEN = "#2f9e44"
PITCH_STRIPE = "#2b9140"
BENCH_BG = "#e9eef2"
LINE = "#e8f5e9"

# Цвет футболки и номера по short_name клуба (лишние ключи безвредны, неизвестный клуб — серый).
TEAM_COLOURS: dict[str, tuple[str, str]] = {
    "ARS": ("#EF0107", "#FFFFFF"), "AVL": ("#670E36", "#95BFE5"), "BOU": ("#DA291C", "#000000"),
    "BRE": ("#E30613", "#FFFFFF"), "BHA": ("#0057B8", "#FFFFFF"), "BUR": ("#6C1D45", "#99D6EA"),
    "CHE": ("#034694", "#FFFFFF"), "COV": ("#78D0F5", "#1C355E"), "CRY": ("#1B458F", "#C4122E"),
    "EVE": ("#003399", "#FFFFFF"), "FUL": ("#F5F5F5", "#000000"), "HUL": ("#F5971D", "#000000"),
    "IPS": ("#3A64A3", "#FFFFFF"), "LEE": ("#F5F5F5", "#1D428A"), "LEI": ("#003090", "#FDBE11"),
    "LIV": ("#C8102E", "#FFFFFF"), "MCI": ("#6CABDD", "#FFFFFF"), "MUN": ("#DA291C", "#FBE122"),
    "NEW": ("#241F20", "#FFFFFF"), "NFO": ("#DD0000", "#FFFFFF"), "SUN": ("#EB172B", "#FFFFFF"),
    "TOT": ("#F5F5F5", "#132257"), "WHU": ("#7A263A", "#1BB1E7"), "WOL": ("#FDB913", "#231F20"),
}  # fmt: skip


@dataclass
class CardSpec:
    name: str  # ровно как на карточке FPL: web_name
    position: str  # GKP/DEF/MID/FWD
    price: float | None = None
    team_short: str = ""
    captain: bool = False
    vice: bool = False
    flagged: bool = False


@dataclass
class PitchSpec:
    starters: list[CardSpec]
    bench: list[CardSpec]  # порядок = порядок на скамейке, первый — вратарь
    gw: int | None = None
    bank: float | None = None
    free_transfers: int | None = None
    title: str = "Pick Team"
    expected_names: list[str] = field(default_factory=list)  # для smoke-оценки: все имена


# Встроенный в Pillow Aileron НЕ содержит «ã», «é», «£», «—» (рисует квадраты), поэтому сначала
# ищем системный TrueType с полной латиницей; переопределить можно переменной FPL_RENDER_FONT.
FONT_CANDIDATES: tuple[str, ...] = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Debian/Ubuntu
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",  # Fedora
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
)
GLYPH_PROBE = "ãé£—"


def _glyph_ok(font: ImageFont.FreeTypeFont, ch: str) -> bool:
    def raster(c: str) -> bytes:
        im = Image.new("L", (64, 48), 0)
        ImageDraw.Draw(im).text((4, 4), c, fill=255, font=font)
        return im.tobytes()

    return raster(ch) != raster("\uffff")  # U+FFFF всегда .notdef


@lru_cache(maxsize=1)
def find_font_path() -> str | None:
    """Первый доступный шрифт, у которого есть все GLYPH_PROBE; None — остаётся дефолт Pillow."""
    override = os.environ.get("FPL_RENDER_FONT")
    for path in ((override,) if override else ()) + FONT_CANDIDATES:
        if path and Path(path).exists():
            try:
                font = ImageFont.truetype(path, 20)
            except OSError:
                continue
            if all(_glyph_ok(font, ch) for ch in GLYPH_PROBE):
                return path
    log.warning("no system font with full Latin coverage — accents/£ will render as boxes")
    return None


@lru_cache(maxsize=64)
def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    path = find_font_path()
    if path:
        return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1: битмап-шрифт без размера
        return ImageFont.load_default()


def _fit_font(text: str, max_width: int, size: int, min_size: int = 10):
    font = _font(size)
    while size > min_size and font.getlength(text) > max_width:
        size -= 1
        font = _font(size)
    return font


def _draw_pitch(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((0, HEADER_H, WIDTH, PITCH_BOTTOM), fill=PITCH_GREEN)
    for y in range(HEADER_H, PITCH_BOTTOM, 120):  # полосы газона
        draw.rectangle((0, y, WIDTH, min(y + 60, PITCH_BOTTOM)), fill=PITCH_STRIPE)
    m = 40
    draw.rectangle((m, HEADER_H + m, WIDTH - m, PITCH_BOTTOM - m), outline=LINE, width=3)
    mid = (HEADER_H + PITCH_BOTTOM) // 2
    draw.line((m, mid, WIDTH - m, mid), fill=LINE, width=3)
    draw.ellipse((WIDTH // 2 - 70, mid - 70, WIDTH // 2 + 70, mid + 70), outline=LINE, width=3)
    box_w, box_h = 320, 90
    draw.rectangle(
        (WIDTH // 2 - box_w // 2, HEADER_H + m, WIDTH // 2 + box_w // 2, HEADER_H + m + box_h),
        outline=LINE,
        width=3,
    )
    draw.rectangle(
        (
            WIDTH // 2 - box_w // 2,
            PITCH_BOTTOM - m - box_h,
            WIDTH // 2 + box_w // 2,
            PITCH_BOTTOM - m,
        ),
        outline=LINE,
        width=3,
    )


def _draw_card(draw: ImageDraw.ImageDraw, x: int, y: int, card: CardSpec) -> None:
    kit, number = TEAM_COLOURS.get(card.team_short.upper(), ("#9e9e9e", "#ffffff"))
    # футболка
    draw.rounded_rectangle((x + 18, y + 6, x + CARD_W - 18, y + KIT_H - 4), radius=12, fill=kit)
    draw.rectangle((x + 4, y + 12, x + 26, y + 44), fill=kit)  # рукава
    draw.rectangle((x + CARD_W - 26, y + 12, x + CARD_W - 4, y + 44), fill=kit)
    draw.rounded_rectangle(
        (x + 4, y + 12, x + CARD_W - 4, y + KIT_H - 4), radius=10, outline="#00000033"
    )
    draw.text(
        (x + CARD_W // 2, y + KIT_H // 2 + 4),
        card.team_short.upper(),
        fill=number,
        font=_font(15),
        anchor="mm",
    )
    # имя
    draw.rectangle((x, y + KIT_H, x + CARD_W, y + KIT_H + NAME_H), fill="#ffffff")
    font = _fit_font(card.name, CARD_W - 10, 17)
    draw.text(
        (x + CARD_W // 2, y + KIT_H + NAME_H // 2),
        card.name,
        fill="#1a1a1a",
        font=font,
        anchor="mm",
    )
    # цена
    py = y + KIT_H + NAME_H
    draw.rectangle((x, py, x + CARD_W, py + PRICE_H), fill=FPL_PURPLE)
    price = f"£{card.price:.1f}m" if card.price is not None else "—"
    draw.text(
        (x + CARD_W // 2, py + PRICE_H // 2), price, fill="#ffffff", font=_font(15), anchor="mm"
    )
    # бейджи капитана / вице
    badge = "C" if card.captain else "V" if card.vice else None
    if badge:
        cx, cy, r = x + CARD_W - 12, y + 12, 15
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill="#000000", outline="#ffffff", width=2)
        draw.text((cx, cy + 1), badge, fill="#ffffff", font=_font(17), anchor="mm")
    if card.flagged:
        cx, cy, r = x + 12, y + 12, 13
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill="#f5c518", outline="#ffffff", width=2)
        draw.text((cx, cy + 1), "!", fill="#000000", font=_font(17), anchor="mm")


def _row_x(n: int) -> list[int]:
    total = n * CARD_W + (n - 1) * CARD_GAP
    x0 = (WIDTH - total) // 2
    return [x0 + i * (CARD_W + CARD_GAP) for i in range(n)]


def render_pick_team(spec: PitchSpec) -> Image.Image:
    im = Image.new("RGB", (WIDTH, HEIGHT), "#ffffff")
    draw = ImageDraw.Draw(im, "RGBA")

    # шапка
    draw.rectangle((0, 0, WIDTH, HEADER_H), fill=FPL_PURPLE)
    title = spec.title + (f" — Gameweek {spec.gw}" if spec.gw else "")
    draw.text((30, HEADER_H // 2), title, fill="#ffffff", font=_font(28), anchor="lm")
    right = []
    if spec.bank is not None:
        right.append(f"In the bank £{spec.bank:.1f}m")
    if spec.free_transfers is not None:
        right.append(f"Free Transfers {spec.free_transfers}")
    for i, line in enumerate(right):
        draw.text((WIDTH - 30, 36 + i * 34), line, fill="#e0ffe0", font=_font(20), anchor="rm")

    _draw_pitch(draw)
    for pos, y in ROW_Y.items():
        row = [c for c in spec.starters if c.position == pos]
        for x, card in zip(_row_x(len(row)), row, strict=True):
            _draw_card(draw, x, y, card)

    # скамейка
    draw.rectangle((0, BENCH_TOP, WIDTH, HEIGHT), fill=BENCH_BG)
    draw.text((30, BENCH_TOP + 22), "Substitutes", fill=FPL_PURPLE, font=_font(22), anchor="lm")
    xs = _row_x(len(spec.bench)) if spec.bench else []
    for i, (x, card) in enumerate(zip(xs, spec.bench, strict=True)):
        _draw_card(draw, x, BENCH_CARD_Y, card)
        label = card.position if i == 0 else f"{i}. {card.position}"
        draw.text(
            (x + CARD_W // 2, BENCH_CARD_Y + CARD_H + 22),
            label,
            fill="#333333",
            font=_font(17),
            anchor="mm",
        )
    return im


def save_png(im: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format="PNG", optimize=True)
    return path


# ---------- эталонные составы из bootstrap ----------


def lookup(bs: Bootstrap, web_name: str, team_short: str | None = None) -> Player:
    hits = [p for p in bs.elements if p.web_name == web_name]
    if team_short:
        hits = [p for p in hits if bs.team(p.team).short_name == team_short]
    if len(hits) != 1:
        raise LookupError(
            f"{web_name!r} ({team_short or 'any club'}): {len(hits)} matches in bootstrap"
        )
    return hits[0]


def card(bs: Bootstrap, web_name: str, team_short: str | None = None, **flags: bool) -> CardSpec:
    p = lookup(bs, web_name, team_short)
    return CardSpec(
        name=p.web_name,
        position=p.position.short,
        price=p.price,
        team_short=bs.team(p.team).short_name,
        **flags,
    )


def sample_specs(bs: Bootstrap) -> dict[str, PitchSpec]:
    """Два эталона: валидный (4-4-2, «João Pedro», «B.Fernandes») и сломанный (14 игроков,
    два капитана, 4 игрока Arsenal, 4 MID)."""
    gw = bs.next_event.id if bs.next_event else None
    valid = PitchSpec(
        starters=[
            card(bs, "Raya", "ARS"),
            card(bs, "Gabriel", "ARS"),
            card(bs, "Virgil", "LIV"),
            card(bs, "Gvardiol", "MCI", flagged=True),
            card(bs, "Senesi", "TOT"),
            card(bs, "B.Fernandes", "MUN", captain=True),
            card(bs, "Semenyo", "MCI"),
            card(bs, "Mbeumo", "MUN"),
            card(bs, "Eze", "ARS"),
            card(bs, "João Pedro", "CHE", vice=True),
            card(bs, "Watkins", "AVL"),
        ],
        bench=[
            card(bs, "Sels", "NFO"),
            card(bs, "Gudmundsson", "LEE"),
            card(bs, "Rogers", "CHE"),
            card(bs, "Wood", "NFO"),
        ],
        gw=gw,
        bank=0.5,
        free_transfers=1,
    )
    broken = PitchSpec(
        starters=[
            card(bs, "Raya", "ARS"),
            card(bs, "Gabriel", "ARS"),
            card(bs, "Virgil", "LIV"),
            card(bs, "Gvardiol", "MCI"),
            card(bs, "Senesi", "TOT"),
            card(bs, "B.Fernandes", "MUN", captain=True),
            card(bs, "Saka", "ARS"),
            card(bs, "Semenyo", "MCI"),
            card(bs, "Haaland", "MCI", captain=True),
            card(bs, "João Pedro", "CHE", vice=True),
        ],
        bench=[
            card(bs, "Sels", "NFO"),
            card(bs, "J.Timber", "ARS"),
            card(bs, "Mbeumo", "MUN"),
            card(bs, "Wood", "NFO"),
        ],
        gw=gw,
        bank=1.2,
        free_transfers=2,
    )
    for spec in (valid, broken):
        spec.expected_names = [c.name for c in (*spec.starters, *spec.bench)]
    return {"pick_team_valid": valid, "pick_team_broken": broken}


def render_samples(bs: Bootstrap, out_dir: Path) -> list[Path]:
    paths = []
    for name, spec in sample_specs(bs).items():
        paths.append(save_png(render_pick_team(spec), out_dir / f"{name}.png"))
    return paths
