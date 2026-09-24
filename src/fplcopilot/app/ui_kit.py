"""HTML-компоненты в стиле референса Figma (чистые функции, без streamlit на импорте).

Иерархия текста на карточке: мелкая подпись секции капсом (`section_label`) → крупный
заголовок (`card_title`) → обычный текст → крупное число (`big_number`). Цвет несёт смысл:
синий — рекомендация и главное, зелёный — плюс, жёлтый — риск, красный — проблема.
Моноширинный шрифт — только у чисел и коротких меток.

Компоненты: подписи, таблетки, аватары, строка трансфера, «VS», новость, блок риска,
наблюдение, столбики (в том числе парные «без трансферов / по плану»), рейтинг, футбольное поле
с футболками клубов (вратарь внизу, нападающие вверху), карточки туров плана. Всё возвращает
строку для `st.markdown(..., unsafe_allow_html=True)`; стили — `KIT_CSS` (внедряется один раз
через `theme.global_css()`), цвета — токены `theme.TOKENS_CSS`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from fplcopilot.agent.tools import LineupOut, PlanOut
from fplcopilot.app import format as fmt

# Сверху вниз: нападение -> полузащита -> защита -> вратарь (скамейка — под полем).
POSITION_ROWS: tuple[str, ...] = ("FWD", "MID", "DEF", "GKP")

# Основной цвет домашней формы (фон футболки, цвет текста на ней) — статичная таблица по
# сокращениям FPL; нет в таблице — нейтральная футболка.
CLUB_COLORS: dict[str, tuple[str, str]] = {
    "ARS": ("#E0202E", "#FFFFFF"),
    "AVL": ("#6E1A3E", "#FFFFFF"),
    "BOU": ("#C8102E", "#FFFFFF"),
    "BRE": ("#D71E28", "#FFFFFF"),
    "BHA": ("#0057B8", "#FFFFFF"),
    "BUR": ("#6C1D45", "#FFFFFF"),
    "CHE": ("#1756A9", "#FFFFFF"),
    "COV": ("#6CB4E4", "#0B2540"),
    "CRY": ("#1B458F", "#FFFFFF"),
    "EVE": ("#003399", "#FFFFFF"),
    "FUL": ("#F4F4F2", "#10191E"),
    "HUL": ("#F5A12D", "#1A1A1A"),
    "IPS": ("#3A64A3", "#FFFFFF"),
    "LEE": ("#F4F4F2", "#1D428A"),
    "LEI": ("#003090", "#FFFFFF"),
    "LIV": ("#C8102E", "#FFFFFF"),
    "MCI": ("#6BB8DF", "#0B2540"),
    "MUN": ("#DA291C", "#FFFFFF"),
    "NEW": ("#241F20", "#FFFFFF"),
    "NFO": ("#DD0000", "#FFFFFF"),
    "SOU": ("#D71920", "#FFFFFF"),
    "SUN": ("#EB172B", "#FFFFFF"),
    "TOT": ("#F4F4F2", "#132257"),
    "WHU": ("#7A263A", "#FFFFFF"),
    "WOL": ("#FDB913", "#231F20"),
}
NEUTRAL_CLUB = ("#8E9CA2", "#FFFFFF")

# Чип FPL -> подпись на карточке тура.
CHIP_LABELS: dict[str, str] = {
    "bboost": "Bench Boost",
    "3xc": "Triple Captain",
    "freehit": "Free Hit",
    "wildcard": "Wildcard",
}


def esc(s: Any) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def club_colors(club: str | None) -> tuple[str, str]:
    """(фон, текст) формы клуба по сокращению FPL; неизвестный клуб — нейтральная."""
    return CLUB_COLORS.get((club or "").upper(), NEUTRAL_CLUB)


def initials(name: str) -> str:
    """«B.Fernandes» -> «BF», «Haaland» -> «HA», «João Pedro» -> «JP», «O'Shea» -> «OS»."""
    cleaned = "".join(ch if ch.isalpha() else " " for ch in name)
    parts = cleaned.split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    return parts[0][:2].upper() if parts else "?"


# ---------- типографика ----------


def eyebrow(text: str) -> str:
    """Подпись над заголовком страницы: «GW6 · БРИФИНГ» (мелко, капсом, синим)."""
    return f'<div class="fpl-eyebrow">{esc(text)}</div>'


def lead(text: str) -> str:
    """Подзаголовок под H1 — одна фраза."""
    return f'<p class="fpl-lead">{esc(text)}</p>'


def section_label(text: str, *, dot: bool = False, tone: str = "") -> str:
    """Подпись секции карточки: «КАПИТАН», «ГЛАВНЫЙ ХОД» (с точкой — рекомендация)."""
    d = '<span class="fpl-pulse"></span>' if dot else ""
    cls = f"fpl-label {tone}".strip()
    return f'<div class="{cls}">{d}{esc(text)}</div>'


def card_title(title: str, sub: str = "") -> str:
    """Крупный заголовок карточки и (необязательно) строка над ним помельче."""
    s = f'<p class="fpl-kicker">{esc(sub)}</p>' if sub else ""
    return f'<div class="fpl-card-title">{s}<h2>{esc(title)}</h2></div>'


def text(body: str, *, muted: bool = False) -> str:
    cls = "fpl-text muted" if muted else "fpl-text"
    return f'<p class="{cls}">{esc(body)}</p>'


def big_number(value: str, note: str = "", *, tone: str = "accent") -> str:
    """Крупное число (моноширинное) и подпись под ним; tone — accent / good / warn / bad / plain."""
    n = f"<small>{esc(note)}</small>" if note else ""
    return f'<div class="fpl-big {esc(tone)}"><b>{esc(value)}</b>{n}</div>'


def pill(label: str, tone: str = "") -> str:
    cls = f"fpl-pill {tone}".strip()
    return f'<span class="{cls}">{esc(label)}</span>'


def pills(items: Sequence[tuple[str, str] | str]) -> str:
    parts = [pill(i) if isinstance(i, str) else pill(i[0], i[1]) for i in items]
    return '<div class="fpl-pills">' + "".join(parts) + "</div>" if parts else ""


def avatar(name: str, club: str | None, *, size: str = "") -> str:
    bg, fg = club_colors(club)
    cls = f"fpl-avatar {size}".strip()
    return f'<span class="{cls}" style="background:{bg};color:{fg}">{esc(initials(name))}</span>'


def card(body: str, *, cls: str = "") -> str:
    return f'<div class="{f"fpl-card {cls}".strip()}">{body}</div>'


def deadline_card(gw: int, countdown: str, when: str) -> str:
    """Карточка дедлайна справа от заголовка: «Дедлайн GW6 · 15 д 16 ч · 10 окт, 15:00»."""
    return (
        '<div class="fpl-deadline">'
        f'<span class="k">Дедлайн GW{gw}</span>'
        f"<strong>{esc(countdown)}</strong>"
        f"<small>{esc(when)}</small></div>"
    )


def news_item(title: str, body: str = "", meta: str = "", *, tone: str = "") -> str:
    """Строка новости: значок, заголовок, текст, «источник · дата»; tone — warn / bad / good."""
    icon = (
        '<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" '
        'stroke-width="1.8" stroke-linecap="round"><path d="M5 4h14v16H5z"/>'
        '<path d="M8 8h8M8 12h8M8 16h5"/></svg>'
    )
    inner = f"<strong>{esc(title)}</strong>"
    if body:
        inner += f"<p>{esc(body)}</p>"
    if meta:
        inner += f"<small>{esc(meta)}</small>"
    return (
        f'<div class="fpl-news"><span class="ico {esc(tone)}">{icon}</span><div>{inner}</div></div>'
    )


def risk_note(title: str, body: str) -> str:
    """Жёлтый блок «Что может пойти не так»."""
    return f'<div class="fpl-risk"><strong>{esc(title)}</strong><p>{esc(body)}</p></div>'


SPARK = (
    '<svg class="spark" viewBox="0 0 24 24" width="18" height="18" fill="none" '
    'stroke="currentColor" stroke-width="1.8" stroke-linejoin="round">'
    '<path d="m12 2 1.8 6.2L20 10l-6.2 1.8L12 18l-1.8-6.2L4 10l6.2-1.8L12 2Z"/></svg>'
)


def observation(label: str, title: str, body: str = "") -> str:
    """Полоса-наблюдение с синей левой границей: подпись, главная фраза, пояснение."""
    inner = f"<small>{esc(label)}</small><strong>{esc(title)}</strong>"
    if body:
        inner += f"<p>{esc(body)}</p>"
    return f'<div class="fpl-obs">{SPARK}<div>{inner}</div></div>'


def summary_strip(cells: Sequence[tuple[str, str, str]]) -> str:
    """Полоса итогов из ячеек [(подпись, значение, пояснение)]."""
    items = "".join(
        f"<div><small>{esc(k)}</small><strong>{esc(v)}</strong><span>{esc(note)}</span></div>"
        for k, v, note in cells
    )
    return f'<div class="fpl-strip">{items}</div>'


def transfer_row(
    outs: Sequence[tuple[str, str, float | None]],
    ins: Sequence[tuple[str, str, float | None]],
    gain: str,
    gain_note: str,
) -> str:
    """Строка трансфера: продаём (аватары) -> покупаем, справа выигрыш зелёным."""

    def side(items: Sequence[tuple[str, str, float | None]], caption: str) -> str:
        out = [f'<span class="cap">{esc(caption)}</span>']
        for name, club, price in items:
            price_t = f" · £{price:.1f}m" if price is not None else ""
            out.append(
                f'<div class="who">{avatar(name, club)}<div><strong>{esc(name)}</strong>'
                f"<small>{esc(club)}{price_t}</small></div></div>"
            )
        return '<div class="side">' + "".join(out) + "</div>"

    arrow = (
        '<span class="arrow"><svg viewBox="0 0 24 24" width="20" height="20" fill="none" '
        'stroke="currentColor" stroke-width="1.8" stroke-linecap="round">'
        '<path d="M5 12h14M13 6l6 6-6 6"/></svg></span>'
    )
    return (
        f'<div class="fpl-transfer">{side(outs, "Продаём")}{arrow}{side(ins, "Покупаем")}'
        f'<div class="gain"><strong>{esc(gain)}</strong><small>{esc(gain_note)}</small></div></div>'
    )


def versus(a: tuple[str, str, str, str], b: tuple[str, str, str, str] | None) -> str:
    """Пара «A VS B»: (имя, клуб, матч, значение); первый — рекомендованный."""

    def one(p: tuple[str, str, str, str], cls: str) -> str:
        name, club, fx, value = p
        return (
            f'<div class="{cls}">{avatar(name, club, size="large")}<strong>{esc(name)}</strong>'
            f"<small>{esc(fx)}</small><b>{esc(value)}</b></div>"
        )

    mid = "<span>или</span>" if b is not None else ""
    return f'<div class="fpl-vs">{one(a, "pick")}{mid}{one(b, "alt") if b else ""}</div>'


def bars(values: Sequence[tuple[str, float]]) -> str:
    """Столбики по турам (высота — доля от максимума), значение и тур под столбиком."""
    if not values:
        return ""
    top = max(v for _, v in values) or 1.0
    cols = []
    for label, v in values:
        h = max(8, round(100 * v / top))
        cols.append(
            f'<div class="col"><div class="stack"><span class="bar" style="height:{h}%"></span>'
            f"</div><b>{v:.1f}</b><small>{esc(label)}</small></div>"
        )
    return '<div class="fpl-bars">' + "".join(cols) + "</div>"


def paired_bars(
    rows: Sequence[tuple[str, float | None, float | None]], legend: tuple[str, str]
) -> str:
    """Два столбика на тур: серый — первая серия («без трансферов»), синий — вторая («по
    плану»); под парой — тур и разница второй серии с первой (зелёным, если больше)."""
    vals = [v for _, a, b in rows for v in (a, b) if v is not None]
    if not vals:
        return ""
    top = max(vals) or 1.0
    cols = []
    for label, a, b in rows:

        def one(v: float | None, cls: str) -> str:
            if v is None:
                return f'<span class="bar {cls} none"></span>'
            return f'<span class="bar {cls}" style="height:{max(6, round(100 * v / top))}%" title="{v:.1f}"></span>'

        diff = ""
        if a is not None and b is not None:
            d = b - a
            tone = "good" if d > 0.05 else "bad" if d < -0.05 else ""
            diff = f'<b class="{tone}">{"+" if d > 0 else ""}{d:.1f}</b>'
        cols.append(
            f'<div class="col"><div class="stack pair">{one(a, "base")}{one(b, "plan")}</div>'
            f"{diff}<small>{esc(label)}</small></div>"
        )
    leg = (
        '<div class="fpl-legend2"><span><i class="base"></i>'
        f'{esc(legend[0])}</span><span><i class="plan"></i>{esc(legend[1])}</span></div>'
    )
    return '<div class="fpl-bars">' + "".join(cols) + "</div>" + leg


def rank_list(items: Sequence[Mapping[str, Any]]) -> str:
    """Рейтинг 1–3: [{name, club, sub, value, tag}] — первый выделен синим."""
    rows = []
    for i, it in enumerate(items, start=1):
        cls = "first" if i == 1 else ""
        tag = pill(str(it["tag"])) if it.get("tag") else ""
        rows.append(
            f'<div class="fpl-rank {cls}"><span class="n">{i}</span>'
            f"{avatar(str(it['name']), str(it.get('club') or ''))}"
            f"<div><strong>{esc(it['name'])}</strong><small>{esc(it.get('sub', ''))}</small></div>"
            f"{tag}<b>{esc(it['value'])}</b></div>"
        )
    return '<div class="fpl-ranks">' + "".join(rows) + "</div>"


def watch_list_html(rows: Sequence[Mapping[str, Any]], player_url: str) -> str:
    """Компактный список «Следить до дедлайна»: цветная точка, имя-ссылка на карточку игрока,
    короткий статус; источник · дата мелко справа."""
    items = "".join(
        f'<div class="fpl-watch-row"><span class="dot {esc(r["tone"])}"></span>'
        f'<a href="{player_url.format(pid=r["id"])}" target="_self">{esc(r["name"])}</a>'
        f'<span class="st">{esc(r["status"])}</span><small>{esc(r["meta"])}</small></div>'
        for r in rows
    )
    return f'<div class="fpl-watch">{items}</div>'


TICKER_SECONDS_PER_ITEM = 7  # скорость ленты: медленно, чтобы успеть прочитать


def news_ticker_html(items: Sequence[Mapping[str, Any]], player_url: str) -> str:
    """Лента новостей фиксированной высоты: записи медленно плывут вверх бесконечно (вторая
    копия списка — для бесшовного цикла), пауза при наведении, мягкое затухание сверху и снизу;
    при prefers-reduced-motion — статичный прокручиваемый список. Меньше трёх записей — без
    анимации."""

    def one(i: Mapping[str, Any]) -> str:
        link = f' · <a href="{esc(i["url"])}" target="_blank">источник</a>' if i.get("url") else ""
        return (
            f'<div class="item"><div class="h"><span class="dot {esc(i["tone"])}"></span>'
            f'<a href="{player_url.format(pid=i["id"])}" target="_self">{esc(i["player"])}</a>'
            f'<span class="st">{esc(i["status"])}</span></div>'
            f"<p>«{esc(i['quote'])}»</p><small>{esc(i['meta'])}{link}</small></div>"
        )

    body = "".join(one(i) for i in items)
    if len(items) < 3:
        return f'<div class="fpl-ticker static"><div class="track">{body}</div></div>'
    dur = TICKER_SECONDS_PER_ITEM * len(items)
    return (
        f'<div class="fpl-ticker"><div class="track" style="animation-duration:{dur}s">'
        f'{body}<div class="dup" aria-hidden="true">{body}</div></div></div>'
    )


def name_chips(names: Sequence[str]) -> str:
    return '<div class="fpl-pills">' + "".join(pill(n) for n in names) + "</div>"


# ---------- футбольное поле ----------


@dataclass(frozen=True)
class PitchPlayer:
    id: int | None
    name: str
    club: str
    pos: str  # GKP / DEF / MID / FWD
    sub: str = ""  # соперник «BRE (H)»
    value: str = ""  # прогноз «5.6»
    role: str = ""  # C / VC / ""
    flag: tuple[str, str] | None = None  # (warn|bad, подсказка) — проблема игрока
    tag: str = ""  # sell / buy — отметка маршрута трансфера


def _shirt(p: PitchPlayer) -> str:
    bg, fg = club_colors(p.club)
    badge = f'<b class="role">{"C" if p.role == "C" else "V"}</b>' if p.role in ("C", "VC") else ""
    return (
        f'<span class="shirt" style="background:{bg};color:{fg}"><i>{esc(p.club)}</i></span>{badge}'
    )


def _token(p: PitchPlayer, player_url: str | None) -> str:
    flag = ""
    title = ""
    if p.flag:
        flag = f'<span class="flag {esc(p.flag[0])}"></span>'
        title = f' title="{esc(p.flag[1])}"'
    tag = ""
    if p.tag == "sell":
        tag = '<span class="mark sell">продать</span>'
    elif p.tag == "buy":
        tag = '<span class="mark buy">купить</span>'
    value = f"<em>{esc(p.value)}</em>" if p.value else ""
    inner = (
        f'<span class="kit">{_shirt(p)}{flag}</span>'
        f"<strong>{esc(p.name)}</strong>"
        f"<small>{esc(p.sub)}{' · ' if p.sub and p.value else ''}{value}</small>{tag}"
    )
    cls = f"fpl-token {esc(p.tag)}".strip()
    if player_url and p.id is not None:
        href = player_url.format(pid=p.id)
        return f'<a class="{cls}" href="{href}" target="_self"{title}>{inner}</a>'
    return f'<div class="{cls}"{title}>{inner}</div>'


def pitch_html(
    starters: Sequence[PitchPlayer],
    bench: Sequence[PitchPlayer],
    *,
    player_url: str | None = None,
    toolbar_left: str = "",
    toolbar_right: tuple[str, str] | None = None,
) -> str:
    """Поле, как его видит тренер со своей половины: нападающие вверху, вратарь внизу у своих
    ворот; скамейка под полем в порядке замен. Футболка — цвета клуба, C / V — бейдж, точка —
    проблема, «продать / купить» — отметки маршрута; имя — ссылка на страницу «Игрок»."""
    rows = []
    for pos in POSITION_ROWS:
        line = [p for p in starters if p.pos == pos]
        if line:
            rows.append(
                f'<div class="line {pos.lower()}">'
                + "".join(_token(p, player_url) for p in line)
                + "</div>"
            )
    toolbar = ""
    if toolbar_left or toolbar_right:
        right = (
            f'<div class="tr"><b>{esc(toolbar_right[0])}</b><span>{esc(toolbar_right[1])}</span></div>'
            if toolbar_right
            else ""
        )
        toolbar = (
            f'<div class="fpl-pitch-bar"><div class="tl">{esc(toolbar_left)}</div>{right}</div>'
        )
    bench_h = ""
    if bench:
        bench_h = (
            '<div class="fpl-bench"><span class="lbl"><b>Скамейка</b><small>в порядке замен</small>'
            "</span>" + "".join(_token(p, player_url) for p in bench) + "</div>"
        )
    return (
        '<div class="fpl-pitch-wrap">'
        + toolbar
        + '<div class="fpl-pitch"><span class="half"></span><span class="circle"></span>'
        '<span class="box top"></span><span class="goal top"></span>'
        '<span class="box bottom"></span><span class="goal bottom"></span>'
        + "".join(rows)
        + "</div>"
        + bench_h
        + "</div>"
    )


def pitch_from_squad_rows(
    xi: Sequence[Mapping[str, Any]], bench: Sequence[Mapping[str, Any]], *, gw: int
) -> tuple[list[PitchPlayer], list[PitchPlayer]]:
    """Строки таблицы «Мой состав» (`format.squad_table_rows`) -> игроки поля: соперник и
    прогноз на тур gw, роль C / VC, точка проблемы."""

    def one(r: Mapping[str, Any]) -> PitchPlayer:
        x = r.get("xPts")
        role = str(r.get("Роль") or "")
        return PitchPlayer(
            id=r.get("_id"),
            name=str(r["Player"]),
            club=str(r.get("_team") or ""),
            pos=str(r.get("_pos") or ""),
            sub=str(r.get(f"GW{gw}") or "—"),
            value="" if x is None else f"{float(x):.1f}",
            role=role if role in ("C", "VC") else "",
            flag=r.get("_flag"),
        )

    return [one(r) for r in xi], [one(r) for r in bench]


def pitch_from_lineup(
    lu: LineupOut, *, sell: Sequence[str] = (), buy: Sequence[str] = ()
) -> tuple[list[PitchPlayer], list[PitchPlayer]]:
    """Лучший состав (`optimize_team`) -> игроки поля; отметки «продать / купить» по маршруту."""
    sell_set = {s.casefold() for s in sell}
    buy_set = {s.casefold() for s in buy}

    def one(p: Any, role: str = "") -> PitchPlayer:
        name = p.name.casefold()
        tag = "sell" if name in sell_set else "buy" if name in buy_set else ""
        return PitchPlayer(
            id=p.id,
            name=p.name,
            club=p.team,
            pos=p.position,
            sub=fixture_short(p.fixture),
            value=f"{p.xpts:.1f}",
            role=role,
            tag=tag,
        )

    starters = [
        one(p, "C" if p.name == lu.captain else "VC" if p.name == lu.vice else "")
        for p in lu.starters
    ]
    return starters, [one(p) for p in lu.bench]


_FIXTURE_TOKEN = re.compile(r"([v@])([A-Z]{2,4})\b")


def fixture_short(fixture: str | None) -> str:
    """`tools.fixture_label`: «vTOT (FSI 2)» -> «TOT (H)», два матча «vTOT (FSI 2) @LIV (FSI 5)»
    -> «TOT (H) · LIV (A)»; «blank» / пусто -> «—»."""
    found = _FIXTURE_TOKEN.findall(fixture or "")
    if not found:
        return "—"
    return " · ".join(f"{club} ({'H' if side == 'v' else 'A'})" for side, club in found)


# ---------- карточки туров плана ----------


def plan_gw_cards(
    plan: PlanOut,
    *,
    deadlines: Mapping[int, datetime] | None = None,
    reason_ru: Mapping[str, str] | None = None,
    chips: Mapping[str, str] | None = None,
) -> str:
    """Карточка на каждый тур горизонта. Главное — ходы «Out → In» с выигрышем за горизонт и
    причиной (и «−4», если ход платный) или «без трансфера»; второстепенно, мелко — прогноз
    состава, капитан, FT и банк. Чип тура (`chips`: «7» -> «bboost») — бейдж в шапке карточки.
    Всё — из `PlanOut`; первый тур помечен «сейчас»."""
    deadlines = deadlines or {}
    reason_ru = reason_ru or {}
    chips = chips or {}
    horizon_word = fmt.plural(plan.horizon, "тур", "тура", "туров")
    gws = sorted({int(g) for g in plan.xi_points_by_gw} | {int(g) for g in plan.moves_by_gw})
    cards = []
    for i, g in enumerate(gws):
        key = str(g)
        moves = plan.moves_by_gw.get(key) or []
        dl = deadlines.get(g)
        date = f"<small>{dl.strftime('%d.%m')}</small>" if dl else ""
        badges = '<b class="now">сейчас</b>' if i == 0 else ""
        if chips.get(key):
            badges += f'<b class="chip">{esc(CHIP_LABELS.get(chips[key], chips[key]))}</b>'
        if moves:
            items = []
            for m in moves:
                reason = (
                    reason_ru.get(m.out_problem or "", m.out_problem or "") or "сильнее по прогнозу"
                )
                paid = '<span class="hit">−4</span>' if m.paid else ""
                gain = fmt.points_text(m.delta_xpts_horizon, signed=True)
                items.append(
                    f'<div class="move"><strong>{esc(m.out)} → {esc(m.in_)}</strong>{paid}'
                    f'<p><span class="up">{esc(gain)}</span> за {esc(horizon_word)} · {esc(reason)}</p></div>'
                )
            move_block = f'<div class="moves">{"".join(items)}</div>'
        else:
            move_block = (
                '<div class="moves"><div class="move idle"><strong>Без трансфера</strong>'
                "<p>Бесплатный трансфер переносится на следующий тур.</p></div></div>"
            )
        xp = plan.xi_points_by_gw.get(key)
        cap = plan.captain_by_gw.get(key, "—") or "—"
        ft = plan.ft_by_gw.get(key)
        bank = plan.bank_by_gw.get(key)
        hits = int(plan.hits_by_gw.get(key, 0) or 0)
        foot = [f"капитан {esc(cap)}", f"FT {'—' if ft is None else ft}"]
        foot.append(f"банк {'—' if bank is None else f'£{bank:.1f}m'}")
        if hits:
            foot.append(f'<span class="hit">хит −{hits}</span>')
        xp_h = f'<span class="xp">{xp:.1f}<small> xPts</small></span>' if xp is not None else ""
        cls = "fpl-gw current" if i == 0 else "fpl-gw"
        cards.append(
            f'<article class="{cls}"><div class="head"><div><span>GW{g}</span>{date}{badges}</div>'
            f"{xp_h}</div>{move_block}"
            f'<div class="foot">{" · ".join(foot)}</div></article>'
        )
    return '<div class="fpl-gw-grid">' + "".join(cards) + "</div>"


# ---------- CSS ----------

KIT_CSS = """
.fpl-eyebrow {
  font-size: 12px; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--fpl-accent); margin: 0 0 4px;
}
.fpl-lead { color: var(--fpl-text-2); font-size: 16px; line-height: 1.5; margin: -2px 0 1.1rem; max-width: 760px; }
.fpl-label {
  font-size: 11.5px; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--fpl-muted); display: flex; align-items: center; gap: 8px; margin: 0 0 8px;
}
.fpl-label.accent { color: var(--fpl-accent); }
.fpl-label.warn { color: var(--fpl-warn); }
.fpl-pulse {
  width: 7px; height: 7px; border-radius: 50%; background: var(--fpl-brand);
  box-shadow: 0 0 0 4px var(--fpl-accent-bg); flex: 0 0 auto;
}
.fpl-card-title { margin: 0 0 10px; }
.fpl-card-title h2 {
  margin: 0; padding: 0; font-size: 22px; font-weight: 750; line-height: 1.25;
  letter-spacing: -0.025em; color: var(--fpl-text);
}
.fpl-kicker { margin: 0 0 2px; font-size: 13px; color: var(--fpl-muted); }
.fpl-text { font-size: 14.5px; line-height: 1.6; color: var(--fpl-text-2); margin: 8px 0; }
.fpl-text.muted { color: var(--fpl-muted); font-size: 13.5px; }
.fpl-big { display: flex; flex-direction: column; gap: 2px; }
.fpl-big b { font-family: var(--fpl-mono); font-size: 30px; font-weight: 600; line-height: 1.1; letter-spacing: -0.03em; }
.fpl-big small { font-size: 13px; color: var(--fpl-muted); }
.fpl-big.accent b { color: var(--fpl-accent); }
.fpl-big.good b { color: var(--fpl-good); }
.fpl-big.warn b { color: var(--fpl-warn); }
.fpl-big.bad b { color: var(--fpl-bad); }
.fpl-big.plain b { color: var(--fpl-text); }
.fpl-card {
  background: var(--fpl-surface); border: 1px solid var(--fpl-line-strong); border-radius: 12px;
  padding: 20px 22px; box-shadow: var(--fpl-shadow); margin: 0 0 12px;
}
.fpl-pills { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0; }
.fpl-pill {
  display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 12.5px; font-weight: 600;
  background: var(--fpl-surface-2); color: var(--fpl-text-2); border: 1px solid var(--fpl-line);
}
.fpl-pill.accent { background: var(--fpl-accent-bg); color: var(--fpl-accent); border-color: transparent; }
.fpl-pill.good { background: var(--fpl-good-bg); color: var(--fpl-good); border-color: transparent; }
.fpl-pill.warn { background: var(--fpl-warn-bg); color: var(--fpl-warn); border-color: transparent; }
.fpl-pill.bad { background: var(--fpl-bad-bg); color: var(--fpl-bad); border-color: transparent; }
.fpl-avatar {
  width: 40px; height: 40px; border-radius: 50%; display: inline-grid; place-items: center;
  font-family: var(--fpl-mono); font-size: 12.5px; font-weight: 700; flex: 0 0 auto;
  box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.08);
}
.fpl-avatar.large { width: 54px; height: 54px; font-size: 14px; }
.fpl-deadline {
  border-left: 1px solid var(--fpl-line-strong); padding: 4px 0 4px 18px; margin-top: 12px;
  display: grid; gap: 2px;
}
.fpl-deadline .k { font-size: 12px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--fpl-muted); }
.fpl-deadline strong {
  font-family: var(--fpl-mono); font-size: 24px; font-weight: 600; color: var(--fpl-text); letter-spacing: -0.02em;
}
.fpl-deadline small { font-size: 13px; color: var(--fpl-muted); }
.fpl-transfer {
  margin: 14px 0; padding: 16px 18px; display: flex; align-items: center; gap: 18px;
  background: var(--fpl-surface-2); border: 1px solid var(--fpl-line); border-radius: 10px; flex-wrap: wrap;
}
.fpl-transfer .side { display: flex; flex-direction: column; gap: 9px; min-width: 150px; }
.fpl-transfer .cap { font-size: 11.5px; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: var(--fpl-faint); }
.fpl-transfer .who { display: flex; align-items: center; gap: 10px; }
.fpl-transfer .who div { display: flex; flex-direction: column; }
.fpl-transfer .who strong { font-size: 15px; font-weight: 700; }
.fpl-transfer .who small { color: var(--fpl-muted); font-size: 12.5px; margin-top: 1px; }
.fpl-transfer .arrow { color: var(--fpl-faint); display: grid; place-items: center; }
.fpl-transfer .gain { margin-left: auto; text-align: right; display: flex; flex-direction: column; }
.fpl-transfer .gain strong { font-family: var(--fpl-mono); font-size: 28px; font-weight: 600; color: var(--fpl-good); }
.fpl-transfer .gain small { font-size: 13px; color: var(--fpl-muted); }
.fpl-vs { display: flex; align-items: center; justify-content: center; gap: 30px; margin: 10px 0 12px; }
.fpl-vs > div { display: grid; justify-items: center; gap: 3px; text-align: center; }
.fpl-vs > span { font-size: 12px; color: var(--fpl-faint); }
.fpl-vs strong { font-size: 15px; font-weight: 700; margin-top: 6px; }
.fpl-vs small { font-size: 12.5px; color: var(--fpl-muted); }
.fpl-vs b { font-family: var(--fpl-mono); font-size: 18px; font-weight: 600; margin-top: 2px; }
.fpl-vs .pick b { color: var(--fpl-accent); }
.fpl-vs .alt b { color: var(--fpl-muted); }
.fpl-news { display: flex; gap: 12px; margin: 12px 0 0; }
.fpl-news .ico {
  width: 34px; height: 34px; display: grid; place-items: center; border-radius: 8px; flex: 0 0 auto;
  background: var(--fpl-accent-bg); color: var(--fpl-accent);
}
.fpl-news .ico.warn { background: var(--fpl-warn-bg); color: var(--fpl-warn); }
.fpl-news .ico.bad { background: var(--fpl-bad-bg); color: var(--fpl-bad); }
.fpl-news .ico.good { background: var(--fpl-good-bg); color: var(--fpl-good); }
.fpl-news div { display: flex; flex-direction: column; min-width: 0; }
.fpl-news strong { font-size: 15px; font-weight: 700; }
.fpl-news p { color: var(--fpl-text-2); font-size: 14px; margin: 3px 0; line-height: 1.5; }
.fpl-news small { font-size: 12.5px; color: var(--fpl-muted); }
.fpl-risk {
  background: var(--fpl-warn-bg); border-left: 3px solid var(--fpl-warn-line); padding: 12px 16px;
  margin: 12px 0; border-radius: 0 8px 8px 0;
}
.fpl-risk strong { font-size: 14px; color: var(--fpl-warn); }
.fpl-risk p { margin: 4px 0 0; font-size: 14px; line-height: 1.5; color: var(--fpl-text-2); }
.fpl-obs {
  display: flex; gap: 12px; align-items: flex-start; margin: 12px 0; padding: 16px 18px;
  background: var(--fpl-surface); border: 1px solid var(--fpl-line-strong);
  border-left: 4px solid var(--fpl-brand); border-radius: 8px;
}
.fpl-obs .spark { color: var(--fpl-brand); flex: 0 0 auto; margin-top: 2px; }
.fpl-obs div { display: flex; flex-direction: column; gap: 4px; }
.fpl-obs small { font-size: 11.5px; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase; color: var(--fpl-accent); }
.fpl-obs strong { font-size: 16px; font-weight: 700; line-height: 1.4; }
.fpl-obs p { margin: 0; font-size: 14px; color: var(--fpl-text-2); line-height: 1.5; }
.fpl-strip {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  background: var(--fpl-surface); border: 1px solid var(--fpl-line-strong); border-radius: 12px;
  margin: 8px 0 14px; box-shadow: var(--fpl-shadow);
}
.fpl-strip > div { padding: 14px 18px; display: flex; flex-direction: column; border-right: 1px solid var(--fpl-line); }
.fpl-strip > div:last-child { border-right: 0; }
.fpl-strip small { font-size: 12px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--fpl-muted); }
.fpl-strip strong { margin-top: 6px; font-family: var(--fpl-mono); font-size: 20px; font-weight: 600; }
.fpl-strip span { color: var(--fpl-muted); font-size: 13px; margin-top: 2px; }
.fpl-bars { display: flex; align-items: flex-end; gap: 14px; height: 130px; margin: 8px 0 4px; }
.fpl-bars .col { flex: 1; height: 100%; display: flex; flex-direction: column; align-items: center; gap: 3px; }
.fpl-bars .stack { flex: 1; width: 100%; display: flex; align-items: flex-end; justify-content: center; gap: 4px; }
.fpl-bars .bar { width: 100%; max-width: 34px; border-radius: 4px 4px 0 0; background: #42B9EE; display: block; }
.fpl-bars .stack.pair .bar { max-width: 22px; }
.fpl-bars .bar.base { background: light-dark(#C9D4DB, #33434D); }
.fpl-bars .bar.plan { background: var(--fpl-brand); }
.fpl-bars .bar.none { height: 0; }
.fpl-bars b { font-family: var(--fpl-mono); font-size: 12.5px; font-weight: 600; color: var(--fpl-text-2); }
.fpl-bars b.good { color: var(--fpl-good); } .fpl-bars b.bad { color: var(--fpl-bad); }
.fpl-bars small { font-size: 12px; font-weight: 600; color: var(--fpl-muted); }
.fpl-legend2 { display: flex; gap: 16px; font-size: 13px; color: var(--fpl-muted); margin: 6px 0 2px; }
.fpl-legend2 i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
.fpl-legend2 i.base { background: light-dark(#C9D4DB, #33434D); } .fpl-legend2 i.plan { background: var(--fpl-brand); }
.fpl-ranks { display: flex; flex-direction: column; gap: 2px; }
.fpl-rank {
  display: grid; grid-template-columns: 16px 40px 1fr auto auto; align-items: center; gap: 10px;
  padding: 10px; border-bottom: 1px solid var(--fpl-line);
}
.fpl-rank.first { border: 1px solid var(--fpl-accent-line); background: var(--fpl-accent-bg); border-radius: 10px; }
.fpl-rank .n { font-family: var(--fpl-mono); font-size: 12px; color: var(--fpl-faint); }
.fpl-rank div { display: flex; flex-direction: column; min-width: 0; }
.fpl-rank strong { font-size: 15px; font-weight: 700; }
.fpl-rank small { font-size: 12.5px; color: var(--fpl-muted); margin-top: 1px; }
.fpl-rank > b { font-family: var(--fpl-mono); font-size: 17px; font-weight: 600; color: var(--fpl-text); }
.fpl-rank.first > b { color: var(--fpl-accent); }

/* «Следить до дедлайна» — строка на игрока */
.fpl-watch { display: flex; flex-direction: column; }
.fpl-watch-row {
  display: grid; grid-template-columns: 10px auto 1fr; column-gap: 10px; row-gap: 1px;
  align-items: baseline; padding: 9px 0; border-bottom: 1px solid var(--fpl-line);
}
.fpl-watch-row:last-child { border-bottom: 0; }
.fpl-watch-row .dot, .fpl-ticker .dot {
  width: 9px; height: 9px; border-radius: 50%; background: var(--fpl-warn); display: inline-block;
  align-self: center;
}
.fpl-watch-row .dot.bad, .fpl-ticker .dot.bad { background: var(--fpl-bad); }
.fpl-watch-row .dot.good { background: var(--fpl-good); }
.fpl-watch-row a { font-size: 15px; font-weight: 700; color: var(--fpl-text) !important; text-decoration: none; }
.fpl-watch-row a:hover { color: var(--fpl-accent) !important; }
.fpl-watch-row .st { font-size: 14px; color: var(--fpl-text-2); overflow-wrap: anywhere; }
.fpl-watch-row small { grid-column: 2 / 4; font-size: 12px; color: var(--fpl-faint); }

/* лента новостей */
.fpl-ticker {
  height: 300px; overflow: hidden; position: relative;
  -webkit-mask-image: linear-gradient(transparent, #000 12%, #000 88%, transparent);
  mask-image: linear-gradient(transparent, #000 12%, #000 88%, transparent);
}
.fpl-ticker.static { height: auto; -webkit-mask-image: none; mask-image: none; }
.fpl-ticker .track { animation: fpl-ticker-up linear infinite; }
.fpl-ticker.static .track { animation: none; }
.fpl-ticker:hover .track { animation-play-state: paused; }
@keyframes fpl-ticker-up { from { transform: translateY(0); } to { transform: translateY(-50%); } }
.fpl-ticker .item { padding: 12px 2px; border-bottom: 1px solid var(--fpl-line); }
.fpl-ticker .h { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.fpl-ticker .h a { font-size: 15px; font-weight: 700; color: var(--fpl-text) !important; text-decoration: none; }
.fpl-ticker .h .st { font-size: 12px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--fpl-muted); }
.fpl-ticker p { margin: 4px 0 3px; font-size: 14px; line-height: 1.5; color: var(--fpl-text-2); overflow-wrap: anywhere; }
.fpl-ticker small { font-size: 12px; color: var(--fpl-faint); }
.fpl-ticker small a { color: var(--fpl-accent) !important; }
@media (prefers-reduced-motion: reduce) {
  .fpl-ticker { overflow-y: auto; -webkit-mask-image: none; mask-image: none; }
  .fpl-ticker .track { animation: none; }
  .fpl-ticker .dup { display: none; }
}

/* строка вопроса ассистенту (Брифинг) — компактная, акцент синим, закреплена внизу экрана:
   по центру области контента (сдвиг на ширину открытого сайдбара — через :has) */
.st-key-ask_bar {
  position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%); z-index: 90;
  width: min(820px, calc(100vw - 48px));
  background: var(--fpl-surface); border: 1.5px solid var(--fpl-brand); border-radius: 999px;
  padding: 4px 6px 4px 16px; box-shadow: 0 8px 24px rgba(6, 168, 253, 0.18);
}
.stApp:has([data-testid="stSidebar"][aria-expanded="true"]) .st-key-ask_bar {
  left: calc(50% + 150px); width: min(820px, calc(100vw - 360px));
}
.stApp:has(.st-key-ask_bar) [data-testid="stMainBlockContainer"] { padding-bottom: 7rem; }
.st-key-ask_bar [data-testid="stForm"] { border: 0; padding: 0; }
.st-key-ask_bar [data-testid="stTextInputRootElement"], .st-key-ask_bar [data-baseweb="input"] {
  border: 0 !important; background: transparent !important;
}
.st-key-ask_bar input { font-size: 15px; background: transparent !important; }
.st-key-ask_bar [data-testid="stMarkdownContainer"] p { margin: 0; line-height: 1; }
.st-key-ask_bar button { border-radius: 999px; min-height: 36px; }

/* поле */
.fpl-pitch-wrap { border-radius: 12px; overflow: hidden; border: 1px solid var(--fpl-line-strong); margin: 4px 0 12px; }
.fpl-pitch-bar {
  display: flex; justify-content: space-between; align-items: center; padding: 12px 18px;
  background: var(--fpl-surface); font-size: 14px;
}
.fpl-pitch-bar .tl { font-weight: 650; color: var(--fpl-text-2); }
.fpl-pitch-bar .tr { display: flex; flex-direction: column; align-items: flex-end; }
.fpl-pitch-bar .tr b { font-family: var(--fpl-mono); font-size: 22px; font-weight: 600; color: var(--fpl-accent); }
.fpl-pitch-bar .tr span { font-size: 12.5px; color: var(--fpl-muted); }
.fpl-pitch {
  position: relative; min-height: 490px; display: flex; flex-direction: column;
  justify-content: space-around; padding: 22px 6px;
  background: repeating-linear-gradient(180deg, var(--fpl-pitch-a) 0 12.5%, var(--fpl-pitch-b) 12.5% 25%);
}
.fpl-pitch:before { content: ""; position: absolute; inset: 10px; border: 1px solid rgba(255,255,255,0.35); pointer-events: none; }
.fpl-pitch .half { position: absolute; left: 10px; right: 10px; top: 50%; border-top: 1px solid rgba(255,255,255,0.35); }
.fpl-pitch .circle {
  position: absolute; width: 96px; height: 96px; border: 1px solid rgba(255,255,255,0.35);
  border-radius: 50%; left: 50%; top: 50%; transform: translate(-50%, -50%);
}
.fpl-pitch .box {
  position: absolute; width: 40%; height: 64px; border: 1px solid rgba(255,255,255,0.35);
  left: 50%; transform: translateX(-50%);
}
.fpl-pitch .box.top { top: 10px; } .fpl-pitch .box.bottom { bottom: 10px; }
.fpl-pitch .goal {
  position: absolute; width: 16%; height: 22px; border: 1px solid rgba(255,255,255,0.35);
  left: 50%; transform: translateX(-50%);
}
.fpl-pitch .goal.top { top: 10px; } .fpl-pitch .goal.bottom { bottom: 10px; }
.fpl-pitch .line { position: relative; z-index: 2; display: flex; justify-content: space-evenly; align-items: flex-start; gap: 4px; padding: 6px 0; }
.fpl-token {
  display: flex; flex-direction: column; align-items: center; min-width: 82px; max-width: 116px;
  text-decoration: none !important; color: inherit !important; position: relative;
}
.fpl-token .kit { position: relative; display: inline-block; }
.fpl-token .shirt {
  width: 46px; height: 46px; display: grid; place-items: center; position: relative;
  clip-path: polygon(23% 7%,37% 0,63% 0,77% 7%,100% 21%,84% 42%,76% 34%,78% 100%,22% 100%,24% 34%,16% 42%,0 21%);
  box-shadow: inset 0 -14px rgba(0,0,0,0.12);
}
.fpl-token .shirt i { font-style: normal; font-family: var(--fpl-mono); font-size: 10px; font-weight: 700; margin-top: 8px; }
.fpl-token .role {
  position: absolute; top: -4px; right: -9px; width: 20px; height: 20px; border-radius: 50%;
  display: grid; place-items: center; background: #F8E85D; color: #1C3024;
  font-family: var(--fpl-mono); font-size: 11px; font-weight: 700; box-shadow: 0 1px 3px rgba(0,0,0,0.3);
}
.fpl-token .flag { position: absolute; top: -2px; left: -9px; width: 13px; height: 13px; border-radius: 50%; border: 2px solid #FFFFFF; }
.fpl-token .flag.warn { background: #E0A320; } .fpl-token .flag.bad { background: #E0424A; }
.fpl-token > strong {
  margin-top: 5px; font-size: 13px; font-weight: 700; white-space: nowrap; max-width: 114px;
  overflow: hidden; text-overflow: ellipsis; padding: 2px 8px; border-radius: 4px;
  background: #FFFFFF; color: #12202A; box-shadow: 0 2px 5px rgba(19, 36, 28, 0.3);
}
.fpl-token > small { margin-top: 3px; font-size: 12px; color: #EAF5EE; white-space: nowrap; text-shadow: 0 1px 1px rgba(0,0,0,0.35); }
.fpl-token > small em { font-style: normal; font-family: var(--fpl-mono); font-weight: 700; color: #FFFFFF; }
.fpl-token .mark { margin-top: 3px; font-size: 11.5px; font-weight: 700; padding: 1px 8px; border-radius: 999px; }
.fpl-token .mark.sell { background: rgba(0,0,0,0.5); color: #FFD7D2; }
.fpl-token .mark.buy { background: #06A8FD; color: #03131B; }
.fpl-token.sell .shirt { opacity: 0.55; }
a.fpl-token:hover > strong { box-shadow: 0 0 0 2px var(--fpl-brand); }
.fpl-bench { display: flex; align-items: center; gap: 18px; padding: 12px 18px; background: var(--fpl-surface); overflow-x: auto; }
.fpl-bench .lbl { display: flex; flex-direction: column; margin-right: auto; min-width: 100px; }
.fpl-bench .lbl b { font-size: 11.5px; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase; color: var(--fpl-muted); }
.fpl-bench .lbl small { font-size: 12.5px; color: var(--fpl-faint); }
.fpl-bench .fpl-token > strong { box-shadow: none; border: 1px solid var(--fpl-line-strong); background: var(--fpl-surface); color: var(--fpl-text); }
.fpl-bench .fpl-token > small { color: var(--fpl-muted); text-shadow: none; }
.fpl-bench .fpl-token > small em { color: var(--fpl-text); }

/* карточки туров плана */
.fpl-gw-grid { display: flex; gap: 12px; align-items: stretch; overflow-x: auto; padding: 2px 0 10px; }
.fpl-gw {
  min-width: 210px; flex: 1; background: var(--fpl-surface); border: 1px solid var(--fpl-line-strong);
  border-radius: 12px; display: flex; flex-direction: column; box-shadow: var(--fpl-shadow);
}
.fpl-gw.current { border-color: var(--fpl-accent-line); box-shadow: inset 0 3px var(--fpl-brand); }
.fpl-gw .head { padding: 14px 16px 6px; display: flex; justify-content: space-between; align-items: center; gap: 6px; }
.fpl-gw .head div { display: flex; align-items: baseline; gap: 7px; flex-wrap: wrap; }
.fpl-gw .head span { font-family: var(--fpl-mono); font-size: 16px; font-weight: 700; }
.fpl-gw .head small { font-size: 12.5px; color: var(--fpl-faint); }
.fpl-gw .head .xp { font-family: var(--fpl-mono); font-size: 15px; font-weight: 600; color: var(--fpl-text-2); }
.fpl-gw .head .xp small { font-family: var(--fpl-sans); font-size: 11.5px; color: var(--fpl-faint); }
.fpl-gw .now, .fpl-gw .chip {
  font-size: 11px; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase;
  padding: 2px 7px; border-radius: 4px;
}
.fpl-gw .now { color: var(--fpl-accent); background: var(--fpl-accent-bg); }
.fpl-gw .chip { color: #FFFFFF; background: var(--fpl-primary); }
.fpl-gw .moves { padding: 4px 16px 14px; flex: 1; }
.fpl-gw .move { margin-top: 8px; }
.fpl-gw .move strong { font-size: 15px; font-weight: 700; }
.fpl-gw .move.idle strong { color: var(--fpl-muted); }
.fpl-gw .move p { margin: 3px 0 0; font-size: 13px; color: var(--fpl-muted); line-height: 1.45; }
.fpl-gw .move .up { color: var(--fpl-good); font-weight: 700; }
.fpl-gw .hit {
  margin-left: 6px; font-family: var(--fpl-mono); font-size: 11.5px; font-weight: 700;
  color: var(--fpl-bad); background: var(--fpl-bad-bg); padding: 1px 6px; border-radius: 4px;
}
.fpl-gw .foot { border-top: 1px solid var(--fpl-line); padding: 10px 16px; font-size: 12.5px; color: var(--fpl-muted); }
.fpl-gw .foot .hit { margin-left: 0; }
"""
