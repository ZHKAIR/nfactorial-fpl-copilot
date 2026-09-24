"""Склейка «Почему»: LLM из одной позиции на игрока + запасной текст.

Числа и имена только из payload. Штампы запрещены. Кэш `.cache/why/`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from fplcopilot.config import settings
from fplcopilot.core.why_facts import _plural

log = logging.getLogger(__name__)

STAMPS: tuple[str, ...] = (
    "это повторяется",
    "на удаче",
    "спокойный выбор для осторожной стратегии",
    "это может повлиять на его результаты",
    "это может повлиять",
    "выглядит интересным вариантом",
    "может быть интересным вариантом",
    "может стать интересным выбором",
    "интересным вариантом",
    "интересным выбором",
    "многообещающе",
    "стоит рассмотреть его продажу",
    "стоит рассмотреть продажу",
    "календарь выглядит",
    "в свою очередь",
    "без конкретной причины",
    "нейтральная позиция",
    "набрал ",
)
_STAMP_SCORED = re.compile(r"набрал\s+\d+\s+за\s+\d+\s+тур", re.IGNORECASE)
_NUM = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)")
_LATIN_NAME = re.compile(r"\b[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’.\-]{1,}\b")
_CYR_WORD = re.compile(r"[А-ЯЁ][а-яёА-ЯЁ-]{2,}")
_CYR_LAT = str.maketrans(
    "абвгдеёжзийклмнопрстуфхцчшщыэюя",
    "abvgdeezziiklmnoprstufhccssyeua",
)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

WHY_SYSTEM = """Ты пишешь «Почему» для карточки трансфера FPL.
Ровно 2–4 коротких предложения по-русски. Каждая фраза несёт факт, без воды.
Только JSON. Цифры копируй как есть (57.3 или 57,3).
Имена игроков — точно web_name из JSON (латиница). Клубы — как в JSON.
Не транслитерируй имена игроков.
Без жаргона xG, CS, λ.
Запрещены штампы: «это повторяется», «на удаче», «спокойный выбор для осторожной стратегии»,
«набрал N за 3 тура, из них», «интересным вариантом», «интересным выбором»,
«это может повлиять», «многообещающе», «стоит рассмотреть его продажу»,
«календарь выглядит».
pairs[] — каждая замена независима. Факты (очки, метка, календарь, пенальти) только
своего игрока. Не складывай очки двух игроков в одну цифру. Не вешай календарь
или пенальти одного на другого.
Если пар две: по 1 предложению на пару sell_i → buy_i, плюс итог.
Если buy.interesting — обязательна фраза с этой цифрой, не только календарь.
Владение (buy.ownership) — не статистика и не замена interesting: не пиши «его выбрали N%»
вместо факта. Можно вскользь, только если интересный факт уже есть.
Если buy.status — обязательно скажи коротко (под вопросом / травма / дисквалификация /
шанс сыграть N%). Без статуса текст не годится.
Вратарям не пиши «везение» по голам.
Если buy.penalty_opponent — назови этот клуб в том же тексте, что и пенальти.
stance — единственная позиция. Не спорь с ней и не добавляй противоположную.
sell.solid_form_bad_fixtures: форма нормальная, причина — календарь.
sell.weak_form: слабая форма, не говори что форма нормальная.
buy.lucky_but_fixtures: очки во многом везение, берём за календарь (перечисли соперников).
buy.repeatable: стабильные источники (сухие / защита / сейвы), не везение и не «не повторится».
Пенальти — только если есть buy.penalty; не выдумывай фолы.
Не начинай все карточки одинаково с «Продаём».
Последнее предложение — итог: состав и банк из after.
Ответ — только текст, без markdown."""

# Вымышленные имена, чтобы few-shot не утекли в ответ по живому составу.
_SHOT_1_FACTS = {
    "sell": {
        "name": "Keller",
        "names": ["Keller"],
        "stance": "weak_form",
        "calendar": "впереди Норвич дома и Уотфорд",
        "detail": "без моментов три тура подряд",
        "form_ok": False,
        "reason": "без моментов три тура подряд",
    },
    "buy": {
        "name": "Novak",
        "names": ["Novak"],
        "stance": "lucky_but_fixtures",
        "calendar": "с Миллуоллом дома, Редингом и Лутоном",
        "opponents": ["Миллуолл", "Рединг", "Лутон"],
        "lucky_points": 16,
        "penalty": None,
    },
    "after": {"xi_points": "57.6", "bank": "2.5", "horizon": 3, "hit": 0},
    "variant": 0,
}
_SHOT_1_TEXT = (
    "Keller без моментов три тура подряд, а впереди Норвич и Уотфорд. "
    "Novak набрал 16 во многом на везении, но берём его за Миллуолл, Рединг и Лутон. "
    "Состав — 57.6 за 3 тура, в банке 2.5."
)
_SHOT_2_FACTS = {
    "sell": {
        "name": "Riedel",
        "names": ["Riedel"],
        "stance": "solid_form_bad_fixtures",
        "calendar": "впереди Норвич в гостях",
        "detail": "",
        "form_ok": True,
        "reason": "",
    },
    "buy": {
        "name": "Duarte",
        "names": ["Duarte"],
        "stance": "repeatable",
        "calendar": "с Миллуоллом дома, Редингом и Лутоном",
        "opponents": ["Миллуолл", "Рединг", "Лутон"],
        "repeatable": "сухих матчей",
        "penalty": None,
    },
    "after": {"xi_points": "54.2", "bank": "1.1", "horizon": 3, "hit": 0},
    "variant": 1,
}
_SHOT_2_TEXT = (
    "У Riedel форма нормальная, но впереди Норвич в гостях — меняем из-за календаря. "
    "Duarte недавно набирал за счёт сухих матчей, дальше Миллуолл, Рединг и Лутон. "
    "Состав — 54.2 за 3 тура, в банке 1.1."
)
_SHOT_3_FACTS = {
    "sell": {
        "name": "Kovac",
        "names": ["Kovac"],
        "stance": "minutes_risk",
        "calendar": "впереди Норвич дома",
        "detail": "под вопросом по статусу FPL",
        "form_ok": False,
        "reason": "под вопросом по статусу FPL",
    },
    "buy": {
        "name": "Berg",
        "names": ["Berg"],
        "stance": "neutral",
        "calendar": "с Миллуоллом дома и Редингом",
        "opponents": ["Миллуолл", "Рединг"],
        "penalty": "примерно 32%, что у Норвича будет пенальти за эти 3 тура, бить будет Berg",
    },
    "after": {"xi_points": "51.0", "bank": "0.4", "horizon": 3, "hit": 4},
    "variant": 2,
}
_SHOT_3_TEXT = (
    "Kovac под вопросом по статусу FPL, ещё и Норвич дома. "
    "Berg сыграет с Миллуоллом дома и Редингом; примерно 32%, что у Норвича будет пенальти, бить будет Berg. "
    "Это платный трансфер за минус 4 очка. Состав — 51.0 за 3 тура, в банке 0.4."
)
_SHOT_4_FACTS = {
    "pairs": [
        {
            "sell": {
                "name": "Holm",
                "names": ["Holm"],
                "stance": "calendar",
                "calendar": "впереди Норвич в гостях",
                "form_ok": False,
            },
            "buy": {
                "name": "Voss",
                "names": ["Voss"],
                "stance": "repeatable",
                "calendar": "с Миллуоллом дома",
                "opponents": ["Миллуолл"],
                "points": 9,
                "interesting": "4 из 9 — сухие матчи",
                "repeatable": "4 из 9 — сухие матчи",
            },
        },
        {
            "sell": {
                "name": "Perez",
                "names": ["Perez"],
                "stance": "weak_form",
                "calendar": "впереди Уотфорд дома",
                "detail": "без моментов три тура подряд",
                "form_ok": False,
            },
            "buy": {
                "name": "Quist",
                "names": ["Quist"],
                "stance": "lucky_but_fixtures",
                "calendar": "с Редингом дома",
                "opponents": ["Рединг"],
                "lucky_points": 20,
                "points": 20,
            },
        },
    ],
    "after": {"xi_points": "56.0", "bank": "1.2", "horizon": 3, "hit": 0},
    "variant": 0,
}
_SHOT_4_TEXT = (
    "Holm меняем из-за Норвича в гостях; у Voss 4 из 9 — сухие матчи, дальше Миллуолл. "
    "Perez без моментов; 20 очков Quist — во многом везение, берём за Рединг. "
    "Состав — 56.0 за 3 тура, в банке 1.2."
)


def payload_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _walk_nums(obj: Any, out: list[str]) -> None:
    if obj is None or isinstance(obj, bool):
        return
    if isinstance(obj, int):
        out.append(str(obj))
        return
    if isinstance(obj, float):
        out.append(str(int(obj)) if obj == int(obj) else f"{obj:.1f}")
        out.append(f"{obj:.2f}".rstrip("0").rstrip("."))
        out.append(str(obj))
        return
    if isinstance(obj, str):
        out.extend(_NUM.findall(obj.replace(",", ".")))
        return
    if isinstance(obj, Mapping):
        for v in obj.values():
            _walk_nums(v, out)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _walk_nums(v, out)


def _walk_strings(obj: Any, out: list[str]) -> None:
    if isinstance(obj, str):
        out.append(obj)
        return
    if isinstance(obj, Mapping):
        for v in obj.values():
            _walk_strings(v, out)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _walk_strings(v, out)


def _walk_names(obj: Any, out: set[str]) -> None:
    if isinstance(obj, str):
        for m in _LATIN_NAME.finditer(obj):
            out.add(m.group())
        return
    if isinstance(obj, Mapping):
        for v in obj.values():
            _walk_names(v, out)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _walk_names(v, out)


def allowed_number_forms(payload: Mapping[str, Any]) -> set[str]:
    raw: list[str] = []
    _walk_nums(payload, raw)
    forms: set[str] = set()
    for tok in raw:
        t = tok.replace(",", ".").lstrip("+-−")
        if not t:
            continue
        forms.add(t)
        forms.add(t.replace(".", ","))
        try:
            x = float(t)
        except ValueError:
            continue
        if x == int(x):
            forms.add(str(int(x)))
        forms.add(f"{x:.1f}")
        forms.add(f"{x:.1f}".replace(".", ","))
    return forms


def extract_numbers(text: str) -> list[str]:
    return [m.replace(",", ".") for m in _NUM.findall(text or "")]


def sentence_count(text: str) -> int:
    parts = [p.strip() for p in _SENT_SPLIT.split((text or "").strip()) if p.strip()]
    return len(parts)


def player_web_names(payload: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for side in ("sell", "buy"):
        block = payload.get(side) or {}
        if block.get("name"):
            names.append(str(block["name"]))
        names.extend(str(x) for x in (block.get("names") or []) if x)
    seen: list[str] = []
    for n in names:
        if n not in seen:
            seen.append(n)
    return seen


def has_stamp(text: str) -> bool:
    low = (text or "").lower()
    if any(s in low for s in STAMPS if s != "набрал "):
        return True
    return bool(_STAMP_SCORED.search(text or ""))


def validate_why_text(text: str, payload: Mapping[str, Any]) -> tuple[bool, str]:
    """Числа, web_name, ≤4 предложения, без штампов и чужой кириллицы."""
    if not (text or "").strip():
        return False, "empty"
    if has_stamp(text):
        return False, "stamp"
    n_sent = sentence_count(text)
    if n_sent > 4:
        return False, f"sentences:{n_sent}"
    allowed = allowed_number_forms(payload)
    for tok in extract_numbers(text):
        if tok not in allowed and tok.replace(".", ",") not in allowed:
            return False, f"number:{tok}"
    names = set()
    _walk_names(payload, names)
    extra = {"FPL"}
    for m in _LATIN_NAME.finditer(text or ""):
        word = m.group()
        if word in extra:
            continue
        if word not in names and word.rstrip(".") not in names:
            return False, f"name:{word}"
    web = player_web_names(payload)
    blob_facts = " ".join(_walk_collect_strings(payload))
    allowed_cyr = set(_CYR_WORD.findall(blob_facts))
    for word in _CYR_WORD.findall(text or ""):
        if word in allowed_cyr:
            continue
        lat = word.lower().translate(_CYR_LAT)
        for name in web:
            nl = name.lower()
            if lat == nl or (len(lat) >= 4 and len(nl) >= 4 and lat[:4] == nl[:4]):
                return False, f"cyr_name:{word}"
    buy = payload.get("buy") or {}
    sell = payload.get("sell") or {}
    pen = str(buy.get("penalty") or "")
    pct = re.search(r"(\d+)\s*%", pen)
    compact = (text or "").replace(" ", "")
    if (
        pct
        and f"{pct.group(1)}%" not in compact
        and pct.group(1) not in extract_numbers(text)
    ):
        return False, "penalty_pct"
    stance = str(sell.get("stance") or "")
    form_ok = bool(sell.get("form_ok")) or stance == "solid_form_bad_fixtures"
    if form_ok and re.search(
        r"не показывает|плохой игр|провал в очках|форма плох|не в хорошей",
        text or "",
        re.IGNORECASE,
    ):
        return False, "form_ok"
    if stance == "weak_form" and re.search(r"форма нормальн", text or "", re.IGNORECASE):
        return False, "weak_vs_solid"
    if str(buy.get("stance") or "") == "lucky_but_fixtures" and re.search(
        r"не был[оа] везени|не везение", text or "", re.IGNORECASE
    ):
        return False, "lucky_contradict"
    if str(buy.get("stance") or "") == "repeatable" and re.search(
        r"не повтори|везени", text or "", re.IGNORECASE
    ):
        return False, "repeatable_contradict"
    raw_facts = json.dumps(payload, ensure_ascii=False)
    if "в последнем матче" in (text or "") and "в последнем матче" not in raw_facts:
        return False, "last_match"
    ok_pts, why_pts = _validate_player_points(text, payload)
    if not ok_pts:
        return False, why_pts
    ok_pen, why_pen = _validate_penalty_named(text, payload)
    if not ok_pen:
        return False, why_pen
    ok_int, why_int = _validate_interesting(text, payload)
    if not ok_int:
        return False, why_int
    ok_st, why_st = _validate_buy_status(text, payload)
    if not ok_st:
        return False, why_st
    ok_own, why_own = _validate_ownership_not_stat(text, payload)
    if not ok_own:
        return False, why_own
    return True, "ok"


def _iter_pairs(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    pairs = list(payload.get("pairs") or [])
    if pairs:
        return [p for p in pairs if isinstance(p, Mapping)]
    sell = payload.get("sell") or {}
    buy = payload.get("buy") or {}
    if sell or buy:
        return [{"sell": sell, "buy": buy}]
    return []


def _player_points_map(payload: Mapping[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for pair in _iter_pairs(payload):
        for side in ("sell", "buy"):
            block = pair.get(side) or {}
            name = str(block.get("name") or "")
            pts = block.get("points")
            if pts is None:
                pts = block.get("lucky_points")
            if name and pts is not None:
                try:
                    out[name] = int(pts)
                except (TypeError, ValueError):
                    continue
    return out


def _validate_player_points(text: str, payload: Mapping[str, Any]) -> tuple[bool, str]:
    """Число очков рядом с именем X = его собственная раскладка, не сумма чужих."""
    pts_map = _player_points_map(payload)
    if not pts_map:
        return True, "ok"
    after = payload.get("after") or {}
    after_blob = f"{after.get('xi_points') or ''} {after.get('bank') or ''}"
    for sent in _SENT_SPLIT.split(text or ""):
        if not sent.strip():
            continue
        if "состав" in sent.lower() or "банке" in sent.lower():
            continue
        names_here = [n for n in pts_map if n in sent]
        found = [int(m.group(1)) for m in re.finditer(r"(\d+)\s+очк", sent)]
        if len(names_here) >= 2 and found:
            own = {pts_map[n] for n in names_here}
            total = sum(pts_map[n] for n in names_here)
            for n in found:
                if n == total or n not in own:
                    return False, f"summed_points:{n}"
            continue
        for name in names_here:
            own = pts_map[name]
            for n in found:
                if n != own and str(n) not in after_blob:
                    return False, f"points:{name}:{n}"
    values = list(pts_map.values())
    if len(values) >= 2:
        summed = sum(values)
        if re.search(rf"(?<![\d.]){summed}\s+очк", text or ""):
            return False, "summed_points"
    return True, "ok"


def _validate_penalty_named(text: str, payload: Mapping[str, Any]) -> tuple[bool, str]:
    """Пенальти-соперник из фактов должен быть назван в тексте, если пенальти упомянуты."""
    if "пенальти" not in (text or ""):
        return True, "ok"
    for pair in _iter_pairs(payload):
        buy = pair.get("buy") or {}
        opp = str(buy.get("penalty_opponent") or "")
        if opp and opp not in (text or ""):
            return False, f"pen_opp:{opp}"
    return True, "ok"


def _validate_interesting(text: str, payload: Mapping[str, Any]) -> tuple[bool, str]:
    """Если у покупаемого есть нумерованный факт — хотя бы одна его цифра должна быть в тексте."""
    blob = extract_numbers(text or "")
    for pair in _iter_pairs(payload):
        buy = pair.get("buy") or {}
        interesting = str(buy.get("interesting") or "")
        nums = extract_numbers(interesting)
        if not nums:
            continue
        if not any(n in blob for n in nums):
            return False, f"interesting:{buy.get('name') or '?'}"
    return True, "ok"


_STATUS_MENTION = re.compile(
    r"под вопрос|травм|дисквалиф|недоступ|не в заявк|шанс сыграть",
    re.IGNORECASE,
)
_OWN_FILLER = re.compile(
    r"выбрали\s+\d|держат\s+\d|владен",
    re.IGNORECASE,
)


def _validate_buy_status(text: str, payload: Mapping[str, Any]) -> tuple[bool, str]:
    """Покупаемый не «доступен» — статус обязан быть в тексте."""
    for pair in _iter_pairs(payload):
        buy = pair.get("buy") or {}
        if not buy.get("status"):
            continue
        if not _STATUS_MENTION.search(text or ""):
            return False, f"buy_status:{buy.get('name') or '?'}"
    return True, "ok"


def _validate_ownership_not_stat(text: str, payload: Mapping[str, Any]) -> tuple[bool, str]:
    """Владение не закрывает требование «хотя бы одна цифра кроме календаря»."""
    has_interesting = any(
        str((pair.get("buy") or {}).get("interesting") or "")
        for pair in _iter_pairs(payload)
    )
    if has_interesting:
        return True, "ok"
    if _OWN_FILLER.search(text or ""):
        return False, "ownership_as_stat"
    return True, "ok"


def _walk_collect_strings(obj: Any) -> list[str]:
    out: list[str] = []
    _walk_strings(obj, out)
    return out


def _cap1(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _join(parts: Sequence[str]) -> str:
    items = [p for p in parts if p]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} и {items[1]}"
    return f"{', '.join(items[:-1])} и {items[-1]}"


def _sell_names(sell: Mapping[str, Any]) -> str:
    if sell.get("name"):
        return str(sell["name"])
    return _join(list(sell.get("names") or []))


def _buy_names(buy: Mapping[str, Any]) -> str:
    if buy.get("name"):
        return str(buy["name"])
    return _join(list(buy.get("names") or []))


def _after_sentence(after: Mapping[str, Any]) -> str:
    xi = after.get("xi_points")
    bank = after.get("bank")
    horizon = int(after.get("horizon") or 3)
    if xi is None or bank is None:
        return ""
    tours = "тур" if horizon == 1 else ("тура" if horizon in (2, 3, 4) else "туров")
    hit = int(after.get("hit") or 0)
    tail = (
        f"После замены ваш состав набирает {xi} очков за {horizon} {tours}, "
        f"в банке останется {bank} миллиона."
    )
    if hit:
        tail = f"Это платный трансфер за минус {hit} очка. {tail}"
    return tail


def _sell_fallback(sell: Mapping[str, Any], variant: int) -> str:
    sell_names = _sell_names(sell)
    sell_cal = str(sell.get("calendar") or "").strip()
    sell_stance = str(sell.get("stance") or "")
    detail = str(sell.get("detail") or sell.get("reason") or "").rstrip(".")
    form_ok = bool(sell.get("form_ok")) or sell_stance == "solid_form_bad_fixtures"
    if form_ok and sell_cal:
        openings = (
            f"У {sell_names} форма нормальная, но {sell_cal} — меняем из-за календаря",
            f"{_cap1(sell_cal)} — форма у {sell_names} нормальная, меняем из-за соперников",
            f"Форма у {sell_names} нормальная, но {sell_cal}",
        )
        return openings[variant] + "."
    if sell_stance == "weak_form" or (detail and not form_ok):
        reason = detail or "форма слабая"
        if sell_cal:
            mix = (
                f"{sell_names}: {reason}; {sell_cal}",
                f"{_cap1(sell_cal)}. {sell_names}: {reason}",
                f"{sell_names} {reason}, ещё и {sell_cal}",
            )
            return mix[variant] + "."
        return f"{sell_names}: {reason}."
    if sell_names and sell_cal:
        return f"{_cap1(sell_cal)} — поэтому {sell_names} уходит."
    if sell_names:
        return f"{sell_names} уходит из состава."
    return ""


def _buy_fallback(buy: Mapping[str, Any], variant: int) -> list[str]:
    buy_names = _buy_names(buy)
    buy_cal = str(buy.get("calendar") or "").strip()
    buy_opps = list(buy.get("opponents") or [])
    buy_stance = str(buy.get("stance") or "")
    sentences: list[str] = []
    lucky = buy.get("lucky_points")
    interesting = str(buy.get("interesting") or "").rstrip(".")
    if buy_stance == "lucky_but_fixtures" or (lucky is not None and buy_stance != "repeatable"):
        if lucky is not None and buy_names:
            opps = _join(buy_opps) if buy_opps else buy_cal
            extra = f"; {interesting}" if interesting and interesting not in str(lucky) else ""
            pts = (
                _plural(lucky, "очко", "очка", "очков")
                if isinstance(lucky, int)
                else f"{lucky} очков"
            )
            if opps:
                sentences.append(
                    f"{pts} {buy_names} — во многом везение, такого снова не ждите"
                    f"{extra}; берём его за календарь: {opps}."
                )
            else:
                sentences.append(
                    f"{pts} {buy_names} — во многом везение, такого снова не ждите{extra}."
                )
    elif buy.get("repeatable") and buy_names:
        src = str(buy["repeatable"])
        if buy_cal:
            sentences.append(f"{buy_names} недавно набирал за счёт {src}. Дальше {buy_cal}.")
        else:
            sentences.append(f"{buy_names} недавно набирал за счёт {src}.")
    elif buy.get("moments") and buy_names:
        bit = f"{buy_names} {buy['moments']}"
        if buy_cal:
            bit += f". Сыграет {buy_cal}"
        extra = buy.get("leaky") or (interesting if interesting else "")
        if extra:
            bit += f" — {extra}"
        sentences.append(bit.rstrip(".") + ".")
    elif interesting and buy_names:
        if buy_cal:
            sentences.append(f"{buy_names}: {interesting}. Дальше {buy_cal}.")
        else:
            sentences.append(f"{buy_names}: {interesting}.")
    elif buy_names and buy_cal:
        cal_open = (
            f"{buy_names} берём под календарь: {buy_cal}",
            f"Вместо него — {buy_names}, впереди {buy_cal}",
            f"{buy_names} сыграет {buy_cal}",
        )
        sentences.append(cal_open[variant] + ".")
    elif buy_names:
        sentences.append(f"Вместо него берём {buy_names}.")
    if buy.get("penalty"):
        clause = str(buy["penalty"]).rstrip(".")
        opp = str(buy.get("penalty_opponent") or "")
        if opp and opp not in clause:
            clause = f"{clause} — как раз {opp}"
        sentences.append(clause[:1].upper() + clause[1:] + ".")
    if buy.get("status"):
        sentences.append(f"{buy_names} {buy['status']}.")
    return sentences


def _pair_fallback(sell: Mapping[str, Any], buy: Mapping[str, Any], variant: int) -> str:
    """Одно предложение sell → buy для двойного трансфера."""
    sell_bit = _sell_fallback(sell, variant).rstrip(".")
    buy_bits = _buy_fallback(buy, variant)
    buy_raw = " ".join(b.rstrip(".") for b in buy_bits[:1])
    buy_first = ""
    if buy_raw:
        buy_first = _SENT_SPLIT.split(buy_raw)[0].rstrip(".")
    if sell_bit and buy_first:
        return f"{sell_bit}; {buy_first}."
    rest = sell_bit or buy_first
    if rest and not rest.endswith((".", "!", "?")):
        return rest + "."
    return rest


def fallback_why(payload: Mapping[str, Any]) -> str:
    """Короткий запасной текст без штампов. variant 0..2 меняет зачин."""
    after = payload.get("after") or {}
    variant = int(payload.get("variant") or 0) % 3
    pairs = _iter_pairs(payload)
    sentences: list[str] = []
    if len(pairs) >= 2:
        ranked = list(pairs)
        if len(ranked) > 2:
            sentences.append(_pair_fallback(ranked[0].get("sell") or {}, ranked[0].get("buy") or {}, variant))
            sentences.append(f"Ещё {len(ranked) - 1} замены по своим фактам.")
        else:
            for pair in ranked:
                sentences.append(
                    _pair_fallback(pair.get("sell") or {}, pair.get("buy") or {}, variant)
                )
        tail = _after_sentence(after)
        if tail:
            sentences.append(tail)
    else:
        sell = (pairs[0].get("sell") if pairs else None) or payload.get("sell") or {}
        buy = (pairs[0].get("buy") if pairs else None) or payload.get("buy") or {}
        sell_s = _sell_fallback(sell, variant)
        if sell_s:
            sentences.append(sell_s)
        sentences.extend(_buy_fallback(buy, variant))
        tail = _after_sentence(after)
        if tail:
            sentences.append(tail)
    if len(sentences) > 4:
        sentences = sentences[:3] + sentences[-1:]
    text = " ".join(s for s in sentences if s)
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    if len(parts) > 4:
        tail = parts[-1]
        head = parts[: 4 - 1]
        text = " ".join(p if p.endswith((".", "!", "?")) else p + "." for p in [*head, tail])
    return text


def _cache_dir() -> Path:
    root = Path(getattr(settings, "cache_dir", Path(".cache")))
    return Path(root) / "why"


def disk_get(key: str) -> tuple[str, str] | None:
    path = _cache_dir() / f"{key}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    text = data.get("text")
    source = data.get("source") or "cache"
    if isinstance(text, str) and text.strip():
        return text, str(source)
    return None


def disk_put(key: str, text: str, source: str) -> None:
    path = _cache_dir()
    try:
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{key}.json").write_text(
            json.dumps({"text": text, "source": source}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        log.debug("why cache write failed", exc_info=True)


def _llm_complete(
    payload: Mapping[str, Any],
    timeout_s: float,
    *,
    reject: str | None = None,
) -> str:
    from fplcopilot.rag.llm import get_openai_client

    model = str(getattr(settings, "why_llm_model", None) or settings.agent_explain_model)
    client = get_openai_client()
    if hasattr(client, "with_options"):
        client = client.with_options(timeout=timeout_s)
    user = "FACTS:\n" + json.dumps(payload, ensure_ascii=False, indent=1, default=str)
    if reject:
        user += f"\n\nПрошлая попытка отклонена ({reject}). Исправь: 2–4 фразы, web_name латиницей, без воды."
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": WHY_SYSTEM},
            {"role": "user", "content": "FACTS:\n" + json.dumps(_SHOT_1_FACTS, ensure_ascii=False)},
            {"role": "assistant", "content": _SHOT_1_TEXT},
            {"role": "user", "content": "FACTS:\n" + json.dumps(_SHOT_2_FACTS, ensure_ascii=False)},
            {"role": "assistant", "content": _SHOT_2_TEXT},
            {"role": "user", "content": "FACTS:\n" + json.dumps(_SHOT_3_FACTS, ensure_ascii=False)},
            {"role": "assistant", "content": _SHOT_3_TEXT},
            {"role": "user", "content": "FACTS:\n" + json.dumps(_SHOT_4_FACTS, ensure_ascii=False)},
            {"role": "assistant", "content": _SHOT_4_TEXT},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_completion_tokens=280,
    )
    return (completion.choices[0].message.content or "").strip()


def _call_llm(
    payload: Mapping[str, Any],
    llm: Callable[..., str] | None,
    wait: float,
    reject: str | None = None,
) -> str:
    if llm is not None:
        try:
            return llm(payload, reject) if reject else llm(payload)
        except TypeError:
            return llm(payload)
    return _llm_complete(payload, wait, reject=reject)


def narrate_why(
    payload: Mapping[str, Any],
    *,
    llm: Callable[..., str] | None = None,
    use_llm: bool = True,
    use_disk: bool = True,
    timeout_s: float | None = None,
    disk_key: str | None = None,
) -> tuple[str, str]:
    """(текст, source: llm|fallback). Один повтор при ошибке валидатора, затем fallback."""
    key = payload_hash(payload)
    if disk_key:
        key = str(disk_key)
    if use_disk:
        hit = disk_get(key)
        if hit is not None:
            src = hit[1] if hit[1] in ("llm", "fallback") else "fallback"
            return hit[0], src
    wait = float(getattr(settings, "why_llm_timeout_s", 8.0) if timeout_s is None else timeout_s)
    text = ""
    source = "fallback"
    if use_llm:
        reject: str | None = None
        for _attempt in range(2):
            try:
                started = time.perf_counter()
                text = _call_llm(payload, llm, wait, reject)
                if time.perf_counter() - started > wait + 1:
                    text = ""
                    break
            except Exception:
                log.debug("why llm failed", exc_info=True)
                text = ""
                break
            if not text:
                break
            ok, reason = validate_why_text(text, payload)
            if ok:
                source = "llm"
                break
            reject = reason
            text = ""
    if not text:
        text = fallback_why(payload)
        source = "fallback"
    if use_disk:
        disk_put(key, text, source)
    return text, source
