"""Скриншот -> ScreenshotSquadRaw (vision LLM, structured output) -> ParsedSquad (детерминированно).

squad_from_image(image, bootstrap, as_of=None):
  1. load_image: bytes / путь -> (bytes, mime); картинки крупнее MAX_SIDE px уменьшаются через
     Pillow — OpenAI всё равно масштабирует до 2048×2048 / короткая сторона 768, а base64 меньше;
  2. call_vision_llm: system-промпт vision/prompts/v1/squad_extraction.system.md + изображение как
     data-URL (detail = VISION_DETAIL), response_format=ScreenshotSquadRaw, temperature 0.
     Модель НЕ резолвит игроков: только читает карточки;
  3. parsed_from_raw: PlayerResolver (resolve.py) -> validate_squad (validate.py) -> ParsedSquad с
     сырым ответом и usage (токены, оценка стоимости, латентность) для отладки и evals.

LangSmith: клиент OpenAI из rag.llm (wrap_openai при наличии ключа), стадии — @traceable.
"""

from __future__ import annotations

import base64
import io
import logging
import re
import time
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from langsmith import traceable

from fplcopilot.config import settings
from fplcopilot.data.schemas import Bootstrap
from fplcopilot.rag.llm import configure_tracing, get_openai_client
from fplcopilot.vision.resolve import PlayerResolver
from fplcopilot.vision.schemas import (
    SQUAD_SIZE,
    ParsedSquad,
    ScreenshotSquadRaw,
    SquadIssue,
    VisionUsage,
)
from fplcopilot.vision.validate import is_valid, validate_squad

log = logging.getLogger(__name__)

PROMPT_VERSION = (
    settings.vision_prompt_version
)  # v2 по умолчанию; v1 сохранён для A/B (docs/vision.md)
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
USER_TEXT = (
    "Extract the FPL squad from this screenshot. Read the header, every pitch row AND the "
    "substitutes strip at the bottom; return every readable player card with captain/vice badges, "
    "bench order, prices, bank and free transfers exactly as shown."
)
MAX_COMPLETION_TOKENS = 2500  # 15 карточек × ~9 полей + служебные токены схемы
MAX_SIDE = 2048  # px: больше OpenAI всё равно ужимает; экономим base64 и время загрузки

# $/1M токенов (input, output); оценка стоимости в VisionUsage. Токены изображения уже включены
# в prompt_tokens ответа API (у gpt-4o-mini картинка «стоит» ~33× больше токенов, чем у gpt-4o,
# при цене токена в 17× ниже — поэтому цена картинки у них сопоставима).
PRICES_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
}

_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF8", "image/gif"),
)


class VisionExtractionError(RuntimeError):
    """Модель отказалась / не вернула структурированный ответ / изображение не читается."""


# ---------- изображение ----------


def sniff_mime(data: bytes) -> str | None:
    for magic, mime in _MAGIC:
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def load_image(source: bytes | str | Path, *, max_side: int = MAX_SIDE) -> tuple[bytes, str]:
    """Байты или путь -> (bytes, mime). Неизвестный формат/слишком крупная картинка -> PNG через Pillow."""
    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    if not data:
        raise VisionExtractionError("empty image")
    mime = sniff_mime(data)
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            width, height = im.size
            if mime is not None and max(width, height) <= max_side:
                return data, mime
            im.load()
            if max(width, height) > max_side:
                scale = max_side / max(width, height)
                im = im.resize((round(width * scale), round(height * scale)))
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGBA" if "A" in im.mode else "RGB")
            buf = io.BytesIO()
            im.save(buf, format="PNG", optimize=True)
            return buf.getvalue(), "image/png"
    except VisionExtractionError:
        raise
    except Exception as exc:  # Pillow не открыл файл
        if mime is not None:
            return data, mime  # формат известен — отдаём как есть, пусть решает API
        raise VisionExtractionError(f"unsupported image: {exc}") from exc


def to_data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


# ---------- промпт и вызов модели ----------


@lru_cache(maxsize=4)
def load_system_prompt(version: str = PROMPT_VERSION) -> str:
    path = PROMPTS_DIR / version / "squad_extraction.system.md"
    return path.read_text(encoding="utf-8").strip()


def build_messages(data_url: str, *, detail: str, version: str = PROMPT_VERSION) -> list[dict]:
    return [
        {"role": "system", "content": load_system_prompt(version)},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_TEXT},
                {"type": "image_url", "image_url": {"url": data_url, "detail": detail}},
            ],
        },
    ]


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Оценка $ по PRICES_PER_1M; неизвестная модель -> ближайший префикс, иначе 0."""
    prices = PRICES_PER_1M.get(model)
    if prices is None:
        for name, p in sorted(PRICES_PER_1M.items(), key=lambda kv: -len(kv[0])):
            if model.startswith(name):
                prices = p
                break
    if prices is None:
        return 0.0
    return (prompt_tokens * prices[0] + completion_tokens * prices[1]) / 1_000_000


def _supports_temperature(model: str) -> bool:
    """Reasoning-модели (o*, gpt-5*) temperature не принимают; для остальных фиксируем 0."""
    return not model.startswith(("o", "gpt-5"))


@traceable(name="squad_vision_llm", run_type="chain")
def call_vision_llm(
    image: bytes,
    mime: str,
    *,
    model: str | None = None,
    detail: str | None = None,
    prompt_version: str = PROMPT_VERSION,
) -> tuple[ScreenshotSquadRaw, VisionUsage]:
    """Один вызов vision-модели со structured output. Отказ модели -> VisionExtractionError."""
    model = model or settings.vision_model
    detail = detail or settings.vision_detail
    configure_tracing()
    client = get_openai_client()
    kwargs: dict = {}
    if _supports_temperature(model):
        kwargs["temperature"] = 0
    started = time.perf_counter()
    completion = client.chat.completions.parse(
        model=model,
        messages=build_messages(to_data_url(image, mime), detail=detail, version=prompt_version),  # type: ignore[arg-type]
        response_format=ScreenshotSquadRaw,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        **kwargs,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    msg = completion.choices[0].message
    u = completion.usage
    prompt_tokens = u.prompt_tokens if u else 0
    completion_tokens = u.completion_tokens if u else 0
    usage = VisionUsage(
        model=model,
        detail=detail,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=round(estimate_cost(model, prompt_tokens, completion_tokens), 6),
        latency_ms=round(latency_ms, 1),
    )
    if msg.parsed is None:
        raise VisionExtractionError(
            f"model returned no structured answer: {msg.refusal or 'empty response'}"
        )
    log.info(
        "vision %s detail=%s: %d cards, %d+%d tokens ≈ $%.4f (%.0f ms)",
        model,
        detail,
        len(msg.parsed.players),
        prompt_tokens,
        completion_tokens,
        usage.cost_usd,
        latency_ms,
    )
    return msg.parsed, usage


# ---------- детерминированная часть ----------


_LAYOUT_TOTAL = re.compile(r"total\D{0,12}?(\d{1,2})\b", re.IGNORECASE)


def layout_total(layout: str | None) -> int | None:
    """«...; substitutes strip: 4; total 15» -> 15. Скретчпад модели vs фактический список карточек:
    расхождение — признак обрезанного/недочитанного экрана (warning layout_mismatch)."""
    m = _LAYOUT_TOTAL.search(layout or "")
    return int(m.group(1)) if m else None


def next_gw(bs: Bootstrap, as_of: datetime | None = None) -> int | None:
    """Ближайший тур, дедлайн которого ещё не прошёл на момент as_of (по умолчанию — сейчас)."""
    now = (as_of or datetime.now(UTC)).astimezone(UTC)
    upcoming = [e for e in bs.events if e.deadline_time > now]
    if upcoming:
        return min(upcoming, key=lambda e: e.deadline_time).id
    return bs.next_event.id if bs.next_event else None


def parsed_from_raw(
    raw: ScreenshotSquadRaw,
    bs: Bootstrap,
    *,
    as_of: datetime | None = None,
    usage: VisionUsage | None = None,
) -> ParsedSquad:
    """Сырой ответ модели -> резолюция по bootstrap -> правила FPL -> ParsedSquad. Без сети."""
    resolver = PlayerResolver.for_bootstrap(bs)
    players = resolver.resolve_all(raw.players)
    issues = validate_squad(players, bank=raw.bank_as_shown)
    # Арифметика скретчпада ненадёжна (smoke: «total 17» при 15 верных карточках), поэтому
    # расхождение показываем только когда карточек и так меньше 15 — как объяснение недостачи.
    counted = layout_total(raw.layout)
    if counted is not None and len(raw.players) < SQUAD_SIZE and counted != len(raw.players):
        issues.append(
            SquadIssue(
                code="layout_mismatch",
                message=(
                    f"model counted {counted} cards in its layout but returned {len(raw.players)} — "
                    "part of the screen may be cut off or misread"
                ),
                blocking=False,
            )
        )
    if raw.notes.strip():
        issues.append(
            SquadIssue(
                code="model_note", message=f"model note: {raw.notes.strip()}", blocking=False
            )
        )

    captain = next((p.player_id for p in players if p.is_captain and p.resolved), None)
    vice = next((p.player_id for p in players if p.is_vice_captain and p.resolved), None)
    starting_ids = [p.player_id for p in players if p.resolved and not p.is_bench]
    bench = sorted(
        (p for p in players if p.is_bench and p.resolved),
        key=lambda p: (p.bench_order is None, p.bench_order or 0),
    )
    return ParsedSquad(
        players=players,
        bank=raw.bank_as_shown,
        free_transfers=raw.free_transfers_as_shown,
        gw=next_gw(bs, as_of),
        screen_type=raw.screen_type,
        captain_id=captain,
        vice_id=vice,
        starting_ids=starting_ids,  # type: ignore[arg-type]
        bench_order=[p.player_id for p in bench],  # type: ignore[misc]
        issues=issues,
        is_valid=is_valid(issues),
        raw=raw,
        usage=usage,
    )


@traceable(name="squad_from_image", run_type="chain")
def squad_from_image(
    image: bytes | str | Path,
    bs: Bootstrap,
    *,
    as_of: datetime | None = None,
    model: str | None = None,
    detail: str | None = None,
    prompt_version: str = PROMPT_VERSION,
) -> ParsedSquad:
    """Полный конвейер: изображение -> vision LLM -> резолюция -> валидация."""
    data, mime = load_image(image)
    raw, usage = call_vision_llm(
        data, mime, model=model, detail=detail, prompt_version=prompt_version
    )
    parsed = parsed_from_raw(raw, bs, as_of=as_of, usage=usage)
    log.info(
        "squad_from_image: %d cards, %d resolved, valid=%s, blocking=%d, warnings=%d",
        len(parsed.players),
        parsed.resolved_count,
        parsed.is_valid,
        len(parsed.blocking_issues),
        len(parsed.warnings),
    )
    return parsed
