"""Сквозной конвейер без сети: сырой ответ модели -> ParsedSquad; squad_from_image с фейковой
моделью; подготовка изображения; рендер эталонов; оценка стоимости. Реальный вызов vision-модели —
отдельный тест с маркерами llm + network."""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest
from PIL import Image
from test_vision_fixtures import AS_OF, make_bootstrap, raw, valid_raw_players

from fplcopilot.vision import extract
from fplcopilot.vision.extract import (
    PRICES_PER_1M,
    VisionExtractionError,
    build_messages,
    estimate_cost,
    layout_total,
    load_image,
    load_system_prompt,
    next_gw,
    parsed_from_raw,
    sniff_mime,
    squad_from_image,
    to_data_url,
)
from fplcopilot.vision.render import render_pick_team, render_samples, sample_specs
from fplcopilot.vision.schemas import ScreenshotSquadRaw, VisionUsage
from fplcopilot.vision.to_squad import to_squad

ROOT = Path(__file__).resolve().parents[1]


def valid_raw(**overrides) -> ScreenshotSquadRaw:
    base = {
        "layout": "pitch: GKP 1, DEF 4, MID 4, FWD 2 = 11; substitutes strip: 4; total 15",
        "players": valid_raw_players(),
        "bank_as_shown": 0.5,
        "free_transfers_as_shown": 1,
        "screen_type": "pick_team",
        "notes": "",
    }
    base.update(overrides)
    return ScreenshotSquadRaw(**base)


# ---------- raw -> parsed ----------


def test_parsed_from_raw_valid_round_trip():
    bs = make_bootstrap()
    parsed = parsed_from_raw(valid_raw(), bs, as_of=AS_OF)

    assert parsed.is_valid and parsed.issues == []
    assert parsed.resolved_count == 15 and len(parsed.players) == 15
    assert parsed.captain_id == 426 and parsed.vice_id == 165
    assert len(parsed.starting_ids) == 11 and 426 in parsed.starting_ids
    assert parsed.bench_order == [467, 331, 40, 490]
    assert parsed.bank == 0.5 and parsed.free_transfers == 1
    assert parsed.gw == 5 and parsed.screen_type == "pick_team"
    assert parsed.raw is not None and parsed.usage is None
    flagged = [p for p in parsed.players if p.is_flagged]
    assert [p.web_name for p in flagged] == ["Gvardiol"]
    by_name = {p.name_as_shown: p for p in parsed.players}
    # «Gabriel» неоднозначен сам по себе: его вытянули ряд DEF + цена £8.0m -> уверенность 0.9
    assert by_name["Gabriel"].match_method == "exact+hints"
    assert by_name["Gabriel"].match_confidence == pytest.approx(0.9)
    assert all(
        p.match_confidence == 1.0 and p.match_method == "exact"
        for p in parsed.players
        if p.name_as_shown != "Gabriel"
    )

    squad = to_squad(parsed, bs)
    assert len(squad.players) == 15 and squad.gw == 5


def test_parsed_from_raw_broken_sample_reports_expected_issues():
    """samples/pick_team_broken.png: 14 игроков, два капитана, 4 игрока Arsenal, 4 MID, старт 10."""
    bs = make_bootstrap()
    players = [
        raw("Raya", "GKP", price=6.0),
        raw("Gabriel", "DEF", price=8.0),
        raw("Virgil", "DEF", price=6.5),
        raw("Gvardiol", "DEF", price=5.7),
        raw("Senesi", "DEF", price=5.8),
        raw("B.Fernandes", "MID", price=12.0, captain=True),
        raw("Saka", "MID", price=9.5),
        raw("Semenyo", "MID", price=8.4),
        raw("Haaland", "FWD", price=15.5, captain=True),
        raw("João Pedro", "FWD", price=7.8, vice=True),
        raw("Sels", "GKP", price=5.0, bench=1),
        raw("J.Timber", "DEF", price=6.5, bench=2),
        raw("Mbeumo", "MID", price=7.9, bench=3),
        raw("Wood", "FWD", price=5.8, bench=4),
    ]
    parsed = parsed_from_raw(
        valid_raw(players=players, bank=1.2, free_transfers_as_shown=2), bs, as_of=AS_OF
    )

    assert not parsed.is_valid
    assert parsed.resolved_count == 14  # все карточки распознаны, ломают правила, а не OCR
    blocking = {i.code: i.message for i in parsed.blocking_issues}
    assert blocking["squad_size"] == "squad size 14 ≠ 15"
    assert blocking["position_count"] == "MID: 4 ≠ 5"
    assert blocking["club_limit"] == "ARS: 4 players > 3"
    warnings = {i.code: i.message for i in parsed.warnings}
    assert warnings["captain"] == "2 captains: B.Fernandes, Haaland"
    assert warnings["xi_count"] == "starting XI has 10 players ≠ 11"
    assert parsed.captain_id == 426  # первый бейдж C
    with pytest.raises(ValueError):
        to_squad(parsed, bs)


def test_parsed_from_raw_ambiguous_card_blocks_and_lists_candidates():
    bs = make_bootstrap()
    players = valid_raw_players()
    players[1] = raw("Gabriel", None, price=None)  # позиция и цена не прочитаны
    parsed = parsed_from_raw(valid_raw(players=players), bs, as_of=AS_OF)
    assert not parsed.is_valid
    assert parsed.resolved_count == 14
    amb = [i for i in parsed.issues if i.code == "ambiguous"]
    assert len(amb) == 1 and "Gudmundsson (LEE DEF £4.5m)" in amb[0].message
    # без позиции карточки DEF стало 4 — это тоже нарушение, и оно честно показано
    assert any(i.code == "position_count" and i.message == "DEF: 4 ≠ 5" for i in parsed.issues)


def test_layout_total_parsing():
    assert layout_total("pitch: GKP 1, DEF 4 = 11; substitutes strip: 4; total 15") == 15
    assert layout_total("Total: 14 cards") == 14
    assert layout_total("total of 10") == 10
    assert layout_total("no numbers here") is None
    assert layout_total("") is None and layout_total(None) is None


def test_layout_mismatch_becomes_a_warning():
    """Обрезанный скриншот (smoke): модель насчитала 14, вернула 10 и ничего не написала в notes."""
    bs = make_bootstrap()
    players = valid_raw_players()[:10]
    parsed = parsed_from_raw(
        valid_raw(players=players, layout="pitch: 10; substitutes strip: 4; total 14"),
        bs,
        as_of=AS_OF,
    )
    assert not parsed.is_valid
    mismatch = [i for i in parsed.issues if i.code == "layout_mismatch"]
    assert len(mismatch) == 1 and not mismatch[0].blocking
    assert "counted 14 cards" in mismatch[0].message and "returned 10" in mismatch[0].message
    # согласованный layout предупреждения не даёт
    ok = parsed_from_raw(valid_raw(layout="pitch 11; bench 4; total 15"), bs, as_of=AS_OF)
    assert not any(i.code == "layout_mismatch" for i in ok.issues)
    # 15 карточек на месте, а арифметика скретчпада врёт («total 17» в smoke) — не шумим
    ok = parsed_from_raw(valid_raw(layout="pitch 13; bench 4; total 17"), bs, as_of=AS_OF)
    assert ok.is_valid and not any(i.code == "layout_mismatch" for i in ok.issues)


def test_model_notes_become_a_warning():
    bs = make_bootstrap()
    parsed = parsed_from_raw(valid_raw(notes="bench prices cut off"), bs, as_of=AS_OF)
    assert parsed.is_valid
    assert [i.code for i in parsed.issues] == ["model_note"]
    assert "bench prices cut off" in parsed.issues[0].message


def test_next_gw_uses_deadlines():
    bs = make_bootstrap()
    assert next_gw(bs, AS_OF) == 5
    from datetime import UTC, datetime

    assert next_gw(bs, datetime(2026, 9, 21, tzinfo=UTC)) == 6
    assert (
        next_gw(bs, datetime(2027, 1, 1, tzinfo=UTC)) == 5
    )  # дедлайны кончились -> next_event API


# ---------- squad_from_image с фейковой моделью ----------


def test_squad_from_image_with_fake_llm(monkeypatch, tmp_path):
    bs = make_bootstrap()
    spec = sample_specs(bs)["pick_team_valid"]
    path = tmp_path / "shot.png"
    render_pick_team(spec).save(path)

    seen: dict = {}

    def fake_call(image: bytes, mime: str, *, model=None, detail=None, prompt_version="v1"):
        seen.update(image=image, mime=mime, model=model, detail=detail)
        return valid_raw(), VisionUsage(
            model="fake",
            detail=detail or "high",
            prompt_tokens=1000,
            completion_tokens=200,
            cost_usd=0.001,
            latency_ms=5,
        )

    monkeypatch.setattr(extract, "call_vision_llm", fake_call)
    parsed = squad_from_image(path, bs, as_of=AS_OF, detail="low")

    assert seen["mime"] == "image/png" and seen["image"][:8] == b"\x89PNG\r\n\x1a\n"
    assert seen["detail"] == "low"
    assert parsed.is_valid and parsed.resolved_count == 15
    assert (
        parsed.usage is not None
        and parsed.usage.model == "fake"
        and parsed.usage.prompt_tokens == 1000
    )
    assert parsed.raw is not None and len(parsed.raw.players) == 15


# ---------- изображение ----------


def _png_bytes(size=(64, 32), mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, "green").save(buf, format="PNG")
    return buf.getvalue()


def test_load_image_passthrough_and_sniff(tmp_path):
    png = _png_bytes()
    assert sniff_mime(png) == "image/png"
    data, mime = load_image(png)
    assert data == png and mime == "image/png"

    buf = io.BytesIO()
    Image.new("RGB", (40, 40), "red").save(buf, format="JPEG")
    jpeg = buf.getvalue()
    assert sniff_mime(jpeg) == "image/jpeg"
    p = tmp_path / "a.jpg"
    p.write_bytes(jpeg)
    data, mime = load_image(str(p))
    assert data == jpeg and mime == "image/jpeg"


def test_load_image_downscales_huge_images():
    data, mime = load_image(_png_bytes(size=(3000, 600)), max_side=1500)
    assert mime == "image/png"
    with Image.open(io.BytesIO(data)) as im:
        assert im.size == (1500, 300)


def test_load_image_rejects_garbage():
    with pytest.raises(VisionExtractionError):
        load_image(b"not an image at all")
    with pytest.raises(VisionExtractionError):
        load_image(b"")


def test_data_url_and_messages():
    url = to_data_url(b"abc", "image/png")
    assert url == "data:image/png;base64,YWJj"
    msgs = build_messages(url, detail="low")
    assert msgs[0]["role"] == "system" and "Output only the JSON object" in msgs[0]["content"]
    assert msgs[1]["content"][0] == {"type": "text", "text": extract.USER_TEXT}
    image_part = msgs[1]["content"][1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"] == {"url": url, "detail": "low"}


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_system_prompts_have_key_rules(version):
    prompt = load_system_prompt(version)
    for needle in (
        "B.Fernandes",
        "João Pedro",
        "is_captain",
        "bench_order",
        "In the bank",
        "Never guess",
    ):
        assert needle in prompt
    if version == "v2":
        assert "`layout` first" in prompt and "substitutes strip" in prompt


def test_layout_is_first_field_in_llm_schema():
    """Порядок полей = порядок генерации: описание экрана должно идти до списка игроков."""
    assert list(ScreenshotSquadRaw.model_fields)[:2] == ["layout", "players"]


def test_default_prompt_version_is_v2():
    from fplcopilot.vision.extract import PROMPT_VERSION

    assert PROMPT_VERSION == "v2"
    assert build_messages("data:image/png;base64,YWJj", detail="high")[0]["content"] == (
        load_system_prompt("v2")
    )


def test_estimate_cost():
    inp, out = PRICES_PER_1M["gpt-4o-mini"]
    assert estimate_cost("gpt-4o-mini", 1_000_000, 0) == pytest.approx(inp)
    assert estimate_cost("gpt-4o-mini", 0, 1_000_000) == pytest.approx(out)
    assert estimate_cost("gpt-4o-mini-2024-07-18", 1_000_000, 0) == pytest.approx(
        inp
    )  # по префиксу
    assert estimate_cost("some-unknown-model", 1000, 1000) == 0.0


# ---------- рендер эталонов ----------


def test_render_samples_are_small_pngs(tmp_path):
    bs = make_bootstrap()
    paths = render_samples(bs, tmp_path)
    assert [p.name for p in paths] == ["pick_team_valid.png", "pick_team_broken.png"]
    for p in paths:
        assert p.stat().st_size < 200 * 1024
        with Image.open(p) as im:
            assert im.format == "PNG" and im.size[0] >= 800
    specs = sample_specs(bs)
    assert len(specs["pick_team_valid"].expected_names) == 15
    assert len(specs["pick_team_broken"].expected_names) == 14
    assert "João Pedro" in specs["pick_team_valid"].expected_names
    assert "B.Fernandes" in specs["pick_team_valid"].expected_names


def test_committed_samples_exist_and_are_small():
    for name in ("pick_team_valid.png", "pick_team_broken.png"):
        p = ROOT / "samples" / name
        assert p.exists(), p
        assert p.stat().st_size < 200 * 1024


# ---------- реальный вызов модели (uv run pytest -m llm tests/test_vision_pipeline.py) ----------


@pytest.mark.llm
@pytest.mark.network
def test_real_vision_call_on_valid_sample():
    from fplcopilot.config import settings
    from fplcopilot.data import FPLClient

    if not settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    bs = FPLClient().bootstrap()
    parsed = squad_from_image(ROOT / "samples" / "pick_team_valid.png", bs)
    assert parsed.usage is not None and parsed.usage.prompt_tokens > 0
    assert parsed.resolved_count >= 14
    assert parsed.captain_id is not None and bs.player(parsed.captain_id).web_name == "B.Fernandes"
    assert not parsed.blocking_issues
