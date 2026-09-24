"""Резолюция имён со скриншота против bootstrap: акценты, инициалы, неоднозначные фамилии, тай-брейки."""

from __future__ import annotations

import pytest
from test_vision_fixtures import make_bootstrap, raw

from fplcopilot.vision.resolve import (
    ACCEPT_SCORE,
    PlayerResolver,
    clean_shown_name,
    normalize,
    player_aliases,
    team_matches_hint,
)
from fplcopilot.vision.validate import validate_squad


@pytest.fixture(scope="module")
def resolver() -> PlayerResolver:
    return PlayerResolver.from_bootstrap(make_bootstrap())


# ---------- нормализация и алиасы ----------


def test_normalize_strips_accents_case_and_separators():
    assert normalize("João Pedro") == "joao pedro"
    assert normalize("B.Fernandes") == "b fernandes"
    assert normalize("B. Fernandes") == "b fernandes"
    assert normalize("Gibbs-White") == "gibbs white"
    assert normalize("O'Riley") == "o riley"
    assert normalize("  Ødegaard ") == "odegaard"


def test_clean_shown_name_strips_badges_and_prices():
    assert clean_shown_name("Haaland (C)") == "Haaland"
    assert clean_shown_name("Gvardiol £5.7m") == "Gvardiol"
    assert clean_shown_name("Saka (V) £9.5m") == "Saka"
    assert clean_shown_name("  Wood ") == "Wood"


def test_player_aliases_cover_expected_forms():
    bs = make_bootstrap()
    bruno = player_aliases(bs.player(426))
    assert {
        "b fernandes",
        "fernandes",
        "bruno fernandes",
        "bruno borges fernandes",
        "bruno",
    } <= bruno
    gabriel = player_aliases(bs.player(4))
    assert {"gabriel", "magalhaes", "gabriel magalhaes", "g magalhaes"} <= gabriel
    costinha = player_aliases(bs.player(119))
    assert "joao pedro" not in costinha  # многословное имя — не алиас: это web_name другого игрока
    assert "costinha" in costinha
    joao = player_aliases(bs.player(165))
    assert "joao pedro" in joao


# ---------- точные совпадения ----------


def test_accent_insensitive_joao_pedro(resolver: PlayerResolver):
    for shown in ("Joao Pedro", "João Pedro", "JOÃO PEDRO"):
        out = resolver.resolve(raw(shown))
        assert out.player_id == 165, shown
        assert out.match_method == "exact"
        assert out.match_confidence == 1.0
        assert out.web_name == "João Pedro" and out.team_short == "CHE" and out.position == "FWD"


def test_initial_dot_surname_b_fernandes(resolver: PlayerResolver):
    for shown in ("B.Fernandes", "B. Fernandes", "B Fernandes", "Bruno Fernandes"):
        out = resolver.resolve(raw(shown))
        assert out.player_id == 426, shown
        assert out.match_method == "exact"


def test_particles_in_surname(resolver: PlayerResolver):
    assert resolver.resolve(raw("Van Dijk")).player_id == 356
    assert resolver.resolve(raw("Virgil")).player_id == 356
    assert resolver.resolve(raw("van dijk")).player_id == 356


def test_web_name_with_initial(resolver: PlayerResolver):
    assert resolver.resolve(raw("J.Timber")).player_id == 5
    assert resolver.resolve(raw("G.Jesus")).player_id == 27


def test_noise_around_name_is_ignored(resolver: PlayerResolver):
    assert resolver.resolve(raw("Haaland (C)")).player_id == 411
    assert resolver.resolve(raw("Gvardiol £5.7m")).player_id == 391


# ---------- неоднозначность и тай-брейки ----------


def test_gabriel_without_hints_is_ambiguous(resolver: PlayerResolver):
    out = resolver.resolve(raw("Gabriel"))
    assert out.player_id is None
    assert out.match_method == "ambiguous"
    assert out.match_confidence == 0.0
    names = " ".join(out.candidates)
    for expected in (
        "Gabriel (ARS DEF",
        "Martinelli (ARS MID",
        "G.Jesus (ARS FWD",
        "Gudmundsson (LEE DEF",
    ):
        assert expected in names
    issues = validate_squad([out])
    assert any(
        i.code == "ambiguous"
        and i.blocking
        and i.message.startswith("ambiguous: Gabriel — candidates [")
        for i in issues
    )


def test_gabriel_position_alone_is_not_enough(resolver: PlayerResolver):
    out = resolver.resolve(raw("Gabriel", "DEF"))  # Gabriel (ARS) и Gudmundsson (LEE) — оба DEF
    assert out.match_method == "ambiguous"
    assert len(out.candidates) == 2


def test_gabriel_resolved_with_club_and_price(resolver: PlayerResolver):
    out = resolver.resolve(raw("Gabriel", club="ARS", price=8.0))
    assert out.player_id == 4
    assert out.match_method == "exact+hints"
    assert out.match_confidence == pytest.approx(0.9)

    out = resolver.resolve(raw("Gabriel", "DEF", club="Arsenal"))
    assert out.player_id == 4

    out = resolver.resolve(raw("Gabriel", "MID", club="ARS"))  # ряд MID -> Martinelli
    assert out.player_id == 18


def test_price_tolerance_breaks_tie(resolver: PlayerResolver):
    assert resolver.resolve(raw("Gabriel", price=8.2)).player_id == 4  # ±0.3
    assert resolver.resolve(raw("Gabriel", price=4.6)).player_id == 331
    assert (
        resolver.resolve(raw("Gabriel", price=7.0)).match_method == "ambiguous"
    )  # никому не подходит


def test_bare_surname_fernandes(resolver: PlayerResolver):
    out = resolver.resolve(raw("Fernandes"))
    assert out.match_method == "ambiguous"
    assert resolver.resolve(raw("Fernandes", club="MUN")).player_id == 426
    assert resolver.resolve(raw("Fernandes", club="Man United")).player_id == 426
    assert resolver.resolve(raw("Fernandes", price=5.8)).player_id == 525
    assert resolver.resolve(raw("Fernandes", club="Spurs")).player_id == 525


def test_silva_needs_club(resolver: PlayerResolver):
    assert resolver.resolve(raw("Silva")).match_method == "ambiguous"
    assert (
        resolver.resolve(raw("Silva", "DEF")).match_method == "ambiguous"
    )  # Silva (BOU), Morato (NFO)
    assert resolver.resolve(raw("Silva", "DEF", club="BOU")).player_id == 566


def test_same_web_name_two_players_split_by_position(resolver: PlayerResolver):
    assert resolver.resolve(raw("Palmer")).match_method == "ambiguous"
    assert resolver.resolve(raw("Palmer", "MID")).player_id == 154
    assert resolver.resolve(raw("Palmer", "GKP")).player_id == 301


def test_bare_first_name_pedro_is_ambiguous(resolver: PlayerResolver):
    out = resolver.resolve(raw("Pedro"))
    assert out.match_method == "ambiguous"
    assert resolver.resolve(raw("Pedro Porro")).player_id == 499


# ---------- нечёткие совпадения ----------


def test_fuzzy_typo_accepted_above_threshold(resolver: PlayerResolver):
    out = resolver.resolve(raw("Haland"))
    assert out.player_id == 411
    assert out.match_method == "fuzzy"
    assert out.match_confidence >= ACCEPT_SCORE / 100

    out = resolver.resolve(raw("Fernandez", club="MUN"))  # опечатка + подсказка клуба
    assert out.player_id == 426
    assert out.match_method == "fuzzy+hints"


def test_unknown_name_is_unresolved(resolver: PlayerResolver):
    out = resolver.resolve(raw("Zzyzx"))
    assert out.player_id is None
    assert out.match_method == "unresolved"
    assert out.candidates == []
    issues = validate_squad([out])
    assert any(i.code == "unresolved" and i.blocking for i in issues)


def test_empty_name_is_unresolved(resolver: PlayerResolver):
    assert resolver.resolve(raw("  ")).match_method == "unresolved"
    assert resolver.resolve(raw("£7.8m")).match_method == "unresolved"


def test_resolution_keeps_card_flags(resolver: PlayerResolver):
    out = resolver.resolve(raw("Sels", "GKP", price=5.0, bench=1, captain=False))
    assert out.is_bench and out.bench_order == 1 and out.position == "GKP"
    out = resolver.resolve(raw("B.Fernandes", "MID", captain=True))
    assert out.is_captain


# ---------- подсказка клуба ----------


@pytest.mark.parametrize(
    ("short", "name", "hint", "expected"),
    [
        ("MUN", "Man Utd", "MUN", True),
        ("MUN", "Man Utd", "Man Utd", True),
        ("MUN", "Man Utd", "Manchester United", True),
        ("MUN", "Man Utd", "man united", True),
        ("CHE", "Chelsea", "Chelsea FC", True),
        ("CHE", "Chelsea", "Chelsea", True),
        ("MCI", "Man City", "MUN", False),
        ("ARS", "Arsenal", "Aston Villa", False),
        ("ARS", "Arsenal", None, False),
        ("ARS", "Arsenal", "", False),
    ],
)
def test_team_matches_hint(short, name, hint, expected):
    assert team_matches_hint(short, name, hint) is expected
