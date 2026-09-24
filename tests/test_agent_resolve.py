"""Детерминированное разрешение имён игроков (agent/resolve.py) на синтетическом bootstrap."""

from __future__ import annotations

from test_agent_fakes import BS, SQUAD_IDS

from fplcopilot.agent.resolve import PlayerResolver, name_tokens


def test_name_tokens_strip_accents_possessives_and_initials():
    assert name_tokens("João Pedro's") == ("joao", "pedro")
    assert name_tokens("M.Salah") == ("salah",)
    assert name_tokens("Gibbs-White") == ("gibbs", "white")


def test_unique_surname_resolves():
    res = PlayerResolver(BS).resolve("Saka")
    assert res.status == "resolved" and res.player["id"] == 12
    # Wan-Bissaka и Sakamoto не подходят: матчинг по токенам, а не по подстроке
    assert PlayerResolver(BS).resolve("Guéhi").player["id"] == 388
    assert PlayerResolver(BS).resolve("guehi").player["id"] == 388


def test_first_name_shared_by_several_players_is_ambiguous():
    res = PlayerResolver(BS, squad_ids=SQUAD_IDS).resolve("Gabriel")
    assert res.status == "ambiguous"
    ids = [c["id"] for c in res.candidates]
    assert set(ids) == {4, 18, 27, 331}
    assert res.candidates[0]["id"] == 4 and res.candidates[0]["in_squad"]  # свой — первым
    assert "first name" in (res.note or "")
    assert PlayerResolver(BS).resolve("Pedro").status == "ambiguous"


def test_surname_collision_resolved_by_squad_membership():
    res = PlayerResolver(BS, squad_ids=SQUAD_IDS).resolve("Palmer")
    assert res.status == "resolved" and res.player["id"] == 154
    assert "in your squad" in (res.note or "")


def test_surname_collision_without_squad_uses_ownership_dominance():
    res = PlayerResolver(BS).resolve("Palmer")  # Cole 26.8% против Alex 4.0%
    assert res.status == "resolved" and res.player["id"] == 154
    assert "assumed" in (res.note or "")


def test_full_name_and_unknown():
    r = PlayerResolver(BS)
    assert r.resolve("Alex Palmer").player["id"] == 301
    assert r.resolve("Cole Palmer").player["id"] == 154
    assert r.resolve("Gabriel Martinelli").player["id"] == 18
    assert r.resolve("Nobody Here").status == "unknown"


def test_resolve_all_uses_entity_matcher_and_router_mentions():
    r = PlayerResolver(BS, squad_ids=SQUAD_IDS)
    players, ambiguous, unknown, notes = r.resolve_all(
        ["João Pedro", "Gabriel", "Zzyzx"], "Is João Pedro fit? What about Gabriel and Zzyzx?"
    )
    assert [p["id"] for p in players] == [165]
    assert [a.mention for a in ambiguous] == ["Gabriel"]
    assert [u.mention for u in unknown] == ["Zzyzx"]
    assert notes == []
