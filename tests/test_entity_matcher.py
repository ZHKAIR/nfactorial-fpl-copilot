"""Матчер игроков/клубов на вымышленном Bootstrap (без сети)."""

import pytest

from fplcopilot.data.schemas import Bootstrap
from fplcopilot.rag.entity_matcher import EntityMatcher, strip_accents, tokenize

ARS, MCI, BOU, NFO, TOT, CHE, NEW = 1, 2, 3, 4, 5, 6, 7


def _player(pid: int, web: str, first: str, second: str, team: int) -> dict:
    return {
        "id": pid,
        "web_name": web,
        "first_name": first,
        "second_name": second,
        "team": team,
        "element_type": 3,
        "now_cost": 50,
    }


@pytest.fixture(scope="module")
def matcher() -> EntityMatcher:
    bs = Bootstrap.model_validate(
        {
            "events": [],
            "teams": [
                {"id": ARS, "name": "Arsenal", "short_name": "ARS"},
                {"id": MCI, "name": "Man City", "short_name": "MCI"},
                {"id": BOU, "name": "Bournemouth", "short_name": "BOU"},
                {"id": NFO, "name": "Nott'm Forest", "short_name": "NFO"},
                {"id": TOT, "name": "Spurs", "short_name": "TOT"},
                {"id": CHE, "name": "Chelsea", "short_name": "CHE"},
                {"id": NEW, "name": "Newcastle", "short_name": "NEW"},
            ],
            "elements": [
                _player(12, "Saka", "Bukayo", "Saka", ARS),
                _player(4, "Gabriel", "Gabriel", "dos Santos Magalhães", ARS),
                _player(18, "Martinelli", "Gabriel", "Martinelli Silva", ARS),
                _player(27, "G.Jesus", "Gabriel", "Fernando de Jesus", ARS),
                _player(8, "Ødegaard", "Martin", "Ødegaard", ARS),
                _player(10, "White", "Benjamin", "White", ARS),
                _player(13, "Rice", "Declan", "Rice", ARS),
                _player(200, "Bernardo", "Bernardo", "Veiga de Carvalho e Silva", MCI),
                _player(
                    566, "Silva", "António João", "Pereira de Albuquerque Tavares da Silva", BOU
                ),
                _player(480, "Gibbs-White", "Morgan", "Gibbs-White", NFO),
                _player(490, "Wood", "Chris", "Wood", NFO),
                _player(165, "João Pedro", "João Pedro", "Junqueira de Jesus", CHE),
                _player(300, "Palmer", "Cole", "Palmer", CHE),
                _player(301, "Palmer", "Alex", "Palmer", NEW),
                _player(452, "Bruno G.", "Bruno", "Guimarães Rodriguez Moura", NEW),
                _player(426, "B.Fernandes", "Bruno", "Borges Fernandes", TOT),
                _player(155, "Enzo", "Enzo", "Fernández", MCI),
                _player(385, "Trafford", "James", "Trafford", NEW),
            ],
        }
    )
    return EntityMatcher.from_bootstrap(bs)


def test_strip_accents_and_tokenize():
    assert strip_accents("Ødegaard João Kadıoğlu Müller") == "Odegaard Joao Kadioglu Muller"
    assert tokenize("Gibbs-White's return: Nott'm Forest!") == [
        "Gibbs", "White", "s", "return", "Nott", "m", "Forest",
    ]  # fmt: skip


def test_unique_surname_is_matched(matcher):
    r = matcher.match("Saka limped off and is a doubt for Gameweek 5.")
    assert r.players == [12]
    assert r.teams == []


def test_accent_insensitive_match(matcher):
    assert matcher.match("Odegaard returned to training").players == [8]
    assert matcher.match("Ødegaard returned to training").players == [8]
    assert matcher.match("Joao Pedro out of Brazil squad").players == [165]


def test_full_name_always_wins(matcher):
    assert matcher.match("Bukayo Saka scores").players == [12]
    assert matcher.match("Gabriel Magalhaes is a doubt").players == [4]
    assert matcher.match("Gabriel dos Santos Magalhães is a doubt").players == [4]
    assert matcher.match("Mohamed Bruno Guimaraes").players == [452]


def test_ambiguous_surname_not_matched_without_team(matcher):
    assert matcher.match("Silva is fit again.").players == []
    assert matcher.match("Palmer scored twice.").players == []


def test_ambiguous_surname_matched_with_team_context(matcher):
    r = matcher.match("Bournemouth defender Silva is fit again.")
    assert r.players == [566]
    assert r.teams == [BOU]
    assert matcher.match("Silva is a doubt for Man City").players == [200]
    assert matcher.match("Chelsea's Palmer scored twice.").players == [300]


def test_ambiguous_surname_still_skipped_if_two_candidates_share_the_team(matcher):
    # "Gabriel" — web_name Магальяеса и first_name Мартинелли/Жезуса: все трое в Arsenal
    assert matcher.match("Arsenal's Gabriel is a doubt.").players == []
    assert matcher.match("Gabriel is a doubt.").players == []


def test_first_name_collision_resolved_by_full_name_only(matcher):
    r = matcher.match("Gabriel Martinelli is back. Gabriel trained fully.")
    assert r.players == [18]  # второй "Gabriel" не приписывается Магальяесу


def test_initials_in_web_name_are_stripped(matcher):
    assert matcher.match("Fernandes late winner for Spurs").players == [426]
    assert matcher.match("Jesus returns for Arsenal").players == [27]


def test_common_word_surname_needs_team_context(matcher):
    assert matcher.match("White is back in training.").players == []
    assert matcher.match("Arsenal's White is back in training.").players == [10]
    assert matcher.match("Wood scores as Forest win").players == [490]
    assert matcher.match("Rice was pushed higher up the pitch").players == []
    assert matcher.match("Declan Rice was pushed higher up the pitch").players == [13]


def test_team_nicknames_and_case_sensitivity(matcher):
    r = matcher.match("Spurs and Forest drew, Man City beat the Gunners; Toon and Cherries lost.")
    assert r.teams == [ARS, MCI, BOU, NFO, TOT, NEW]
    assert matcher.match("the wolves ran through the forest").teams == []
    assert matcher.match("FOREST WIN AT SPURS").teams == [NFO, TOT]


def test_short_codes_only_uppercase_and_not_common_words(matcher):
    assert matcher.match("ARS v MCI preview").teams == [ARS, MCI]
    assert matcher.match("a new signing, ars technica, tot").teams == []
    assert matcher.match("NEW SIGNING arrives").teams == []  # NEW в стоп-листе кодов


def test_adjacent_proper_noun_means_someone_else(matcher):
    # менеджеры/стадионы с тем же именем, что у игрока
    assert matcher.match("Enzo Maresca praised his Chelsea side.").players == []
    assert matcher.match("Enzo Fernandez scored for Man City.").players == [155]
    assert matcher.match("Enzo scored for Man City.").players == [155]
    assert matcher.match("United were poor at Old Trafford again.").players == []
    assert matcher.match("Trafford saved twice for Newcastle.").players == [385]
    # неоднозначная фамилия + контекст клуба всё равно не спасает "Marco Silva"
    assert matcher.match("Marco Silva press conference: Fulham vs Bournemouth").players == []


def test_adjacent_rule_exemptions(matcher):
    # запятые между фамилиями — граница, список имён работает
    assert matcher.match("Saka, Odegaard, Martinelli appeal").players == [8, 12, 18]
    # Title Case заголовки: обычные слова заголовка не считаются чужим именем
    assert matcher.match("Saka Injury Update Ahead Of Gameweek 5").players == [12]
    # соседнее слово — название клуба
    assert matcher.match("Saka Arsenal contract talks").players == [12]
    # ALL-CAPS заголовок
    assert matcher.match("SAKA OUT FOR SIX WEEKS").players == [12]
    # перенос строки между заголовком и текстом — граница
    assert matcher.match("Saka\nArsenal winger trained").players == [12]


def test_hyphenated_and_compound_names(matcher):
    r = matcher.match("Gibbs-White limps off for Nott'm Forest")
    assert r.players == [480]
    assert r.teams == [NFO]
    # без клуба "White" из "Gibbs-White" не утекает к Бену Уайту
    assert matcher.match("Gibbs-White limps off").players == [480]
