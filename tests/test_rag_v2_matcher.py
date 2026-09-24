"""Матчер v2: границы слов (Egan/Keegan, Sels/Brussels), обычные слова из выведенного списка
("White" требует контекста, "Ben White"/"B. White"/"Arsenal's White" — нет), text_mentions_player
и chunk_mentions_player с EntityMatcher. Без сети."""

from datetime import UTC, datetime

import pytest

from fplcopilot.data.schemas import Bootstrap
from fplcopilot.rag.entity_matcher import (
    EntityMatcher,
    common_english_words,
    is_common_word_surname,
    text_mentions_player,
)
from fplcopilot.rag.extract import chunk_mentions_player
from fplcopilot.rag.retrieve import RetrievedChunk

ARS, NFO, SHU, CHE, TOT = 1, 4, 8, 6, 5


def _player(pid, web, first, second, team, pos=3):
    return {
        "id": pid, "web_name": web, "first_name": first, "second_name": second,
        "team": team, "element_type": pos, "now_cost": 50,
    }  # fmt: skip


@pytest.fixture(scope="module")
def bs() -> Bootstrap:
    return Bootstrap.model_validate(
        {
            "events": [],
            "teams": [
                {"id": ARS, "name": "Arsenal", "short_name": "ARS"},
                {"id": NFO, "name": "Nott'm Forest", "short_name": "NFO"},
                {"id": SHU, "name": "Sheffield Utd", "short_name": "SHU"},
                {"id": CHE, "name": "Chelsea", "short_name": "CHE"},
                {"id": TOT, "name": "Spurs", "short_name": "TOT"},
            ],
            "elements": [
                _player(10, "White", "Benjamin", "White", ARS, 2),
                _player(12, "Saka", "Bukayo", "Saka", ARS),
                _player(600, "Egan", "John", "Egan", SHU, 2),
                _player(601, "Sels", "Matz", "Sels", NFO, 1),
                _player(142, "James", "Reece", "James", CHE, 2),
                _player(480, "Gibbs-White", "Morgan", "Gibbs-White", NFO),
                _player(4, "Gabriel", "Gabriel", "dos Santos Magalhães", ARS, 2),
                _player(496, "Kinský", "Antonín", "Kinský", TOT, 1),
            ],
        }
    )


@pytest.fixture(scope="module")
def matcher(bs) -> EntityMatcher:
    return EntityMatcher.from_bootstrap(bs)


def test_common_word_list_is_derived_not_hand_written():
    words = common_english_words()
    assert len(words) > 5000
    for w in ("white", "james", "hall", "wood", "grant", "king", "young", "long", "best", "timber"):
        assert is_common_word_surname(w), w
    for w in ("saka", "egan", "sels", "haaland", "doku", "kinsky"):
        assert not is_common_word_surname(w), w
    assert is_common_word_surname("Groß")  # диакритика снимается -> "gross"


def test_word_boundaries_no_substring_hits(matcher, bs):
    assert matcher.match("Keegan praised the Sheffield United defence.").players == []
    assert matcher.match("Egan limped off for Sheffield Utd.").players == [600]
    assert matcher.match("The final is in Brussels next May.").players == []
    assert matcher.match("Sels saved a penalty for Forest.").players == [601]
    egan, sels = bs.player(600), bs.player(601)
    assert not text_mentions_player("Keegan praised the defence", egan)
    assert text_mentions_player("Egan limped off", egan)
    assert not text_mentions_player("the final is in Brussels", sels)
    assert text_mentions_player("Sels saved twice", sels)


def test_white_needs_context_ben_white_matches(matcher, bs):
    white, arsenal = bs.player(10), bs.team(ARS)
    # тегирование статей
    assert matcher.match("White is back in training.").players == []
    assert matcher.match("Ben White is back in training.").players == [10]
    assert matcher.match("B. White is back in training.").players == [10]
    assert matcher.match("Arsenal's White is back in training.").players == [10]
    assert matcher.match("Fans in white T-shirts filled the stand.").players == []
    # проверка «чанк про игрока»
    assert not text_mentions_player("White missed training on Thursday", white)
    assert text_mentions_player("Ben White missed training on Thursday", white)
    assert text_mentions_player("Benjamin White missed training", white)
    assert text_mentions_player("B White missed training", white)
    assert text_mentions_player("White missed training; Arsenal face Brighton", white, arsenal)
    assert text_mentions_player("Gunners defender White is a doubt", white, arsenal)
    assert not text_mentions_player("White missed training; Arsenal face Brighton", white, None)
    # дефисная фамилия — один токен: "Gibbs-White" не упоминание Бена Уайта даже с Arsenal
    assert not text_mentions_player("Gibbs-White scored against Arsenal", white, arsenal)
    assert text_mentions_player("Gibbs-White scored against Arsenal", bs.player(480))


def test_unique_surname_needs_no_context_and_first_name_only_web_name_does(bs):
    saka, gabriel = bs.player(12), bs.player(4)
    assert text_mentions_player("Saka trained fully", saka)
    assert text_mentions_player("SAKA OUT FOR SIX WEEKS", saka)  # регистр не важен
    assert not text_mentions_player("Sakai trained fully", saka)  # граница слова
    # web_name "Gabriel" = имя -> сам по себе не упоминание; фамилия или полное имя — да
    assert not text_mentions_player("Gabriel joined United from Brazil", gabriel)
    assert text_mentions_player("Gabriel Magalhaes was rested", gabriel)
    assert text_mentions_player("Magalhães was rested", gabriel)
    assert not text_mentions_player(
        "Andrey Santos signed for Chelsea", gabriel
    )  # "Santos" не алиас


def test_chunk_mentions_player_prefers_tag_then_matcher(matcher, bs):
    kinsky, spurs = bs.player(496), bs.team(TOT)
    james, chelsea = bs.player(142), bs.team(CHE)
    now = datetime(2026, 9, 17, tzinfo=UTC)

    def ch(text, players=()):
        return RetrievedChunk(
            chunk_id=1, article_id=1, text=text, source="x", url="u", title="t",
            published_at=now, players=list(players),
        )  # fmt: skip

    assert chunk_mentions_player(
        ch("anything", players=[496]), kinsky, spurs, matcher
    )  # тег решает
    assert not chunk_mentions_player(
        ch("Spurs injury update: Tonali, Porro latest"), kinsky, spurs, matcher
    )
    assert chunk_mentions_player(ch("Kinsky kept a clean sheet"), kinsky, spurs, matcher)
    assert chunk_mentions_player(ch("Kinský kept a clean sheet"), kinsky, None, None)  # fallback
    # "James" — обычное слово: James Garner в чанке про Челси не превращается в Риса Джеймса
    assert not chunk_mentions_player(
        ch("Chelsea host Everton; James Garner is fit"), james, chelsea, matcher
    )
    assert chunk_mentions_player(
        ch("Reece James was not seen in training"), james, chelsea, matcher
    )
    assert chunk_mentions_player(
        ch("Chelsea's James was not seen in training"), james, chelsea, matcher
    )
