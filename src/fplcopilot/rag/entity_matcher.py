"""Детерминированный поиск игроков и клубов FPL в тексте новости (без LLM).

Алгоритм:
1. Текст очищаем от диакритики (NFKD: "Ødegaard" -> "Odegaard", "João" -> "Joao"),
   пунктуацию считаем разделителем, регистр СОХРАНЯЕМ.
2. Идём по токенам слева направо и ищем n-граммы (от длинных к коротким, без пересечений)
   в словаре алиасов. Алиасы клуба: name/short_name из API + прозвища (TEAM_ALIASES).
   Алиасы игрока: web_name, фамилия (second_name целиком и её последняя/первая часть),
   полные имена ("Bukayo Saka", "Gabriel Magalhaes", "Gabriel Martinelli").
3. Регистр: заглавная буква алиаса требует заглавной в тексте — "Wolves"/"Forest"/"White"
   не срабатывают на "wolves"/"forest"/"white". Строчная буква алиаса принимает любой регистр,
   поэтому ALL-CAPS заголовки и "Van Dijk" в начале предложения проходят.
4. Правило неоднозначности. Алиас считается неоднозначным, если указывает на >1 игрока
   (общая фамилия "Silva"; имя "Gabriel", которое у одного игрока web_name, а у других — first_name)
   или является обычным английским словом ("White", "Wood", "Rice", "James", "Grant", "King").
   Список обычных слов ВЫВОДИТСЯ, а не пишется руками: ручной стоп-лист COMMON_WORD_SURNAMES
   плюс data/english_common_words.txt (~9.5k самых частых слов английского веба, google-10000-english;
   образовательное использование). Такой алиас присваивается игроку ТОЛЬКО если выполняется одно из:
     а) в тексте есть его полное имя ("Gabriel Magalhaes") или инициал + фамилия ("B. White");
     б) ровно один из кандидатов уже найден в этом тексте по однозначному алиасу;
     в) ровно один кандидат играет за клуб, упомянутый в этом же тексте.
   Иначе алиас игнорируется — лучше пропустить упоминание, чем приписать травму не тому игроку.
5. Чужие имена. Однословный алиас, к которому через один пробел примыкает другое слово с заглавной
   ("Enzo Maresca", "Old Trafford", "Marco Silva", "Keith Andrews"), считается упоминанием другого
   человека/объекта и пропускается. Исключения: соседнее слово — часть названия клуба, часть имени
   самого игрока, ALL-CAPS, или обычное слово заголовка (HEADLINE_WORDS: "Injury", "Update", "Boost").
   Знаки препинания между словами ("Doku, Sarr, Foden") снимают правило.
6. Границы слов везде. Матчинг идёт по токенам, поэтому "Egan" не находится в "Keegan", а "Sels" —
   в "Brussels". Та же логика вынесена в text_mentions_player() — проверка «чанк про игрока» для
   валидации цитат (rag/extract.py) вместо прежнего поиска подстроки.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from fplcopilot.data.schemas import Bootstrap, Player, Team

COMMON_WORDS_FILE = Path(__file__).resolve().parent / "data" / "english_common_words.txt"

# Прозвища и полные названия по стабильному short_name (клубы меняются по сезонам,
# лишние ключи безвредны). Намеренно нет "United", "City", "Blues", "Reds" — неоднозначны.
TEAM_ALIASES: dict[str, tuple[str, ...]] = {
    "ARS": ("Arsenal", "Gunners"),
    "AVL": ("Aston Villa", "Villa", "Villans"),
    "BOU": ("Bournemouth", "AFC Bournemouth", "Cherries"),
    "BRE": ("Brentford", "Bees"),
    "BHA": ("Brighton", "Brighton & Hove Albion", "Brighton and Hove Albion", "Seagulls"),
    "BUR": ("Burnley", "Clarets"),
    "CHE": ("Chelsea",),
    "COV": ("Coventry", "Coventry City", "Sky Blues"),
    "CRY": ("Crystal Palace", "Palace", "Eagles"),
    "EVE": ("Everton", "Toffees"),
    "FUL": ("Fulham", "Cottagers"),
    "HUL": ("Hull", "Hull City", "Tigers"),
    "IPS": ("Ipswich", "Ipswich Town", "Tractor Boys"),
    "LEE": ("Leeds", "Leeds United"),
    "LEI": ("Leicester", "Leicester City", "Foxes"),
    "LIV": ("Liverpool",),
    "LUT": ("Luton", "Luton Town", "Hatters"),
    "MCI": ("Man City", "Manchester City", "Citizens"),
    "MUN": ("Man Utd", "Man United", "Manchester United", "Red Devils"),
    "NEW": ("Newcastle", "Newcastle United", "Magpies", "Toon"),
    "NFO": ("Nott'm Forest", "Nottm Forest", "Nottingham Forest", "Forest"),
    "SHU": ("Sheffield United", "Sheffield Utd", "Blades"),
    "SOU": ("Southampton", "Saints"),
    "SUN": ("Sunderland", "Black Cats"),
    "TOT": ("Spurs", "Tottenham", "Tottenham Hotspur"),
    "WHU": ("West Ham", "West Ham United", "Hammers", "Irons"),
    "WOL": ("Wolves", "Wolverhampton", "Wolverhampton Wanderers"),
}

# Трёхбуквенные коды, совпадающие с обычными словами в ALL-CAPS заголовках, — не используем.
SHORT_CODE_STOPLIST = frozenset({"NEW", "SUN", "EVE", "TOT", "CRY"})

# Фамилии-омонимы обычных слов: даже с заглавной буквы (начало предложения, заголовок)
# требуют контекста клуба или полного имени. Ручной список — минимум; основной источник —
# data/english_common_words.txt (см. common_english_words / is_common_word_surname).
COMMON_WORD_SURNAMES = frozenset(
    {
        "white", "wood", "young", "king", "hill", "cook", "grant", "price", "ward", "bell",
        "long", "rice", "gray", "grey", "green", "brown", "black", "walker", "hall", "best",
        "day", "park", "rose", "march", "may", "case", "chance", "wright", "stone", "cash",
        "hope", "love", "little", "small", "strong", "bright", "summer", "winter", "forward",
        "back", "field", "bridge", "wall", "ball", "goal", "kick", "shot", "save", "cross",
        "hunt", "fox", "wolf", "lion", "bird", "marsh", "moore", "rich", "power", "nelson",
    }
)  # fmt: skip

# Слова с заглавной, которые часто стоят рядом с фамилией в заголовках и НЕ означают другого человека.
HEADLINE_WORDS = frozenset(
    {
        "injury", "injured", "injuries", "update", "updates", "news", "latest", "fpl", "gw",
        "gameweek", "premier", "league", "fantasy", "return", "returns", "boost", "blow", "doubt",
        "doubtful", "fit", "fitness", "out", "in", "on", "off", "for", "to", "and", "or", "the",
        "a", "an", "is", "are", "was", "has", "have", "will", "set", "could", "ready", "back",
        "scores", "signs", "starts", "start", "ruled", "suffers", "misses", "sidelined", "facing",
        "faces", "verdict", "reveals", "explains", "confirms", "provides", "press", "conference",
        "captain", "captaincy", "transfer", "transfers", "deal", "move", "cup", "carabao",
        "champions", "europa", "world", "euro", "euros", "nations", "international", "v", "vs",
        "live", "video", "watch", "highlights", "preview", "review", "report", "analysis",
        "reaction", "team", "squad", "xi", "lineup", "lineups", "predicted", "tips", "picks",
        "differential", "differentials", "price", "rise", "rises", "fall", "falls", "penalty",
        "goal", "goals", "assist", "assists", "hat", "trick", "red", "yellow", "card", "ban",
        "banned", "suspended", "suspension", "england", "brazil", "portugal", "spain", "wales",
        "scotland", "ireland", "france", "germany", "italy", "netherlands", "argentina",
        "nigeria", "ghana", "senegal", "denmark", "norway", "sweden", "belgium", "croatia",
        "uruguay", "colombia", "japan", "egypt", "morocco", "cameroon", "monday", "tuesday",
        "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february", "march",
        "april", "may", "june", "july", "august", "september", "october", "november",
        "december", "i", "we", "he", "they", "it", "after", "before", "as", "at", "with",
        "from", "but", "who", "why", "how", "what", "when", "where", "which", "not", "no",
        "yes", "new", "big", "best", "top", "first", "last", "next", "this", "that", "his",
        "her", "their", "our", "my", "ahead", "of", "against", "over", "under", "into", "than",
        "still", "now", "here", "there", "one", "two", "three",
    }
)  # fmt: skip

MIN_ALIAS_LEN = 3  # однословные алиасы короче — не индексируем ("Ba", "Li")

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_NO_DECOMPOSITION = str.maketrans(
    {
        "ø": "o", "Ø": "O", "ł": "l", "Ł": "L", "ß": "ss", "æ": "ae", "Æ": "Ae",
        "œ": "oe", "Œ": "Oe", "ð": "d", "Ð": "D", "þ": "th", "Þ": "Th", "đ": "d",
        "Đ": "D", "ı": "i",
    }
)  # fmt: skip


def strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).translate(_NO_DECOMPOSITION)


def tokenize(text: str) -> list[str]:
    """Токены без диакритики и пунктуации, регистр сохранён."""
    return _TOKEN_RE.findall(strip_accents(text))


def _tokenize_with_spans(text: str) -> tuple[str, list[str], list[tuple[int, int]]]:
    stripped = strip_accents(text)
    matches = list(_TOKEN_RE.finditer(stripped))
    return stripped, [m.group() for m in matches], [(m.start(), m.end()) for m in matches]


def _case_ok(alias_tok: str, text_tok: str) -> bool:
    if len(alias_tok) != len(text_tok):
        return False
    for a, t in zip(alias_tok, text_tok, strict=True):
        if a.isupper():
            if a != t:
                return False
        elif a.lower() != t.lower():
            return False
    return True


@dataclass(frozen=True)
class _Alias:
    tokens: tuple[str, ...]
    entity_id: int
    full: bool = False  # полное имя игрока: снимает неоднозначность само по себе


@dataclass(frozen=True)
class _Hit:
    start: int  # индекс первого токена
    n: int  # длина n-граммы
    aliases: list[_Alias]

    @property
    def positions(self) -> range:
        return range(self.start, self.start + self.n)


@dataclass
class MatchResult:
    players: list[int] = field(default_factory=list)
    teams: list[int] = field(default_factory=list)


class _Index:
    def __init__(self) -> None:
        self.by_key: dict[tuple[str, ...], list[_Alias]] = {}
        self.max_n = 1

    def add(self, toks: Iterable[str], entity_id: int, *, full: bool = False) -> None:
        toks = tuple(toks)
        if not toks or (len(toks) == 1 and len(toks[0]) < MIN_ALIAS_LEN):
            return
        alias = _Alias(toks, entity_id, full)
        bucket = self.by_key.setdefault(tuple(t.lower() for t in toks), [])
        if alias not in bucket:
            bucket.append(alias)
        self.max_n = max(self.max_n, len(toks))

    def scan(self, toks: list[str]) -> list[_Hit]:
        """Жадный поиск n-грамм слева направо; каждая позиция текста используется один раз."""
        hits: list[_Hit] = []
        i, n_tokens = 0, len(toks)
        while i < n_tokens:
            advanced = False
            for n in range(min(self.max_n, n_tokens - i), 0, -1):
                window = toks[i : i + n]
                bucket = self.by_key.get(tuple(t.lower() for t in window))
                if not bucket:
                    continue
                matched = [
                    a
                    for a in bucket
                    if all(_case_ok(x, y) for x, y in zip(a.tokens, window, strict=True))
                ]
                if matched:
                    hits.append(_Hit(i, n, matched))
                    i += n
                    advanced = True
                    break
            if not advanced:
                i += 1
        return hits


MIN_NICKNAME_LEN = 3  # "Ben" ⊂ "Benjamin", "Matt" ⊂ "Matthew": префикс имени считается именем


def first_name_matches(token: str, first_tokens: Iterable[str]) -> bool:
    """Токен — имя игрока, его префикс-уменьшительное (>= MIN_NICKNAME_LEN) или инициал."""
    t = strip_accents(token).lower()
    for f in first_tokens:
        f = strip_accents(f).lower()
        if (
            t == f
            or (len(t) >= MIN_NICKNAME_LEN and f.startswith(t))
            or (len(t) == 1 and f[:1] == t)
        ):
            return True
    return False


class EntityMatcher:
    def __init__(self, teams: Iterable[Team], players: Iterable[Player]) -> None:
        self._teams = _Index()
        self._players = _Index()
        self._team_of: dict[int, int] = {}
        self._first_of: dict[int, tuple[str, ...]] = {}
        self._own_cache: dict[frozenset[int], set[str]] = {}
        for t in teams:
            self._add_team(t)
        players = list(players)
        for p in players:
            self._team_of[p.id] = p.team
            self._first_of[p.id] = tuple(tokenize(p.first_name))
            self._add_player(p)
        self._add_first_name_collisions(players)

    @classmethod
    def from_bootstrap(cls, bs: Bootstrap) -> EntityMatcher:
        return cls(bs.teams, bs.elements)

    # ---------- построение словаря ----------

    def _add_team(self, t: Team) -> None:
        self._teams.add(tokenize(t.name), t.id)
        if t.short_name.upper() not in SHORT_CODE_STOPLIST:
            self._teams.add((t.short_name.upper(),), t.id)
        for alias in TEAM_ALIASES.get(t.short_name.upper(), ()):
            self._teams.add(tokenize(alias), t.id)

    def _add_player(self, p: Player) -> None:
        idx = self._players
        first, second, web = tokenize(p.first_name), tokenize(p.second_name), tokenize(p.web_name)

        idx.add(web, p.id)  # "Saka", "Joao Pedro", "M Salah", "Bruno G"
        without_initials = [
            t for t in web if len(t) > 1
        ]  # "M Salah" -> "Salah", "P M Sarr" -> "Sarr"
        if without_initials and len(without_initials) < len(web):
            idx.add(without_initials, p.id)

        caps = [t for t in second if t[0].isupper()]  # без частиц "de", "van", "dos"
        if second:
            idx.add(second, p.id)  # "dos Santos Magalhaes"
        if caps:
            idx.add((caps[-1],), p.id)  # "Magalhaes", "Silva"
            idx.add((caps[0],), p.id)  # "Martinelli" из "Martinelli Silva"

        if not first:
            return
        if second:
            idx.add(first + second, p.id, full=True)  # официальное полное имя
        for surname in dict.fromkeys(caps[-1:] + caps[:1]):
            idx.add([*first, surname], p.id, full=True)  # "Gabriel Magalhaes", "Bruno Guimaraes"
            idx.add([first[0][0], surname], p.id, full=True)  # инициал: "B White", "B. White"
        if len(web) == 1 and web[0].lower() not in {t.lower() for t in [*first, *caps]}:
            idx.add([*first, web[0]], p.id, full=True)  # "Gabriel Martinelli" при web="Martinelli"

    def _add_first_name_collisions(self, players: list[Player]) -> None:
        """Однословный алиас, равный имени других игроков, становится неоднозначным ("Gabriel")."""
        by_first: dict[str, list[tuple[str, int]]] = {}
        for p in players:
            ft = tokenize(p.first_name)
            if len(ft) == 1 and len(ft[0]) >= MIN_ALIAS_LEN:
                by_first.setdefault(ft[0].lower(), []).append((ft[0], p.id))
        for key, bucket in list(self._players.by_key.items()):
            if len(key) != 1:
                continue
            present = {a.entity_id for a in bucket}
            for tok, pid in by_first.get(key[0], []):
                if pid not in present:
                    bucket.append(_Alias((tok,), pid))

    # ---------- поиск ----------

    def match(self, text: str) -> MatchResult:
        stripped, toks, spans = _tokenize_with_spans(text)

        team_hits = self._teams.scan(toks)
        teams = {a.entity_id for hit in team_hits for a in hit.aliases}
        team_pos = {i for hit in team_hits for i in hit.positions}

        player_hits = self._players.scan(toks)
        player_pos = {i for hit in player_hits for i in hit.positions}

        sure: set[int] = set()
        pending: list[set[int]] = []
        for hit in player_hits:
            ids = {a.entity_id for a in hit.aliases}
            full_ids = {a.entity_id for a in hit.aliases if a.full}
            if len(full_ids) == 1:
                sure |= full_ids
                continue
            if hit.n == 1:
                named = self._named_by_first_name(hit, toks, spans, stripped, ids)
                if len(named) == 1:  # "Ben White": уменьшительное имя перед фамилией
                    sure |= named
                    continue
            if hit.n == 1 and self._adjacent_proper_noun(
                hit, toks, spans, stripped, exempt=team_pos | player_pos, ids=ids
            ):
                continue  # "Enzo Maresca", "Old Trafford" — не наш игрок
            if len(ids) == 1 and not _is_common_word(hit.aliases[0].tokens):
                sure |= ids
            else:
                pending.append(ids)

        for ids in pending:
            if ids & sure:  # кандидат(ы) уже найдены однозначно — ничего не добавляем
                continue
            by_team = {pid for pid in ids if self._team_of.get(pid) in teams}
            if len(by_team) == 1:
                sure |= by_team

        return MatchResult(players=sorted(sure), teams=sorted(teams))

    @staticmethod
    def _space_adjacent(
        i: int, j: int, toks: list[str], spans: list[tuple[int, int]], stripped: str
    ) -> bool:
        """Токены i и j разделены только пробелами (без знаков препинания и переносов)."""
        if j < 0 or j >= len(toks):
            return False
        lo, hi = (spans[j][1], spans[i][0]) if j < i else (spans[i][1], spans[j][0])
        return stripped[lo:hi].strip(" ") == ""

    def _named_by_first_name(
        self,
        hit: _Hit,
        toks: list[str],
        spans: list[tuple[int, int]],
        stripped: str,
        ids: set[int],
    ) -> set[int]:
        """Кандидаты, чьё имя (или уменьшительное/инициал) стоит прямо перед фамилией."""
        j = hit.start - 1
        if not self._space_adjacent(hit.start, j, toks, spans, stripped):
            return set()
        nb = toks[j]
        if not nb[:1].isupper():
            return set()
        return {pid for pid in ids if first_name_matches(nb, self._first_of.get(pid, ()))}

    def _adjacent_proper_noun(
        self,
        hit: _Hit,
        toks: list[str],
        spans: list[tuple[int, int]],
        stripped: str,
        *,
        exempt: set[int],
        ids: set[int],
    ) -> bool:
        """Рядом (через пробелы) стоит чужое слово с заглавной: "Enzo Maresca", "Old Trafford"."""
        i = hit.start
        own = self._own_tokens(ids)
        for j in (i - 1, i + 1):
            if j in exempt or not self._space_adjacent(i, j, toks, spans, stripped):
                continue
            nb = toks[j]
            if len(nb) < 2 or not nb[0].isupper() or nb.isupper():
                continue  # инициалы, строчные, ALL-CAPS не считаем
            if nb.lower() in own or nb.lower() in HEADLINE_WORDS:
                continue
            if j == i - 1 and any(
                first_name_matches(nb, self._first_of.get(pid, ())) for pid in ids
            ):
                continue  # "Ben White": уменьшительное имя самого игрока
            return True
        return False

    def _own_tokens(self, ids: set[int]) -> set[str]:
        """Все токены всех алиасов игроков-кандидатов ("Bukayo", "Saka")."""
        key = frozenset(ids)
        if key not in self._own_cache:
            self._own_cache[key] = {
                t.lower()
                for bucket in self._players.by_key.values()
                for a in bucket
                if a.entity_id in ids
                for t in a.tokens
            }
        return self._own_cache[key]


def _is_common_word(toks: tuple[str, ...]) -> bool:
    return len(toks) == 1 and is_common_word_surname(toks[0])


# ---------- обычные слова (выведенный список) ----------


@lru_cache(maxsize=1)
def common_english_words() -> frozenset[str]:
    """~9.5k самых частых слов английского веба (строчные, только буквы, длина >= MIN_ALIAS_LEN)."""
    try:
        raw = COMMON_WORDS_FILE.read_text(encoding="utf-8").split()
    except OSError:
        return frozenset()
    return frozenset(w.lower() for w in raw if len(w) >= MIN_ALIAS_LEN and w.isalpha())


def is_common_word_surname(token: str) -> bool:
    """Фамилия, совпадающая с обычным словом ("White", "James", "Grant", "King", "Long", "Best"):
    ручной стоп-лист ∪ частотный словарь. Требует контекста (имя, инициал, клуб)."""
    t = strip_accents(token).lower()
    return t in COMMON_WORD_SURNAMES or t in common_english_words()


# ---------- «текст про игрока?» с границами слов (для валидации цитат) ----------

_WORD_TOKEN = re.compile(r"[a-z0-9]+")
_INNER_HYPHEN = re.compile(r"(?<=[A-Za-z])-(?=[A-Za-z])")


def _join_hyphens(text: str) -> str:
    """ "Gibbs-White" -> "GibbsWhite": дефисная фамилия — один токен, "White" из неё не утекает."""
    return _INNER_HYPHEN.sub("", text)


def _norm_tokens(text: str) -> list[str]:
    return _WORD_TOKEN.findall(strip_accents(_join_hyphens(text)).lower())


def _contains_seq(tokens: list[str], seq: tuple[str, ...]) -> bool:
    n = len(seq)
    if n == 0 or n > len(tokens):
        return False
    if n == 1:
        return seq[0] in tokens
    return any(tuple(tokens[i : i + n]) == seq for i in range(len(tokens) - n + 1))


def player_name_tokens(player: Player) -> dict[str, tuple[str, ...]]:
    """Нормализованные варианты имени игрока.

    first — имя; surnames — последняя заглавная часть фамилии ("Magalhaes" из "dos Santos
    Magalhães"; частицы "van"/"de"/"dos" и первая часть "Santos" — не алиасы: слишком много
    чужих Santos); web — заглавные токены web_name без инициалов ("M.Salah" -> ("salah",),
    "van Ewijk" -> ("ewijk",), "Gibbs-White" -> ("gibbswhite",)); full — полное имя.
    """
    first = tuple(_norm_tokens(player.first_name))
    caps = [t for t in tokenize(_join_hyphens(player.second_name)) if t[:1].isupper()]
    surnames = tuple(strip_accents(t).lower() for t in caps[-1:])
    web = tuple(
        strip_accents(t).lower()
        for t in tokenize(_join_hyphens(player.web_name))
        if t[:1].isupper() and len(t) > 1
    )
    return {
        "first": first,
        "surnames": surnames,
        "web": web,
        "full": tuple(_norm_tokens(player.full_name)),
    }


def team_context_tokens(team: Team | None) -> list[tuple[str, ...]]:
    if team is None:
        return []
    aliases = [team.name, *TEAM_ALIASES.get(team.short_name.upper(), ())]
    seqs = [tuple(_norm_tokens(a)) for a in aliases]
    return [s for s in seqs if s and not (len(s) == 1 and len(s[0]) < MIN_ALIAS_LEN)]


def text_mentions_player(text: str, player: Player, team: Team | None = None) -> bool:
    """Упоминает ли текст игрока — по границам слов, без подстрок ("Egan" ≠ "Keegan").

    Полное имя, web_name из 2+ слов, инициал+фамилия ("B. White", "B White") — всегда достаточно.
    Однословная фамилия/web_name достаточна, если это не обычное английское слово; для
    "White"/"James"/"Grant" нужен контекст в том же тексте: имя игрока, инициал перед фамилией
    или упоминание его клуба.
    """
    toks = _norm_tokens(text)
    if not toks:
        return False
    names = player_name_tokens(player)
    first, surnames, web, full = names["first"], names["surnames"], names["web"], names["full"]

    if len(full) > 1 and _contains_seq(toks, full):
        return True
    if len(web) > 1 and _contains_seq(toks, web):
        return True
    # однословные алиасы: фамилия и web_name, если он не просто имя игрока ("Gabriel", "Bruno")
    singles = {s for s in surnames if len(s) >= MIN_ALIAS_LEN}
    if len(web) == 1 and len(web[0]) >= MIN_ALIAS_LEN and web[0] not in first:
        singles.add(web[0])
    if not singles:
        return False
    if first:
        for s in singles:  # "Benjamin White", "Ben White", "B White"
            if _contains_seq(toks, (*first, s)):
                return True
            for i in range(len(toks) - 1):
                if toks[i + 1] == s and first_name_matches(toks[i], first[:1]):
                    return True
    present = {s for s in singles if s in toks}
    if not present:
        return False
    if any(not is_common_word_surname(s) for s in present):
        return True
    # только «обычные слова» ("White") — нужен контекст: имя или клуб в том же тексте
    if first and _contains_seq(toks, first) and len(first[0]) >= MIN_ALIAS_LEN:
        return True
    return any(_contains_seq(toks, seq) for seq in team_context_tokens(team))
