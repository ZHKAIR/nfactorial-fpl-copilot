"""Кириллические написания латинских имён FPL: транслитерация, снятие падежа, нечёткое сравнение.

Пользователь пишет «Саки», «Холанда», «Жоау Педро»; в bootstrap — Saka, Haaland, João Pedro.
Сравниваются не строки, а фонетические ключи: обе стороны сводятся к одному грубому алфавиту —
латиница без диакритики, ch / sh / zh / j -> j, c -> k или s, w -> v, y -> i, x -> ks, ou -> u,
удвоенные буквы схлопываются; кириллица сначала транслитерируется (ж -> zh, ч -> ch, ц -> ts, ...).
У латинского токена бывает несколько ключей — по тому, как имя читается в языке оригинала:
Ø / Ö -> «e» (Ødegaard -> «Эдегор»), скандинавское aa -> «o» (Haaland -> «Холанд»),
португальское ão -> «au» (João -> «Жоау»), голландское ij -> «ei» (Dijk -> «Дейк»), мягкое g -> «j»
(Virgil -> «Вирджил»), š / č / ž -> «sh / ch / zh» (Šeško -> «Шешко»), английские oo -> «u»,
ea / ee -> «i» (Mainoo, Heaton).
Сходство — среднее difflib-ratio полного ключа и его согласных (гласные при передаче чужих имён
плывут сильнее согласных) при ratio полного ключа >= MIN_FULL_RATIO и равном числе слогов.
Короткие ключи (<= SHORT_KEY букв) — только точное совпадение («Исак» ≠ Saka), ключи кириллицы
короче MIN_KEY не читаются вовсе («весь» = Wes, «уже» = Uche): трёхбуквенные фамилии остаются
роутеру.
Падеж: у слова перебираются окончания («Палмера» -> «Палмер», «Саки» -> «Сака»,
«Вирцем» -> «Вирц»); основа кончается согласной («данные» не даёт «данн»), а перед «я» — гласной
или «ь» («Раю» -> «Рая», «Куньи» -> «Кунья»; «даже» -> «дажя» — нет).
Обычные слова FPL-чата и русские названия клубов (RU_STOPWORDS, в любом падеже) за имена не
принимаются никогда. Только stdlib.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from difflib import SequenceMatcher

from fplcopilot.rag.entity_matcher import strip_accents

MATCH_THRESHOLD = 0.8  # мин. сходство слова с токеном имени
MIN_FULL_RATIO = 0.78  # и мин. ratio полного ключа (одинаковых согласных мало: «тройка» ~ Tyrick)
MARGIN = 0.05  # токены в пределах MARGIN от лучшего считаются одинаково близкими
SHORT_KEY = 4  # ключ такой длины и короче сравнивается только точно
MIN_KEY = 4  # ключи кириллицы короче не читаются: «весь» = Wes, «уже» = Uche, «мать» = Matt

_CYR_CHAR = re.compile(r"[А-Яа-яЁё]")
CYR_WORD = re.compile(r"(?<![\w-])[А-ЯЁа-яё]+(?![\w-])")
LETTERS = re.compile(r"[^\W\d_]+")

_CYR_TO_LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z",
    "и": "i", "й": "i", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sh",
    "ъ": "", "ы": "i", "ь": "", "э": "e", "ю": "iu", "я": "ia",
}  # fmt: skip
_CYR_VOWELS = frozenset("аеёиоуыэюя")
_CYR_NOT_CONSONANT = _CYR_VOWELS | frozenset("ьъй")
_BEFORE_YA = frozenset("аеиоуэь")  # Рая, Кунья; «даже» -> «дажя», «данные» -> «данныя» — нет

# Окончания косвенных падежей иностранных фамилий -> как восстановить именительный.
_ENDINGS: tuple[tuple[str, str], ...] = (
    ("ом", ""), ("ем", ""), ("ём", ""),  # Палмером, Вирцем, Габриэлем
    ("ой", "а"), ("ей", "я"),  # Сакой
    ("ы", "а"), ("и", "а"), ("и", "я"),  # Саки
    ("у", "а"), ("ю", "я"), ("у", ""), ("ю", ""),  # Саку, Раю, Палмеру, Габриэлю
    ("е", "а"), ("е", "я"), ("е", ""),  # Саке, Рае, Палмере
    ("а", ""), ("я", ""),  # Палмера, Габриэля
)  # fmt: skip

# Слова FPL-чата и русские названия клубов, фонетически совпадающие с чьим-то именем
# («туре» = Touré, «Халл» = Hall, «Вилла» ~ Will, «команда» ~ Kamada): кириллическое слово,
# у которого одна из падежных основ в списке, за имя игрока не принимается. Игрока с таким
# именем роутер передаёт латиницей — латинский путь резолвера этот список не видит.
RU_STOPWORDS = frozenset(
    {
        "тур", "туре", "туры", "туров", "гол", "голы", "голов", "матч", "матчи", "сейв", "сейвы",
        "хит", "хиты", "план", "форма", "сила", "роль", "риск", "шанс", "бонус", "бонусы", "очко",
        "очки", "очков", "капитан", "вице", "кэп", "состав", "команда", "команд", "лига", "фишка",
        "фишки", "вратарь", "защитник", "полузащитник", "нападающий", "форвард", "игрок", "стоит",
        "сравни", "данные", "вердикт",
        "арсенал", "астон", "вилла", "борнмут", "брентфорд", "брайтон", "бернли", "челси",
        "ковентри", "кристал", "пэлас", "эвертон", "фулхэм", "фулхем", "халл", "ипсвич", "лидс",
        "лестер", "ливерпуль", "манчестер", "сити", "юнайтед", "ньюкасл", "ноттингем", "форест",
        "тоттенхэм", "тоттенхем", "шпоры", "сандерленд", "саутгемптон", "вест", "хэм",
        "вулверхэмптон", "вулвз", "волки",
    }
)  # fmt: skip

_KEY_RULES: tuple[tuple[str, str], ...] = (
    ("tsch", "ch"), ("sch", "sh"), ("tch", "ch"), ("dzh", "j"), ("dj", "j"), ("zh", "j"),
    ("kh", "h"), ("ph", "f"), ("th", "t"), ("ck", "k"), ("qu", "k"), ("tz", "ts"), ("cz", "ch"),
    ("sz", "s"), ("ch", "j"), ("sh", "j"),
)  # fmt: skip
_SOFT_C = re.compile(r"c(?=[eiy])")
_DOUBLE = re.compile(r"(.)\1+")
_VOWELS = re.compile(r"[aeiou]")
_SYLLABLE = re.compile(r"[aeiou]+")
# Варианты чтения латинского токена (до снятия диакритики): (что ищем, на что меняем).
_READINGS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"[øöœ]", re.IGNORECASE), "e"),  # Ødegaard -> Эдегор, Gyökeres -> Дьёкереш
    (re.compile(r"aa", re.IGNORECASE), "o"),  # Haaland -> Холанд (норв. aa = å)
    (re.compile(r"ão", re.IGNORECASE), "au"),  # João -> Жоау
    (re.compile(r"ij", re.IGNORECASE), "ei"),  # van Dijk -> ван Дейк
    (re.compile(r"g(?=[eiy])", re.IGNORECASE), "j"),  # Virgil -> Вирджил
    (re.compile(r"[šś]", re.IGNORECASE), "sh"),  # Šeško -> Шешко (strip_accents дал бы «s»)
    (re.compile(r"[čć]", re.IGNORECASE), "ch"),  # Kovačić -> Ковачич
    (re.compile(r"ž", re.IGNORECASE), "zh"),
    (re.compile(r"oo", re.IGNORECASE), "u"),  # Mainoo -> Мейну, Wood -> Вуд
    (re.compile(r"e[ea]", re.IGNORECASE), "i"),  # Heaton -> Хитон, Reece -> Рис
)


def has_cyrillic(text: str) -> bool:
    return bool(_CYR_CHAR.search(text or ""))


# Кириллические буквы, неотличимые от латинских, и обратно («О'Shea» с кириллической «О»).
_CYR_HOMOGLYPHS = "АВЕКМНОРСТХаеорсух"
_LAT_HOMOGLYPHS = "ABEKMHOPCTXaeopcyx"
_TO_LATIN = str.maketrans(_CYR_HOMOGLYPHS, _LAT_HOMOGLYPHS)
_TO_CYRILLIC = str.maketrans(_LAT_HOMOGLYPHS, _CYR_HOMOGLYPHS)
_MIXED_WORD = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*")


def fix_mixed_script(text: str) -> str:
    """Слово из латиницы и кириллицы сразу приводится к алфавиту большинства его букв, если
    буквы меньшинства — двойники («О'Shea» -> «O'Shea», «Cака» -> «Сака»); иначе не трогается."""

    def fix(m: re.Match[str]) -> str:
        word = m.group()
        cyr = len(_CYR_CHAR.findall(word))
        lat = sum(1 for ch in word if "a" <= ch.lower() <= "z")
        if not cyr or not lat:
            return word
        fixed = word.translate(_TO_LATIN if lat > cyr else _TO_CYRILLIC)
        mixed = has_cyrillic(fixed) and any("a" <= ch.lower() <= "z" for ch in fixed)
        return word if mixed else fixed

    return _MIXED_WORD.sub(fix, text or "")


def cyr_to_latin(word: str) -> str:
    """«Семеньо» -> «semenio»: ь перед гласной — «i» (ньо = nyo), иначе опускается."""
    w = word.lower()
    out = []
    for i, ch in enumerate(w):
        if ch == "ь" and i + 1 < len(w) and w[i + 1] in _CYR_VOWELS:
            out.append("i")
        else:
            out.append(_CYR_TO_LAT.get(ch, ch))
    return "".join(out)


def phonetic_key(latin: str) -> str:
    """Грубый фонетический ключ латинской строки (уже без диакритики)."""
    s = "".join(LETTERS.findall(latin.lower()))
    for a, b in _KEY_RULES:
        s = s.replace(a, b)
    s = _SOFT_C.sub("s", s).replace("c", "k")
    s = s.replace("x", "ks").replace("w", "v").replace("y", "i").replace("q", "k")
    s = s.replace("ou", "u")
    return _DOUBLE.sub(r"\1", s)


def latin_keys(token: str) -> tuple[str, ...]:
    """Ключи латинского токена имени: «Ødegaard» -> ('odegard', 'edegard', 'odegord', ...)."""
    variants = [token]
    for pattern, repl in _READINGS:
        variants += [pattern.sub(repl, v) for v in variants if pattern.search(v)]
    keys = (phonetic_key(strip_accents(v)) for v in variants)
    return tuple(dict.fromkeys(k for k in keys if k))


def case_stems(word: str) -> list[str]:
    """Слово и его возможные именительные формы: «Палмера» -> [палмера, палмер]."""
    w = word.lower()
    out = [w]
    if len(w) <= 3:  # «или» -> «иля» = Ilia: у коротких слов падеж не снимаем
        return out
    for suffix, repl in _ENDINGS:
        base = w[: len(w) - len(suffix)]
        if not w.endswith(suffix) or len(base) < 2:
            continue
        if (base[-1] not in _BEFORE_YA) if repl == "я" else (base[-1] in _CYR_NOT_CONSONANT):
            continue
        stem = base + repl
        if len(stem) >= 3 and stem not in out:
            out.append(stem)
    return out


def is_stopword(word: str) -> bool:
    return any(s in RU_STOPWORDS for s in case_stems(word))


def cyrillic_keys(word: str) -> list[str]:
    """Ключи падежных основ слова длиной >= MIN_KEY («Саки» -> ['saki', 'saka'])."""
    keys = (phonetic_key(cyr_to_latin(s)) for s in case_stems(word))
    return list(dict.fromkeys(k for k in keys if len(k) >= MIN_KEY))


def similarity(a: str, b: str) -> float:
    """Среднее difflib-ratio полного ключа и его согласных. 0, если: ключ <= SHORT_KEY и не равен,
    ratio полного ключа < MIN_FULL_RATIO или разное число слогов («troik» / «tirik»)."""
    if a == b:
        return 1.0
    if min(len(a), len(b)) <= SHORT_KEY:
        return 0.0
    full = SequenceMatcher(None, a, b).ratio()
    if full < MIN_FULL_RATIO or len(_SYLLABLE.findall(a)) != len(_SYLLABLE.findall(b)):
        return 0.0
    ca, cb = _VOWELS.sub("", a), _VOWELS.sub("", b)
    cons = SequenceMatcher(None, ca, cb).ratio() if ca and cb else full
    return (full + cons) / 2


def closest(scored: list[tuple[str, float]], margin: float = MARGIN) -> list[tuple[str, float]]:
    """Лучший и все, кто в пределах margin от него (сортировка по убыванию сходства)."""
    if not scored:
        return []
    top = max(s for _, s in scored)
    return sorted(((t, s) for t, s in scored if s >= top - margin), key=lambda x: (-x[1], x[0]))


class NameIndex:
    """Латинские токены имён -> фонетические ключи; поиск близких к кириллическому слову."""

    def __init__(self, tokens: Iterable[str]) -> None:
        self._entries: dict[str, tuple[str, tuple[str, ...]]] = {}
        for tok in tokens:
            norm = strip_accents(tok).lower()
            if len(norm) < 3 or not norm.isalpha():
                continue
            shown, keys = self._entries.get(norm, (tok, ()))
            self._entries[norm] = (shown, tuple(dict.fromkeys((*keys, *latin_keys(tok)))))

    def __len__(self) -> int:
        return len(self._entries)

    def match(self, word: str, *, threshold: float = MATCH_THRESHOLD) -> list[tuple[str, float]]:
        """Все [(латинский токен как в источнике, сходство)] не ниже порога, лучшие первыми."""
        if is_stopword(word):
            return []
        keys = cyrillic_keys(word)
        scored: list[tuple[str, float]] = []
        for token, lat_keys in self._entries.values():
            best = 0.0
            for a in keys:
                for b in lat_keys:
                    if a == b:
                        best = 1.0
                        break
                    short, total = min(len(a), len(b)), len(a) + len(b)
                    if short <= SHORT_KEY or 2 * short / total < MIN_FULL_RATIO:
                        continue  # верхняя граница ratio по длинам — без SequenceMatcher
                    best = max(best, similarity(a, b))
                if best == 1.0:
                    break
            if best >= threshold:
                scored.append((token, round(best, 3)))
        return sorted(scored, key=lambda x: (-x[1], x[0]))
