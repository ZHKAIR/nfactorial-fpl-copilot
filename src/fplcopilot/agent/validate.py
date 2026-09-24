"""Детерминированная проверка ответа объяснителя: имена и числа — только из фактов.

Правила (docs/agent.md):
- каждое слово с заглавной буквы в ответе, похожее на имя (>= 3 букв, не ALL-CAPS, без цифр),
  должно встречаться среди токенов фактов/доказательств/известных имён (игроки, клубы), либо
  быть обычным английским словом (частотный словарь rag/entity_matcher + собственный стоп-лист
  слов форматирования); URL и цитаты `[source, dd.mm]` перед проверкой вырезаются;
- каждое десятичное число в ответе должно совпадать с каким-то числом фактов с точностью до
  округления до показанного числа знаков (6.2 против 6.20 — ок; 6.3 против 6.20 — нет);
  даты доказательств dd.mm разрешены как строки; целые числа не проверяются (GW, ранги, %);
- ссылки на стратегическую KB `[n]` (в т.ч. `[1][3]`, `[2, 4]`) допустимы только с номерами
  цитат из `facts.strategy_answer.citations`; неизвестный номер — нарушение (`unknown_refs`,
  попадает и в `bad_citations`), после исчерпания регенераций такие ссылки вырезаются;
- «−4» в строке вердикта допустимо только если в фактах есть платный трансфер (`hit_cost > 0`,
  `hits_by_gw > 0`, `PAID -4`) — иначе `hit_mismatch` (демо v2: «You should take a -4 …» при
  маршруте без хита — вердикт, перевёрнутый относительно оптимизатора; в промптах правило
  «never mention -4 unless a hit_cost of 4 is in FACTS» есть с v1, валидатор его закрепляет);
- имя игрока из фактов, написанное кириллицей («Сака», «Холанда», «Паскаль Гросс»), —
  `cyrillic_names`: слово с заглавной сравнивается только с именами игроков из «именных» полей
  фактов и extra_names (клубы исключены) по фонетическим ключам agent/translit.py; после
  исчерпания регенераций такие слова детерминированно заменяются написанием из фактов;
- номера туров (`GW7`, «7 тур», «в 7-м туре») — только из фактов или из самого вопроса
  (`unknown_gws`); если тур назван только в вопросе, а посчитан другой, ответ обязан назвать и
  посчитанный (`gw_uncovered`: «состав на GW7» при фактах GW6 — перевёрнутая подпись данных);
- оценка «N из 10» / «N/10» — число должно быть в фактах (иначе в `unknown_numbers`);
- капитан: в первых строках ответа на вопрос о капитане назван другой кандидат из
  `captain_options`, а капитан модели — нет (`captain_mismatch`);
- игрок из `facts.not_in_your_squad` (пользователь просил «не продавать» / «продать» того, кого в
  составе нет) упоминается только в строке с отрицанием («нет в вашем составе», «not in your
  squad»); иначе — `not_in_squad_claims`, после исчерпания регенераций такие строки вырезаются;
- атрибуция (`misattributed_numbers`): в строке таблицы / пункте списка / предложении ровно один
  игрок и число метрики (xPts, £, %, очки; в таблице — столбец с таким заголовком), которое в
  фактах есть только у других игроков — ни у этого, ни вне игрока (суммы плана, банк, маршруты);
  фрагменты с двумя+ игроками не проверяются. Прогон v4-run2 c20: xPts João Pedro и Van Hecke
  в строках Haaland, Palmer, Hall… — старые проверки этот ответ пропускали.
Нарушения возвращаются списком — граф даёт объяснителю одну регенерацию с этим фидбеком, при
повторном провале ответ остаётся, но получает видимую оговорку.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from fplcopilot.agent.translit import CYR_WORD, LETTERS, NameIndex
from fplcopilot.rag.entity_matcher import TEAM_ALIASES, common_english_words, strip_accents

_URL = re.compile(r"https?://\S+")
_CITATION = re.compile(r"\[[^\]\n]{1,80}\]")
# Цитаты-заглушки, которые модель пишет вместо реального источника: [FACTS], [source, dd.mm].
_BAD_CITATION = re.compile(
    r"\[\s*(facts?|source|src|evidence|n/?a|citation|dd\.mm)[^\]\n]{0,40}\]", re.IGNORECASE
)
_CAP_WORD = re.compile(r"(?<![\w'’-])[A-ZÀ-ÖØ-Þ][\w'’-]*")
_DECIMAL = re.compile(r"(?<![\w.])[-+−]?\d+\.\d+(?!\w|\.\d)")
_WORD = re.compile(r"[a-z0-9]+")
# Ссылки на документы стратегической KB: [1], [1][3], [2, 4]
_KB_REF = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_VERDICT_LINE = re.compile(r"^\W*\**\s*(Verdict|Вердикт)\b.*$", re.IGNORECASE | re.MULTILINE)
_MINUS_FOUR = re.compile(r"(?<![\d.])[-−–]\s?4\b(?![.,]\d)")
_PAID_MOVE = re.compile(r"PAID\s*-4", re.IGNORECASE)
# Номера туров: GW7 / GW 7 / GW6–GW8 (второй — отдельным совпадением), «7 тур», «в 7-м туре»
# Номера туров: GW7 / GW 7; по-русски — только номер, не количество: «7-й тур», «в 7 туре»,
# «тур 7»; «3 тура» / «5 туров» — это период, а не тур №3
_GW_REF = re.compile(
    r"\bGW\s?(\d{1,2})\b"
    r"|(?<![\d.,])(\d{1,2})-?(?:й|го|м|ом|ый|ой)\s+тур\w*"
    r"|(?<![\d.,])(\d{1,2})\s+туре\b"
    r"|\bтур(?:е|а|у)?\s+(?:№\s?)?(\d{1,2})\b",
    re.IGNORECASE,
)
_GW_RANGE = re.compile(r"\bGW(\d{1,2})\s*[–—-]\s*GW(\d{1,2})\b", re.IGNORECASE)
_RATING = re.compile(
    r"(?<![\d.])(\d{1,2}(?:[.,]\d)?)\s*(?:/\s*10\b|из\s+10\b|out\s+of\s+10\b)", re.IGNORECASE
)
_CAPTAIN_WORD = re.compile(r"captain|капитан", re.IGNORECASE)
MAX_GW = 38

# Слова форматирования/предметной области, которые пишутся с заглавной в начале строк и ячеек
# таблиц и не являются именами (страховка на случай отсутствия частотного словаря).
_FORMAT_WORDS_TEXT = """
    verdict why sources caveats caveat option options xpts delta cost risk note notes data as of
    yes no hold go safe balanced differential total gameweek gameweeks transfer transfers captain
    vice bench start starting fixture fixtures doubtful injured suspended available unavailable
    status chance expected points point hit hits wildcard free bank squad team home away clean
    sheet minutes probability ownership owned horizon next plan route routes move moves roll keep
    sell buy strategy model news official source evidence summary recommendation baseline
    alternative scenario premier league fantasy fpl gw fsi xi ft the a an and or for with without
    this that these those over under between against versus vs from to in on at by of if then else
    when while because but so not no none null nan yet still only also both either neither each
    every all any some most more less than much many few same other another such about after before
    during since until unless via per within into onto out off up down left right first second
    third last final current previous new old high low higher lower best worst better worse good
    bad top bottom key main primary secondary direct indirect recommended suggested confirmed
    rejected confirm reject pending action decision user manager player players club clubs match
    matches game games week weeks day days deadline kick kickoff table row column value values
    number numbers estimate estimated approx approximately roughly around about nearly almost
    exactly likely unlikely possible probable expected unexpected risky safe safer safest
    monday tuesday wednesday thursday friday saturday sunday january february march april may june
    july august september october november december
    gkp def mid fwd defender defenders midfielder midfielders forward forwards goalkeeper
    goalkeepers keeper keepers striker strikers winger wingers
    go hold hit_not_worth
    """
FORMAT_WORDS = frozenset(_FORMAT_WORDS_TEXT.split())
MIN_NAME_LEN = 3


@dataclass
class ValidationResult:
    passed: bool
    unknown_names: list[str] = field(default_factory=list)
    unknown_numbers: list[str] = field(default_factory=list)
    bad_citations: list[str] = field(default_factory=list)  # заглушки + неизвестные [n]
    unknown_refs: list[str] = field(default_factory=list)  # только неизвестные [n] (подмножество)
    missing_refs: bool = False  # ответ KB покрыт цитатами, но маркеры [n] из текста пропали
    hit_mismatch: bool = False  # «-4» в вердикте, хотя в фактах нет платного трансфера
    # имя игрока из фактов, написанное кириллицей: {как в ответе: как в фактах} («Саки»: «Saka»)
    cyrillic_names: dict[str, str] = field(default_factory=dict)
    unknown_gws: list[str] = field(
        default_factory=list
    )  # «GW9», которого нет ни в фактах, ни в вопросе
    gw_uncovered: list[str] = field(default_factory=list)  # тур из вопроса выдан за посчитанный
    captain_mismatch: str | None = None  # капитан в ответе ≠ капитан модели
    bench_mismatch: str | None = None  # «на скамейку X», а X в стартовом XI модели
    not_in_squad_claims: list[str] = field(default_factory=list)  # строки «X в составе» без X
    # число другого игрока при единственном игроке фрагмента: {player, number, owners}
    misattributed_numbers: list[dict[str, Any]] = field(default_factory=list)
    checked_names: int = 0
    checked_numbers: int = 0
    checked_refs: int = 0

    @property
    def feedback(self) -> str:
        parts = []
        if self.unknown_gws:
            parts.append(
                "Gameweeks not present in FACTS: "
                + ", ".join(self.unknown_gws)
                + ". Name only the gameweeks FACTS were computed for."
            )
        if self.gw_uncovered:
            parts.append(
                "The answer names "
                + ", ".join(self.gw_uncovered)
                + " from the question, but FACTS were computed for another gameweek: say "
                "explicitly which gameweek the numbers are for (the one in FACTS) and that the "
                "asked one is not computed — never present them as the asked gameweek."
            )
        if self.not_in_squad_claims:
            parts.append(
                "These players are NOT in the user's squad (FACTS.not_in_your_squad): "
                + ", ".join(dict.fromkeys(self.not_in_squad_claims))
                + ". Never list them as squad / team / lineup members; mention them only to say "
                "they are not in the squad."
            )
        if self.bench_mismatch:
            parts.append(
                f"{self.bench_mismatch} is in the model's starting XI (FACTS.best_xi.starters): the "
                "players to bench are FACTS.best_xi.bench — name those."
            )
        if self.captain_mismatch:
            parts.append(
                f"The recommended captain must be {self.captain_mismatch} (FACTS.best_xi.captain / "
                "captain_options[0]); do not recommend another player as captain."
            )
        if self.cyrillic_names:
            parts.append(
                "Player names must stay in Latin script exactly as in FACTS, without Russian case "
                "endings: "
                + ", ".join(f"'{w}' -> '{n}'" for w, n in self.cyrillic_names.items())
                + " (write e.g. 'вместо Saka', not 'вместо Саки')."
            )
        if self.hit_mismatch:
            parts.append(
                "The verdict mentions a -4 hit, but FACTS contain no paid transfer (every hit_cost "
                "is 0 / no hits in the plan): the recommended route uses free transfers only. Restate "
                "the verdict from FACTS.headline — no hit is taken, do not write '-4' in the verdict."
            )
        if self.missing_refs:
            parts.append(
                "The knowledge-base citation markers [n] are missing from the answer text. Keep "
                "every [n] marker of FACTS.strategy_answer.answer next to the claim it supports "
                "(write '… [3]', not '… [source]'); the Sources block alone is not enough."
            )
        placeholders = [c for c in self.bad_citations if c not in self.unknown_refs]
        if placeholders:
            parts.append(
                "Placeholder citations are not allowed: "
                + ", ".join(placeholders)
                + ". Numbers from FACTS need no citation; only news bullets get a citation with the "
                "real source and date from EVIDENCE, e.g. [bbc_football, 17.09]; a rules_context "
                "excerpt is cited as [source] with its literal source name."
            )
        if self.unknown_refs:
            parts.append(
                "Knowledge-base references not present in FACTS.strategy_answer.citations: "
                + ", ".join(self.unknown_refs)
                + ". Keep only the [n] markers that exist in the KB answer; never renumber or "
                "invent references."
            )
        if self.unknown_names:
            parts.append(
                "Names not present in FACTS/EVIDENCE: "
                + ", ".join(self.unknown_names)
                + ". Remove them or use only players/clubs that appear in FACTS."
            )
        if self.misattributed_numbers:
            parts.append(
                "Numbers attached to the wrong player: "
                + "; ".join(
                    f"{m['player']} {m['number']} (in FACTS {m['number']} belongs to "
                    f"{', '.join(m['owners'])}, not {m['player']})"
                    for m in self.misattributed_numbers
                )
                + ". Give each player only his own values from FACTS (xPts, price, ownership, "
                "points) or drop the number."
            )
        if self.unknown_numbers:
            parts.append(
                "Numbers not present in FACTS: "
                + ", ".join(self.unknown_numbers)
                + ". Copy values exactly as given in FACTS (do not recompute, sum or re-round) "
                "or drop them."
            )
        return "\n".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "unknown_names": self.unknown_names,
            "unknown_numbers": self.unknown_numbers,
            "bad_citations": self.bad_citations,
            "unknown_refs": self.unknown_refs,
            "missing_refs": self.missing_refs,
            "hit_mismatch": self.hit_mismatch,
            "cyrillic_names": self.cyrillic_names,
            "unknown_gws": self.unknown_gws,
            "gw_uncovered": self.gw_uncovered,
            "captain_mismatch": self.captain_mismatch,
            "bench_mismatch": self.bench_mismatch,
            "not_in_squad_claims": self.not_in_squad_claims,
            "misattributed_numbers": self.misattributed_numbers,
            "checked_names": self.checked_names,
            "checked_numbers": self.checked_numbers,
            "checked_refs": self.checked_refs,
        }


def strip_bad_citations(answer: str, unknown_refs: Iterable[str] = ()) -> str:
    """Детерминированная зачистка заглушек вида [FACTS] (и неизвестных ссылок [n]) после
    исчерпания регенераций."""
    cleaned = _BAD_CITATION.sub("", answer)
    for ref in unknown_refs:
        cleaned = cleaned.replace(ref, "")
    return re.sub(r"[ \t]+\n", "\n", re.sub(r" {2,}", " ", cleaned))


_SOURCES_HEAD = re.compile(r"^\W*\**\s*(Sources|Источники)\**\W*$", re.IGNORECASE)
_SECTION_HEAD = re.compile(
    r"^\W*\**\s*(Caveats|Оговорки|Why|Почему|Verdict|Вердикт)", re.IGNORECASE
)
_ANY_URL = re.compile(r"https?://\S+|fpl://\S+")


def _mentioned(player: str, body_folded: str) -> bool:
    toks = [t for t in _WORD.findall(fold(player)) if len(t) > 2]
    return bool(toks) and all(t in body_folded for t in toks)


def prune_undiscussed_sources(
    answer: str, evidence: Iterable[dict[str, Any]], facts: Any = None
) -> tuple[str, list[str]]:
    """Детерминированно убрать из раздела Sources строки о новостях игроков, которых ответ не
    обсуждает (слабость промпта v1 №1: «Sources» с цитатами про проблемных игроков состава).
    Строка остаётся, если её URL — из rules_context / strategy_answer, неизвестен (судить нечем)
    или относится к игроку, чьё имя встречается в тексте вне раздела Sources.
    Возвращает (ответ, удалённые строки)."""
    lines = (answer or "").splitlines()
    start = next((i for i, ln in enumerate(lines) if _SOURCES_HEAD.match(ln)), None)
    if start is None:
        return answer, []
    end = next(
        (i for i in range(start + 1, len(lines)) if _SECTION_HEAD.match(lines[i])), len(lines)
    )
    by_url: dict[str, set[str]] = {}
    for e in evidence or []:
        # цитата клуба (scope=club) «обсуждается», если ответ называет клуб (имя или код)
        who = [e.get("club"), e.get("team")] if e.get("scope") == "club" else [e.get("player")]
        by_url.setdefault(str(e.get("url")), set()).update(str(w) for w in who if w)
    keep_urls: set[str] = set()
    if isinstance(facts, dict):
        for x in (facts.get("rules_context") or {}).get("excerpts") or []:
            keep_urls.add(str(x.get("url")))
        for c in (facts.get("strategy_answer") or {}).get("citations") or []:
            keep_urls.add(str(c.get("url")))
    body_folded = fold("\n".join(lines[:start] + lines[end:]))
    removed: list[str] = []
    kept: list[str] = []
    for ln in lines[start + 1 : end]:
        urls = [u.rstrip(").,;") for u in _ANY_URL.findall(ln)]
        drop = False
        for u in urls:
            if u in keep_urls or u not in by_url:
                continue
            if not any(_mentioned(p, body_folded) for p in by_url[u]):
                drop = True
        (removed if drop else kept).append(ln)
    if not removed:
        return answer, []
    if any(_ANY_URL.search(ln) for ln in kept):
        new_lines = lines[: start + 1] + kept + lines[end:]
    else:  # раздел опустел — убираем и заголовок
        new_lines = lines[:start] + lines[end:]
    text = "\n".join(new_lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip(), [ln.strip() for ln in removed]


def allowed_kb_refs(facts: Any) -> set[int] | None:
    """Номера цитат KB из фактов; None — в фактах нет ответа KB (любая [n] неизвестна)."""
    if not isinstance(facts, dict):
        return None
    sa = facts.get("strategy_answer")
    if not isinstance(sa, dict):
        return None
    refs: set[int] = set()
    for c in sa.get("citations") or []:
        try:
            refs.add(int(c.get("n")))
        except (TypeError, ValueError, AttributeError):
            continue
    return refs


def unknown_kb_refs(answer: str, facts: Any) -> tuple[list[str], int]:
    """(неизвестные ссылки [n] как написаны, число проверенных)."""
    allowed = allowed_kb_refs(facts)
    unknown: list[str] = []
    checked = 0
    for m in _KB_REF.finditer(answer or ""):
        checked += 1
        numbers = [int(p.strip()) for p in m.group(1).split(",")]
        bad = allowed is None or any(n not in allowed for n in numbers)
        if bad and m.group() not in unknown:
            unknown.append(m.group())
    return unknown, checked


def facts_have_paid_transfer(facts: Any) -> bool:
    """Есть ли в фактах платный трансфер: hit_cost > 0, hits_by_gw > 0 или ход «PAID -4»."""

    def walk(obj: Any, key: str | None = None) -> bool:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "hit_cost" and isinstance(v, (int, float)) and not isinstance(v, bool):
                    if v > 0:
                        return True
                elif k == "hits_by_gw" and isinstance(v, dict):
                    if any(isinstance(x, (int, float)) and x > 0 for x in v.values()):
                        return True
                elif walk(v, k):
                    return True
        elif isinstance(obj, (list, tuple)):
            return any(walk(v, key) for v in obj)
        elif isinstance(obj, str) and key != "question":
            return bool(_PAID_MOVE.search(obj))
        return False

    if not isinstance(facts, dict):
        return False
    # решение пользователя по хиту (confirm / reject) — хит обсуждается, упоминать «-4» можно
    decided = ((facts.get("user_decision") or {}).get("on") or {}).get("action") or {}
    if isinstance(decided, dict) and (decided.get("cost") or 0) > 0:
        return True
    return walk(facts)


def verdict_line(answer: str) -> str:
    m = _VERDICT_LINE.search(answer or "")
    if m:
        return m.group()
    return next((ln for ln in (answer or "").splitlines() if ln.strip()), "")


def verdict_hit_mismatch(answer: str, facts: Any) -> bool:
    """«-4» в строке вердикта при отсутствии платного трансфера в фактах."""
    if not isinstance(facts, dict) or not facts:
        return False
    if not _MINUS_FOUR.search(verdict_line(answer)):
        return False
    return not facts_have_paid_transfer(facts)


_SOURCE_LINE = re.compile(r"^\s*(?:[-*]\s*)?\[\d+\]\s", re.MULTILINE)


def kb_refs_missing(answer: str, facts: Any) -> bool:
    """True, если ответ KB покрыт цитатами, а в тексте (вне строк Sources вида «- [n] …») не
    осталось ни одного маркера [n]: объяснитель заменил номера на имена источников или выкинул."""
    allowed = allowed_kb_refs(facts)
    if not allowed or not isinstance(facts, dict):
        return False
    if not (facts.get("strategy_answer") or {}).get("covered"):
        return False
    body = "\n".join(line for line in (answer or "").splitlines() if not _SOURCE_LINE.match(line))
    return not any(
        int(p.strip()) in allowed for m in _KB_REF.finditer(body) for p in m.group(1).split(",")
    )


# ---------- имена игроков кириллицей («Сака» вместо Saka) ----------

# Слово свободного текста — строже, чем упоминание от роутера (0.8): 0.88 ловит «Холанн» (0.883)
# и «Эдегор» (0.89), которые объяснитель копирует из вопроса.
ANSWER_NAME_MATCH = 0.88
_NAME_KEYS = frozenset(
    {"name", "full_name", "web_name", "player", "out", "in", "in_", "captain", "vice",
     "vice_captain", "current_captain"}
)  # fmt: skip
_NAME_DICTS = frozenset({"players", "squad_news_signals"})  # {имя игрока: данные}


def fact_player_names(facts: Any) -> list[str]:
    """Имена игроков из «именных» полей фактов и ключей словарей players / squad_news_signals."""
    out: list[str] = []

    def walk(obj: Any, key: str | None = None) -> None:
        if isinstance(obj, dict):
            if key in _NAME_DICTS:
                out.extend(str(k) for k in obj if str(k)[:1].isupper())
            for k, v in obj.items():
                if k in _NAME_KEYS and isinstance(v, str):
                    out.append(v)
                elif k in _NAME_KEYS and isinstance(v, list) and all(isinstance(x, str) for x in v):
                    out.extend(v)
                else:
                    walk(v, k)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v, key)

    walk(facts)
    return out


def _club_tokens() -> set[str]:
    return {
        fold(t) for aliases in TEAM_ALIASES.values() for a in aliases for t in LETTERS.findall(a)
    }


def cyrillic_player_names(text: str, facts: Any, extra_names: Iterable[str] = ()) -> dict[str, str]:
    """Слова кириллицей с заглавной, читающиеся как имя игрока из фактов: {«Саки»: «Saka»}.
    Сравнение — только с именами игроков (клубы исключены) по фонетическим ключам
    agent/translit.py с порогом ANSWER_NAME_MATCH; обычные слова FPL-чата — стоп-лист там же."""
    words = [
        m.group()
        for m in CYR_WORD.finditer(text or "")
        if m.group()[:1].isupper() and not m.group().isupper()
    ]
    if not words:
        return {}
    clubs = _club_tokens()
    index = NameIndex(
        t
        for s in [*fact_player_names(facts), *extra_names]
        for t in LETTERS.findall(str(s))
        if t[:1].isupper() and fold(t) not in clubs
    )
    found: dict[str, str] = {}
    for w in dict.fromkeys(words):
        hits = index.match(w, threshold=ANSWER_NAME_MATCH)
        if hits:
            found[w] = hits[0][0]
    return found


def restore_latin_names(answer: str, names: dict[str, str]) -> str:
    """Замена кириллических имён на написание из фактов — когда регенерации исчерпаны."""
    for word, latin in names.items():
        answer = re.sub(rf"(?<![\w-]){re.escape(word)}(?![\w-])", lambda _, s=latin: s, answer)
    return answer


# ---------- сбор допустимых токенов и чисел ----------


def _walk(obj: Any, strings: list[str], numbers: list[float]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            strings.append(str(k))
            _walk(v, strings, numbers)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            _walk(v, strings, numbers)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        numbers.append(float(obj))
    elif isinstance(obj, str):
        strings.append(obj)


def fold(word: str) -> str:
    return strip_accents(word).lower()


def known_tokens(*sources: Any, extra_names: Iterable[str] = ()) -> set[str]:
    """Все словарные токены (без диакритики, строчные) из фактов/доказательств/имён."""
    strings: list[str] = list(extra_names)
    numbers: list[float] = []
    for src in sources:
        _walk(src, strings, numbers)
    tokens: set[str] = set()
    for s in strings:
        tokens.update(_WORD.findall(fold(s)))
    for aliases in TEAM_ALIASES.values():
        for alias in aliases:
            tokens.update(_WORD.findall(fold(alias)))
    return tokens


def allowed_numbers(*sources: Any) -> tuple[list[float], set[str]]:
    """(числа фактов, строковые формы: сами числа с 0–3 знаками и десятичные внутри строк)."""
    strings: list[str] = []
    numbers: list[float] = []
    for src in sources:
        _walk(src, strings, numbers)
    forms: set[str] = set()
    for s in strings:
        forms.update(m.lstrip("+-−") for m in _DECIMAL.findall(s))
    for n in numbers:
        for d in range(4):
            forms.add(f"{abs(n):.{d}f}")
    return numbers, forms


def _number_ok(token: str, numbers: list[float], forms: set[str]) -> bool:
    raw = token.lstrip("+-−")
    if raw in forms:
        return True
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    try:
        x = float(raw)
    except ValueError:
        return True
    for n in numbers:
        if abs(round(abs(n), decimals) - x) < 1e-9:
            return True
    return False


def _strip_noise(answer: str) -> str:
    text = _URL.sub(" ", answer)
    text = _CITATION.sub(" ", text)
    return text.replace("|", " ").replace("*", " ").replace("#", " ")


def _gw_ints(text: str) -> set[int]:
    out: set[int] = set()
    for m in _GW_RANGE.finditer(text or ""):
        lo, hi = sorted((int(m.group(1)), int(m.group(2))))
        out.update(range(lo, hi + 1))
    for m in _GW_REF.finditer(text or ""):
        n = int(next(g for g in m.groups() if g))
        if 1 <= n <= MAX_GW:
            out.add(n)
    return out


def facts_gws(facts: Any, evidence: Any = None) -> set[int]:
    """Номера туров, для которых что-то посчитано: «GW6» в строках фактов (диапазоны
    раскрываются), целые значения ключей с «gw» в имени, ключи-номера словарей `*_by_gw`.
    Поле `question` не входит — это то, о чём спросили, а не то, что посчитано."""
    out: set[int] = set()

    def walk(obj: Any, key: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                ks = str(k)
                if ks == "question":
                    continue
                if key.endswith("by_gw") and ks.isdigit() and 1 <= int(ks) <= MAX_GW:
                    out.add(int(ks))
                out.update(_gw_ints(ks))
                walk(v, ks)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v, key)
        elif isinstance(obj, bool):
            return
        elif isinstance(obj, int) and "gw" in key.lower() and 1 <= obj <= MAX_GW:
            out.add(obj)
        elif isinstance(obj, str):
            out.update(_gw_ints(obj))

    walk(facts)
    walk(evidence or [])
    return out


def gw_violations(answer: str, facts: Any, evidence: Any = None) -> tuple[list[str], list[str]]:
    """(unknown_gws, gw_uncovered). Без фактов (нечего сверять) — пусто."""
    if not isinstance(facts, dict) or not facts:
        return [], []
    computed = facts_gws(facts, evidence)
    if not computed:
        return [], []
    asked = _gw_ints(str(facts.get("question") or ""))
    in_answer = _gw_ints(answer)
    unknown = [f"GW{g}" for g in sorted(in_answer - computed - asked)]
    only_asked = sorted((in_answer & asked) - computed)
    main = {
        g for g in (facts.get("gw"), (facts.get("best_xi") or {}).get("gw")) if isinstance(g, int)
    }
    uncovered = (
        [f"GW{g}" for g in only_asked]
        if only_asked and not (in_answer & (main or computed))
        else []
    )
    return unknown, uncovered


def invented_ratings(answer: str, facts: Any) -> list[str]:
    """«7 из 10» / «6/10» — оценку «из 10» считает только инструмент: число должно стоять в поле
    фактов с rating / score в имени (сейчас такого поля нет — любая оценка выдумана)."""
    rated: list[float] = []

    def walk(obj: Any, key: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, str(k))
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v, key)
        elif (
            isinstance(obj, (int, float))
            and not isinstance(obj, bool)
            and ("rating" in key.lower() or "score" in key.lower())
        ):
            rated.append(float(obj))

    walk(facts)
    out = []
    for m in _RATING.finditer(answer or ""):
        x = float(m.group(1).replace(",", "."))
        if not any(abs(x - r) < 1e-9 for r in rated) and m.group(0).strip() not in out:
            out.append(m.group(0).strip())
    return out


_BENCH_WORD = re.compile(r"bench|скамейк|запасн", re.IGNORECASE)


def bench_mismatch(answer: str, facts: Any) -> str | None:
    """«Посадите на скамейку X», где X — в стартовом XI модели (best_xi.starters)."""
    if not isinstance(facts, dict) or facts.get("intent") != "lineup":
        return None
    xi = facts.get("best_xi") or {}
    starters = [str(s).split(" (")[0] for s in xi.get("starters") or []]
    bench = [str(b).split(" (")[0] for b in xi.get("bench") or []]
    if not starters or not bench:
        return None
    first = next((ln for ln in (answer or "").splitlines() if ln.strip()), "")
    if not _BENCH_WORD.search(first):
        return None
    folded = fold(first)
    wrong = [
        s
        for s in starters
        if _mentioned(s, folded) and not any(_mentioned(b, folded) for b in bench)
    ]
    return wrong[0] if wrong else None


_NEGATION = re.compile(
    r"\bнет\b|\bне\s+в\b|\bне\s+применен|not\s+in\b|isn['’]t|is\s+not|not\s+applied|"
    r"\bне\s+входит|\bотсутств",
    re.IGNORECASE,
)


def not_in_squad_lines(answer: str, facts: Any) -> tuple[list[str], list[int]]:
    """(имена, номера строк): строки ответа, где игрок не из состава упомянут без отрицания."""
    names = list((facts or {}).get("not_in_your_squad") or []) if isinstance(facts, dict) else []
    if not names:
        return [], []
    bad_names: list[str] = []
    bad_lines: list[int] = []
    for i, ln in enumerate((answer or "").splitlines()):
        folded = fold(ln)
        hit = [n for n in names if _mentioned(n, folded)]
        if hit and not _NEGATION.search(ln):
            bad_names += hit
            bad_lines.append(i)
    return list(dict.fromkeys(bad_names)), bad_lines


_LIST_ROW = re.compile(r"^\s*(?:[-*•]|\d+[.)]|\|)")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def strip_lines(answer: str, line_numbers: Iterable[int], names: Iterable[str] = ()) -> str:
    """Строки-пункты списка / таблицы с ложным утверждением убираются целиком; в прозе —
    только предложения, где назван игрок (остальные предложения строки сохраняются)."""
    drop = set(line_numbers)
    folded_names = list(names)
    out = []
    for i, ln in enumerate((answer or "").splitlines()):
        if i not in drop:
            out.append(ln)
            continue
        if _LIST_ROW.match(ln) or not folded_names:
            continue
        kept = [
            sent
            for sent in _SENTENCE.split(ln)
            if not any(_mentioned(n, fold(sent)) for n in folded_names) or _NEGATION.search(sent)
        ]
        if kept:
            out.append(" ".join(kept))
    return "\n".join(out)


def strip_ratings(answer: str, ratings: Iterable[str]) -> str:
    """После исчерпания регенераций: убрать предложения с выдуманной оценкой «N из 10»."""
    out = answer or ""
    for r in ratings:
        out = re.sub(r"\s*[^.!?\n]*" + re.escape(r) + r"[^.!?\n]*[.!?]?", "", out)
    return out


def fix_name_typos(
    answer: str, unknown: Iterable[str], known: Iterable[str]
) -> tuple[str, dict[str, str]]:
    """«Caka» -> «Saka»: неизвестное латинское имя, отличающееся от ровно одного известного имени
    игрока одной буквой той же длины (>= 4 букв), заменяется детерминированно."""
    names = sorted({k for k in known if k and len(k) >= 4 and k.isalpha()})
    fixed: dict[str, str] = {}
    for u in unknown:
        cands = [
            k
            for k in names
            if len(k) == len(u)
            and sum(a != b for a, b in zip(k.lower(), u.lower(), strict=True)) == 1
        ]
        if len(u) >= 4 and len(cands) == 1:
            fixed[u] = cands[0]
    for u, k in fixed.items():
        answer = re.sub(rf"\b{re.escape(u)}\b", k, answer)
    return answer, fixed


def captain_mismatch(answer: str, facts: Any) -> str | None:
    """Капитан модели не назван в начале ответа о капитане, а назван другой кандидат."""
    if not isinstance(facts, dict) or facts.get("intent") != "captain":
        return None
    xi = facts.get("best_xi") or {}
    opts = [str(o.get("name")) for o in facts.get("captain_options") or [] if o.get("name")]
    cap = str(xi.get("captain") or (opts[0] if opts else "") or "")
    if not cap or len(opts) < 2:
        return None
    head = "\n".join(ln for ln in (answer or "").splitlines() if ln.strip())[:300]
    m = _CAPTAIN_WORD.search(head)
    if not m:
        return None
    # предложение о капитане без части про вице-капитана («капитан — X, вице — Y»)
    sentence = re.split(r"(?<=[.!?\n])\s", head[max(0, m.start() - 120) :])
    about = next((x for x in sentence if _CAPTAIN_WORD.search(x)), head)
    about = re.split(r"вице|vice", about, maxsplit=1, flags=re.IGNORECASE)[0]
    folded = fold(about)
    if _mentioned(cap, folded):
        return None
    if any(_mentioned(o, folded) for o in opts if o != cap):
        return cap
    return None


# ---------- атрибуция: число одного игрока приписано другому («Palmer 9.5 xPts» — цена Saka) ----------

_OWNER_KEYS = ("name", "web_name", "player", "full_name")
_FACT_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?!\.\d)")
_METRIC_NUM = re.compile(
    r"(?P<pre>£\s?|\bxpts?\s*[:=]?\s*)?(?<![\w.])(?P<num>\d+(?:\.\d+)?)(?!\.\d)"
    r"(?P<post>\s*\(?\s*(?:x\s?pts?\b|xp\b|очк|ожид\w*\.?\s+очк|pts\b|points?\b|%|m\b|м\b|млн))?",
    re.IGNORECASE,
)
_METRIC_HEADER = re.compile(
    r"x\s?pts?|xp\b|очк|pts\b|points?|£|цен|price|%|владен|own", re.IGNORECASE
)
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}")
# «Saka (ARS, MID, £9.5)», «Sels (NFO, GKP) 2.91 xPts», «Forster (GKP) 0.25» — имя в начале строки
_LEADING_NAME = re.compile(
    r"^\s*(?P<n>[^\W\d_][\w'’.\-]*(?:\s[^\W\d_][\w'’.\-]*){0,3}?)\s?\([A-Z]{3}\b"
)
_LEADING_COLON_NAME = re.compile(r"^\s*(?P<n>[A-ZÀ-ÖØ-Þ][\w'’.\-]*):\s")

NameKey = tuple[tuple[str, bool], ...]  # (токен без диакритики, с заглавной ли в имени)


def _name_key(name: str) -> NameKey:
    return tuple((fold(t), t[:1].isupper()) for t in LETTERS.findall(name) if len(t) >= 2)


def _tokens(key: NameKey) -> frozenset[str]:
    return frozenset(t for t, _ in key)


class _Words:
    """Слова фрагмента: все (без диакритики, строчные) и те, что написаны с заглавной."""

    def __init__(self, text: str) -> None:
        words = LETTERS.findall(text)
        self.all = {fold(w) for w in words}
        self.cap = {fold(w) for w in words if w[:1].isupper()}

    def has(self, key: NameKey) -> bool:
        return bool(key) and all(t in (self.cap if cap else self.all) for t, cap in key)


def _mentions(words: _Words, keys: dict[NameKey, str]) -> list[NameKey]:
    """Упомянутые игроки без вложенных дублей («Saka» внутри «Bukayo Saka» — один игрок)."""
    hit = [k for k in keys if words.has(k)]
    return [k for k in hit if not any(_tokens(k) < _tokens(o) for o in hit)]


def attribution_player_names(
    facts: Any, evidence: Any = None, extra: Iterable[str] = ()
) -> list[str]:
    """Имена игроков для проверки атрибуции: «именные» поля фактов и доказательств, ведущие
    имена строк вида «Saka (ARS, MID, £9.5)» / «Palmer: doubtful …», extra (без клубов).
    «Name:» в начале строки — слабый признак: обычное слово («Rule:», «Note:») именем не считается."""
    strong = [*fact_player_names(facts), *fact_player_names(evidence or []), *extra]
    weak: list[str] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)
        elif isinstance(obj, str):
            if m := _LEADING_NAME.match(obj):
                strong.append(m.group("n"))
            elif m := _LEADING_COLON_NAME.match(obj):
                weak.append(m.group("n"))

    walk(facts)
    common = common_english_words() | FORMAT_WORDS
    clubs = _club_tokens()
    out: list[str] = []
    for n in dict.fromkeys(str(x).strip() for x in [*strong, *weak]):
        toks = [fold(t) for t in LETTERS.findall(n) if len(t) >= 2]
        if not toks or not n[:1].isupper() or n.isupper() or any(ch.isdigit() for ch in n):
            continue
        if all(t in clubs for t in toks):
            continue  # extra_names графа содержат и клубы
        if n not in strong and all(t in common for t in toks):
            continue
        out.append(n)
    return out


def _owned_numbers(
    facts: Any, evidence: Any, keys: dict[NameKey, str]
) -> list[tuple[float, frozenset[str]]]:
    """(число, чьё оно) по фактам и доказательствам. Владелец — игрок словаря (name / player …),
    ключ-имя словаря или игроки, названные в строке (в «Palmer (£9.7) -> Groß (£5.8)» — оба).
    Пустое множество — не число игрока: сумма плана, банк, маршрут трансферов, вопрос."""
    out: list[tuple[float, frozenset[str]]] = []

    def key_of(name: str) -> str | None:
        k = _name_key(name)
        if k in keys:
            return keys[k]
        return next((v for kk, v in keys.items() if _tokens(kk) == _tokens(k)), None)

    def walk(obj: Any, owner: frozenset[str], key: str = "") -> None:
        if isinstance(obj, dict):
            own = frozenset(
                n for k in _OWNER_KEYS if isinstance(obj.get(k), str) and (n := key_of(obj[k]))
            )
            owner = own or owner
            for k, v in obj.items():
                ks = str(k)
                if ks == "question":
                    walk(v, frozenset(), ks)
                    continue
                named = key_of(ks) if ks[:1].isupper() else None
                walk(v, frozenset({named}) if named else owner, ks)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v, owner, key)
        elif isinstance(obj, bool):
            return
        elif isinstance(obj, (int, float)):
            out.append((abs(float(obj)), owner))
        elif isinstance(obj, str):
            if key == "question":
                out.extend((float(m.group()), frozenset()) for m in _FACT_NUM.finditer(obj))
                return
            mentioned = [keys[k] for k in _mentions(_Words(obj), keys)]
            who = owner | frozenset(mentioned)
            # «Saka (6.2) > Palmer (4.76)»: число в скобках сразу за именем — этого игрока
            spans: list[tuple[int, int, str]] = []
            if not owner and len(mentioned) > 1:
                for n in mentioned:
                    rx = rf"(?<![\w'’.-]){re.escape(n)}\s*\(([^()]*)\)"
                    spans += [(m.start(1), m.end(1), n) for m in re.finditer(rx, obj)]
            for m in _FACT_NUM.finditer(obj):
                inside = {n for a, b, n in spans if a <= m.start() < b}
                out.append((float(m.group()), frozenset(inside) if len(inside) == 1 else who))

    walk(facts, frozenset())
    walk(evidence or [], frozenset())
    return out


def _answer_units(answer: str) -> list[tuple[str, list[str]]]:
    """(фрагмент, числа с признаком метрики): строка таблицы, пункт списка или предложение прозы.
    В таблице число метрики — любое число в столбце, чей заголовок — xPts / цена / % / очки."""
    units: list[tuple[str, list[str]]] = []
    header: list[bool] | None = None
    for raw in (answer or "").splitlines():
        line = raw.strip()
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if _TABLE_SEP.match(line):
                continue
            if header is None:
                header = [bool(_METRIC_HEADER.search(c)) for c in cells]
                continue
            nums: list[str] = []
            for i, cell in enumerate(cells):
                clean = _strip_noise(cell)
                if i < len(header) and header[i]:
                    nums += [m.group() for m in _FACT_NUM.finditer(clean)]
                else:
                    nums += [
                        m.group("num")
                        for m in _METRIC_NUM.finditer(clean)
                        if m.group("pre") or m.group("post")
                    ]
            units.append((_strip_noise(" ".join(cells)), nums))
            continue
        header = None
        if not line:
            continue
        parts = [line] if _LIST_ROW.match(line) else _SENTENCE.split(line)
        for part in parts:
            clean = _strip_noise(part)
            nums = [
                m.group("num")
                for m in _METRIC_NUM.finditer(clean)
                if m.group("pre") or m.group("post")
            ]
            if nums:
                units.append((clean, nums))
    return units


def misattributed_numbers(
    answer: str, facts: Any, evidence: Any = None, *, extra_names: Iterable[str] = ()
) -> list[dict[str, Any]]:
    """Число метрики игрока (xPts, £, %, очки) во фрагменте ровно с одним игроком, которое в
    фактах есть ТОЛЬКО у других игроков и ни разу — у этого или вне игрока.

    Консервативно: фрагменты с двумя+ игроками, с неизвестным словом-именем с заглавной или
    с кириллическим именем игрока пропускаются; число, которого нет в фактах, — дело
    unknown_numbers; совпадение числа у двух игроков или с суммой / банком — не нарушение."""
    if not isinstance(facts, dict) or not facts:
        return []
    names = attribution_player_names(facts, evidence, extra_names)
    keys: dict[NameKey, str] = {}
    token_sets: set[frozenset[str]] = set()
    for n in names:
        k = _name_key(n)
        if k and _tokens(k) not in token_sets:  # «Van de Ven» и «van de Ven» — один игрок
            keys[k] = n
            token_sets.add(_tokens(k))
    if not keys:
        return []
    owned = _owned_numbers(facts, evidence, keys)
    player_tokens = {t for k in keys for t in _tokens(k)}
    allowed_words = common_english_words() | FORMAT_WORDS | _club_tokens()
    cyrillic = set(cyrillic_player_names(answer or "", facts, names))
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for text, nums in _answer_units(answer):
        words = _Words(text)
        hit = _mentions(words, keys)
        if len(hit) != 1:
            continue
        player = keys[hit[0]]
        own_tokens = _tokens(hit[0])
        stray = [
            w
            for w in _CAP_WORD.findall(text)
            if not w.isupper()
            and not any(ch.isdigit() for ch in w)
            and any(
                len(p) >= MIN_NAME_LEN
                and fold(p) not in own_tokens
                and (fold(p) in player_tokens or fold(p) not in allowed_words)
                for p in re.split(r"[-'’]", w)
            )
        ]
        if stray or any(re.search(rf"(?<![\w-]){re.escape(c)}(?![\w-])", text) for c in cyrillic):
            continue
        for tok in nums:
            decimals = len(tok.split(".")[1]) if "." in tok else 0
            x = float(tok)
            occ = [who for n, who in owned if abs(round(n, decimals) - x) < 1e-9]
            if not occ or any(not who for who in occ):
                continue
            mine = own_tokens
            if any(
                _tokens(_name_key(o)) <= mine or mine <= _tokens(_name_key(o))
                for who in occ
                for o in who
            ):
                continue
            if (player, tok) in seen:
                continue
            seen.add((player, tok))
            owners = sorted({o for who in occ for o in who})
            out.append({"player": player, "number": tok, "owners": owners})
    return out


def validate_answer(
    answer: str,
    facts: Any,
    evidence: Any = None,
    *,
    extra_names: Iterable[str] = (),
) -> ValidationResult:
    bad_citations = list(dict.fromkeys(m.group() for m in _BAD_CITATION.finditer(answer or "")))
    unknown_refs, checked_refs = unknown_kb_refs(answer or "", facts)
    bad_citations += [r for r in unknown_refs if r not in bad_citations]
    missing_refs = kb_refs_missing(answer or "", facts)
    hit_mismatch = verdict_hit_mismatch(answer or "", facts)
    text = _strip_noise(answer or "")
    extra_names = list(extra_names)
    cyrillic_names = cyrillic_player_names(text, facts, extra_names)
    tokens = known_tokens(facts, evidence or [], extra_names=extra_names)
    common = common_english_words() | FORMAT_WORDS
    unknown_names: list[str] = []
    seen: set[str] = set()
    checked_names = 0
    for m in _CAP_WORD.finditer(text):
        word = m.group()
        if any(ch.isdigit() for ch in word) or word.isupper():
            continue
        for part in re.split(r"[-'’]", word):
            if len(part) < MIN_NAME_LEN or not part[0].isupper():
                continue
            key = fold(part)
            if not key.isalpha():
                continue
            checked_names += 1
            if key in tokens or key in common or key in seen:
                continue
            seen.add(key)
            unknown_names.append(part)

    numbers, forms = allowed_numbers(facts, evidence or [])
    unknown_numbers: list[str] = []
    checked_numbers = 0
    for m in _DECIMAL.finditer(text):
        tok = m.group()
        checked_numbers += 1
        if not _number_ok(tok, numbers, forms) and tok not in unknown_numbers:
            unknown_numbers.append(tok)
    unknown_numbers += [r for r in invented_ratings(text, facts) if r not in unknown_numbers]
    unknown_gws, gw_uncovered = gw_violations(text, facts, evidence)
    cap_mismatch = captain_mismatch(answer or "", facts)
    bench_wrong = bench_mismatch(answer or "", facts)
    squad_claims, _ = not_in_squad_lines(answer or "", facts)
    misattributed = misattributed_numbers(answer or "", facts, evidence, extra_names=extra_names)

    return ValidationResult(
        passed=(
            not unknown_names
            and not unknown_numbers
            and not misattributed
            and not bad_citations
            and not missing_refs
            and not hit_mismatch
            and not cyrillic_names
            and not unknown_gws
            and not gw_uncovered
            and cap_mismatch is None
            and bench_wrong is None
            and not squad_claims
        ),
        unknown_names=unknown_names,
        unknown_numbers=unknown_numbers,
        bad_citations=bad_citations,
        unknown_refs=unknown_refs,
        missing_refs=missing_refs,
        hit_mismatch=hit_mismatch,
        cyrillic_names=cyrillic_names,
        unknown_gws=unknown_gws,
        gw_uncovered=gw_uncovered,
        captain_mismatch=cap_mismatch,
        bench_mismatch=bench_wrong,
        not_in_squad_claims=squad_claims,
        misattributed_numbers=misattributed,
        checked_names=checked_names,
        checked_numbers=checked_numbers,
        checked_refs=checked_refs,
    )
