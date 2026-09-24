"""Детерминированное сопоставление имени со скриншота с игроком bootstrap (без LLM).

Алгоритм:
1. Нормализация обеих сторон: NFKD без диакритики («João» -> «Joao»), нижний регистр, точки/дефисы/
   апострофы -> пробел («B.Fernandes» -> «b fernandes», «Gibbs-White» -> «gibbs white»).
2. Алиасы игрока: web_name; web_name без инициалов («J.Timber» -> «timber»); second_name целиком;
   последняя заглавная часть фамилии («dos Santos Magalhães» -> «magalhaes»); «имя фамилия» и полное
   имя; инициал + фамилия («b fernandes»); однословное имя («gabriel», «pedro») — модель иногда
   печатает только его. Многословное имя («João Pedro» у Costinha) алиасом НЕ становится: такое
   написание на карточке — это web_name другого игрока.
3. Точное совпадение алиаса -> кандидаты со score 100. Иначе rapidfuzz fuzz.ratio по всем алиасам с
   порогом ACCEPT_SCORE=88 («Haland» -> «haaland» 92, «Fernandez» -> «fernandes» 89).
4. Кандидатов больше одного — тай-брейки по подсказкам со скриншота, каждый применяется, только если
   оставляет хотя бы одного: позиция (ряд на поле) -> клуб -> цена ±PRICE_TOLERANCE.
5. Один кандидат — resolved; ноль — unresolved; несколько — ambiguous с перечнем кандидатов.
   «Gabriel» без позиции/клуба/цены не присваивается никому — лучше спросить пользователя, чем
   оптимизировать чужой состав.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from fplcopilot.data.schemas import Bootstrap, Player, Team
from fplcopilot.rag.entity_matcher import TEAM_ALIASES, strip_accents, tokenize
from fplcopilot.vision.schemas import RawPlayer, ResolvedPlayer

ACCEPT_SCORE = 88  # порог fuzz.ratio (0..100) для нечёткого совпадения
TIE_WINDOW = 3  # кандидаты в пределах TIE_WINDOW от лучшего score считаются равноценными
PRICE_TOLERANCE = 0.3  # £m: цена на скриншоте может отставать от now_cost на пару изменений
CLUB_HINT_SCORE = 85  # fuzz.ratio для подсказки клуба («Chelsea FC» ~ «chelsea»)
FUZZY_LIMIT = 50  # сколько алиасов берём из rapidfuzz до группировки по игроку
HINT_CONFIDENCE = 0.9  # множитель уверенности, если понадобились тай-брейки

_SEPARATORS = re.compile(r"[.\-'’`´_/]+")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]")
_SPACES = re.compile(r"\s+")
_PRICE_TAG = re.compile(r"£\s*\d+(?:[.,]\d+)?\s*m?", re.IGNORECASE)
_BADGE_TAG = re.compile(r"\(\s*(?:c|v|vc|tc)\s*\)", re.IGNORECASE)
_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")


def normalize(text: str) -> str:
    """«B.Fernandes» -> «b fernandes», «João Pedro» -> «joao pedro», «Gibbs-White» -> «gibbs white»."""
    t = strip_accents(text or "").lower()
    t = _SEPARATORS.sub(" ", t)
    t = _NON_ALNUM.sub(" ", t)
    return _SPACES.sub(" ", t).strip()


def clean_shown_name(name: str) -> str:
    """Убирает то, что модель могла приклеить к имени: «(C)», «£7.8m», числа."""
    s = _PRICE_TAG.sub(" ", name or "")
    s = _BADGE_TAG.sub(" ", s)
    s = _NUMBER.sub(" ", s)
    return " ".join(s.split()).strip(" -–—,;:")


def player_aliases(p: Player) -> set[str]:
    """Нормализованные написания, под которыми игрок может встретиться на карточке."""
    out: set[str] = set()
    web = normalize(p.web_name)
    first = normalize(p.first_name)
    second = normalize(p.second_name)
    web_tokens = web.split()
    first_tokens = first.split()

    if web:
        out.add(web)
        without_initials = [t for t in web_tokens if len(t) > 1]  # «j timber» -> «timber»
        if without_initials and len(without_initials) < len(web_tokens):
            out.add(" ".join(without_initials))
    if second:
        out.add(second)
    caps = [t for t in tokenize(p.second_name) if t[:1].isupper()]  # без частиц «de», «van», «dos»
    surname = normalize(caps[-1]) if caps else ""
    if surname:
        out.add(surname)
    if first and second:
        out.add(f"{first} {second}")
    if first and surname:
        out.add(f"{first} {surname}")
        out.add(f"{first_tokens[0][0]} {surname}")  # инициал: «b fernandes»
    if len(first_tokens) == 1:
        out.add(first)  # «gabriel», «pedro» — модель могла напечатать только имя
    if first and len(web_tokens) == 1 and web not in (first, surname):
        out.add(f"{first} {web}")  # «gabriel martinelli» при web_name «Martinelli»
        out.add(f"{first_tokens[0][0]} {web}")
    return {a for a in out if a}


def _team_names(short_name: str, name: str) -> set[str]:
    names = {normalize(short_name), normalize(name)}
    names.update(normalize(a) for a in TEAM_ALIASES.get(short_name.upper(), ()))
    return {n for n in names if n}


def team_matches_hint(short_name: str, name: str, hint: str | None) -> bool:
    """Подсказка клуба со скриншота («MUN», «Man Utd», «Manchester United», «Chelsea FC») про этот клуб?"""
    h = normalize(hint or "")
    if not h:
        return False
    names = _team_names(short_name, name)
    if h in names:
        return True
    for n in names:
        if len(n) >= 4 and len(h) >= 4 and (h.startswith(n) or n.startswith(h)):
            return True
        if len(n) >= 4 and fuzz.ratio(h, n) >= CLUB_HINT_SCORE:
            return True
    return False


def price_matches(price: float, shown: float | None, tolerance: float = PRICE_TOLERANCE) -> bool:
    return shown is None or abs(price - shown) <= tolerance + 1e-9


@dataclass(frozen=True)
class Candidate:
    player: Player
    team: Team
    score: float  # 0..100
    alias: str

    def describe(self) -> str:
        return (
            f"{self.player.web_name} ({self.team.short_name} {self.player.position.short} "
            f"£{self.player.price:.1f}m)"
        )


class PlayerResolver:
    """Индекс алиасов по bootstrap; один экземпляр на bootstrap (см. for_bootstrap)."""

    def __init__(self, teams: Iterable[Team], players: Iterable[Player]) -> None:
        self._teams = {t.id: t for t in teams}
        self._players: dict[int, Player] = {}
        self._aliases: dict[str, set[int]] = {}
        for p in players:
            self._players[p.id] = p
            for alias in player_aliases(p):
                self._aliases.setdefault(alias, set()).add(p.id)
        self._keys = list(self._aliases)

    @classmethod
    def from_bootstrap(cls, bs: Bootstrap) -> PlayerResolver:
        return cls(bs.teams, bs.elements)

    @classmethod
    def for_bootstrap(cls, bs: Bootstrap) -> PlayerResolver:
        """Кэш на объекте bootstrap — индекс строится один раз на процесс."""
        resolver = bs.__dict__.get("_vision_resolver")
        if resolver is None:
            resolver = cls.from_bootstrap(bs)
            bs.__dict__["_vision_resolver"] = resolver
        return resolver

    # ---------- кандидаты ----------

    def _candidate(self, pid: int, score: float, alias: str) -> Candidate:
        p = self._players[pid]
        return Candidate(p, self._teams[p.team], score, alias)

    def candidates(self, shown: str) -> list[Candidate]:
        """Точные совпадения (score 100) либо нечёткие >= ACCEPT_SCORE; пусто — не найден."""
        q = normalize(clean_shown_name(shown))
        if not q:
            return []
        exact = self._aliases.get(q)
        if exact:
            return [self._candidate(pid, 100.0, q) for pid in sorted(exact)]
        best: dict[int, tuple[float, str]] = {}
        for alias, score, _ in process.extract(
            q, self._keys, scorer=fuzz.ratio, score_cutoff=ACCEPT_SCORE, limit=FUZZY_LIMIT
        ):
            for pid in self._aliases[alias]:
                if pid not in best or score > best[pid][0]:
                    best[pid] = (float(score), alias)
        return [self._candidate(pid, s, a) for pid, (s, a) in sorted(best.items())]

    # ---------- резолюция ----------

    def resolve(self, raw: RawPlayer) -> ResolvedPlayer:
        base = ResolvedPlayer(
            name_as_shown=raw.name_as_shown,
            position=raw.position,
            position_as_shown=raw.position,
            price_as_shown=raw.price_as_shown,
            club_hint=raw.club_hint,
            is_captain=raw.is_captain,
            is_vice_captain=raw.is_vice_captain,
            is_bench=raw.is_bench,
            bench_order=raw.bench_order,
            is_flagged=raw.is_flagged,
        )
        cands = self.candidates(raw.name_as_shown)
        if not cands:
            return base

        top = max(c.score for c in cands)
        pool = [c for c in cands if c.score >= top - TIE_WINDOW]
        used_hints = False
        if len(pool) > 1:
            filters: list[Callable[[Candidate], bool]] = []
            if raw.position:
                filters.append(lambda c: c.player.position.short == raw.position)
            if raw.club_hint:
                filters.append(
                    lambda c: team_matches_hint(c.team.short_name, c.team.name, raw.club_hint)
                )
            if raw.price_as_shown is not None:
                filters.append(lambda c: price_matches(c.player.price, raw.price_as_shown))
            for keep in filters:
                narrowed = [c for c in pool if keep(c)]
                if narrowed and len(narrowed) < len(pool):
                    pool = narrowed
                    used_hints = True
                if len(pool) == 1:
                    break

        if len(pool) > 1:
            return base.model_copy(
                update={
                    "match_method": "ambiguous",
                    "candidates": [c.describe() for c in pool],
                }
            )

        c = pool[0]
        exact = c.score >= 100
        confidence = (1.0 if exact else c.score / 100) * (HINT_CONFIDENCE if used_hints else 1.0)
        method = ("exact" if exact else "fuzzy") + ("+hints" if used_hints else "")
        return base.model_copy(
            update={
                "player_id": c.player.id,
                "web_name": c.player.web_name,
                "position": c.player.position.short,
                "team_short": c.team.short_name,
                "team_name": c.team.name,
                "price": c.player.price,
                "match_confidence": round(confidence, 3),
                "match_method": method,
                "candidates": [c.describe()],
            }
        )

    def resolve_all(self, raws: Iterable[RawPlayer]) -> list[ResolvedPlayer]:
        return [self.resolve(r) for r in raws]
