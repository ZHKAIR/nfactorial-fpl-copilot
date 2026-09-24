"""Детерминированное разрешение упоминаний игроков из запроса в id bootstrap (без LLM).

LLM-роутер лишь перечисляет упоминания («Palmer», «João Pedro», «Gabriel»; с v3 — латиницей и в
именительном падеже, даже если пользователь писал «Палмера»); кто это — решает код:

1. `EntityMatcher` (rag/entity_matcher.py, та же логика, что тегирует новости) по всему тексту
   запроса даёт «уверенные» id: полные имена, инициалы, контекст клуба.
2. Каждое упоминание роутера ищется по токенам (без диакритики и регистра) как непрерывная
   подпоследовательность токенов web_name или полного имени: «Saka» -> Bukayo Saka, но не
   Wan-Bissaka/Sakamoto.
3. Ровно один кандидат -> разрешено. Несколько:
   a) упоминание из одного слова, которое является ИМЕНЕМ у >= 2 кандидатов («Gabriel»: Magalhães,
      Martinelli, Jesus, Gudmundsson; «Pedro») -> неоднозначно: имя само по себе игрока не
      идентифицирует, даже если у кого-то оно совпадает с web_name;
   b) иначе (совпадение фамилий: Cole Palmer / Alex Palmer) — ровно один кандидат в составе
      менеджера -> он («Should I sell Palmer?» — про того Палмера, которым владеешь);
   c) иначе доминирование по владению: лидер >= DOMINANCE_RATIO × второго и >= MIN_OWNERSHIP % ->
      он, с пометкой в ответе; иначе неоднозначно -> ветка clarify.
4. Ноль кандидатов -> «не найден» (попадает в оговорки ответа).
5. Упоминание с кириллицей (роутер не перевёл его в латиницу) читается страховочно
   (agent/translit.py): каждое слово -> ближайшие токены имён bootstrap по фонетическому ключу
   со снятым падежом («Саки» -> Saka, «Паскаля Гросса» -> Pascal Groß); нечёткое (не точное)
   совпадение — только для игроков с владением >= MIN_FUZZY_OWNERSHIP. Одно прочтение -> шаги
   2–3 как для латиницы; несколько одинаково близких («Саки»: Saka / Sakyi) -> их кандидаты
   вместе через те же правила 3 (состав, доминирование, иначе clarify).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from itertools import islice, product
from typing import Any, Literal

from fplcopilot.agent.translit import LETTERS, MARGIN, NameIndex, closest, has_cyrillic
from fplcopilot.data import Bootstrap, Player
from fplcopilot.rag.entity_matcher import EntityMatcher, strip_accents

log = logging.getLogger(__name__)

DOMINANCE_RATIO = 5.0
MIN_OWNERSHIP = 5.0
MIN_FUZZY_OWNERSHIP = 5.0  # кириллица, похожая, но не точная («Эдегор»), — только популярные
MAX_READINGS = 8
_TOKEN = re.compile(r"[a-z0-9]+")
_POSSESSIVE = re.compile(r"['’]s\b", re.IGNORECASE)

Status = Literal["resolved", "ambiguous", "unknown"]


def name_tokens(text: str) -> tuple[str, ...]:
    """«João Pedro's» -> ('joao', 'pedro'); инициалы («M.» в «M.Salah») отбрасываются."""
    cleaned = strip_accents(_POSSESSIVE.sub("", text)).lower()
    return tuple(t for t in _TOKEN.findall(cleaned) if len(t) > 1)


def _contains(seq: tuple[str, ...], sub: tuple[str, ...]) -> bool:
    n = len(sub)
    if n == 0 or n > len(seq):
        return False
    return any(seq[i : i + n] == sub for i in range(len(seq) - n + 1))


@dataclass
class Resolution:
    mention: str
    status: Status
    player: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    note: str | None = None


class PlayerResolver:
    def __init__(self, bs: Bootstrap, *, squad_ids: Iterable[int] = ()) -> None:
        self.bs = bs
        self.squad = set(squad_ids)
        self._matcher: EntityMatcher | None = None
        self._names: NameIndex | None = None
        self._token_own: dict[str, float] = {}
        self._web: dict[int, tuple[str, ...]] = {}
        self._full: dict[int, tuple[str, ...]] = {}
        self._first: dict[int, str | None] = {}
        for p in bs.elements:
            self._web[p.id] = name_tokens(p.web_name)
            self._full[p.id] = name_tokens(p.full_name)
            first = name_tokens(p.first_name)
            self._first[p.id] = first[0] if first else None

    @property
    def matcher(self) -> EntityMatcher:
        if self._matcher is None:
            self._matcher = EntityMatcher.from_bootstrap(self.bs)
        return self._matcher

    @property
    def names(self) -> NameIndex:
        """Токены имён bootstrap для кириллического пути (строится при первом обращении)."""
        if self._names is None:
            tokens: list[str] = []
            for p in self.bs.elements:
                own = float(p.selected_by_percent or 0.0)
                for tok in LETTERS.findall(f"{p.web_name} {p.first_name} {p.second_name}"):
                    tokens.append(tok)
                    key = strip_accents(tok).lower()
                    self._token_own[key] = max(own, self._token_own.get(key, 0.0))
            self._names = NameIndex(tokens)
        return self._names

    # ---- представление ----

    def describe(self, p: Player) -> dict[str, Any]:
        return {
            "id": p.id,
            "name": p.web_name,
            "full_name": p.full_name,
            "team": self.bs.team(p.team).short_name,
            "position": p.position.short,
            "price": p.price,
            "status": p.status,
            "ownership": float(p.selected_by_percent or 0.0),
            "in_squad": p.id in self.squad,
        }

    # ---- поиск ----

    def sure_in_text(self, text: str) -> list[int]:
        """Уверенные упоминания по всему тексту (EntityMatcher: полные имена, контекст клуба)."""
        try:
            return list(self.matcher.match(text).players)
        except Exception:  # матчер — вспомогательный слой; без него работает шаг 2
            log.warning("EntityMatcher failed on %r", text, exc_info=True)
            return []

    def candidates(self, mention: str) -> list[Player]:
        toks = name_tokens(mention)
        if not toks:
            return []
        joined = "".join(toks)
        out: list[Player] = []
        for p in self.bs.elements:
            web, full = self._web[p.id], self._full[p.id]
            if (
                _contains(web, toks)
                or _contains(full, toks)
                or joined == "".join(web)  # «Gibbs-White» против web «GibbsWhite»
            ):
                out.append(p)
        return out

    def _lookup(self, mention: str) -> list[Player]:
        """Кандидаты по токенам; если их нет — уверенные id матчера по самому упоминанию
        («Bruno Fernandes»: в полном имени между ними «Miguel Borges», матчер берёт имя+фамилию)."""
        return self.candidates(mention) or [
            self.bs.player(pid) for pid in self.sure_in_text(mention)
        ]

    def cyrillic_readings(self, mention: str) -> list[tuple[str, float]]:
        """Латинские прочтения упоминания с кириллицей, лучшие первыми:
        «Паскаля Гросса» -> [('Pascal Groß', 1.0)], «Саки» -> [('Saka', 1.0), ('Sakyi', 1.0)].
        Нераспознанное слово перед распознанной фамилией отбрасывается («Коул Палмер» ->
        Palmer); нераспознанное последнее слово — прочтений нет."""
        words = [w for w in LETTERS.findall(mention) if len(w) > 1]
        options: list[list[tuple[str, float]]] = []
        for i, word in enumerate(words):
            if not has_cyrillic(word):
                options.append([(word, 1.0)])
                continue
            alts = closest(
                [
                    (tok, score)
                    for tok, score in self.names.match(word)
                    if score >= 1.0
                    or self._token_own.get(strip_accents(tok).lower(), 0.0) >= MIN_FUZZY_OWNERSHIP
                ]
            )
            if alts:
                options.append(alts)
            elif i == len(words) - 1:
                return []
        readings: dict[str, float] = {}
        for combo in islice(product(*options), MAX_READINGS) if options else ():
            readings.setdefault(" ".join(t for t, _ in combo), min(s for _, s in combo))
        return sorted(readings.items(), key=lambda x: -x[1])

    def _resolve_cyrillic(self, mention: str, sure: Iterable[int]) -> Resolution:
        hits = [
            (latin, score, cands)
            for latin, score in self.cyrillic_readings(mention)
            if (cands := self._lookup(latin))
        ]
        if not hits:
            return Resolution(mention, "unknown", note=f"'{mention}' not found in FPL bootstrap")
        hits = [h for h in hits if h[1] >= hits[0][1] - MARGIN]
        latin = hits[0][0]
        if len(hits) == 1:
            res = self.resolve(latin, sure=[*sure, *self.sure_in_text(latin)])
        else:
            merged = list({p.id: p for _, _, cands in hits for p in cands}.values())
            res = self._decide(latin, merged, sure)
        return replace(res, mention=mention)

    def resolve(self, mention: str, *, sure: Iterable[int] = ()) -> Resolution:
        if has_cyrillic(mention):
            return self._resolve_cyrillic(mention, sure)
        cands = self._lookup(mention)
        if not cands:
            return Resolution(mention, "unknown", note=f"'{mention}' not found in FPL bootstrap")
        return self._decide(mention, cands, sure)

    def _decide(self, mention: str, cands: list[Player], sure: Iterable[int]) -> Resolution:
        """Правила шага 3 для непустого списка кандидатов."""
        if len(cands) == 1:
            return Resolution(mention, "resolved", player=self.describe(cands[0]))
        sure_set = set(sure)
        by_sure = [p for p in cands if p.id in sure_set]
        if len(by_sure) == 1:
            return Resolution(mention, "resolved", player=self.describe(by_sure[0]))
        toks = name_tokens(mention)
        described = sorted(
            (self.describe(p) for p in cands),
            key=lambda d: (not d["in_squad"], -d["ownership"], d["name"]),
        )
        # (a) одно слово, которое у >= 2 кандидатов — имя: имя игрока не идентифицирует
        if len(toks) == 1 and sum(1 for p in cands if self._first[p.id] == toks[0]) >= 2:
            return Resolution(
                mention,
                "ambiguous",
                candidates=described,
                note=f"'{mention}' is a first name shared by {len(cands)} players",
            )
        # (b) ровно один из кандидатов — в составе менеджера
        in_squad = [d for d in described if d["in_squad"]]
        if len(in_squad) == 1:
            return Resolution(
                mention,
                "resolved",
                player=in_squad[0],
                note=f"'{mention}' resolved to the one in your squad ({in_squad[0]['full_name']})",
            )
        # (c) доминирование по владению
        top, second = described[0], described[1]
        if top["ownership"] >= MIN_OWNERSHIP and top["ownership"] >= DOMINANCE_RATIO * max(
            second["ownership"], 1e-9
        ):
            return Resolution(
                mention,
                "resolved",
                player=top,
                note=(
                    f"'{mention}' assumed to be {top['full_name']} ({top['team']}, "
                    f"{top['ownership']:.1f}% owned); say the full name for "
                    + ", ".join(d["full_name"] for d in described[1:4])
                ),
            )
        return Resolution(
            mention,
            "ambiguous",
            candidates=described,
            note=f"'{mention}' matches {len(cands)} players",
        )

    def resolve_all(
        self, mentions: Iterable[str], query: str
    ) -> tuple[list[dict[str, Any]], list[Resolution], list[Resolution], list[str]]:
        """(разрешённые игроки, неоднозначные, неизвестные, заметки)."""
        sure = self.sure_in_text(query)
        resolved: dict[int, dict[str, Any]] = {}
        ambiguous: list[Resolution] = []
        unknown: list[Resolution] = []
        notes: list[str] = []
        for m in mentions:
            m = m.strip()
            if not m:
                continue
            res = self.resolve(m, sure=sure)
            if res.status == "resolved" and res.player is not None:
                resolved.setdefault(res.player["id"], res.player)
                if res.note:
                    notes.append(res.note)
            elif res.status == "ambiguous":
                ambiguous.append(res)
            else:
                unknown.append(res)
        # уверенные упоминания матчера, которых роутер не назвал (или назвал иначе)
        for pid in sure:
            if pid not in resolved:
                resolved[pid] = self.describe(self.bs.player(pid))
        return list(resolved.values()), ambiguous, unknown, notes
