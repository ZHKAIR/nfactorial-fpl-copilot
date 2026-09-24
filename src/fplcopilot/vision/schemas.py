"""Схемы шага vision: сырой вывод модели (ровно то, что видно на экране) и валидированный состав.

Два слоя намеренно разделены:
- ScreenshotSquadRaw — контракт structured output. Все поля обязательны (strict-режим OpenAI
  не допускает default), «неизвестно» = null. Модель НЕ резолвит игроков и не знает id —
  она только читает пиксели.
- ParsedSquad — результат детерминированного кода: id из bootstrap, уверенность сопоставления,
  список нарушений правил FPL с флагом blocking. Именно его потребляют to_squad/оптимизатор/агент.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PositionShort = Literal["GKP", "DEF", "MID", "FWD"]
ScreenType = Literal["pick_team", "transfers", "points", "unknown"]
MatchMethod = Literal["exact", "exact+hints", "fuzzy", "fuzzy+hints", "ambiguous", "unresolved"]

POSITION_ORDER: tuple[str, ...] = ("GKP", "DEF", "MID", "FWD")
SQUAD_SIZE = 15
XI_SIZE = 11
BENCH_SIZE = 4
MAX_PER_CLUB = 3
SQUAD_BY_POSITION: dict[str, int] = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
# Допустимая расстановка стартового состава: (min, max) по позиции.
FORMATION_LIMITS: dict[str, tuple[int, int]] = {
    "GKP": (1, 1),
    "DEF": (3, 5),
    "MID": (2, 5),
    "FWD": (1, 3),
}


# ---------- слой 1: что видит модель ----------


class RawPlayer(BaseModel):
    """Одна карточка игрока так, как она напечатана на экране."""

    name_as_shown: str = Field(
        description=(
            "Player name exactly as printed on the card, keeping dots, hyphens and accents: "
            "'Haaland', 'B.Fernandes', 'João Pedro', 'Gibbs-White'."
        )
    )
    position: PositionShort | None = Field(
        description="GKP/DEF/MID/FWD from the pitch row or the card label; null if not visible."
    )
    club_hint: str | None = Field(
        description="Club name or abbreviation if written on the card/kit; null otherwise."
    )
    price_as_shown: float | None = Field(
        description="Price in £m as printed on the card ('£7.8m' -> 7.8); null if no price shown."
    )
    is_captain: bool = Field(description="True if the card carries a round 'C' badge.")
    is_vice_captain: bool = Field(description="True if the card carries a round 'V' badge.")
    is_bench: bool = Field(
        description="True if the card is in the substitutes strip below the pitch."
    )
    bench_order: int | None = Field(
        description="1..4 left-to-right on the bench (1 = bench goalkeeper); null for starters."
    )
    is_flagged: bool = Field(
        description="True if the card shows an injury/doubt/suspension icon (yellow/orange/red)."
    )


class ScreenshotSquadRaw(BaseModel):
    """Структурированный ответ vision-модели — ровно то, что просим в промпте.

    layout идёт ПЕРВЫМ полем намеренно: модель генерирует JSON по порядку схемы, и описание
    экрана «ряды + скамейка + счёт карточек» до списка игроков работает как структурированный
    chain-of-thought (промпт v2). В v1 без него модель на одном из эталонов приняла ряд FWD за
    скамейку и не прочитала полосу «Substitutes» вовсе (10 карточек из 14, без notes).
    """

    layout: str = Field(
        description=(
            "Before listing players: describe the screen top to bottom — header text, each pitch row "
            "with its card count (e.g. 'GKP row: 1 card; DEF row: 4 cards; ...'), then the separate "
            "substitutes strip below the pitch with its card count, and the total number of cards."
        )
    )
    players: list[RawPlayer]
    bank_as_shown: float | None = Field(
        description="'In the bank £0.5m' / 'ITB' / 'Bank' / 'Budget' value in £m; null if not shown."
    )
    free_transfers_as_shown: int | None = Field(
        description="Free transfers number if shown ('Free Transfers 2' -> 2); null otherwise."
    )
    screen_type: ScreenType
    notes: str = Field(
        description="Short remark about anything unreadable, cut off or unusual; '' if nothing."
    )


# ---------- слой 2: результат детерминированного кода ----------


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ResolvedPlayer(_Model):
    """Карточка после сопоставления с bootstrap. player_id=None — не найден или неоднозначен."""

    name_as_shown: str
    player_id: int | None = None
    web_name: str | None = None
    position: PositionShort | None = None  # по bootstrap, если найден; иначе как показано
    position_as_shown: PositionShort | None = None
    team_short: str | None = None
    team_name: str | None = None
    price: float | None = None  # now_cost/10 по bootstrap
    price_as_shown: float | None = None
    club_hint: str | None = None
    match_confidence: float = 0.0  # 0..1
    match_method: MatchMethod = "unresolved"
    candidates: list[str] = Field(default_factory=list)  # для ambiguous: "Gabriel (ARS DEF £8.0m)"
    is_captain: bool = False
    is_vice_captain: bool = False
    is_bench: bool = False
    bench_order: int | None = None
    is_flagged: bool = False

    @property
    def resolved(self) -> bool:
        return self.player_id is not None

    @property
    def is_starting(self) -> bool:
        return not self.is_bench

    @property
    def label(self) -> str:
        return self.web_name or self.name_as_shown


class SquadIssue(_Model):
    """Нарушение правила. blocking — состав нельзя передать дальше без исправления пользователем."""

    code: str
    message: str
    blocking: bool

    def __str__(self) -> str:
        return self.message


class VisionUsage(_Model):
    model: str
    detail: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0  # оценка по таблице цен в extract.PRICES_PER_1M
    latency_ms: float = 0.0


class ParsedSquad(_Model):
    """Валидированный состав со скриншота. is_valid == нет blocking-нарушений."""

    players: list[ResolvedPlayer]
    bank: float | None = None
    free_transfers: int | None = None
    gw: int | None = None  # ближайший тур с непрошедшим дедлайном (по as_of)
    screen_type: ScreenType = "unknown"
    captain_id: int | None = None
    vice_id: int | None = None
    starting_ids: list[int] = Field(default_factory=list)
    bench_order: list[int] = Field(default_factory=list)  # id в порядке скамейки, первый — вратарь
    issues: list[SquadIssue] = Field(default_factory=list)
    is_valid: bool = False
    raw: ScreenshotSquadRaw | None = None  # для отладки/evals
    usage: VisionUsage | None = None

    @property
    def blocking_issues(self) -> list[SquadIssue]:
        return [i for i in self.issues if i.blocking]

    @property
    def warnings(self) -> list[SquadIssue]:
        return [i for i in self.issues if not i.blocking]

    @property
    def resolved_players(self) -> list[ResolvedPlayer]:
        return [p for p in self.players if p.resolved]

    @property
    def resolved_count(self) -> int:
        return len(self.resolved_players)

    @property
    def has_bench(self) -> bool:
        return any(p.is_bench for p in self.players)
