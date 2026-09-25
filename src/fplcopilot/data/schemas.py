"""Типизированные схемы публичного FPL API (сезон 2026/27).

Принципы:
- extra="ignore": API добавляет поля без предупреждения, мы берём только нужные;
- всё, что может отсутствовать, — Optional с дефолтом;
- числа-строки ("9.2", "0.45") pydantic приводит к float сам.
"""

from __future__ import annotations

from datetime import datetime
from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class Position(IntEnum):
    GKP = 1
    DEF = 2
    MID = 3
    FWD = 4

    @property
    def short(self) -> str:
        return self.name


# ---------- bootstrap-static ----------


class Event(_Model):
    id: int
    name: str
    deadline_time: datetime
    finished: bool = False
    is_previous: bool = False
    is_current: bool = False
    is_next: bool = False
    average_entry_score: int | None = None
    most_captained: int | None = None
    most_selected: int | None = None


class Team(_Model):
    id: int
    name: str
    short_name: str
    strength: int | None = None
    strength_overall_home: int | None = None
    strength_overall_away: int | None = None
    strength_attack_home: int | None = None
    strength_attack_away: int | None = None
    strength_defence_home: int | None = None
    strength_defence_away: int | None = None


class Player(_Model):
    id: int
    code: int | None = None
    web_name: str
    first_name: str = ""
    second_name: str = ""
    team: int
    element_type: Position
    now_cost: int  # в десятых долях миллиона: 55 -> 5.5
    cost_change_event: int = 0
    cost_change_start: int = 0  # now_cost − цена на старте сезона

    # доступность
    status: str = "a"  # a=available d=doubtful i=injured s=suspended u=unavailable n=not in squad
    chance_of_playing_next_round: int | None = None
    chance_of_playing_this_round: int | None = None
    news: str = ""
    news_added: datetime | None = None
    can_select: bool | None = None
    can_transact: bool | None = None

    # прогноз и форма от FPL (бейзлайн)
    ep_next: float | None = None
    ep_this: float | None = None
    form: float | None = None
    points_per_game: float | None = None
    total_points: int = 0
    minutes: int = 0
    starts: int = 0
    selected_by_percent: float | None = None
    transfers_in_event: int = 0
    transfers_out_event: int = 0

    # сезонные суммы для компонентного xPts
    goals_scored: int = 0
    assists: int = 0
    clean_sheets: int = 0
    goals_conceded: int = 0
    saves: int = 0
    bonus: int = 0
    bps: int = 0
    expected_goals: float | None = None
    expected_assists: float | None = None
    expected_goal_involvements: float | None = None
    expected_goals_conceded: float | None = None
    tackles: int = 0
    recoveries: int = 0
    clearances_blocks_interceptions: int = 0
    defensive_contribution: float | None = None
    defensive_contribution_per_90: float | None = None
    yellow_cards: int = 0
    red_cards: int = 0

    # индексы FPL (ICT) — карточка игрока в UI
    influence: float | None = None
    creativity: float | None = None
    threat: float | None = None
    ict_index: float | None = None

    # стандарты FPL (пенальтист / штатник / угловые) — UI и слабый бонус xPts, core/ext_stats.py
    penalties_order: int | None = None
    corners_and_indirect_freekicks_order: int | None = None
    direct_freekicks_order: int | None = None

    @property
    def price(self) -> float:
        return self.now_cost / 10

    @property
    def position(self) -> Position:
        return self.element_type

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.second_name}".strip() or self.web_name

    @property
    def is_available(self) -> bool:
        return self.status == "a"


class Chip(_Model):
    id: int | None = None
    name: str
    start_event: int
    stop_event: int


class Bootstrap(_Model):
    events: list[Event]
    teams: list[Team]
    elements: list[Player]
    chips: list[Chip] = Field(default_factory=list)
    total_players: int | None = None

    # ---- удобные индексы ----
    def player(self, player_id: int) -> Player:
        return self._players_by_id[player_id]

    def team(self, team_id: int) -> Team:
        return self._teams_by_id[team_id]

    @property
    def _players_by_id(self) -> dict[int, Player]:
        cache = self.__dict__.get("_pbi")
        if cache is None:
            cache = {p.id: p for p in self.elements}
            self.__dict__["_pbi"] = cache
        return cache

    @property
    def _teams_by_id(self) -> dict[int, Team]:
        cache = self.__dict__.get("_tbi")
        if cache is None:
            cache = {t.id: t for t in self.teams}
            self.__dict__["_tbi"] = cache
        return cache

    @property
    def current_event(self) -> Event | None:
        return next((e for e in self.events if e.is_current), None)

    @property
    def next_event(self) -> Event | None:
        return next((e for e in self.events if e.is_next), None)

    def find_players(self, query: str) -> list[Player]:
        q = query.lower()
        return [p for p in self.elements if q in p.web_name.lower() or q in p.full_name.lower()]


# ---------- fixtures ----------


class Fixture(_Model):
    id: int
    code: int | None = None
    event: int | None = None  # None = матч ещё не назначен на тур (перенос)
    team_h: int
    team_a: int
    kickoff_time: datetime | None = None
    team_h_difficulty: int | None = None
    team_a_difficulty: int | None = None
    finished: bool = False
    started: bool | None = None
    team_h_score: int | None = None
    team_a_score: int | None = None


# ---------- entry / picks / history ----------


class Entry(_Model):
    id: int
    player_first_name: str = ""
    player_last_name: str = ""
    name: str = ""  # название команды менеджера
    summary_overall_points: int | None = None
    summary_overall_rank: int | None = None
    current_event: int | None = None
    started_event: int | None = None  # первый тур команды; picks появляются после его дедлайна
    joined_time: datetime | None = None
    player_region_name: str | None = None
    last_deadline_bank: int | None = None
    last_deadline_value: int | None = None
    last_deadline_total_transfers: int | None = None


class Pick(_Model):
    element: int
    position: int  # 1-11 старт, 12-15 скамейка (12 — запасной вратарь)
    multiplier: int  # 0 скамейка, 1 играет, 2 капитан, 3 TC
    is_captain: bool = False
    is_vice_captain: bool = False
    element_type: Position | None = None

    @property
    def is_starting(self) -> bool:
        return self.position <= 11


class EntryHistoryRow(_Model):
    event: int
    points: int = 0
    total_points: int = 0
    rank: int | None = None
    overall_rank: int | None = None
    bank: int = 0  # десятые миллиона
    value: int = 0  # стоимость состава + банк, десятые миллиона
    event_transfers: int = 0
    event_transfers_cost: int = 0
    points_on_bench: int = 0


class PicksResponse(_Model):
    active_chip: str | None = None
    entry_history: EntryHistoryRow
    picks: list[Pick]
    automatic_subs: list[dict] = Field(default_factory=list)


class ChipPlay(_Model):
    name: str
    time: datetime
    event: int


class ManagerHistory(_Model):
    current: list[EntryHistoryRow]
    past: list[dict] = Field(default_factory=list)
    chips: list[ChipPlay] = Field(default_factory=list)


# ---------- element-summary ----------


class PlayerGWHistory(_Model):
    """Одна строка истории игрока за один матч (element-summary.history)."""

    element: int
    fixture: int
    opponent_team: int | None = None
    round: int
    kickoff_time: datetime | None = None
    was_home: bool
    minutes: int = 0
    starts: int = 0
    total_points: int = 0
    goals_scored: int = 0
    assists: int = 0
    clean_sheets: int = 0
    goals_conceded: int = 0
    saves: int = 0
    bonus: int = 0
    bps: int = 0
    expected_goals: float | None = None
    expected_assists: float | None = None
    expected_goals_conceded: float | None = None
    tackles: int = 0
    recoveries: int = 0
    clearances_blocks_interceptions: int = 0
    defensive_contribution: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    own_goals: int = 0
    penalties_missed: int = 0
    penalties_saved: int = 0
    value: int | None = None
    selected: int | None = None
    team_h_score: int | None = None
    team_a_score: int | None = None


class PlayerFixture(_Model):
    id: int | None = None
    event: int | None = None
    event_name: str | None = None
    opponent_team: int | None = None
    is_home: bool
    difficulty: int | None = None
    kickoff_time: datetime | None = None


class ElementSummary(_Model):
    fixtures: list[PlayerFixture]
    history: list[PlayerGWHistory]
    history_past: list[dict] = Field(default_factory=list)


# ---------- leagues ----------


class LeagueEntry(_Model):
    entry: int
    player_name: str
    entry_name: str
    rank: int
    last_rank: int | None = None
    total: int
    event_total: int | None = None


class LeagueStandings(_Model):
    league_id: int
    league_name: str
    results: list[LeagueEntry]
    has_next: bool = False


# ---------- наши производные объекты ----------


class SquadPlayer(_Model):
    """Игрок в составе менеджера: pick + карточка игрока + команда."""

    pick: Pick
    player: Player
    team: Team

    @property
    def role(self) -> str:
        if self.pick.is_captain:
            return "C"
        if self.pick.is_vice_captain:
            return "VC"
        return "XI" if self.pick.is_starting else f"B{self.pick.position - 11}"


class Squad(_Model):
    manager_id: int
    gw: int
    players: list[SquadPlayer]
    bank: float  # миллионы
    team_value: float  # миллионы (состав + банк)
    active_chip: str | None = None
    free_transfers: int | None = None  # оценка по истории, см. fpl_client.estimate_free_transfers
    chips_used: list[str] = Field(default_factory=list)

    @property
    def starting_xi(self) -> list[SquadPlayer]:
        return [p for p in self.players if p.pick.is_starting]

    @property
    def bench(self) -> list[SquadPlayer]:
        return sorted(
            (p for p in self.players if not p.pick.is_starting), key=lambda p: p.pick.position
        )

    @property
    def captain(self) -> SquadPlayer | None:
        return next((p for p in self.players if p.pick.is_captain), None)

    def count_by_team(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for p in self.players:
            out[p.team.id] = out.get(p.team.id, 0) + 1
        return out
