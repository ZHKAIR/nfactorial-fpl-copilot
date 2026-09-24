"""Pydantic schemas of the golden dataset (evals/golden/*.jsonl) and loaders.

Labels are validated on load: ranges must be sane, as_of timezone-aware, tags non-empty,
`no_coverage` examples must expect `unknown` without evidence. Anything that fails here is a
labeling bug, not a model bug.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from evals import GOLDEN_DIR

Availability = Literal["fit", "doubtful", "injured", "suspended", "unavailable", "unknown"]
AVAILABILITY_CLASSES: tuple[str, ...] = (
    "fit", "doubtful", "injured", "suspended", "unavailable", "unknown",
)  # fmt: skip


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("as_of must be timezone-aware (use a trailing Z)")
    return value.astimezone(UTC)


class ExpectedSignal(BaseModel):
    availability: Availability
    # also-acceptable labels for genuinely two-sided cases (tag `conflict`); scored as "lenient"
    availability_alt: list[Availability] = Field(default_factory=list)
    start_probability_min: float = Field(ge=0, le=1)
    start_probability_max: float = Field(ge=0, le=1)
    expected_minutes_min: int | None = Field(default=None, ge=0, le=90)
    expected_minutes_max: int | None = Field(default=None, ge=0, le=90)
    return_gw: int | None = Field(default=None, ge=1, le=38)

    @model_validator(mode="after")
    def _ranges(self) -> ExpectedSignal:
        if self.start_probability_min > self.start_probability_max:
            raise ValueError("start_probability_min > start_probability_max")
        if (self.expected_minutes_min is None) != (self.expected_minutes_max is None):
            raise ValueError("expected_minutes_min/max must be given together")
        if (
            self.expected_minutes_min is not None
            and self.expected_minutes_max is not None
            and self.expected_minutes_min > self.expected_minutes_max
        ):
            raise ValueError("expected_minutes_min > expected_minutes_max")
        if self.availability in self.availability_alt:
            raise ValueError("availability_alt must not repeat the primary label")
        return self

    @property
    def accepted(self) -> set[str]:
        return {self.availability, *self.availability_alt}


class SignalExample(BaseModel):
    id: str = Field(pattern=r"^sig_[a-z0-9_]+$")
    player_name: str = Field(min_length=2)
    player_id: int = Field(gt=0)
    as_of: datetime
    expected: ExpectedSignal
    must_have_evidence: bool
    tags: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=40)
    source_urls: list[str]

    _as_of = field_validator("as_of")(_aware_utc)

    @model_validator(mode="after")
    def _consistency(self) -> SignalExample:
        if "no_coverage" in self.tags:
            if self.must_have_evidence:
                raise ValueError("no_coverage examples cannot require evidence")
            if self.expected.availability != "unknown":
                raise ValueError("no_coverage examples must expect availability=unknown")
            if self.source_urls:
                raise ValueError("no_coverage examples must not list source_urls")
        elif self.must_have_evidence and not self.source_urls:
            raise ValueError("must_have_evidence examples need at least one source_url")
        if self.expected.availability in {"injured", "suspended", "unavailable"} and (
            self.expected.start_probability_max > 0
        ):
            raise ValueError("injured/suspended/unavailable expect start_probability 0")
        return self


class RetrievalExample(BaseModel):
    id: str = Field(pattern=r"^ret_[a-z0-9_]+$")
    query: str = Field(min_length=8)
    player_id: int | None = Field(default=None, gt=0)
    team_id: int | None = Field(default=None, ge=1, le=20)
    as_of: datetime
    relevant_article_ids: list[int]
    relevant_urls: list[str]
    rationale: str = Field(min_length=40)

    _as_of = field_validator("as_of")(_aware_utc)

    @model_validator(mode="after")
    def _consistency(self) -> RetrievalExample:
        ids = self.relevant_article_ids
        if len(ids) != len(set(ids)):
            raise ValueError("relevant_article_ids contains duplicates")
        if len(ids) != len(self.relevant_urls):
            raise ValueError("relevant_urls must align 1:1 with relevant_article_ids")
        if self.player_id is None and self.team_id is None and ids:
            # allowed (pure semantic query) but flag obviously broken rows
            pass
        return self

    @property
    def has_relevant(self) -> bool:
        return bool(self.relevant_article_ids)


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_no}: invalid JSON: {exc}") from exc


def _check_unique_ids(rows: list[SignalExample] | list[RetrievalExample], name: str) -> None:
    ids = [r.id for r in rows]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"{name}: duplicate ids {dupes}")


def load_signals(path: Path | None = None) -> list[SignalExample]:
    path = path or GOLDEN_DIR / "signals.jsonl"
    rows = [SignalExample.model_validate(r) for r in _read_jsonl(path)]
    _check_unique_ids(rows, path.name)
    return rows


def load_retrieval(path: Path | None = None) -> list[RetrievalExample]:
    path = path or GOLDEN_DIR / "retrieval.jsonl"
    rows = [RetrievalExample.model_validate(r) for r in _read_jsonl(path)]
    _check_unique_ids(rows, path.name)
    return rows
