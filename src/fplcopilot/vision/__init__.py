"""Мультимодальный вход: скриншот «Pick Team»/«Transfers» -> строгий JSON -> валидированный состав.

Зачем: публичный FPL API отдаёт picks менеджера только за ЗАВЕРШЁННЫЕ туры. До дедлайна
черновик состава (после трансферов) без логина недоступен, поэтому пользователь загружает
скриншот, vision-модель читает карточки, а детерминированный код сопоставляет имена с живым
bootstrap и проверяет правила FPL. Подробности — docs/vision.md.
"""

from fplcopilot.vision.extract import parsed_from_raw, squad_from_image
from fplcopilot.vision.resolve import PlayerResolver
from fplcopilot.vision.schemas import (
    ParsedSquad,
    RawPlayer,
    ResolvedPlayer,
    ScreenshotSquadRaw,
    SquadIssue,
    VisionUsage,
)
from fplcopilot.vision.to_squad import to_squad
from fplcopilot.vision.validate import validate_squad

__all__ = [
    "ParsedSquad",
    "PlayerResolver",
    "RawPlayer",
    "ResolvedPlayer",
    "ScreenshotSquadRaw",
    "SquadIssue",
    "VisionUsage",
    "parsed_from_raw",
    "squad_from_image",
    "to_squad",
    "validate_squad",
]
