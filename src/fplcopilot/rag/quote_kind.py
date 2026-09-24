"""Консервативный словарь «о чём цитата»: доступность / только форма / другое.

Используется (1) сигналом v4 как страховка поверх классификации модели: цитата со словами о
форме и БЕЗ единого слова о доступности не может быть доказательством доступности и уходит в
form_notes, а заметка о форме со словами о травме / бане / выборе состава — неверно разложенная;
(2) evals как метрика A/B #4. Приоритет у доступности: «played the full 90», «scored before
limping off» остаются доказательствами, поэтому словарь не выбрасывает то, из-за чего фильтр по
словам был отвергнут в v2 (docs/rag.md). Доля «формы» в evidence по этому словарю у v4 равна 0
по построению — поэтому в A/B #4 она проверяется ещё и ручной разметкой.
"""

from __future__ import annotations

import re
from typing import Literal

QuoteKind = Literal["availability", "form", "other"]

AVAILABILITY_TERMS = re.compile(
    r"injur|\bfit\b|fitness|train|doubt|knock|strain|tear|hamstring|calf|groin|knee|ankle|\bfoot\b"
    r"|thigh|muscl|\bill\b|illness|concussion|surgery|setback|recover|suspen|\bban|red card"
    r"|available|unavailable|absen|\bmiss|ruled out|sidelined|\binvolved\b|\bready\b|quite close"
    r"|\breturn(?:ed|ing)?\b|\breturns? (?:to|from)\b|return date|back in (?:training|the)"
    r"|expected back|chance of playing|\bstarted\b|\bstarts?\b (?:for|against|in|on)|to start\b"
    r"|starting (?:xi|line|eleven|role|spot|place)|line-?up|\bbench|\brest|night off|rotat"
    r"|substitut|\bsub\b|subbed|replaced|came on|half-time|\bminutes\b|\bplayed\b|feature"
    r"|squad|\bnamed\b|selected|selection|dropped|\bloan\b|joined|transfer|registered"
    r"|managed|look after|cotton wool|planned|fatigue|issue|precaution|scan|assess|behind",
    re.IGNORECASE,
)
FORM_TERMS = re.compile(
    r"scor|goal|assist|brace|hat-?trick|\bform\b|returns|\bpoints?\b|\bhaul|clean sheet|£"
    r"|\bprice|ownership|owned|projected|projection|\bxg|\bxa\b|shots?\b|chances? created"
    r"|dangerous|\bbest\b|impress|genius|brilliant|excellent|performance|rating|involvement"
    r"|captain|differential|penalt|\bpens\b|set-?piece|free-?kick|corner|firing|triumvirate"
    r"|inside forward|number 10|false nine|wing-?back|\(\d+(?:st|nd|rd|th) minute\)",
    re.IGNORECASE,
)


def quote_kind(quote: str) -> QuoteKind:
    """'availability' | 'form' (только форма / роль / цена) | 'other' (ни то, ни другое)."""
    if AVAILABILITY_TERMS.search(quote or ""):
        return "availability"
    if FORM_TERMS.search(quote or ""):
        return "form"
    return "other"
