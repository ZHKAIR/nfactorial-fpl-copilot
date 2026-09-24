# Output template (fpl-transfer-analyst)

Every answer to an FPL decision question has the same five blocks, in this order. Fill every
number from a tool result; never derive one.

```markdown
**Verdict:** <one sentence — the tool's `recommendation` / `verdict` / `verdict_on_forced_sale`
in plain words; for a hit or chip: "Option (needs your confirmation): ...">

| option | xPts next GW | Δ vs hold (horizon N GW) | cost | risk |
|---|---|---|---|---|
| <route: out -> in> | <xi_points_after> | <gain_horizon, signed> | <hit_cost or "free"> | <risk_note> |
| Hold (roll the FT) | <baseline_xi_points / hold_xi_points> | 0 | free | <issue of the player kept> |
| <player A> (captain / compare) | <xpts> | <total_xpts over horizon> | — | <status, p_start> |

**Why**
- <news claim> [<source>, <dd.mm>]
- <model fact: xPts, p_start, fixture FSI, ownership> (no citation needed)

**Sources:** <source, dd.mm — url>; <source, dd.mm — url>

**Caveats:** data as of <as_of>; squad = picks of GW<squad_gw> (last finished GW) unless a
screenshot squad was provided; xPts model v0 (fixture provider team_rating); <tool notes,
`resolution_notes`, error hints>; only the next GW's move is actionable.
```

## Column sources

| column | tool field |
|---|---|
| xPts next GW (route) | `routes[i].xi_points_after` (XI after the route, incl. captain) |
| xPts next GW (hold) | `baseline_xi_points` (recommend_transfers) or `hold_xi_points` (simulate_scenario) |
| Δ vs hold | `gain_horizon` (sum over the horizon, vs no transfer); mention `gain_next_gw` in Why |
| cost | `hit_cost` (0 -> "free"; 4 -> "-4 hit") |
| risk | `risk_note`, or `issues[].kind` + `detail` of the player sold |
| captain rows | `captain_options[].xpts`, `captain_points` (2·xPts), `ownership`, `tag`, `fixture` |
| compare rows | `ranking[].total_xpts`, per-GW `by_gw[].xpts` |
| plan rows | `moves_by_gw[gw][]` (out, in, price_out, price_in, delta_xpts_horizon, paid) |

## Filled example (transfer, no hit)

**Verdict:** Sell João Pedro — the optimizer's best route sells him with a free transfer
(Konsa, João Pedro -> Barry, Guéhi; verdict go).

| option | xPts next GW | Δ vs hold (3 GW) | cost | risk |
|---|---|---|---|---|
| Konsa, João Pedro -> Barry, Guéhi | 60.17 | +9.78 | free | out João Pedro: doubtful (75%) |
| Palmer, João Pedro -> Barry, B.Fernandes | 59.70 | +9.72 | free | out Palmer: low_xpts |
| Hold (roll the FT) | 56.14 | 0 | free | João Pedro may miss GW5 |

**Why**
- João Pedro pulled out of the Brazil squad ahead of the internationals [ffscout, 16.09]; FPL
  lists him at 75% [fpl_api, 16.09].
- Guéhi: FSI 2 fixtures, p_start 0.95; Barry: 5.1 xPts next GW at £6.0.

**Sources:** ffscout, 16.09 — https://…; fpl_api, 16.09 — https://fantasy.premierleague.com/…

**Caveats:** data as of 17.09 11:00Z; squad = GW4 picks (transfers made this GW are not
visible before the deadline); xPts model v0; 2 free transfers are used, none banked.

## Paid action (hit / chip) — confirmation pattern

```markdown
**Option (needs your confirmation):** a -4 hit for Konsa, Palmer, João Pedro -> Saka, Barry,
Guéhi: +14.15 xPts over 3 GWs, but only −0.19 above the best free route after the 4-point cost
(verdict hit_not_worth).

| option | ... |
| -4 route | ... | -4 hit | ... |
| best free route | ... | free | ... |

Do you want to take the hit anyway? If not, I'll keep the free route as the recommendation.
```
