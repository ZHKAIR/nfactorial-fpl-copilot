# FPL 2026/27 rules digest (for the fpl-transfer-analyst Skill)

Reference only: the optimizer already encodes these rules; use this file to answer rules
questions and to sanity-check tool output. Numbers below are the official game rules, not model
outputs.

## Squad, budget, lineup

- 15 players: 2 GKP, 5 DEF, 5 MID, 3 FWD; £100.0m initial budget; max 3 players per club.
- Starting XI: 1 GKP, 3–5 DEF, 2–5 MID, 1–3 FWD. Bench slots 12–15 (12 = reserve GKP) set the
  order of automatic substitutions when a starter plays 0 minutes.
- Captain scores double; the vice-captain takes over only if the captain plays 0 minutes.
- Selling price: half of any price rise since purchase (rounded down to £0.1m) is kept; price
  falls are taken in full. The public API shows current prices, not the personal selling price —
  the optimizer's bank can be off by a few tenths for managers with big team-value growth.

## Transfers, free transfers, hits

- After GW1 one free transfer (FT) is added per gameweek; unused FTs bank up to a maximum of 5.
- Each transfer beyond the available FTs costs -4 points in that gameweek (a "hit").
- Transfers made with a Wildcard or Free Hit are free and do not consume banked FTs (the bank
  is kept and still +1 next GW, capped at 5).
- Tool fields: `free_transfers` (estimated from transfer history — the public API does not
  expose it), `hit_cost` = 4 × paid transfers, `paid: true` marks the paid move in a plan.

## Chips: two sets, GW19 expiry

- Every chip exists twice: one for the first half (GW1–19; Wildcard from GW2) and one for the
  second half (GW20–38). An unused first-half chip **expires after GW19**.
- Wildcard: unlimited free transfers for one GW, squad changes persist.
- Free Hit: unlimited transfers for one GW, squad reverts afterwards.
- Bench Boost: bench points count. Triple Captain: captain ×3 instead of ×2.
- One chip per gameweek; Wildcard/Free Hit cannot be combined with another chip in the same GW.
- Tool fields: `chips_available` (every chip the manager still has at the upcoming deadline:
  wildcard, freehit, bboost, 3xc), `wildcard` alternative in `build_gameweek_plan` /
  `simulate_scenario(use_wildcard=true)`; the optimizer recommends the Wildcard only when it
  beats the transfer plan by the strategy margin (3.0 / 2.0 / 1.0 pts) and at least 3 squad
  players have serious problems.
- Bench Boost and Triple Captain are modelled in `build_gameweek_plan(chips=[{gw, chip}])` for
  the gameweek the user names (`bboost` — the bench counts that GW, `3xc` — captain ×3): the
  transfers are optimised for it and `chips` in the result gives its xPts contribution. A chip
  in the first plan GW disables the Wildcard alternative. Free Hit is not modelled.

## Deadlines

- Deadline = 90 minutes before the first kick-off of the gameweek; all changes lock then.
- The public API shows a manager's transfers of the current GW only after its deadline, so the
  squad the tools see is the picks of the last **finished** GW (`squad_gw`) — always say so.
- `get_gameweek_context` returns `deadline` (UTC) and `as_of` (when the data was read).

## Scoring (per match)

| event | GKP | DEF | MID | FWD |
|---|---|---|---|---|
| playing (any minutes) | 1 | 1 | 1 | 1 |
| 60+ minutes | +1 | +1 | +1 | +1 |
| goal | 6 | 6 | 5 | 4 |
| assist | 3 | 3 | 3 | 3 |
| clean sheet (60+ min) | 4 | 4 | 1 | 0 |
| every 2 goals conceded | -1 | -1 | — | — |
| every 3 saves | 1 | — | — | — |
| penalty save / miss | +5 / -2 | | | |
| own goal | -2 | -2 | -2 | -2 |
| yellow / red card | -1 / -3 | | | |
| bonus (top-3 BPS in the match) | 3 / 2 / 1 | | | |

**Defensive contribution ("DefCon", since 2025/26):** DEF earn 2 points for 10+ clearances,
blocks, interceptions and tackles (CBIT) in a match; MID and FWD earn 2 points for 12+ CBIT +
recoveries (CBIRT). Max 2 points per match. The xPts model has a `defcon` component.

## Strategy heuristics used by the tools (not official rules)

- Hit verdict: `hit_marginal_gain` = extra discounted gain over the best free route − 4 must be
  ≥ the preset threshold (conservative 2.0, balanced 1.0, aggressive 0.25) for `go`.
- Rolling an FT is valued at ~1.5 pts; £1m in the bank ~0.08 pts/GW; future GWs are discounted
  by 0.84 per GW.
- Captain tags by ownership: >30% safe, 10–30% balanced, <10% differential.
