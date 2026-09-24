# FPL 2026/27 rules digest (v1, built-in; a proper strategy knowledge base is a later step)

Squad and budget
- 15 players: 2 goalkeepers, 5 defenders, 5 midfielders, 3 forwards; £100.0m initial budget;
  maximum 3 players from one club.
- Starting XI each gameweek: 1 GKP, 3–5 DEF, 2–5 MID, 1–3 FWD (11 players). Bench order 12–15
  (slot 12 = reserve goalkeeper) drives automatic substitutions when a starter does not play.
- Captain scores double; vice-captain takes over if the captain plays 0 minutes.
- Selling price: you receive half of any price rise since purchase (rounded down to £0.1m);
  price falls are taken in full. Public API shows current price, not the personal selling price.

Transfers and hits
- After GW1, 1 free transfer (FT) is added each gameweek; unused FTs roll over and accumulate
  up to a maximum of 5.
- Every transfer beyond the available FTs costs -4 points (a "hit") in that gameweek.
- Transfers made on a Wildcard or Free Hit are free and do not consume banked FTs (the FT count is
  preserved and still +1 next gameweek, capped at 5).
- Deadline: 90 minutes before the first kick-off of the gameweek; changes lock at the deadline.

Chips (each available once per half-season: first half GW1–19, second half GW20–38;
Wildcard windows start at GW2)
- Wildcard: unlimited free transfers for one gameweek; the first Wildcard expires after GW19.
- Free Hit: unlimited transfers for one gameweek, squad reverts afterwards.
- Bench Boost: bench players' points count for that gameweek.
- Triple Captain: captain scores triple instead of double.
- One chip per gameweek; Wildcard/Free Hit cannot be combined with another chip in the same GW.

Scoring (per match)
- Appearance: 1 point for playing, +1 for 60+ minutes.
- Goal: GKP/DEF 6, MID 5, FWD 4. Assist: 3.
- Clean sheet (60+ minutes): GKP/DEF 4, MID 1, FWD 0. Goals conceded: -1 per 2 goals (GKP/DEF).
- Saves: 1 point per 3 saves (GKP); penalty save 5; penalty miss -2; own goal -2.
- Defensive contribution ("DefCon", since 2025/26): DEF 2 points for 10+ clearances, blocks,
  interceptions and tackles (CBIT) in a match; MID/FWD 2 points for 12+ CBIT + recoveries
  (CBIRT). Maximum 2 points per match.
- Yellow card -1, red card -3. Bonus points 3/2/1 to the top three BPS scores in each match.

Strategy heuristics used by this project's optimizer (see docs/optimizer.md)
- A hit is worth taking only if the extra transfer gains more than 4 points over the horizon
  compared with the best free alternative (threshold depends on the strategy preset:
  conservative 2.0, balanced 1.0, aggressive 0.25 points above the 4-point cost).
- Rolling a transfer (banking an FT) is valued at about 1.5 points; £1m in the bank about
  0.08 points per gameweek; future gameweeks are discounted by 0.84 per gameweek.
- Wildcard is recommended when the wildcard squad beats the transfer plan by the preset's margin
  (3.0 / 2.0 / 1.0 points) AND at least 3 squad players have serious problems.
- Captaincy tags by ownership: >30% safe, 10–30% balanced, <10% differential.
