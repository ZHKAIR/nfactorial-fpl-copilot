# Golden dataset changelog

Every label change is listed here with the reason, so that results files from different dates can
be compared honestly. Format: date — file — row — what changed — why.

## 2026-09-17 — RAG v2 (A/B #2)

- `signals.jsonl` — `sig_haaland_fit` — `expected_minutes_min` 75 → **68**;
  `sig_raya_fit_gk` — `expected_minutes_min` 85 → **76**. *Definitional change, not a relabel.*
  In v2 `expected_minutes` is no longer an LLM guess ("minutes if he plays") but the unconditional
  expectation computed in code: `E[min] = p_start · START_MINUTES[pos] + (1 − p_start) · BENCH_MINUTES[pos]`
  with START = {GKP 90, DEF 85, MID 78, FWD 79} (mean minutes when starting, `player_gw_history`
  2026/27 GW1–4) and BENCH = {GKP 0, else 9} (≈ 0.5 chance of a ~18-minute sub appearance). The old
  lower bounds were inconsistent with the rows' own `start_probability_min` (a 0.85 start chance cannot
  give 85 expected minutes). New lower bound = formula(p_min), upper bound unchanged (90).
  `sig_gabriel_fit_ambiguous` (DEF, p_min 0.8 → 69.8 ≈ 70) already satisfied the formula and is unchanged.
  The v1 baseline outputs (Haaland 75, Raya 90) remain inside the new ranges, so A/B #1 numbers are not
  affected by this change.
- `no_coverage` rows (Kinsky, Tarkowski, van Ewijk, Botman) were re-checked after the entity-matcher
  re-tag (17 articles / 87 chunks changed): still zero tagged articles and zero word-boundary mentions
  before `as_of` — labels unchanged.
- No other label, tag or relevance change. Retrieval labels (`retrieval.jsonl`) untouched.

## 2026-09-23 — reproducible corpus (A/B #3 rerun, docs/EVALS.md §4.5)

- **No label change.** Labels describe the corpus the system had at `as_of`. On the live corpus
  `sig_van_ewijk_no_coverage_promoted` now has a mention: FFScout "FPL notes: Le Fee still on pens + £4.0m
  defender injury" (article 2286, RSS `pubDate` 17.09 01:30Z, first fetched 17.09 17:23Z — back-dated,
  public after `as_of`) says he was "fully rested in midweek". The row stays `no_coverage` because it is
  scored on the operational replay (`evals.run_rag --corpus-cutoff`, default `as_of`: only articles
  fetched and FPL snapshots taken by `as_of`), where the article does not exist yet.
- The FPL prior of past-`as_of` rows is now read by `snapshot_at` (fixes a leak of later FPL states into
  Caicedo, Doku, Amenda, Mosquera, Gomes, Tonali, Reinildo); labels were made from the snapshot observed at
  `as_of` and did not change.
