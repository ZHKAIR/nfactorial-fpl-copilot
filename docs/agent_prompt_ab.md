# Mini A/B: agent explain prompt v1 vs v2

Generated 2026-09-17 13:09Z by `scripts/agent_demo.py --ab`. Facts computed ONCE per scenario by the v2 graph (router v2, tools, KB context); the explainer (gpt-4o-mini, T=0.2) then ran twice on identical FACTS / EVIDENCE / CAVEATS — with `prompts/v1/explain.system.md` and `prompts/v2/explain.system.md`. Validation = the deterministic `agent/validate.py` (names, decimals, placeholder citations, KB markers); one regeneration with feedback on failure, as in the graph.

## Per scenario

| scenario | v | violations (1st) | regen | final | answer tok | cost $ | placeholders | Sources lines ok | Δ labelled | rule cited |
|---|---|---|---|---|---|---|---|---|---|---|
| 1. player_status | v1 | 0 | 0 | pass | 385 | 0.0006 | no | 3/3 | n/a | - |
| 1. player_status | v2 | 0 | 0 | pass | 401 | 0.0008 | no | 3/3 | n/a | - |
| 2. transfer | v1 | 0 | 0 | pass | 599 | 0.0009 | no | 5/5 | no | - |
| 2. transfer | v2 | 0 | 0 | pass | 616 | 0.0011 | no | 5/5 | yes | - |
| 3. captain | v1 | 0 | 0 | pass | 441 | 0.0007 | no | 3/3 | n/a | - |
| 3. captain | v2 | 0 | 0 | pass | 449 | 0.0009 | no | 3/3 | n/a | - |
| 4. plan | v1 | 0 | 0 | pass | 425 | 0.0007 | no | 3/3 | no | - |
| 4. plan | v2 | 0 | 0 | pass | 599 | 0.0009 | no | 3/3 | yes | - |
| 5a. what_if reject | v1 | 0 | 0 | pass | 338 | 0.0007 | no | 1/1 | n/a | - |
| 5a. what_if reject | v2 | 0 | 0 | pass | 479 | 0.0009 | no | 1/1 | yes | - |
| 5b. what_if confirm | v1 | 0 | 0 | pass | 372 | 0.0008 | no | 1/1 | n/a | - |
| 5b. what_if confirm | v2 | 0 | 0 | pass | 550 | 0.0011 | no | 2/2 | yes | yes |
| 7. clarification | v1 | 0 | 0 | pass | 640 | 0.0009 | no | 1/4 (undiscussed: João Pedro) | no | - |
| 7. clarification | v2 | 0 | 0 | pass | 478 | 0.0010 | no | 1/1 | yes | - |

## Totals

| metric | v1 | v2 |
|---|---|---|
| scenarios | 7 | 7 |
| validator violations before regeneration | 0 | 0 |
| regenerations | 0 | 0 |
| passed after ≤ 1 regeneration | 7/7 | 7/7 |
| answer tokens (completion, mean per scenario incl. regenerations) | 457 | 510 |
| explain cost, $ (all attempts) | 0.0054 | 0.0068 |
| answers with placeholder citations ([FACTS], [source, dd.mm]) | 0 | 0 |
| answers whose Sources list only discussed players / cited URLs | 6/7 | 7/7 |
| non-compliant Sources lines (undiscussed player or unknown URL) | 3/20 | 0/18 |
| tables with an explicitly labelled Δ column | 0/3 | 5/5 |
| hit/chip answers citing a rules_context excerpt as [source] | 0 | 1 |

## Reading the table

- **Same facts, different prompt.** The router, tools and `rules_context` retrieval ran once per
  scenario (v2 graph); both explainers saw byte-identical FACTS / EVIDENCE / CAVEATS. v1 received
  its original user message (no `LANGUAGE` line, no RULES CONTEXT block — exactly the shipped
  step-7 message); v2 received the v2 system prompt plus the `LANGUAGE` line and the RULES
  CONTEXT block (the same facts rendered explicitly). The comparison is therefore "v1 as shipped"
  vs "v2 as shipped" on identical computations.
- **Δ labelled** — the table header carries `Δ next GW` / `Δ horizon (3 GW)` instead of a bare
  `Δ` mixed into the xPts column (v1 weakness #2). v1: 0/3 tables, v2: 5/5 (v2 also produces a
  table in the two what_if cases where v1 gave prose only).
- **Rule cited** — the hit decision is grounded in one knowledge-base excerpt (`[premierleague]`
  "each additional transfer … will deduct 4 points"). Only 5b (confirmed hit) carries a
  `rules_context`; v1 ignores it (it has no instruction to use it), v2 cites it.
- **Sources compliance** — every Sources line must point to an EVIDENCE URL whose player is
  discussed in the text (or to a rules / KB URL). In this run v1 listed three João Pedro links in
  the clarification scenario without discussing him (weakness #1), v2 was clean. Over the three
  A/B runs made today (T = 0.2, so answers vary) the picture was: v1 non-compliant in 1/3 runs,
  v2 non-compliant in 1/3 runs (7a, the same three João Pedro lines) — the prompt alone does not
  make this reliable, which is why the graph now also removes such lines deterministically
  (`validate.prune_undiscussed_sources`, recorded in `validation.pruned_sources`).
- **Violations / regenerations** — in this run neither version needed a regeneration; in two of
  the three runs v1 fabricated a decimal in the transfer scenario (`2.67`, not in FACTS) and was
  regenerated once, v2 never did. Placeholders (`[FACTS]`, `[source, dd.mm]`) did not appear in
  either version in any run.
- **Cost** — v2 answers are ~10–15 % longer (explicit tables, one rule bullet) and the system
  prompt is ~0.7k tokens longer: +$0.0002 per answer (0.0068 vs 0.0054 for seven answers).

## Conclusion — v2 ships

`AGENT_PROMPT_VERSION=v2` is the default: same or fewer validator violations on identical facts,
labelled Δ columns in every table, a cited rule behind hit / chip verdicts, answers in the user's
language (not measurable in this English-only A/B; see scenarios 9–10 in
`docs/agent_demo_output_v2.md`), and no placeholder citations — for about +$0.0002 per answer.
v1 stays in `agent/prompts/v1/` for reproducibility (`--prompt-version v1`). What the A/B does
not settle: Sources hygiene is now enforced by code rather than by the prompt, and the sample is
seven scenarios at T = 0.2 — single-run differences of one scenario are noise, the three-run
pattern above is the evidence.

## Chat A/B: prompts v3 → v4 (24 Sep 2026) — the current version

Evolution at a glance (`agent/prompts/<v>/`; a file missing in a version falls back to the older
one, v4 → v3 → v2 → v1; the router schema follows the version — `llm.router_schema`):

| version | router | explainer |
|---|---|---|
| v1 | 10 intents, names as written | Verdict / Why / Sources / Caveats, English |
| v2 | + intent `player_ranking` (11), `language`, `ranking` | answer in the question's language, fixed table per intent, rules_context citations |
| v3 | names re-spelled to Latin, squad review → transfer | + NEWS block (candidate and club news), form notes |
| **v4** | + `standalone_query` from CONVERSATION, intents `squad_review` / `fixtures` / `chips` / `general_fpl`, `target_gw`, `chips` by gameweek, `team_mentions`, forecast ranking metrics (`xpts`, `xpts_per_million`, `horizon_gws`, `max_ownership`), a narrow off-topic | flexible format (answer first, table only to compare options), "say what FACTS cover — never relabel a gameweek", every reason is a FACTS field, captain / bench follow the optimizer, per-intent guidance for the new intents, conversation is context not facts, headings strictly in LANGUAGE, NEWS optional |

v4 router — the rules that did the work (the eval rows are not quoted; examples use other
players and phrasings):
- **Follow-ups**: rewrite the query into a complete question in the user's language, carrying over
  the task, players, gameweek, chips and constraints — *only what the user asked*, never the
  assistant's own suggestions (run 1 inherited "Wissa → Thiago" from the previous answer).
- **No catch-all refusal**: every FPL / Premier League question that fits no intent is
  `general_fpl` (partial answer from the gameweek data); `off_topic` = weather, recipes, coding,
  other fantasy games, and any request to ignore instructions / reveal the prompt.
- **Time in rankings**: "will score / next N gameweeks" → forecast metrics; "was / scored / last N"
  → past metrics (the v3 router put "next 3" into `last_n_gws`).
- **Chips and gameweeks**: `chips=[{chip, gw}]`, `target_gw` for "на gw7", `allow_hit=false` for
  «платные трансферы нельзя», `keep` for «не убирать X»; a lineup with a planned chip is a plan.

v4 explainer — beyond the v3 rules (numbers only from FACTS, citations, KB markers, NEWS):
- no fixed template; one or two sentences of answer first; no computed columns (the v3 captain
  table forced "×2" for every player — the 7.22 of the owner's screenshot);
- scope honesty: another gameweek / an unavailable chip / a missing player is the first sentence;
- no strategy talk of its own ("3-5-2 maximises the bench"), no ratings, no per-player number
  that FACTS do not give for that player;
- a closing line in the user message fixes the language of prose *and* headings (the system
  prompt's Russian examples leaked into English answers in runs 1–2).

Result (`docs/EVALS.md` §9, same code, `--prompt-version v3` vs `v4`, 60 rows): intent 45 → 59,
false refusals 10/55 → 0/55, off-topic still 5/5, p95 33.3 → 25.5 s; manual review of v4:
49 helpful / 4 bad (baseline 17 / 26). **v4 ships as the default**; v1–v3 stay for A/B.
