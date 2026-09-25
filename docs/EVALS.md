# Evals: golden dataset v1, automated RAG metrics, A/B #1 (retrieval modes), A/B #2 (RAG v2), A/B #3 (summary check, prompt v3, reproducible corpus), A/B #4 (form_notes, prompt v4), hyperparameters, model comparison, chat A/B, live GW5 xPts test

Everything lives in `evals/` (package) + `evals/golden/` (labels) + `evals/results/` (JSON
evidence). §3 (A/B #1) uses `evals/results/20260917T091040Z_retrieval.json` and
`evals/results/20260917T091040Z_signals.json` (corpus snapshot: 1102 articles / 1622 chunks visible at
`as_of`, `max(published_at)` = 08:30Z, 198 status snapshots).
§4 (A/B #2, RAG v2) uses the `20260917T09515*`–`20260917T1014*` files listed there. §7
(hyperparameters: temperature / top_p / max_tokens / determinism) and §8 (model comparison) use the
`20260917T1157*`–`20260917T1206*` files (`evals.run_hparams`, `evals.run_models`,
corpus 1109 / 1639); the per-role decisions they support are collected in
[`docs/LLM_CHOICE.md`](LLM_CHOICE.md). §4.5–§4.6 use `20260923T18*` files. `evals.run_rag` pins a run to
the corpus the system had at `as_of` (`--corpus-cutoff`, default `as_of`), so reruns reproduce the
numbers instead of drifting with the live corpus; the `20260917T*` files were produced without a cutoff (§5).

```
golden/*.jsonl ──▶ evals.run_rag [--prompt v1..v4 --retrieval v1|v2 --abstain on|off --corpus-cutoff as_of|<ISO>|none]
                                  ──▶ Retriever(mode, RetrievalConfig).search / extract_signal(save=False)
                                  ├─ retrieval: dedupe chunks→articles, P@k R@k hit@k MRR, top-1 score, distinct articles, latency
                                  ├─ signals:   availability acc / macro-F1, ranges, evidence, return GW (+source), abstentions, LLM calls, cost
                                  ├─ faithfulness: gpt-4o-mini judge over (summary, quotes, FPL status)
                                  └─ results/<UTC>_<suite>_p<prompt>-r<retrieval>-a<on|off>.json (rows + aggregates + git_commit + corpus + arm config)
```

## 1. Golden dataset

Fixed `as_of = 2026-09-17T09:00:00Z` (next GW = 5). Protocol, per-row rationale and known weaknesses:
[`evals/golden/README.md`](../evals/golden/README.md). 45 rows total.

**`signals.jsonl` — 27 players**

| expected class | n | who / why it is hard |
|---|---|---|
| doubtful | 9 | João Pedro 75% (knee, out of Brazil squad), Mosquera 50% (`stale_flag`, 13 days), White, Tonali (`headline_only`), Amenda (flag 90 min before `as_of`, promoted club), Gomes / Hincapie / James / Emersonn (`conflict`: FPL and news disagree) |
| injured | 5 | Caicedo ("expected back 18 Sep" = GW5 fixture, yet 0% and not training), Doku ("quite close"), D. Henderson (GK; Jordan Henderson also injured), Dasilva + Milenković (`fpl_api_only`) |
| fit | 5 | Haaland (heavy coverage), Gabriel (half-time sub was "planned"; a 15-year-old United "Gabriel" in the corpus), Porro (`conflict`: flag `d` vs De Zerbi "available"), Timber ("didn't feature" = rest), Raya (GK, "had the night off") |
| unknown | 4 | Kinsky (GK), Tarkowski, van Ewijk (promoted), Botman — `no_coverage`: no article, no mention |
| suspended | 3 | Foden (3 matches → GW7), Awoniyi (promoted, → GW7), Reinildo (ban is domestic only; "available" for Europe → GW6) |
| unavailable | 1 | Vicario (GK, loan) |

Tags: `return_date` 8 · `ambiguous_name` 7 · `conflict` 5 · `misleading_news` 4 · `goalkeeper` 4 ·
`promoted_club` 4 · `no_coverage` 4 · `heavy_coverage` 4 · `stale_flag` 2 · `fpl_api_only` 2 ·
`headline_only` 1 · `fresh_flag` 1. Conflict cases carry an alternate acceptable label
(`availability_alt`); the strict metric uses the primary label.

**`retrieval.jsonl` — 18 queries**, relevance labelled at *article* level (chunk ids are unstable):
injury (Caicedo, João Pedro, Mosquera, Doku, Shaw, Tonali), suspension (Foden, Awoniyi, Reinildo),
return date (Henderson, Doku, Caicedo), rotation/lineup (Man City v Norwich), press conference
(Arsenal injuries), a team-level "who missed training" query (Chelsea), a conflict case (Gomes),
an ambiguous-name case (Porro vs João Pedro), and **3 queries with no relevant article**
(Kinsky, Tarkowski, Botman). 15 answerable queries carry 69 relevant article labels (2–10 per query).

## 2. Metrics — what each measures, and what it does not

**Retrieval (per mode, article level, k = 8).** Chunks are deduped by `article_id` before scoring.
- `precision@k` = relevant articles among the ≤ k distinct articles in the top-k chunks, divided by k
  (so several chunks of one article count once — a mode that fills the top-8 with 4 chunks of the same
  article is penalised). Averaged over the 15 answerable queries only.
- `recall@k`, `hit@k`, `MRR` — standard, over answerable queries. Recall is capped for queries with
  > 8 relevant articles (João Pedro: 10).
- `top1_score_relevant` vs `top1_score_no_relevant` — mean score of the first chunk (dense cosine /
  RRF×decay / cross-encoder logit depending on mode) on answerable vs no-answer queries. A large gap means
  the score could be thresholded to abstain; a small gap means the mode is equally confident when wrong.
  *Not comparable across modes* (different scales) — read the gap within a mode.
- `mean_top1_age_days` — age of the top-1 chunk relative to `as_of`: a freshness proxy.
- latency p50/p95 — wall time of `Retriever.search`, warm process, one retriever per mode (so the
  query-embedding cache cannot flatter the second mode). Includes the OpenAI embedding call (~250 ms).

What retrieval metrics do **not** show: whether the retrieved text is *sufficient* to answer (that is
the signals suite), and label quality on the long tail — relevance is freshness-aware by protocol, so a
mode that surfaces a correct-but-superseded August headline is "wrong" here by design.

**Signals (per mode).** `extract_signal(save=False)`, prompt per arm (`--prompt`, default v2), gpt-4o-mini, T = 0.
- `availability_accuracy` (strict, primary label) and `availability_accuracy_lenient` (alternate
  labels accepted on `conflict` rows); `availability_macro_f1` over classes present (6 classes,
  1–9 rows each — a single `unavailable` row swings its F1 between 0 and 1).
- `start_probability_in_range_rate`, `expected_minutes_in_range_rate` (only where a range is labelled).
- `evidence_present_rate` over `must_have_evidence` rows; `false_evidence_rate` over `no_coverage` rows
  (any quote at all = failure — the validator should already make this 0).
- `return_gw_exact_rate` / `return_gw_within_1_rate` over the 8 `return_date` rows.
- `mean_confidence_expected_unknown` vs `_expected_known` — calibration sanity: unknown rows should sit
  near 0.
- `mean_validation_fixes` — how much the deterministic validator had to repair (lower = model output
  already obeyed the rules).
- tokens and `cost_usd` (list price $0.15 / $0.60 per 1M for gpt-4o-mini; an estimate, not a bill),
  latency p50/p95 of the full pipeline (retrieval + LLM).
- Added for A/B #2: `return_gw_breakdown` (exact / within ±1 / wrong / missing, by source `fpl_news` /
  `llm_date` / `llm_gw`, plus false-positive GWs on rows labelled null), `abstained_count`
  (`abstained_on_no_coverage` vs `abstained_false` — an abstention on a row that is *not* `no_coverage` is a
  false abstention), `llm_calls` (27 minus abstentions), `cost_usd_total`. Each row records `abstain_reason`
  and `abstain_top_score`. Retrieval rows record `n_distinct_articles`, `top1_final_score` (the ordering
  score, = rerank × date multiplier under retrieval v2) and `max_stage_score` over all candidates.

**Faithfulness** (LLM-as-judge, `evals/prompts/faithfulness_judge.md`, gpt-4o-mini, T = 0,
structured output, ≤ 25 calls per mode): for each signal with non-empty evidence, is every claim in
`summary` supported by the quoted evidence *or the official FPL status line the extractor was given*?
`supported` = 1, `partially_supported` = 0.5, `unsupported` = 0. It measures what a reader can verify
from the citations alone. It does **not** measure correctness (a faithful summary of a stale headline
is still faithful), it is a same-family judge (gpt-4o-mini judging gpt-4o-mini — shared blind spots),
and it never sees the full documents, so an unquoted-but-true detail is scored as unsupported.

## 3. A/B #1 — retrieval strategy

**Hypothesis.** `hybrid_rerank` (BM25 + dense → RRF → time-decay → flashrank cross-encoder) beats
`dense` (pgvector cosine + entity prefilter) on freshness-sensitive injury queries: higher recall@8 and
MRR, better downstream availability accuracy and lower false-evidence rate — at the cost of ~0.7 s per
query. `hybrid` (same fusion, no reranker) is the intermediate arm.

**Setup.** Same golden set, same `as_of`, k = 8, 30 candidates per list, half-life 7 days,
`ms-marco-MiniLM-L-12-v2`, prompt v1, gpt-4o-mini T = 0, one retriever per mode (no shared query cache),
synthetic warm-up query before timing. 18 × 3 searches, 27 × 3 extractions, 66 judge calls.
Total cost ≈ $0.05 (estimate from tokens). Wall time 5.5 min.

### 3.1 Retrieval (18 queries; P/R/hit/MRR over the 15 answerable ones)

| metric | dense | hybrid | hybrid_rerank |
|---|---|---|---|
| precision@8 (article level) | 0.433 | 0.383 | **0.458** |
| recall@8 | 0.794 | 0.658 | **0.837** |
| hit@8 | 1.000 | 1.000 | 1.000 |
| MRR | **0.947** | 0.933 | **0.947** |
| top-1 score, answerable queries | 0.724 (cosine) | 0.029 (RRF×decay) | 0.986 (rerank) |
| top-1 score, no-answer queries | 0.584 | 0.025 | **0.310** |
| mean top-1 age (days before `as_of`) | 10.0 | **1.0** | 4.9 |
| latency p50 / p95 (ms) | **284 / 661** | 284 / 348 | 944 / 1112 |
| stage means (ms) | embed 288, dense 24 | embed 250, dense 21, bm25 15 | + rerank **656** |

Per-query recall@8 (dense / hybrid / hybrid_rerank): Caicedo 0.50/0.50/**0.83**, Foden 1/0.67/1,
João Pedro 0.50/**0.70**/0.30, Mosquera 1/0.40/1, Doku 0.83/0.50/0.83, Porro 0.83/0.17/**1.0**,
Arsenal presser 0.22/**0.44**/0.22, Man City rotation 0.60/**1.0**/0.80, Awoniyi 1/0.75/1,
Henderson 1/1/1, Reinildo 1/0.50/1, Gomes 1/1/1, Shaw 1/0.67/1, Tonali 1/1/1, Chelsea training 0.43/0.57/0.57.

Observations:
- **Reranker adds recall (+4 pp vs dense, +18 pp vs hybrid) and precision, MRR is a tie** — every
  mode puts a relevant article first on 14/15 queries; the entity prefilter already does most of the work
  for player-scoped queries.
- **Plain `hybrid` is the worst arm.** RRF × time-decay promotes *fresh* chunks regardless of topic:
  Porro's top-8 is 7 chunks of one FFScout article plus a price-change table; Mosquera/Reinildo/Foden
  lose the official `fpl_api` item because it is 4–13 days old (decay 0.28–0.67 on an RRF score of
  ~0.03). Time-decay without a relevance stage after it is harmful.
- **Abstention signal.** On the 3 no-answer queries the cross-encoder scores 0.09 / 0.09 / 0.74 vs
  0.96–1.00 for answerable ones; dense cosine gives 0.54–0.66 vs 0.57–0.76 (overlapping — not
  thresholdable). A rerank-score threshold (~0.2) would abstain on 2 of 3 no-answer queries for free.
  The 0.74 outlier is an Everton injury headline for the Tarkowski query: team-adjacent noise still
  scores high.
- **No article diversity.** João Pedro / hybrid_rerank: 8/8 retrieved chunks are relevant but 6 belong
  to one FFScout round-up → 3 distinct articles, recall@8 = 0.30 (dense 0.50). Chunk-level precision is
  100 %; article-level recall is what the LLM actually gets. A per-article cap (≤ 2–3 chunks) or MMR is
  the obvious next change.
- **The reranker is date-blind.** It re-sorts the 30 fused candidates by text relevance only, undoing
  the decay ordering: on the Arsenal press-conference query its top-1 is a 27 Aug headline (age 20.8 d
  vs 1.9 d for `hybrid`). Freshness then depends on the LLM reading the `published_at` tags.
- Team-level queries (Arsenal presser 9 relevant, Chelsea training 7) are hard for all modes
  (R@8 0.22–0.57): many relevant articles, generic headlines, untagged items.

### 3.2 Signals (27 players, strict = primary label; lenient accepts `availability_alt`)

| metric | dense | hybrid | hybrid_rerank |
|---|---|---|---|
| availability accuracy (strict) | **0.963** (26/27) | 0.815 (22/27) | 0.889 (24/27) |
| availability accuracy (lenient) | **1.000** | 0.852 | 0.963 |
| availability macro-F1 | **0.973** | 0.694 | 0.918 |
| start_probability in range | **1.000** | 0.889 | 0.926 |
| expected_minutes in range (n = 12) | **1.000** | 0.833 | **1.000** |
| evidence present when required (n = 23) | **1.000** | 0.870 | **1.000** |
| false-evidence rate on `no_coverage` (n = 4) | 0.000 | 0.000 | 0.000 |
| return_gw exact / ±1 (n = 8) | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| mean confidence: expected unknown / known | 0.00 / 0.84 | 0.00 / 0.72 | 0.00 / 0.84 |
| mean validation fixes (all / excl. no_coverage) | 1.56 / 0.35 | 2.33 / 1.35 | 1.59 / 0.30 |
| faithfulness (judge), n judged | 0.761, 23 | 0.700, 20 | 0.739, 23 |
| judge verdicts supported / partial / unsupported | 12 / 11 / 0 | 8 / 12 / 0 | 11 / 12 / 0 |
| tokens prompt / completion | 73.2k / 3.8k | 83.4k / 4.0k | 70.5k / 4.1k |
| cost estimate signals + judge (USD) | 0.013 + 0.003 | 0.015 + 0.003 | 0.013 + 0.003 |
| latency p50 / p95 (ms), full pipeline | **2007 / 2598** | 2076 / 3007 | 2790 / 3758 |
| of which retrieval / LLM (mean ms) | 342 / 1772 | 310 / 1788 | 1041 / 1818 |

Strict misses: **dense** — Porro (`doubtful`, label `fit`; alternate accepted). **hybrid_rerank** —
Porro (same), Emersonn (`fit`; alternate accepted), Timber (`doubtful` ✗: "didn't feature … we need to
look after him" read as a fitness doubt). **hybrid** — Porro, Emersonn (`injured` ✗), and three
`unknown` on players with a valid official item (Timber, **Vicario `u`**, **Dasilva `i`**) because
time-decay pushed the only relevant `fpl_api` chunk out of the top-8 — exactly the retrieval failure
above, now visible downstream.

Confusion (strict, hybrid_rerank): injured 5/5, suspended 3/3, unavailable 1/1, unknown 4/4,
doubtful 8/9 (→ fit 1), fit 3/5 (→ doubtful 2). Errors concentrate on `conflict`/`misleading_news`
rows (strict 0.60–0.75), i.e. where the label itself is two-sided.

### 3.3 Conclusion and decision

The hypothesis is **half confirmed**. On retrieval, `hybrid_rerank` is the best arm: highest recall@8
and precision, tied MRR, fresher top-1 than dense (4.9 vs 10 days) and — the most useful side effect —
a rerank score that separates answerable from unanswerable queries. On the downstream signal it did
**not** beat `dense`: 24/27 vs 26/27 strict (2 rows, both two-sided conflict cases; 1 row ≈ 3.7 pp),
identical false-evidence rate (0 %), identical evidence-presence (100 %), faithfulness within noise
(0.74 vs 0.76). The intermediate arm `hybrid` is clearly worst on both suites and must never be the default.

Why the retrieval gain did not propagate: for player-scoped queries the **entity prefilter** already
gives dense a near-perfect top-1 (MRR 0.947), the LLM only needs one good chunk plus the official FPL
line, and the reranker's extra recall arrives as *more chunks of the same article*, not more sources.

**Decision.**
- `hybrid_rerank` **stays the default for the interactive path** (one player, one question): +0.7 s on a
  ~2.8 s request is acceptable, recall/freshness are better, and the rerank score is the only usable
  abstention signal we have (candidate for A/B #2: threshold ≈ 0.2 → `unknown` without an LLM call, saving the cost
  on no-coverage players).
- **`dense` becomes the batch-path mode** (refreshing signals for hundreds of players before a deadline):
  no accuracy loss on this set, 3× cheaper retrieval (0.7 s × 600 players ≈ 7 min saved per refresh).
- The ~0.7 s rerank is **justified by retrieval quality and abstention, not (yet) by accuracy**. It only
  pays off after two fixes: a per-article cap / MMR so extra recall means extra *articles*, and either a
  date-aware rerank (score × decay) or a fix so time-decay cannot demote the sole entity-matched
  official item. A/B #2 should test `ms-marco-TinyBERT-L-2` and N = 15 candidates to cut the 656 ms.
- Retire plain `hybrid` as a shipped option (keep it as an ablation arm in evals).

### 3.4 What else the run revealed (prompt v1 / matcher / judge)

1. **`return_gw` is broken in prompt v1**: 0/8 in every mode. The model returns `null` even when the
   official text says "Expected back 18 Sep" or "Suspended until 17 Oct", and when it does answer it
   returns the **day of the month** (Henderson → 11, Dasilva → 10, Milenković → 11 for "11/10 Oct")
   instead of the gameweek (6). Fix in v2: give the GW calendar in the prompt and derive GW in code.
2. **The `unknown` path is carried by the validator, not the model.** All 12 no-coverage extractions
   (4 players × 3 modes) needed 7–9 fixes: the model asserted a status and cited team headlines that do
   not mention the player; rule (f) dropped them. False-evidence rate 0 % is a pipeline property. A
   pre-LLM abstention (no entity-matched chunk → `unknown`) would save the call and the risk.
3. **Faithfulness ≈ 0.74 with zero `unsupported`**: half the summaries are `partially_supported`,
   mostly for (a) "for Gameweek 5" — the GW comes from the request, not from a quote, (b) "is available"
   for status-`a` players whose quotes are about form, not fitness (Haaland, Raya), (c) a real source
   conflict: FPL says Henderson "foot injury", Goal.com says "ankle" — the summary repeats FPL, the judge
   flags it. (a) is a judge-design artefact (quote-only judging); (b) and (c) are genuine v2 targets
   ("prefer fitness quotes; surface source disagreements").
4. **Entity matcher**: `players` tags miss untagged-but-relevant articles (Evening Standard items 741,
   560 with `players = []`) and over-tag surname substrings ("James", "White", "Gabriel" — a 15-year-old
   United forward named Gabriel is tagged to other players). `chunk_mentions_player` uses substring
   aliases, so "Egan"/"Sels" would match "Keegan"/"Brussels"; those players were kept out of the
   `no_coverage` set for that reason.
5. **Label weaknesses surfaced by the models**: Porro is called `doubtful` by all three modes with the
   16 Sep "available" quote in hand — the models follow the FPL prior harder than the label does; either
   the label should be `doubtful` with `fit` as alternate, or prompt rule 4 is too strong for `d`.
   Emersonn/Hincapie/Timber are the rows where reasonable readers disagree; they are marked as such.

## 4. A/B #2: prompt v1 → v2 + retrieval v2 + pre-LLM abstention

Every item in §3.4 became a change; this section measures them. Code: `src/fplcopilot/rag/`
(`extract.py`, `retrieve.py`, `entity_matcher.py`), prompts `src/fplcopilot/prompts/v2/`, design notes in
[`docs/rag.md`](rag.md). Golden label changes made for this run: two `expected_minutes` lower bounds
(definitional, see `evals/golden/CHANGELOG.md`); the four `no_coverage` rows were re-verified after the
matcher re-tag (still zero tagged articles, zero word-boundary mentions).

**Hypothesis.** (H1) Giving the model the gameweek calendar and asking for a *date* (`return_date`) while
the code converts date → GW (official FPL text first) lifts `return_gw` from 0/8 to ≥ 6/8. (H2) Rules
about player-specific, fitness-only, verbatim evidence plus a deterministic `expected_minutes` formula cut
validator fixes and raise faithfulness without hurting availability accuracy. (H3) Abstaining before the
LLM when no candidate mentions the player (or the top score is low) reproduces the 4/4 `unknown` rows at
zero LLM cost with no false abstentions. (H4) A per-article cap of 2 and a date-aware rerank order raise
article-level recall/diversity and top-1 freshness for `hybrid_rerank` at no latency cost.

**Setup.** Same golden set and `as_of` (2026-09-17T09:00Z), k = 8, 30 candidates, half-life 7 d,
`ms-marco-MiniLM-L-12-v2`, gpt-4o-mini T = 0, one retriever per mode, judge budget 20 per mode.
Corpus after the matcher re-tag: 1104 articles / 1625 chunks visible at `as_of` for the v1 arm
(`max(published_at)` 09:08Z); the ingest loop back-filled **one** more pre-`as_of` article (1105 / 1630)
before the v2 arms ran — each results file records what it saw. Reproducible with
`--corpus-cutoff 2026-09-17T10:14:17Z`: the shipped arm replays with the same prior on 27/27 rows and the
same chunks on 25/27, at 24–25/27 (§4.5). Arms:

| arm | prompt | retrieval | abstain | file |
|---|---|---|---|---|
| **v1** (baseline reproduction) | v1 | v1 (no cap, date-blind) | off | `20260917T095150Z_signals_pv1-rv1-aoff.json` |
| **v2 (score 0.2)** — as specified | v2 | v2 (cap 2, date-aware) | on, rerank threshold 0.2 / dense 0.40 | `20260917T100542Z_signals_pv2-rv2-aon.json` |
| **v2 (shipped)** — score gate off | v2 | v2 | on, mention gate only | `20260917T101417Z_signals_pv2-rv2-aon-score-off.json` (hybrid_rerank; dense is identical to the row above — its 0.40 gate never fired) |
| retrieval v1 / v2 | — | v1 / v2 | — | `20260917T100821Z_retrieval_pv2-rv1-aon.json`, `20260917T100849Z_retrieval_pv2-rv2-aon.json` |

135 extractions (27 × 5 mode-arms), 99 judge calls, ≈ $0.08 total (estimate from tokens).

### 4.1 Signals (27 players; strict = primary label, lenient accepts `availability_alt`)

| metric | dense v1 | dense v2 | hybrid_rerank v1 | hybrid_rerank v2 (score 0.2) | hybrid_rerank v2 (shipped) |
|---|---|---|---|---|---|
| availability accuracy (strict) | **0.963** (26/27) | 0.889 (24/27) | **0.889** (24/27) | 0.778 (21/27) | 0.852 (23/27) |
| availability accuracy (lenient) | **1.000** | 0.963 | **0.963** | 0.815 | 0.926 |
| availability macro-F1 | **0.973** | 0.907 | **0.918** | 0.640 | 0.733 |
| start_probability in range | **1.000** | 0.963 | **0.926** | 0.852 | 0.889 |
| expected_minutes in range (n = 12) | 1.000 | 1.000 | 1.000 | 0.750 | 0.917 |
| evidence present when required (n = 23) | **1.000** | 0.957 | **1.000** | 0.826 | 0.957 |
| false-evidence rate on `no_coverage` (n = 4) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| **return_gw exact / ±1 (n = 8)** | 0.000 / 0.000 | **1.000 / 1.000** | 0.000 / 0.000 | **1.000 / 1.000** | **1.000 / 1.000** |
| return_gw source (exact hits) | — | fpl_news 8 | — | fpl_news 8 | fpl_news 8 |
| false-positive return_gw (label null) | 1 | 0 | 0 | 0 | 0 |
| mean confidence: expected unknown / known | 0.00 / 0.83 | 0.00 / 0.82 | 0.00 / 0.84 | 0.00 / 0.69 | 0.00 / 0.81 |
| **mean validation fixes** | 1.56 | **0.41** | 1.41 | **0.04** | **0.26** |
| abstained: total / on `no_coverage` / false | 0 / 0 / 0 | **4 / 4 / 0** | 0 / 0 / 0 | 8 / 4 / **4** | **4 / 4 / 0** |
| LLM calls | 27 | **23** | 27 | 19 | **23** |
| **faithfulness (judge), n judged** | 0.725, 20 | **0.800**, 20 | 0.700, 20 | **0.816**, 19 | 0.750, 20 |
| judge verdicts supported / partial / unsupported | 9 / 11 / 0 | 13 / 6 / 1 | 8 / 12 / 0 | 12 / 7 / 0 | 10 / 10 / 0 |
| tokens prompt / completion | 81.3k / 3.8k | 81.3k / 3.0k | 78.6k / 4.1k | 65.9k / 2.7k | 80.7k / 3.2k |
| cost estimate signals + judge (USD) | 0.014 + 0.002 | 0.014 + 0.002 | 0.014 + 0.003 | 0.012 + 0.002 | 0.014 + 0.002 |
| latency p50 / p95 (ms), full pipeline | 2034 / 2900 | **1898** / 3005 | 2646 / 3509 | 2956 / 3630 | **2536** / 4614 |
| of which retrieval / LLM (mean ms) | 355 / 1767 | 382 / 1704 | 907 / 1768 | 1158 / 1924 | 940 / 1871 |

Strict misses — **dense v1**: Porro (`doubtful`, alternate accepted). **dense v2**: Porro (same), Emersonn
(`fit`, alternate accepted), **Timber → `unknown`** (7 fixes: the model's only quote was not verbatim, rule
(d) dropped it, rule (c) forced `unknown`; same 8 chunks as v1). **hybrid_rerank v1**: Porro, Emersonn
(alternate accepted), Timber (`doubtful` ✗). **hybrid_rerank v2 (shipped)**: the same three plus
**Vicario → `unknown`** (no evidence returned; his only official `fpl_api` chunk is 29 days old and the
fusion-stage time-decay leaves it outside the 30 candidates in *both* v1 and v2 — v1 got by on a Guardian
transfer-verdict line the v2 model did not cite). **hybrid_rerank v2 (score 0.2)** additionally abstained
on Gabriel, Emersonn, Vicario and Raya — see 4.3.

Confusion (strict, hybrid_rerank v2 shipped): injured 5/5, suspended 3/3, unknown 4/4, doubtful 8/9 (→ fit 1),
fit 3/5 (→ doubtful 2), unavailable 0/1 (→ unknown). Dense v2: injured 5/5, suspended 3/3, unavailable 1/1,
unknown 4/4, doubtful 8/9, fit 3/5 (→ doubtful 1, unknown 1).

### 4.2 Retrieval (18 queries; P/R/hit/MRR over the 15 answerable; retrieval v1 vs v2, same reranker)

| metric | hybrid_rerank v1 | hybrid_rerank v2 | dense v1 | dense v2 |
|---|---|---|---|---|
| precision@8 (article level) | 0.458 | **0.467** | 0.433 | **0.450** |
| recall@8 | 0.837 | **0.840** | 0.794 | **0.808** |
| hit@8 | 1.000 | 1.000 | 1.000 | 1.000 |
| MRR | **0.947** | 0.889 | 0.947 | 0.947 |
| mean distinct articles in top-8 | 6.83 | **7.22** | 6.33 | **7.22** |
| mean top-1 age (days before `as_of`) | 4.9 | **1.5** | 10.0 | 10.0 |
| top-1 raw score, answerable / no-answer | 0.986 / 0.310 | 0.957 / 0.304 | 0.724 / 0.584 | 0.724 / 0.584 |
| max raw rerank score, answerable / no-answer | 0.986 / 0.310 | 0.986 / 0.310 | — | — |
| latency p50 / p95 (ms) | 1019 / 1370 | 1021 / 1313 | 302 / 344 | 261 / 427 |

Per-query changes (hybrid_rerank v1 → v2): **João Pedro recall 0.30 → 0.50, 3 → 6 distinct articles** (the
cap did exactly what §3.1 asked); Arsenal press conference RR 0.2 → 1.0 and top-1 age 20.8 → 1 d
(date-aware order); Caicedo top-1 age 17 → 0 d, Mosquera 12 → 1 d; Porro recall 1.0 → 0.83 (one
5-day-old relevant headline pushed out by a fresher João Pedro round-up). MRR lost 0.058 on three queries
where several items score ≈ 0.99 and the date multiplier now decides: Shaw (top-1 became the same-day
Sky "could leave next summer" gossip; the relevant MEN round-up is #3), Tonali (a same-week BBC Garner
feature that mentions him overtook the 3-day-old official `fpl_api` flag), Chelsea training (17 Sep BBC
Lavia piece over the 16 Sep "TRIPLE injury blow" headline). Recall is unchanged on all three — the relevant
article is still in the top-8, just not first. Dense: the cap alone adds +1.4 pp recall, +1.7 pp precision,
+0.9 distinct articles, no MRR change.

### 4.3 What improved, what regressed, why

**Improved.**
- **`return_gw`: 0/8 → 8/8 exact in both modes** (H1 confirmed, stronger than hoped). All eight come from
  the deterministic path (`fpl_news`): "Expected back 18 Sep" / "Suspended until 17 Oct" parsed from the
  official FPL text and mapped to the club's first fixture on/after that date. The model's own
  `return_date` was never needed on this set (it is the fallback when the official text has no date). Zero
  false-positive return GWs (v1 dense had one).
- **Validator fixes 1.4–1.6 → 0.04–0.41 per row** (H2). The 12 no-coverage extractions that needed 7–9
  fixes each are now abstentions (0 fixes, 0 LLM calls); the remaining fixes are single quote alignments.
- **Faithfulness 0.70–0.73 → 0.75–0.82**, `supported` 8–9 → 10–13 of 20 (H2). Summaries now name the source
  and date ("according to the official FPL news (2026-09-13)") and no longer assert "for Gameweek 5" from
  the request. Remaining `partially_supported` verdicts are mostly the judge wanting the FPL status line to
  *confirm* availability for status-`a` players, and the genuine foot-vs-ankle source conflict (Henderson).
  One `unsupported` (dense, Haaland): "According to the official FPL status, he is fit" — status `a` is
  "available", not a fitness statement; the v2 rule about form quotes did not stop the model from leaning
  on the FPL line when the only player-specific documents are about goals.
- **Abstention (mention gate): 4/4 `no_coverage` rows, 0 false, 4 LLM calls saved per mode** (H3 for the
  mention gate). Same `unknown` output as before, but as a pipeline decision with a recorded reason instead
  of a validator rescue.
- **Retrieval v2** (H4): more distinct articles (+0.4/+0.9), João Pedro recall +20 pp, top-1 freshness
  4.9 → 1.5 d for hybrid_rerank, precision/recall up in both modes, latency flat.
- `expected_minutes` is now a documented formula (`E[min] = p·START[pos] + (1−p)·BENCH[pos]`), so the
  in-range metric is a consistency check on `start_probability`, not a second guess by the model.

**Regressed.**
- **Strict accuracy −1 row (hybrid_rerank) / −2 rows (dense)**; lenient 1.000 → 0.963 and 0.963 → 0.926.
  Both dense misses are one-offs of the kind the n = 27 caveat in §5 warns about: Emersonn flipped
  doubtful → fit (a two-sided `conflict` row, alternate accepted), and Timber lost his single quote to the
  verbatim rule under the new prompt (same retrieved chunks; T = 0 does not make the model deterministic
  across prompts). The hybrid_rerank miss (Vicario → `unknown`) is a real, pre-existing retrieval gap now
  exposed: **the official `fpl_api` item can be older than the decay horizon and fall out of the candidate
  set** — v1 only scored the row because the model cited an unrelated transfer line.
- **The rerank-score gate at 0.2 is harmful** (H3 for the score gate rejected): 4 false abstentions
  (Raya max raw rerank 0.156, Gabriel 0.066, Emersonn 0.017, Vicario 0.010). The cross-encoder answers
  "is this about *injury/fitness*?", not "is this about the player?": a fit player whose news is a planned
  half-time sub, a night off or a loan move scores ≈ 0 for the injury-shaped query, exactly like a player
  with no news at all (Kinsky 0.008, Botman 0.008). The §3.1 observation (0.09/0.09/0.74 vs 0.92–1.0) came
  from *question-shaped* retrieval queries whose relevant documents were injury news; it does not transfer
  to the signals path. The dense cosine is not separable either (calibration in `docs/rag.md`).
- MRR −0.058 for hybrid_rerank v2 (three top-1 flips among ≈ 0.99-scored items, recall unchanged).
- `evidence_present_rate` 1.000 → 0.957 in both modes: the Timber (dense) and Vicario (hybrid_rerank) rows above.
- No cost or latency win beyond the saved LLM calls: prompt v2 is longer (calendar block, ~+300 tokens per
  call), so tokens per run are flat at ≈ 81k despite 4 fewer calls.

**Decision — what ships (defaults in `config.py`).**
- **Prompt v2** (`RAG_PROMPT_VERSION=v2`): +8 return GWs, −1.2 fixes per row, +0.05–0.08 faithfulness for
  a within-noise change in strict accuracy. v1 stays on disk for the diff and as an eval arm.
- **Retrieval v2** (`RAG_PER_ARTICLE_CAP=2`, `RAG_DATE_AWARE_RERANK=true`): better recall, diversity and
  freshness in both modes; the MRR dip is three same-day ties. Candidate for v3: apply the date multiplier
  only within a score band (tie-break) so ≈ 0.99 relevant items are not overtaken by fresher gossip.
- **Abstention on, score gates off** (`RAG_ABSTAIN=true`, `RAG_ABSTAIN_SCORE=0`, `RAG_ABSTAIN_DENSE_SCORE=0`):
  the deterministic "no candidate mentions the player" rule does the whole job (4/4, 0 false); both score
  thresholds stay as settings for recalibration but are disabled because neither separates "no coverage"
  from "covered but not injured" on this set.
- Mode defaults unchanged from §3.3 (`hybrid_rerank` interactive, `dense` batch).

**Next fixes (v3), in priority order.** (1) Pin the player's own `fpl_api` chunk into the candidate set
when it exists before `as_of` (Vicario; the §3.3 "sole entity-matched official item" problem), or exempt
`fpl_api` from the fusion-stage decay. (2) A form-quote filter or a separate `form_evidence` field so
Haaland-type rows do not cite "projected attacking returns" as fitness evidence. (3) Date multiplier as
tie-break only. (4) Re-check Timber with a small T = 0 repeat to see whether the verbatim drop is stable.

### 4.4 Prompt evolution (v1 → v2)

Files: [`prompts/v1/signal_extraction.system.md`](../src/fplcopilot/prompts/v1/signal_extraction.system.md),
[`prompts/v1/signal_extraction.user.md`](../src/fplcopilot/prompts/v1/signal_extraction.user.md) →
[`prompts/v2/signal_extraction.system.md`](../src/fplcopilot/prompts/v2/signal_extraction.system.md),
[`prompts/v2/signal_extraction.user.md`](../src/fplcopilot/prompts/v2/signal_extraction.user.md)
(`diff -r src/fplcopilot/prompts/v1 src/fplcopilot/prompts/v2`). The structured-output schema changed with it
(`SignalDraft` → `SignalDraftV2`: `expected_minutes` removed, `return_date` added).

| change in v2 | eval failure that motivated it (A/B #1) | effect in A/B #2 |
|---|---|---|
| Gameweek calendar block in the user prompt (deadline, match window, the club's own fixture date for the next 8 GWs); rule 10: copy the stated date into `return_date`, `return_gw` only when a document names the GW, never a day of the month; code converts date → GW (official FPL text first) | `return_gw` 0/8 in every mode; the model returned `null` or the **day of the month** (Henderson → 11, Dasilva → 10) | 8/8 exact, 0 false positives |
| `expected_minutes` removed from the output; computed by formula from availability, start_probability and position | doubtful players always got 30 minutes; fit players 75/90 "by feel" | metric now checks `start_probability` consistency; no invented numbers |
| Rule 2 with examples: teammates, manager, club round-up headlines are not evidence ("Spurs injury update: Tonali, Porro…" ≠ Kinsky) | 12 no-coverage extractions cited team headlines; only rule (f) saved them | now moot for no-coverage rows (abstention), fixes on covered rows 0.04–0.41 |
| Rule 3: only fitness / injury / training / selection / suspension / transfer quotes; form, goals, projections, prices are not evidence, with a positive and a negative example | faithfulness `partially_supported` for Haaland/Raya "is available" backed by form quotes | partially effective: Haaland still cites goal reports; the summary now separates "available (FPL)" from the goal fact |
| Rule 6: fixed summary "No player-specific news found." with all-neutral fields | model wrote "is available" with no documents | rows with no player-specific chunks never reach the model any more; the fixed text is the fallback if they do |
| Rule 7: verbatim *including typos, prices and brackets*, with the "(£7.8m)" example | `align_quote` had to repair quotes where the model dropped "(£7.8m)" or fixed typos | single quote fixes remain on 4–5 rows per mode (Foden, João Pedro, Porro, Henderson, Haaland) and one verbatim drop cost the Timber row — quoting long FFScout sentences remains imperfect |
| Rule 11: do not restate the request as a fact ("for Gameweek 5"), name source/date, state source disagreements; user prompt no longer says "for GW{n}" | judge flagged "for Gameweek 5" as unsupported (it came from the request); FPL "foot" vs Goal.com "ankle" repeated without comment | summaries name source and date; the foot/ankle conflict is still repeated rather than flagged |

### 4.5 A/B #3: summary check + prompt v3 — and why the first run showed 0.741

**Hypothesis.** (1) A deterministic check of the signal summary (`rag/summary_check.py`: dates, numbers and
proper names only from the cited documents, the FPL prior and the GW calendar) removes unsupported
sentences — live example (Cherki): "… scoring in the last match against Sunderland (BBC, 2026-09-20)"
with no cited document saying so — without touching availability. (2) Prompt v3 (= v2 + rule 13:
relative periods such as "the next two matches" count from the document's date; the FPL date wins)
does not lower availability accuracy.

**The first run (0.741) measured a leak, not the arms.** The first A/B #3 files
(`20260923T175751Z_…-nocheck`, `…175931Z_…-check`, `…180114Z_pv3-…-check`) gave 0.741 / 0.667 / 0.741,
down from 0.852. Golden `as_of` is fixed and retrieval filters `published_at <= as_of`, so corpus growth
alone cannot explain the drop. Row-by-row against the A/B #2 shipped arm (`20260917T101417Z`), two inputs
had changed for the same `as_of`:

1. **FPL prior from the future — code bug, main cause (−3 rows).** `fpl_prior` for a past `as_of` took the
   latest `player_status_snapshots` row with `coalesce(news_added, snapshot_at) <= as_of`. FPL does **not**
   update `news_added` when it changes the chance or the text or clears the news (i → a), so a snapshot taken
   after `as_of` keeps an old `news_added`, passes the filter and wins the `id DESC` tie. 34 of the 55
   snapshots taken after `as_of` leak this way; 7 golden rows got a future prior (priors identical on
   20/27 rows between the buggy and the fixed run, chunks identical on 27/27):
   Caicedo `i/0 "Expected back 18 Sep"` → `d/50 "50% chance"` (snapshot 17.09 13:53Z, `news_added` 30.08),
   Doku `i "Expected back 20 Sep"` → `a/100` (22.09, `news_added` 18.08), Amenda `d/50` → `i/0 "Unknown
   return date"` (22.09), Mosquera, Gomes, Tonali `d` → `a`, Reinildo `s` → `a`. Labels flipped on three:
   Caicedo → doubtful, Doku → fit, Amenda → injured; return GW exact 8/8 → 6–7/8 (the new texts carry no
   date). The A/B #2 runs were clean only because those snapshots did not exist when they ran. **Fix:** the
   time of an FPL state is `snapshot_at` (when the ingest observed it); the prior is the last snapshot
   observed by `as_of` (`rag/extract.py: prior_from_snapshots`, tests `tests/test_asof_replay.py`).
   `core/signals.py` had the same mistake (the rule "FPL status `a` newer than the signal cancels it" could
   not fire for a cleared status while `changed_at` was the old `news_added`) and uses `snapshot_at` too.
2. **Articles published before `as_of` but collected later — data (−1 row).** 14 articles / 61 chunks with
   `published_at <= as_of` were fetched after the A/B #2 cutoff (2026-09-17T10:14:17Z): 1105 + 14 = 1119,
   exactly the count in the first A/B #3 files. Two are FFScout "FPL notes" with RSS `pubDate`
   17.09 01:00Z / 01:30Z that the 30-minute ingest loop first saw at 16:53Z / 17:23Z, while FFScout
   items with later `pubDate`s were picked up within 0.2–2 h — the `pubDate` is back-dated and the articles went public after `as_of`. Article 2286 names van
   Ewijk ("fully rested in midweek") → the `no_coverage` row becomes `fit` (false evidence 0/4 → 1/4,
   abstentions 4 → 3), and the new chunks changed the retrieved top-8 of 11/27 rows.
3. **Rejected.** Retrieval / matcher code: `retrieve.py`, `entity_matcher.py` are unchanged from A/B #2;
   the only `--rematch` ran just before the A/B #2 runs. Code: the committed revision named in the file
   (`20260923T181616Z_…-oldcode52936c7-livedb.json`), run in a separate worktree against the same DB,
   reproduces the first-run v2 arm row for row — identical chunks, prior and label on 27/27 — so the
   uncommitted working-tree changes are not the cause. Golden: `signals.jsonl` is unchanged from A/B #2.
   LLM variation: ±1–2 rows, below.

**Reproducible corpora.** `evals.run_rag --corpus-cutoff` also requires `news_articles.fetched_at <= cutoff`
(dense SQL and BM25 corpus) and `snapshot_at <= cutoff` (prior); without a cutoff the production SQL is
unchanged. Default `as_of` = **operational replay**: only what the system had collected at each row's
`as_of` (1102 articles / 1622 chunks — exactly the A/B #1 corpus). `2026-09-17T10:14:17Z` = the A/B #2
corpus (1105 / 1630). `none` = **research reconstruction**: everything published by `as_of` in the live
corpus — it drifts with every back-fill and back-dated `pubDate` and can contain what was not public at
`as_of`. Replaying the A/B #2 shipped arm on its own corpus (`20260923T182050Z_pv2-…-cut0917T1014-nocheck`):
prior identical 27/27, chunks identical 25/27 (Dasilva, Botman differ in the 8th chunk only), labels
identical 25/27; the two flips (Timber doubtful → fit, Vicario unknown → unavailable) have identical chunks
and prior — run-to-run variation of gpt-4o-mini between runs six days apart (§7.1: 1 of 23 between
same-day runs). The A/B #2 number 0.852 reproduces as 0.889–0.926.

**Arms on identical data** (hybrid_rerank, retrieval v2, abstention on, no judge; strict · lenient of 27):

| data | v2, no check | v2 + check | v3 + check |
|---|---|---|---|
| A/B #2 corpus, `--corpus-cutoff 2026-09-17T10:14:17Z` (1105 / 1630) | **25 · 27** | 24 · 26 | **25 · 27** |
| operational replay, default cutoff `as_of` (1102 / 1622) | 24 · 26 | — | **25 · 27** (twice) |
| live corpus, `--corpus-cutoff none` (1119 / 1691) | 23 · 25 | 23 · 25 | 22 · 24 |
| live corpus, prior as in the first run (bug) | 20 · 22 | 18 · 20 | 20 · 22 |

A/B #2-corpus row in detail (v2 / v2 + check / v3 + check): macro-F1 0.948 / 0.763 / 0.948; evidence
present 23/23, 22/23, 23/23; false evidence 0/4; return GW exact 8/8; mean validation fixes 0.07 / 0.44 /
0.11; ≈ $0.014 per arm. Misses on the frozen corpora: Porro and Emersonn in every arm (two-sided
`conflict` rows, alternate label accepted) plus at most one noisy row — Vicario (`unknown`, v2 + check),
Timber (`unknown` after a dropped non-verbatim quote, v2 replay). On the live corpus add van Ewijk (the
back-dated article), Timber (`doubtful` in all 7 runs on the live chunks) and Raya (v3: quote dropped).
Files: `20260923T182050Z_*-cut0917T1014-*`, `20260923T182334Z_*-cutasof-check`,
`20260923T184726Z_*-cutasof-{nocheck,check-rep2}`, `20260923T182334Z_*-livecorpus-*`.

**Reading.** v2 → v2 + check differences are LLM variation by construction: the check runs after
validation and only edits `summary` (never availability, evidence or numbers), the prompt is identical,
and the rows that differ have identical chunks (Vicario here; Hincapié and Vicario in the first run).
v3 vs v2 on identical data: 25 vs 25, 25 (×2) vs 24, 22 vs 23 — within one row, in both directions.
**On the same data neither prompt v3 nor the summary check is worse than v2**; the 0.741 / 0.667 of the
first run measured the prior leak and the back-dated articles, not the arms. The one corrected summary in
the golden runs (Dasilva) removed "(document id 3055)", an internal chunk id leaked into user-facing
text; on the live Cherki signal the check removed the unsupported "Carabao / Norwich" and "Sunderland,
2026-09-20" sentences (docs/rag.md «Новости в советах»).

**Decision.** Defaults: prompt v3 (`RAG_PROMPT_VERSION=v3`) and the summary check
(`RAG_SUMMARY_CHECK=true`); v1 / v2 remain eval arms (`--prompt v2`). Evals default to the operational
replay, so a rerun reproduces the table above instead of drifting with the corpus; the first-run files stay
in `evals/results/` as evidence of the leak. Not measured by golden: the relative-date rule of v3 itself
(no golden row has a stale "next N matches" phrase) — shown only on the live Cherki example. Form quotes
as availability evidence ("Rayan Cherki scored") — §4.6.

### 4.6 A/B #4: `form_notes` — form and role apart from availability evidence (prompt v4)

**Problem.** For fit players the model cites form as availability evidence and sets confidence 1.0. Live
(prompt v3, `rag.signal --no-save`): Saka `fit`, confidence **1.00**, all three quotes are form —
"have found some early-season form", "looks back to his dangerous best with the ball", "forming a rather
terrifying … triumvirate when they all start firing"; Cherki `fit`, 0.80, quotes "Rayan Cherki scored" and
"can boast as many goal involvements as Haaland". Golden: Haaland `fit` at 1.0 on "the only player to
exceed 1.0 projected attacking returns" (rule 3 of v2/v3 names exactly this quote as NOT evidence). The
information itself is useful to the user — it just is not availability.

**Hypothesis.** A separate field for form / role / position / set pieces gives these facts a legitimate
place: they leave the availability evidence, the confidence of fit players without health news stops
being inflated, and availability accuracy does not drop below v3.

**What v4 is.** Prompt [`prompts/v4/`](../src/fplcopilot/prompts/v4/signal_extraction.system.md) = v3
plus four insertions (`diff -r src/fplcopilot/prompts/v3 src/fplcopilot/prompts/v4`): rule 3 — form, role,
position and set-piece facts go to `form_notes`, a goal is form even though it implies he played
("Rayan Cherki scored" is a form note); rule 7 — each evidence quote carries `about` = fitness / injury /
training / selection / rotation / suspension / transfer; rule 12 — form never raises confidence, status
`a` with only an indirect selection quote → at most 0.8; rule 14 — `form_notes` (kind, one-sentence text,
chunk id, verbatim quote that names the player). Schema `SignalDraftV4` (`draft_schema("v4")`): v2/v3 fields
in the same order, evidence items with `about`, `form_notes` last. Code (`rag/extract.py`), on top of
(b)–(f):

- form notes pass (d) verbatim and a stricter (f): the chunk mentions the player **and the quote itself
  names him** (FFScout round-ups mention 20 players; "Martin Odegaard scored twice" is not a note about Saka);
  the note text may only restate its quote — dates / numbers / names via `summary_check` and ≥ 60 % of
  its content words from the quote, else the quote is shown;
- a quote with form words and no availability word (`rag/quote_kind.py`; availability terms win, so
  "played 53 minutes", "had the night off", "scored before limping off" stay evidence) moves from evidence
  to `form_notes`; a note with availability words ("missed out after tweaking his groin") is dropped as
  misfiled;
- FPL status `a` + `fit` + only `selection` / `rotation`-labelled evidence → confidence ≤ 0.8;
- rule (c) unchanged: no availability evidence → `unknown`, confidence 0, even when form notes exist (the
  summary then says "No fitness or selection news … Form and role notes are listed separately");
- the minutes / xPts model never reads `form_notes` (`core/signals.py` does not select the column; test).

Storage: `player_signals.form_notes jsonb` (migration `010_signal_form_notes.sql`). Shown in the player
card («Новости об игроке» → «Форма и контекст», quotes and links) and to the explainer as
`FACTS.form_and_context` / `candidate_news.players[…].news_signal.form_and_context` with the quotes in
EVIDENCE under `"scope": "form"` (explain prompt v3: context, never availability; the NEWS block cites only
non-form quotes for the availability line).

**Arms.** Same data for all: operational replay (default cutoff = `as_of`, 1102 / 1622), hybrid_rerank,
retrieval v2, abstention on, summary check on, no judge, each arm run twice. v3:
`20260923T182334Z_signals_pv3-rv2-aon-cutasof-check.json`, `20260923T184726Z_…-cutasof-check-rep2.json`.
v4: `20260923T185556Z_signals_pv4-rv2-aon-cutasof-check-final-rep1.json`, `20260923T185558Z_…-final-rep2.json`.

**How "form quotes in availability evidence" is counted.** (1) Dictionary `rag/quote_kind.py`: a quote is
*form* if it has form words (scored, goal, form, returns, projected, £, dangerous, best, penalties, …) and
no availability word. v4 uses the same dictionary to move quotes, so for v4 it is 0 by construction —
hence (2) manual labelling of all 53 distinct evidence quotes of the four runs (one annotator; a quote is
*not availability* if it reports goals / form / projections / prices or neither fitness nor selection;
a goal report counts as form even though it implies he played — the definition of v4). The two methods
disagree on two quotes: "Erling Haaland's goal helps Manchester City win … despite playing with 10 men for
more than 70 minutes" (dictionary: availability via "minutes"; manual: form) and "Pedro Porro and Tonali
out of Anfield clash" (dictionary: other; manual: availability).

| metric | v3 · run 1 | v3 · run 2 | v4 · run 1 | v4 · run 2 |
|---|---|---|---|---|
| availability accuracy strict | **25/27** (0.926) | **25/27** | 23/27 (0.852) | 23/27 |
| lenient | 27/27 | 27/27 | 25/27 | 25/27 |
| macro-F1 | 0.948 | 0.948 | 0.737 | 0.737 |
| start_probability in range | 26/27 | 26/27 | 25/27 | 25/27 |
| evidence present when required | 23/23 | 23/23 | 21/23 | 21/23 |
| false evidence (`no_coverage`) · return GW exact | 0/4 · 8/8 | 0/4 · 8/8 | 0/4 · 8/8 | 0/4 · 8/8 |
| not-availability quotes in evidence — manual | 1/38 | 3/40 | **1/36** | **1/37** |
| form quotes in evidence — dictionary | 1/38 | 1/40 | 0/36 | 0/37 |
| rows whose evidence is only form (manual) · their confidence | Haaland · 1.0 | Haaland · 1.0 | Haaland · **0.8** | Haaland · **0.8** |
| mean confidence, `fit` rows with FPL status `a` (Haaland, Gabriel, Timber, Raya) | 0.925 | 0.975 | **0.825** | **0.825** |
| … of them at confidence 1.0 | 2 | 3 | **0** | **0** |
| form notes · rows with notes · kinds | — | — | 6 · 4 · form 3, role 3 | 6 · 4 · form 3, role 3 |
| quotes moved evidence → notes · confidence capped at 0.8 | — | — | 1 · 4 | 1 · 4 |
| mean validation fixes | 0.11 | 0.07 | 1.11 | 1.07 |
| prompt tokens (27 rows) · cost | 83.1k · $0.014 | 83.1k · $0.014 | 115.7k · $0.020 | 115.7k · $0.020 |

Misses: v3 — Porro, Emersonn (two-sided `conflict` rows, alternate label accepted) in both runs. v4 —
the same two plus **James** (`unknown`: in every v4 variant the model files "Reece James were not see in the
part of training in which Sky cameras were present" as a *form* note; the code drops it as misfiled, no
evidence is left, rule (c) → `unknown`; v3 cites the same sentence and says `doubtful`) and **Vicario**
(`unknown`, the noisy row of §4.1 / §7.1). The rise in validation fixes is the misfiled notes and the (c)
fields they trigger.

Two earlier v4 drafts are not reproducible (their prompt text was replaced) and are listed only as the
path to the final one: draft 1 — `form_notes` first in the schema, rules for FPL-squad articles, "a quote
in both fields is form": 21/27 twice (`20260923T184726Z_…-cutasof-check-rep{1,2}.json`) — 11 of 14 notes
were injuries, bans or rest filed as "role"/"form", and the both-fields rule removed real evidence
(Gomes, Awoniyi, Doku, Timber); draft 2 — evidence before notes, verdict last: 23/27 twice
(`…185235Z_…-check-r2rep{1,2}.json`, James and Raya `unknown`). The final v4 keeps rules 1–13 of v3
verbatim apart from the four insertions.

**Live check** (`RAG_PROMPT_VERSION=v4 uv run python -m fplcopilot.rag.signal --player … --timings --no-save`;
before = the same command with v3, about 30 minutes earlier on the live corpus):

| player | v3 (before) | v4 (after) |
|---|---|---|
| Saka | `fit`, 1.00; evidence = 3 form quotes | `unknown`, 0.00 — rule (c): no fitness / selection news; FPL `a` is the only availability signal. Form notes: "have found some early-season form that they lacked last term" (Sky 21.09), "looks back to his dangerous best with the ball" (Sky 17.09); 3 form quotes moved out of evidence |
| Cherki | `fit`, 0.80; evidence "Rayan Cherki scored", "as many goal involvements as Haaland" | `fit`, 0.80; evidence "with Phil Foden (£7.0m) suspended, he should be more likely to start the next two Premier League games" (BBC 18.09, selection); form notes: goal involvements (Sky 21.09, form), "more of an inside forward, as a number 10 or playing narrow" (BBC 17.09, role) |

**Decision.** Criteria 2 and 3 are met: form quotes in availability evidence 4/78 → 2/73 (manual), the one
form-backed row (Haaland) falls from 1.0 to 0.8, the four fit / status-`a` rows average 0.83 instead of
0.93–0.98 and none sits at 1.0 on indirect evidence. **Criterion 1 is not met**: 23/27 vs 25/27 in both
repeats on identical data (lenient 25 vs 27) — two rows, below the ~10 pp noise line of §5, but consistent
across repeats and all three drafts (James). **v4 is not the default**: `RAG_PROMPT_VERSION` stays `v3`;
v4 is opt-in (`RAG_PROMPT_VERSION=v4`, `--prompt v4` in `evals.run_rag` and `rag.signal`). The column,
the card block and the explainer facts are in place and fill in for v4 signals; v1–v3 signals have no
notes and the block stays hidden. Next: put misfiled availability notes back into evidence with a verdict
re-check (the James failure), or run v4 on gpt-5.4-mini (§8.1: verbatim quotes, best F1; ≈ 5.7× cost).

## 5. Known limitations

- Labels: single annotator, weak FPL-derived ground truth for `fpl_api_only`/`doubtful` rows, ranges are
  judgment calls, outcomes (who actually played GW5) unknown — see the golden README.
- n = 27 / 18: one flipped row ≈ 4–7 pp. Treat differences below ~10 pp as noise unless the per-row
  diff (`rows[]` in the results JSON) shows a consistent pattern.
- Corpus is live: the ingest loop back-fills and some feeds back-date `pubDate` (FFScout, §4.5), so the
  set with `published_at <= as_of` grows after labelling. `--corpus-cutoff` pins a run (default `as_of`:
  `fetched_at` / `snapshot_at` <= `as_of`); the §4.5–§4.6 replay files use it, the `20260917T*` files of
  §3, §4.1–§4.4 and §7–§8 were produced without it, and §7–§8 also saw the leaky FPL prior of §4.5 for one
  row (Amenda `d/25` from a snapshot at 11:23Z instead of `d/50`). The cutoff cannot undo re-tagging
  (`--rematch` rewrites `players` in place) or deleted rows.
- The FPL status before the ingest started (first snapshot 2026-09-17T07:34Z) is approximated by the first
  observation if its news predates `as_of` — exact only from that snapshot on.
- Latency measured on one laptop, warm process, sequential; the embedding call is network-bound.
- Faithfulness judge is same-family (gpt-4o-mini) and quote-only; no human calibration of the judge.
- Cost is an estimate from token counts × list price.
- T = 0 is not determinism: the same chunks under a different prompt can flip one quote (Timber, §4.1).
  Single-row differences between arms need a repeat run before they count as a trend.
- The signals-path query (`expand_player_query`, injury-shaped) and the golden retrieval queries
  (natural questions) live on different score scales; thresholds calibrated on one do not transfer (§4.3).

## 6. How to run

```bash
# everything with the shipped defaults (prompt v3, retrieval v2, abstention on, summary check on,
# corpus pinned to as_of):
# ≈ 81 extraction calls (minus abstentions) + ≤ 75 judge calls, ~$0.05 with gpt-4o-mini, ~10 min
uv run python -m evals.run_rag --suite all --modes dense,hybrid,hybrid_rerank --k 8 --out evals/results/

# A/B #2 arms (results file gets the suffix _p<prompt>-r<retrieval>-a<on|off>[-<tag>])
uv run python -m evals.run_rag --suite signals --modes dense,hybrid_rerank --prompt v1 --retrieval v1 --abstain off --judge-budget 20
uv run python -m evals.run_rag --suite signals --modes dense,hybrid_rerank --prompt v2 --retrieval v2 --abstain on --judge-budget 20
uv run python -m evals.run_rag --suite retrieval --modes hybrid_rerank,dense --retrieval v1
uv run python -m evals.run_rag --suite retrieval --modes hybrid_rerank,dense --retrieval v2
# the score gate is a setting, not a flag: RAG_ABSTAIN_SCORE=0.2 uv run python -m evals.run_rag ... --tag score02

# corpus pinning (§4.5): default --corpus-cutoff as_of (operational replay, reproducible);
# the A/B #2 corpus; the live corpus (research reconstruction — drifts, report separately)
uv run python -m evals.run_rag --suite signals --modes hybrid_rerank --prompt v2 --no-judge --corpus-cutoff 2026-09-17T10:14:17Z
RAG_SUMMARY_CHECK=false uv run python -m evals.run_rag --suite signals --modes hybrid_rerank --prompt v2 --no-judge --tag nocheck
uv run python -m evals.run_rag --suite signals --modes hybrid_rerank --prompt v3 --no-judge --corpus-cutoff none --tag livecorpus
# A/B #4 (§4.6): v3 vs v4 on the replay, ≈ $0.014 / $0.020 per run
uv run python -m evals.run_rag --suite signals --modes hybrid_rerank --prompt v3 --no-judge
uv run python -m evals.run_rag --suite signals --modes hybrid_rerank --prompt v4 --no-judge

# one suite / subset / no judge
uv run python -m evals.run_rag --suite retrieval --modes dense,hybrid_rerank
uv run python -m evals.run_rag --suite signals --no-judge --ids sig_caicedo_injured_return,sig_foden_suspended
uv run python -m evals.run_rag --suite signals --limit 5 --out /tmp/evals

# hyperparameters (§7): extraction T ∈ {0,0.3,0.7} (T=0 and T=0.7 twice), top_p 0.5 at T=0.7 — ≈ $0.10, 9 min;
# explain T × max_tokens grid on 6 agent queries (facts captured once) — ≈ $0.07, 6 min
uv run python -m evals.run_hparams --suite extraction
uv run python -m evals.run_hparams --suite explain --manager 895045
uv run python -m evals.run_hparams --suite explain --explain-requests evals/results/<ts>_hparams_explain.json --tag rep2   # replicate on identical facts
uv run python -m evals.run_hparams --rejudge evals/results/<ts>_hparams_explain.json --judge-model gpt-4.1-mini            # re-score answers with another judge
# model comparison (§8): extraction per model at T=0 + router on the 8 demo queries + explain on the captured facts — ≈ $0.52
uv run python -m evals.run_models --explain-requests evals/results/<ts>_hparams_explain.json
uv run python -m evals.run_models --models gpt-4o-mini,gpt-4.1 --skip-router --budget-usd 0.5
# every runner prints the running list-price spend and stops at --budget-usd (default 1.5)

# unit tests for metrics, golden schema/protocol, judge parsing (no network)
uv run pytest -q tests/test_evals_metrics.py tests/test_evals_golden.py tests/test_evals_judge.py
# hyperparameter / model-comparison helpers: cost tables, truncation rate, determinism agreement, explain aggregates
uv run pytest -q tests/test_evals_hparams.py
# RAG v2 unit tests: calendar date→GW, FPL date parsing, minutes formula, abstention path, article cap,
# date-aware order, word-boundary matcher
uv run pytest -q tests/test_rag_v2_extract.py tests/test_rag_v2_retrieval.py tests/test_rag_v2_matcher.py
# as_of replay (prior by snapshot_at, corpus cutoff), prompt v4 / form_notes, card + explainer facts
uv run pytest -q tests/test_asof_replay.py tests/test_rag_v4_form_notes.py tests/test_form_notes_facts.py
# with Postgres up: relevant article ids/URLs still exist; with network: player ids match names
uv run pytest -q -m db tests/test_evals_golden.py && uv run pytest -q -m network tests/test_evals_golden.py
```

Requirements: Postgres with the indexed corpus (`docker compose up -d db`, ingest + index done, migrations
applied — `007_signals_v2.sql` adds `player_signals.abstained`, `010_signal_form_notes.sql` adds
`player_signals.form_notes`), `OPENAI_API_KEY` in `.env` (embeddings,
extraction, judge). `player_signals` is never written by evals. Results land in
`evals/results/<UTC timestamp>_<suite>_<arm>.json`; the files for the runs reported above are kept in the
repo as evidence. After changing the entity matcher run `uv run python -m fplcopilot.rag.ingest --rematch` first
(re-tags articles and copies the tags to chunks) and re-check the `no_coverage` rows.

## 7. Hyperparameters: temperature, top_p, max_tokens, determinism

Runner `evals/run_hparams.py`. Two suites; all other settings are the A/B #2 decision (prompt v2,
retrieval v2, abstention on, `hybrid_rerank`, k = 8). Prices for all cost figures: `run_hparams.PRICES`
(OpenAI list prices at run time, also stored in each results file).
Total spend for this section ≈ $0.27 (extraction $0.10, explain grid $0.07 + replicate $0.06,
re-judge $0.05).

**Design.** Extraction arms share one retriever (same store, same query-embedding cache), so every arm
sees identical chunks for identical players — verified per row (`retrieval_agreement_vs_first_arm =
27/27` in every arm; the corpus did not move: 1109 articles / 1639 chunks before and after). Latency
is therefore reported for the LLM stage only (`llm_ms`). For explain, the live agent ran once per
query with a capturing `Deps.explain_llm`; the captured `ExplainRequest` (facts JSON, evidence,
caveats) was then replayed through the production prompt for every (T, max_tokens) cell, so all
cells saw byte-identical facts. `as_of` for the agent is "now" (11:57Z), not the golden `as_of`; the
facts are stored in the results file. The graph's normal path was used to compute the facts, so stale
signals were re-extracted and saved exactly as in `scripts/agent_demo.py`.

### 7.1 Extraction temperature and top_p (gpt-4o-mini, 27 players)

`20260917T115720Z_hparams_extraction.json`. Arms: T = 0 twice, T = 0.3, T = 0.7 twice, T = 0.7 with
top_p = 0.5 (top_p = 1.0 is the API default and is not sent otherwise). 4 rows abstain before the LLM
in every arm; 23 calls per arm; judge budget 20 per arm.

| metric | T 0 · run 1 | T 0 · run 2 | T 0.3 | T 0.7 · run 1 | T 0.7 · run 2 | T 0.7 · top_p 0.5 |
|---|---|---|---|---|---|---|
| availability accuracy (strict) | 0.815 (22/27) | 0.852 (23/27) | 0.852 | 0.852 | **0.926** (25/27) | 0.815 |
| availability accuracy (lenient) | 0.889 | 0.926 | 0.926 | 0.926 | 1.000 | 0.889 |
| availability macro-F1 | 0.707 | 0.733 | 0.733 | 0.737 | 0.948 | 0.707 |
| start_probability in range | 0.889 | 0.926 | 0.926 | 0.926 | 0.963 | 0.926 |
| return_gw exact (n = 8) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| evidence present when required (n = 23) | 0.913 | 0.957 | 0.957 | 0.913 | 1.000 | 0.913 |
| false-evidence rate (n = 4) | 0 | 0 | 0 | 0 | 0 | 0 |
| mean validation fixes | 0.481 | 0.222 | 0.333 | 0.481 | 0.148 | 0.481 |
| abstained / LLM calls | 4 / 23 | 4 / 23 | 4 / 23 | 4 / 23 | 4 / 23 | 4 / 23 |
| faithfulness (judge), n = 20 | 0.775 | 0.750 | 0.775 | 0.775 | 0.650 | 0.750 |
| completion tokens per call | 140 | 139 | 136 | 136 | 139 | 137 |
| finish_reason = length | 0 | 0 | 0 | 0 | 0 | 0 |
| cost signals + judge (USD) | 0.014 + 0.002 | same | same | same | same | same |
| LLM latency p50 / p95 (ms) | 1638 / 2563 | 1607 / 2251 | 1689 / 2473 | 1723 / 2515 | 1624 / 2350 | 1638 / 3538 |

Strict misses per arm: **Porro** and **Emersonn** in all six (two-sided `conflict` rows; the alternate
label is accepted), **Timber** in five (`misleading_news`), **Vicario** in five (the 29-day-old official
item falls out of the candidate set, §4.3), **Hincapié** in three (`conflict`, flips to `unknown`).
By tag (strict): `conflict` 0.4–0.6, `misleading_news` 0.75–1.0, `ambiguous_name` 0.86, all other
tags 1.0 in every arm.

**Determinism** (same settings, run 1 vs run 2; "LLM rows" = the 23 that called the model — the 4
abstentions are deterministic by construction):

| identical between runs | T = 0, all 27 | T = 0, LLM rows | T = 0.7, all 27 | T = 0.7, LLM rows |
|---|---|---|---|---|
| availability label | 26/27 (0.963) | **22/23 (0.957)** | 25/27 (0.926) | 21/23 (0.913) |
| start_probability | 25/27 | 21/23 (0.913) | 24/27 | 20/23 (0.870) |
| cited chunk ids | 25/27 | 21/23 (0.913) | 19/27 | 15/23 (0.652) |
| summary text | 16/27 | 12/23 (0.522) | 4/27 | **0/23 (0.000)** |
| whole output | 15/27 | 11/23 (0.478) | 4/27 | 0/23 (0.000) |
| retrieved chunks | 27/27 | | 27/27 | |

Rows that differed at T = 0: Hincapié (label and p_start), Timber (p_start), Awoniyi (quote set).

**Reading and decision.**
- Temperature does not move accuracy, fixes, faithfulness, tokens, cost or latency beyond the noise
  floor: two runs at identical settings differ by 1–2 rows (22 vs 23 at T = 0, 23 vs 25 at T = 0.7),
  which is the whole spread of the table. T = 0.7 · run 2 is the best arm on paper and its twin is
  three rows worse — a single run of any arm cannot rank temperatures on n = 27.
- What temperature does change is reproducibility: at T = 0, 96 % of labels, 91 % of quote sets and
  52 % of summaries are identical between runs; at T = 0.7, 91 %, 65 % and 0 %. **T = 0 stays the
  extraction default** — for stability, not for accuracy.
- **T = 0 is not determinism.** One label in 23 still flips and half the summaries are reworded. This
  is why the label-bearing parts of the signal are pinned by code (validator rules, deterministic
  return-GW, abstention) and why single-row differences between arms in §3–§4 are not trends.
- `top_p`: at T = 0.7, 0.5 vs 1.0 changed nothing distinguishable from noise (one run each); at T = 0
  it is irrelevant. Left at the API default (not sent).
- `max_completion_tokens = 800`: 138/138 calls finished with `stop`, 130–190 tokens used. Not a binding
  constraint; kept as a safety cap.

What these metrics do **not** show: whether a T > 0 sample is ever *better* on the two-sided rows
(the golden set has one primary label; the arms disagree only on rows where annotators would too),
and how stable T = 0 is across prompt versions (§4.1: the same chunks under a different prompt flipped
Timber).

### 7.2 Explain temperature × max_tokens (gpt-4o-mini, 6 queries)

Files: `20260917T115722Z_hparams_explain.json` (grid, run 1, with the captured requests),
`20260917T120327Z_hparams_explain_rep2.json` (same requests replayed: run 2),
`20260917T120432Z_hparams_explain_judge-gpt-4.1-mini.json` (run 1 re-judged). Queries: "Is João Pedro
fit for GW5?" (player_status), "Should I sell Palmer?" (transfer), "Who should I captain this week?"
(captain), "Plan my transfers for the next 5 gameweeks" (plan), "Compare Haaland and Salah for the
next 3 gameweeks" (compare_players), "Who should I start this week?" (lineup). Grid: T ∈ {0, 0.2, 0.7}
× max_tokens ∈ {300, 600, 1000} plus the production cell of the run (0.2, 1400). 60 answers per run.
Explain prompt **v1** (`agent/prompts/v1/explain.system.md`); the runner records the active
`AGENT_PROMPT_VERSION`, so rerunning the grid with another prompt version compares it with v1 on the
same captured facts.

Metrics per answer: `finish_reason` (`length` = truncated; the SDK raises and in production the
explain node falls back to a JSON dump of the facts), the graph's own validator
(`agent/validate.py`: unknown names, unknown decimals, placeholder citations — computed **before**
any regeneration, on completed answers), format compliance (Verdict / Why / Caveats sections, Sources
iff evidence), length (completion tokens, words), faithfulness judge over (FACTS, EVIDENCE, ANSWER)
(`evals/prompts/explain_faithfulness_judge.md`; two judge models, see 7.3), cost, latency.

**By max_tokens** (both runs pooled, all temperatures):

| max_tokens | answers | truncated | validator pass (completed) | violations / completed answer | completion tokens (completed) | words | judge gpt-4.1-mini (n) | cost / answer | latency p50 |
|---|---|---|---|---|---|---|---|---|---|
| 300 | 36 | **36 (100 %)** | – | – | – | – | – | $0.00062 | 2.8 s |
| 600 | 36 | **9 (25 %)** | 0.926 | 0.07 | 431 | 171 | 0.893 (14) | $0.00073 | 4.3 s |
| 1000 | 36 | 0 | 0.833 | 0.17 | 480 | 192 | 0.917 (18) | $0.00073 | 4.1 s |
| 1400 (production cell) | 12 | 0 | 0.750 | 0.25 | 470 | 191 | 0.833 (6) | $0.00072 | 3.6 s |

Completed answers span 328–736 completion tokens (median 441) for the prompt's 150–350 words plus a
table. The 9 truncations at 600 are all transfer (565–610 tokens) and lineup (581–736) answers — the
two intents with a routes table or a full XI.

**By temperature**, cells with max_tokens ≥ 1000 only (no truncation confound; both runs):

| temperature | answers | validator pass | violations / answer | placeholder citations | unknown numbers | unknown names | completion tokens | judge gpt-4.1-mini | judge gpt-4o-mini | latency p50 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 12 | **0.917** | **0.08** | 1 | 0 | 0 | 473 | 0.917 | 0.583 | 3.8 s |
| 0.2 (production cell) | 24 | 0.792 | 0.21 | 5 | 0 | 0 | 476 | 0.875 | 0.604 | 4.0 s |
| 0.7 | 12 | 0.750 | 0.25 | 2 | 1 | 0 | 486 | 0.917 | 0.583 | 4.1 s |

Per query (both runs, completed answers): captain 14/14 and plan 14/14 pass the validator; transfer
9/10, compare 13/14, player_status 12/14; **lineup 2/9** — 7 placeholder citations
(`[source not available]`, `[FACTS]`, `[source: n/a]`) at every temperature. Run 1 vs run 2 at the
same cell differ by 0–2 validator verdicts; the production cell went 4/6 and 5/6.

**Decision.**
- **max_tokens: 1400 for this prompt** (prompts with the candidate-news block use 2000, `agent/llm.py`).
  300 is unusable (100 % truncation), 600 loses a quarter of the answers,
  1000 is the floor; 1400 ≈ 2× the median completed length costs nothing extra (billing is per token
  used) and covers the 736-token lineup answer with margin. The deciding metric is the truncation
  rate: a truncated structured answer is a hard failure, not a shorter answer.
- **temperature: 0** (`AGENT_EXPLAIN_TEMPERATURE = 0` in `config.py`; the production cell of the run
  used 0.2). Length, judge score, cost and latency are flat; the only metric that moves is the validator's
  first-pass violation count, and it favours T = 0 (1/12 vs 5/24 vs 3/12 answers with a violation).
  The absolute effect is small and concentrated in the lineup query, whose placeholder-citation habit
  is a prompt defect (v2 candidate: "issues from FACTS need no citation") rather than a sampling one.

What these metrics do **not** show: readability or user preference (no human labels), whether the
answer's verdict matches `FACTS.headline` (the judge checks claims, not emphasis), and behaviour on
long answers (the prompt caps at 350 words).

### 7.3 Judge model check (38 completed explain answers of run 1)

gpt-4o-mini judging a ≈ 3k-token facts JSON flagged claims that are literally in the facts
("probability of starting 0.64", "expected minutes 30", "medium rotation risk"), even after being
given the list of numbers already verified by code (the hybrid judge in `run_hparams.judge_explain`).
Re-judging the same answers with gpt-4.1-mini (`--rejudge`):

| judge | supported / partial / unsupported | mean | flagged claims per answer | verdict agreement |
|---|---|---|---|---|
| gpt-4o-mini | 12 / 26 / 0 | 0.658 | 1.37 | 53 % |
| gpt-4.1-mini | 30 / 8 / 0 | 0.895 | 0.47 | |

gpt-4.1-mini's remaining flags are expansions beyond the facts (a club name written out for a fixture
code, "very easy" for FSI 1, statements about an unresolved player) — real, and invisible to the
validator because club names are whitelisted. Consequence for this document: judge means are relative
numbers; the extraction judge (short summary vs ≤ 6 quotes) keeps gpt-4o-mini, the explain judge uses
gpt-4.1-mini; neither is human-calibrated.

## 8. Model comparison

Runner `evals/run_models.py`; file `20260917T120615Z_models.json`; spend $0.52 (gpt-4.1 alone $0.29).
Same golden set, same prompts, same structured-output schemas, T = 0, one shared retriever
(retrieval agreement 27/27 for every model; corpus 1109 / 1639 before and after). Candidates:
gpt-4o-mini (shipped), gpt-4.1-nano (cheapest chat model on the key), gpt-4.1-mini, gpt-5.4-mini
(newest small model; accepts `temperature=0`, 0 reasoning tokens in a probe), gpt-4.1 (strongest
non-reasoning model). Reasoning models (o-series, full gpt-5.x) were excluded: no temperature/top_p
control, prompt-dependent reasoning-token cost. Prices $/1M in/out: 0.15/0.60 · 0.10/0.40 · 0.40/1.60
· 0.75/4.50 · 2.00/8.00.

### 8.1 Extraction (27 players, T = 0)

| metric | gpt-4o-mini | gpt-4.1-nano | gpt-4.1-mini | gpt-5.4-mini | gpt-4.1 |
|---|---|---|---|---|---|
| availability accuracy (strict) | **0.852** | 0.556 | 0.778 | **0.889** | **0.852** |
| availability accuracy (lenient) | 0.926 | 0.556 | 0.889 | 0.963 | **1.000** |
| availability macro-F1 | 0.733 | 0.506 | 0.680 | **0.924** | 0.903 |
| start_probability in range | 0.926 | 0.778 | 0.889 | 0.926 | 0.889 |
| return_gw exact (n = 8) | 1.000 | 0.750 | 1.000 | 1.000 | 1.000 |
| evidence present when required (n = 23) | 0.957 | 0.478 | 0.913 | **1.000** | **1.000** |
| false-evidence rate (n = 4) | 0 | 0 | 0 | 0 | 0 |
| mean validation fixes | 0.222 | 2.556 | 0.481 | **0.074** | **0.074** |
| abstained / LLM calls | 4 / 23 | 4 / 23 | 4 / 23 | 4 / 23 | 4 / 23 |
| faithfulness (judge gpt-4o-mini), n | 0.725, 20 | 0.818, 11 | 0.650, 20 | 0.800, 20 | 0.775, 20 |
| mean confidence, expected-known rows | 0.81 | 0.40 | 0.75 | 0.89 | 0.85 |
| label agreement with gpt-4o-mini | — | 15/27 | 24/27 | 22/27 | 23/27 |
| prompt / completion tokens per call | 3500 / 140 | 3500 / 130 | 3500 / 164 | 3496 / 189 | 3500 / 171 |
| cost signals + judge (USD) | **0.014** + 0.002 | 0.009 + 0.001 | 0.038 + 0.003 | 0.080 + 0.003 | 0.193 + 0.003 |
| LLM latency p50 / p95 (ms) | **1648 / 2089** | 2942 / 6506 | 2038 / 2821 | 1845 / 2179 | 2719 / 7090 |

Strict misses: gpt-4o-mini — Porro, Emersonn (conflict), Timber, Vicario. gpt-4.1 — Porro, Hincapié,
James, Emersonn (all `conflict`, hence lenient 1.000). gpt-5.4-mini — Gomes, Emersonn (conflict),
Doku (`doubtful` vs `injured`). gpt-4.1-mini — the 4o-mini four plus Hincapié (`fit`) and James
(`unknown`). gpt-4.1-nano — 12 rows `unknown`: its quotes are not verbatim (2.6 fixes/row), the
validator drops them and rule (c) forces `unknown`; two return GWs are lost the same way.

Reading: the three usable models are within one row of each other on the label — the run-to-run
noise of §7.1. The consistent gains of the larger models are behavioural: verbatim quotes (0.07
fixes/row), evidence on every covered player, better-separated confidence. gpt-4.1-mini is worse
than gpt-4o-mini at 2.7× the price; gpt-4.1-nano is unusable for this role.

### 8.2 Router (8 demo queries, router prompt v1 `agent/prompts/v1/router.system.md`)

| metric | gpt-4o-mini | gpt-4.1-nano | gpt-4.1-mini | gpt-5.4-mini | gpt-4.1 |
|---|---|---|---|---|---|
| intent accuracy | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| agreement with gpt-4o-mini | — | 8/8 | 8/8 | 8/8 | 8/8 |
| prompt / completion tokens | 1388 / 64 | 1388 / 66 | 1388 / 67 | 1386 / 74 | 1388 / 69 |
| cost per call | $0.00025 | $0.00017 | $0.00066 | $0.00137 | $0.00333 |
| latency p50 / p95 (ms) | 1026 / 1679 | 1111 / 1227 | 1196 / 1329 | 1070 / 2366 | 1166 / 1496 |

The demo queries are easy (one intent each, including betting refusal and off-topic); every model is
perfect, so cost decides. Horizon / scenario fields were not scored.

### 8.3 Explain (6 captured requests, production cell T = 0.2 / max_tokens 1400)

| metric | gpt-4o-mini | gpt-4.1-nano | gpt-4.1-mini | gpt-5.4-mini | gpt-4.1 |
|---|---|---|---|---|---|
| truncated | 0 | 0 | 0 | 0 | 0 |
| validator pass (first attempt) | 5/6 | 5/6 | 4/6 | 5/6 | 5/6 |
| unknown names / numbers / placeholder citations | 0 / 0 / 1 | 0 / 1 / 0 | 1 / 2 / 0 | 0 / 1 / 0 | 0 / 2 / 0 |
| format complete (Verdict, Why, Caveats) | 6/6 | 4/6 | 6/6 | 6/6 | 6/6 |
| words / completion tokens | 192 / 477 | 186 / 367 | 285 / 609 | 245 / 594 | 262 / 645 |
| judge gpt-4o-mini (relative only) | 0.583 | 0.750 | 0.583 | 0.667 | 0.667 |
| cost per answer | **$0.00073** | $0.00044 | $0.00215 | $0.00488 | $0.01105 |
| latency p50 / p95 | **3.5 / 5.4 s** | 2.5 / 3.4 s | 5.0 / 6.9 s | 4.2 / 6.0 s | 6.5 / 11.3 s |

gpt-4.1 and gpt-4.1-mini both invented the same derived numbers (`4.60`, `3.30`) in the lineup answer,
gpt-5.4-mini `4.49`; gpt-4.1-nano wrote a bare date `17.09` (validator false positive: dates are only
allowed inside citations) and dropped a required section twice. The larger models are longer, slower
and not more faithful; the validator + one regeneration equalise the user-visible result.

### 8.4 Cost projections (measured tokens × list price)

| | gpt-4o-mini | gpt-4.1-nano | gpt-4.1-mini | gpt-5.4-mini | gpt-4.1 |
|---|---|---|---|---|---|
| extraction, per call | $0.00061 | $0.00040 | $0.00166 | $0.00347 | $0.00837 |
| batch refresh, 659 players, all call the LLM | **$0.40** | $0.27 | $1.10 | $2.29 | $5.52 |
| batch with the golden abstention share (23/27) | $0.34 | $0.23 | $0.93 | $1.95 | $4.70 |
| user query = router + explain | **$0.0010** | $0.0006 | $0.0028 | $0.0063 | $0.0144 |
| query + one fresh extraction | $0.0016 | $0.0010 | $0.0045 | $0.0097 | $0.0227 |

Consistent with the demo footers in `docs/agent.md` ($0.0009–0.0019 per query). The real pool's
abstention share is higher than the golden set's (most of the 659 players have no news), so the
batch column is an upper bound.

### 8.5 Decision

gpt-4o-mini stays for router, extraction, explain, grader and vision (`docs/LLM_CHOICE.md` §1). The
data would justify **gpt-5.4-mini for extraction** if the budget allowed 5.7× the batch cost (best F1,
verbatim quotes, same latency); gpt-4.1 buys the same behaviour at 14×. The judge for answer-vs-facts
checks is gpt-4.1-mini (§7.3). The comparison changes no production model default.

Limitations specific to §7–§8: n = 27 / 6 / 8; one run per model (the §7.1 noise floor of 1–2 rows
applies to every column of §8.1); explain judged by a same-vendor model; the demo router queries do
not exercise ambiguous intents (transfer vs what_if) or horizon extraction; latency measured from one
laptop, sequential, network-bound; costs are list prices with no caching discount (cached input is
50–90 % cheaper on every model listed, and the 3.5k-token extraction prompt shares its system part
across players).

## 9. Chat A/B: agent prompts v3 → v4 (chat regression set)

**Question.** With agent prompts v3 the chat is the weak spot: false `off_topic` refusals, no dialogue
memory, English service texts in a Russian product, answers that relabel gameweeks or invent reasoning
(diagnosis: `docs/chat_diagnosis.md`). Does agent prompt set v4 (router + explainer, plus the
chat code it enables) fix that without new failure modes?

**Data.** `evals/golden/chat.jsonl` — 60 realistic questions (50 RU / 10 EN; colloquial, typos,
mixed RU/EN): squad review, transfers, captain, lineup, plan, chips for *my* squad, fixtures,
player status / forecast, rankings past vs future, differentials, prices, rules, 7 multi-turn
follow-ups (canned previous turns) and 5 true off-topic / prompt-injection rows. Each row: the
acceptable intents, the ideal capability, the expected behaviour (answer / partial / clarify /
refuse) and the answer language. `expect.intents` includes the v4 intents where they implement the
row's declared `ideal` (squad_review, fixtures, chips, general_fpl); the baseline is scored on the
same expectations (`--metrics --refresh-expect`).
Holdout: `evals/golden/chat_holdout.jsonl` — 15 new questions written after the last prompt edit
and run once.

**Runner.** `evals/run_chat.py` calls the live agent exactly like the chat page
(`Agent.stream(prompt, manager_id, strategy, history)` → snapshot), manager 895045, balanced,
gpt-4o-mini router + explainer at T = 0. Paid-action HITL is confirmed automatically, a name
clarification counts as behaviour `clarify`. Results with raw answers, tool logs, validation and
FACTS: `evals/results/<ts>_chat_<label>.json`; manual review: `<…>_review.json` (helpful /
partial / bad with a reason per row). `--router-probe N` runs the router alone (cheap intent /
refusal checks between full runs).

| | baseline (v3 prompts, pre-v4 code) | v3 prompts on v4 code | **v4 run 3 (final)** | v4 holdout (15 new) |
|---|---|---|---|---|
| intent accuracy | 45/60 | 45/60 | **59/60** | 15/15 |
| false refusals | 10/55 | 10/55 | **0/55** | 0/13 |
| off-topic / injection refused | 5/5 | 5/5 | **5/5** | 2/2 |
| answer language = question language | 46/60 | 60/60 | **58/60** | 15/15 |
| refusals in the question's language | 1/15 | 15/15 | **5/5** | 2/2 |
| raw English validator footer shown | 2/43 | 0 (localized note 3/43) | **0/53** | 0/13 |
| validator rejected 1st attempt | 5/43 | 20/43 | 11/53 | 3/13 |
| manual review helpful / partial / bad | 17 / 17 / 26 | — | **49 / 7 / 4** | 13 / 2 / 0 |
| LLM judge helpful / partial / bad (trend only) | 27 / 14 / 19 | 30 / 15 / 15 | 42 / 15 / 3 | — |
| multi-turn rows helpful (manual) | 0/7 | — | 4/7 | 3/3 |
| latency p50 / p95, s | 10.2 / 32.7 | 12.3 / 33.3 | **9.3 / 25.5** | 7.6 / 23.4 |
| cost per 60 rows | $0.144 | $0.168 | $0.151 | $0.038 (15 rows) |

"v3 prompts on v4 code" isolates the prompts: same graph, tools, localized refusals and stricter
validator as v4 run 3, only `--prompt-version v3` (history is passed, but the v3 router has no field for it).
The stricter validator is why v3's first-attempt rejections rose from 5 to 20.

**Iterations (v4).** Run 1: helpful 43, bad 5, multi-turn 4/7 — follow-ups inherited the
*assistant's* suggestions as user constraints, a missing player (Salah is not in the 2026/27 game)
was only a caveat, one answer benched a model starter, Russian headings leaked into English
answers. Run 2 (router: carry over only the user's constraints; headline note for missing players;
bench check; language/heading rules): helpful 47, bad 4, multi-turn 6/7. Run 3 (a gameweek count
«на 3 тура» is no longer read as GW3 by the validator; the rating sentence «N из 10» is removed
after a failed regeneration; one-letter name typos are fixed; no per-player numbers without a
FACTS source; closing language line in the explainer message): helpful 49, bad 4, multi-turn 4/7.

**Decision: `AGENT_PROMPT_VERSION=v4` is the default** — better than v3 on every automatic metric
on identical code and on the manual review; v1–v3 stay for reproducibility.

**Not reached / honest caveats.**
- Multi-turn ≥ 6/7 was reached in run 2 but not in run 3 (4/7): c51 repeated the canned history's
  «капитан Haaland» against FACTS (Saka) — the captain check missed it because Saka was named as
  vice — and c56 hit the explainer's max_tokens (fallback). The run-4 code fixes both (vice-aware
  captain check; one concise retry on `LengthFinishReasonError`); a check on those two rows alone
  had both helpful (`…_chat_v4-postfix-c51-c56.json`), and run 4 below reaches 6/7.
- Failures persistent across runs 1–3 (addressed in run 4, below): a "review of last gameweek"
  question (c03) got an invented "your mistake was …" in 3/3 runs — past gameweeks were not
  modelled; «кто подорожает» (c45) once presented this gameweek's price rises as tonight's prediction.
- The plan's per-player xPts for the target-GW squad (a manual check on 895045 found a plan table
  with invented numbers) is absent from runs 1–3 and measured only in run 4, plus a spot check on
  c20 / c52.
- Canned history turns are text written by hand; with real history the previous answer agrees
  with FACTS, so c51-type contradictions are rarer in the UI. `--history replay` runs the previous
  turns live.
- One manager (895045, all first-half chips used) plus UI checks on 6856911; manual grading by the
  author of the prompts (grades and reasons are in the review files); the gpt-4o-mini
  judge agrees with the manual grades on 43/60 rows and is a trend indicator only.

```bash
uv run python -m evals.run_chat --manager 895045 --label after-x               # full run, ≈ $0.15
uv run python -m evals.run_chat --router-probe 1 --label after-x               # routing only, ≈ $0.05
uv run python -m evals.run_chat --rows evals/golden/chat_holdout.jsonl --label holdout
uv run python -m evals.run_chat --compare evals/results/A.json evals/results/B.json
```

**Run 4 — confirmation on the final code.** Compared with run 3, the chat code adds:
- a deterministic review of a finished gameweek (intent `gw_review`: picks of that GW, actual
  points from `player_gw_history`, the model's forecast saved before the deadline, captain,
  points left on the bench);
- keep / sell of a player who is not in the squad is dropped with a note in the question's
  language, and the validator does not let him be presented as a squad member;
- prices are marked as not predicted;
- plus the two post-run-3 fixes above (vice-aware captain check, concise retry on the length
  limit) and per-player xPts for the target-GW squad.

One full run on this code, same 60 questions, same manual scale (`…_chat_v4-run4.json` /
`_review.json`); the holdout was not re-run.

| | v4 run 3 | **v4 run 4** |
|---|---|---|
| intent accuracy | 59/60 | 58/60 (c08 → player_status, c56 → what_if: defensible readings, counted as misses) |
| false refusals / off-topic refused | 0/55 / 5/5 | 0/55 / 5/5 |
| answer language = question | 58/60 | **60/60** |
| manual helpful / partial / bad | 49 / 7 / 4 | **54 / 4 / 2** |
| multi-turn helpful | 4/7 | **6/7** |
| validator rejected 1st attempt | 11/53 | 11/53 |
| latency p50 / p95, s | 9.3 / 25.5 | **7.5 / 14.8** |
| cost per 60 rows | $0.151 | $0.157 |

What was fixed:
- c03 «где была ошибка в прошлом туре» now answers from facts (63 pts vs average 48, captain
  Haaland 0 vs 7.13 forecast, biggest shortfalls vs forecast) — no invented "mistake";
- c45 starts with "the system does not predict price changes" and shows transfer activity as
  facts;
- c20 no longer lists João Pedro in the 895045 team (only the "not in your squad — condition not
  applied" note).

Remaining bad rows:
- c12 — says Gabriel (£8.0) fits a £7m budget; the routes respect the budget, the prose does not;
- c55 — the follow-up still sometimes inherits the assistant's earlier suggestion from the canned
  history, despite the router rule; nondeterministic: helpful in run 2, bad in runs 1 and 4.

Intent accuracy is one row lower than in run 3; everything else is equal or better. The latency
drop is not claimed as a code effect: both runs made the same number of LLM calls (6 news
extractions, 63–64 explanations); run 3 ran with LangSmith tracing on (uploads rejected with
429), run 4 with tracing off, and OpenAI latency varies between runs.

## 10. Live test: xPts v0 vs FPL `ep_next` on the played GW5

Not an LLM eval, but the one pre-registered out-of-sample test of the deterministic core. The GW5
forecast was frozen at 2026-09-17T12:50Z (`xpts_predictions`, `model_version = v0`, 659 players, each
with the official `ep_next` captured at the same moment); the questions to answer were written
down in `docs/xpts.md` before the round was played. Run: `uv run python -m fplcopilot.core.history
--sync` (667 players, 3 216 season rows), then `uv run python scripts/gw_review.py --gw 5`; the
bootstrap CI and the disagreement count are a one-off script over the same rows.

| method | n | MAE | RMSE | Spearman all | Spearman played | top-20 mean actual | top-20 hits |
|---|---|---|---|---|---|---|---|
| **xpts_v0** | 659 | **1.146** | **2.174** | **0.740** | **0.413** | 4.00 | 2 |
| ep_next (FPL) | 659 | 1.166 | 2.286 | 0.734 | 0.383 | **4.50** | **3** |

- MAE difference over all 659 players: −0.020, paired-bootstrap 95 % CI [−0.082, +0.041] — **a
  tie**; among the 301 players who played: 2.023 vs 2.219, −0.196, CI [−0.317, −0.073] — v0 is
  better where points were actually scored.
- Where the two forecasts disagree by ≥ 2 points (35 players), v0 was closer to the outcome 25
  times, `ep_next` 10 times; of the ten disagreements listed before the round, v0 was on the right
  side in 7 (wrong on Brighton vs Arsenal — De Cuyper 6, Groß 14 — and on Maitland-Niles' rotation).
- Components (sum over 659 players, predicted vs actual): clean sheets **121 vs 184** (−63, the
  Poisson under-count of 0–0 / 1–0 already seen in the GW1–4 calibration), goals 144 vs 124 (+20),
  DefCon 59 vs 72 (−13); total 885 vs 966 (−8 %).
- Biggest misses are unforecastable hauls at 90 minutes (Brobbey 2.19 → 17, Semenyo 4.12 → 17).

**Conclusion, honestly.** On average v0 is not better than FPL's own `ep_next` — the MAE/RMSE/ρ edge
is inside the noise. It is better among players who played and wins most disagreements, which is
what the component model is for (availability and opponent, not form). But at the top of the
ranking — where captain and transfer decisions are made — `ep_next` picked the better top-20 on GW5
(4.5 vs 4.0 points, 3 vs 2 hits); 20 players and one round prove nothing either way, and the claim
"v0 beats the official forecast" is **not** supported. The live numbers sit inside the walk-forward
range (MAE 1.07–1.19, ρ 0.67–0.74 on GW2–4), so there is no out-of-sample degradation. Next: a
clean-sheet correction (Dixon–Coles or odds-based; a `v0-odds` forecast is stored for GW6) and the
same review on GW6. GW4 cannot be reviewed against `ep_next`: no pre-deadline `ep_next` snapshot
exists for it (only the `v0-backtest` walk-forward rows). Details: `docs/xpts.md`, «Итог GW5».
