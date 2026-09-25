# Golden dataset v1 (news RAG)

Two hand-labelled suites, one row per line (JSONL), all fixed to **`as_of = 2026-09-17T09:00:00Z`**
(the retriever only sees documents with `published_at <= as_of`, the FPL prior is taken from
`player_status_snapshots` at that moment). Next gameweek at `as_of` is **GW5** (deadline 18 Sep 17:30Z);
GW6 deadline is 10 Oct, GW7 is 17 Oct — return-date labels are expressed in these numbers.

| file | rows | what is labelled |
|---|---|---|
| `signals.jsonl` | 27 | expected `PlayerSignal` per player: availability class (+ alternates), start-probability range, expected-minutes range, return GW, whether evidence must be present |
| `retrieval.jsonl` | 18 | relevant **articles** per query (15 with answers, 3 with none) |

Schemas: `evals/schemas.py` (`SignalExample`, `RetrievalExample`); validated by `tests/test_evals_golden.py`.

## Labeling protocol

1. **Read the corpus, not the headlines.** For every candidate player the annotator dumped all
   articles tagged to the player (`news_articles.players`) *and* every chunk mentioning the name
   (regex over `news_chunks.text`, accent-insensitive) with `published_at <= as_of`, plus the latest
   `player_status_snapshots` row. Labels cite what was actually read (`rationale`, `source_urls`).
2. **Official FPL status is the prior, news can move it.** `i`/`s`/`u` are never overridden
   (system rule). `d` is treated as *doubtful* unless a **newer** manager quote confirms availability
   (Porro → `fit`, alternate `doubtful`). `a` becomes *doubtful* only when a full-text source reports
   a concrete problem (missed training, "tight quad", "little issues") with no later all-clear.
3. **Recency wins.** When a flag and an article disagree, the later timestamp decides
   (Gomes: article 15 Sep "available" vs flag 16 Sep "calf 75%" → `doubtful`).
4. **Two-sided cases get an alternate label** (`expected.availability_alt`, tag `conflict`).
   Reported both ways: strict accuracy uses the primary label; lenient accuracy accepts either.
5. **Ranges, not points.** `start_probability_min/max` encode what a careful analyst would accept:
   0 for out, 0.85–1.0 for nailed starters, wide for doubtful players, 0.5/0.5 (the system's
   neutral placeholder) for `unknown`. `expected_minutes` is only given where defensible
   (0 for out; for fit starters the lower bound is the v2 formula at `start_probability_min`,
   see `CHANGELOG.md` — `expected_minutes` is the unconditional expectation, not
   "minutes if he plays").
6. **`return_gw`** = first GW whose fixture date is on/after the official "Expected back"/"Suspended
   until" date (Foden: until 17 Oct → GW7; Henderson: back 11 Oct → GW6). Scored exactly and ±1.
7. **`no_coverage`**: status `a`, zero tagged articles, name absent from every chunk before `as_of`.
   Expected `unknown`, `must_have_evidence=false`; *any* returned quote is a false positive.
8. **Retrieval relevance is at article level** (chunk ids change on reindex) and **freshness-aware**:
   an article is relevant if a human answering the query *as of `as_of`* would use it. Superseded
   items (August headlines about an injury that has since been updated) are *not* relevant, and neither
   are pieces that merely mention the player (transfer gossip, tactics features, price-change tables).
   Headline-only `google_news` items count as relevant when the headline itself answers the query.
   Untagged-but-relevant articles (entity matcher misses) are included on purpose (e.g. 741, 560).
9. Three retrieval queries have **no relevant article** by construction (players with no coverage);
   they measure false positives via the top-1 score gap against answerable queries.

## Composition (signals)

| expected class | n | examples |
|---|---|---|
| doubtful | 9 | João Pedro (75%), Mosquera (stale 50%), White, Tonali, Amenda (fresh flag, promoted club), Gomes / Hincapie / James / Emersonn (`conflict`) |
| injured | 5 | Caicedo (return 18 Sep), Doku, D. Henderson (GK), Dasilva + Milenković (`fpl_api_only`) |
| fit | 5 | Haaland, Gabriel (`ambiguous_name`), Porro (`conflict`), Timber, Raya (GK) |
| unknown | 4 | Kinsky (GK), Tarkowski, van Ewijk (promoted club), Botman — `no_coverage` |
| suspended | 3 | Foden (3 matches → GW7), Awoniyi (promoted club → GW7), Reinildo (domestic ban only → GW6) |
| unavailable | 1 | Vicario (loan to Juventus) |

Tags: `return_date` 8 · `ambiguous_name` 7 · `conflict` 5 · `misleading_news` 4 · `goalkeeper` 4 ·
`promoted_club` 4 · `no_coverage` 4 · `heavy_coverage` 4 · `stale_flag` 2 · `fpl_api_only` 2 ·
`headline_only` 1 · `fresh_flag` 1.

## Known weaknesses (read before trusting a number)

- **Weak labels from the FPL API.** For 7 players the only ground truth is the FPL flag itself
  (`fpl_api_only`, several `doubtful`). The flag can be stale (Mosquera's 50% is 13 days old) or
  optimistic (João Pedro 75% while the CBF calls him injured). Where the news disagreed, the range was
  widened rather than the class changed. The weakest label is Hincapie (`doubtful` on "little issues").
- **Single annotator, no adjudication.** All 45 rows were labelled by one person in one sitting;
  inter-annotator agreement is unknown. Conflict cases carry an alternate label to make the
  disagreement explicit rather than hide it.
- **Corpus snapshot.** Labels describe the corpus as ingested on 17 Sep 2026 ~08:00Z (1100 articles /
  1618 chunks). The ingest loop keeps back-filling; an article published before `as_of` but fetched
  later would be unlabelled and counted as a false positive. Every results file stores the corpus
  counts and `max(published_at)` it saw.
- **Outcome is unknown.** Labels are "what a careful analyst would say at `as_of`", not what happened
  in GW5. A player labelled `doubtful` who then started does not make the label wrong.
- **Ranges are judgment calls** (e.g. 0.30–0.75 for João Pedro). They are deliberately wide; the
  metric is "inside the range", not distance.
- **Small n.** 27 signal rows / 18 queries: a single flipped row moves accuracy by ~4 pp; use the
  per-row output in `evals/results/*.json` rather than the aggregate alone when comparing modes.
- **Entity tags in the corpus are noisy.** Surname substrings ("James", "White", "Gabriel") are
  matched to the wrong player in places; the golden set includes such players on purpose
  (`ambiguous_name`) so the effect is measured, not hidden.

## Regenerating / extending

Rows are plain JSON; add a line, keep `as_of` fixed, run `uv run pytest -q tests/test_evals_golden.py`
(schema + protocol checks) and, with Postgres up, `uv run pytest -q -m db tests/test_evals_golden.py`
(article ids/URLs still exist). Resolve `relevant_urls` from `news_articles.url` by id — never by hand.
Record every label change in `CHANGELOG.md` (date, row, old → new, why).
