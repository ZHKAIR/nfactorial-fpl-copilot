# LLM choice per role: cost / latency / quality, hyperparameters, fallbacks

Every LLM call in FPL Copilot has one of seven roles. This document records, per role, which model
is used, which alternatives were measured, what the measurements say (quality, latency, cost) and
how a fallback would be wired. The numbers come from the experiments in
[`docs/EVALS.md`](EVALS.md) §7–§8 (hyperparameters, model comparison; runner code
`evals/run_hparams.py`, `evals/run_models.py`; evidence files under `evals/results/`), from the
vision smoke runs in [`docs/vision.md`](vision.md) and from the retrieval A/Bs in `EVALS.md` §3–§4.
All were run on 17 Sep 2026 against the fixed golden set (`as_of = 2026-09-17T09:00Z`, 27 players)
and the demo manager 895045. Prices are OpenAI list prices per 1M tokens (input / output) as of
17 Sep 2026, so every dollar figure is an estimate from measured token counts, not a bill.

## 1. Decision table

| role | where | chosen model | T / top_p / max_tokens | alternatives measured | why this one (one line) | fallback (design, not implemented) |
|---|---|---|---|---|---|---|
| **router** (intent, mentions, scenario) | `agent/llm.py: router_llm` | **gpt-4o-mini** | 0 / default / 400 | gpt-4.1-nano, gpt-4.1-mini, gpt-5.4-mini, gpt-4.1 | 8/8 intents for *every* model, 100 % agreement; 4o-mini is the cheapest with a proven structured-output path ($0.00025/call, p50 1.0 s) | gpt-4.1-nano (same 8/8, $0.00017); cross-vendor: Claude Haiku / Gemini Flash via the same client factory |
| **signal extraction** (news → `PlayerSignal`) | `rag/extract.py` | **gpt-4o-mini** | **0** / not sent / 800 | T 0.3, 0.7; top_p 0.5; gpt-4.1-nano, gpt-4.1-mini, gpt-5.4-mini, gpt-4.1 | strict accuracy 23/27 = gpt-4.1, 1 row behind gpt-5.4-mini; the deterministic validator repairs its non-verbatim quotes (0.22 fixes/row); 5.7–14× cheaper than the two better-behaved models; $0.40 per full 659-player refresh | gpt-5.4-mini (best F1 0.92, 0.07 fixes/row) for conflict rows or if the budget allows; cross-vendor: Claude Sonnet / Gemini Flash with pydantic JSON mode |
| **explain** (facts JSON → Markdown answer) | `agent/llm.py: explain_llm` | **gpt-4o-mini** | **0** (shipped 17.09, was 0.2) / default / **1400** | T 0, 0.7; max_tokens 300, 600, 1000; the four models above | stronger models write longer answers with *more* invented derived numbers (gpt-4.1: 2 unknown numbers, 262 words, 6.5 s, 15× the price); 4o-mini passes the validator 83 % first time at 3.5 s and $0.0007 | any of the above — the validator + one regeneration make the role model-agnostic; gpt-4.1-nano is the emergency cheap fallback (misses sections in 1/3 answers) |
| **grader** (signal sufficiency, rare) | `agent/llm.py: grader_llm` | **gpt-4o-mini** | 0 / default / 120 | — (fires only when a signal has quotes but confidence < 0.4; never fired in the demo) | same model as the router: tiny structured yes/no, cost negligible | gpt-4.1-nano |
| **faithfulness judge** (evals only) | `evals/judge.py`, `evals/run_hparams.py` | **gpt-4o-mini** for short summary-vs-quotes checks; **gpt-4.1-mini** for answer-vs-facts-JSON checks | 0 / default / 200–400 | gpt-4.1-mini re-judged the same 38 explain answers | over a 3k-token facts JSON gpt-4o-mini flags numbers that *are* in the facts (1.37 false claims/answer, mean 0.66); gpt-4.1-mini agrees with it on only 53 % of verdicts and is the one whose remaining flags are real (mean 0.89, 0.47 claims/answer) | gpt-4.1 for a human-calibration round; a second-vendor judge to break the same-family blind spot |
| **vision** (Pick Team screenshot → squad JSON) | `vision/extract.py` | **gpt-4o-mini** | 0 / default / — ; `detail=high` | `detail=low`; prompt v1 vs v2 | `high` reads 15/15 cards, prices 15/15; `low` is 4.5× cheaper ($0.0015 vs $0.0066) but guesses prices (1/15), misses C/V badges and once swapped a player for a real other player — the worst silent failure | gpt-4.1-mini / gpt-5.4-mini (both multimodal) if real screenshots break 4o-mini; temperature is already skipped for reasoning models |
| **embeddings** (news + KB chunks, queries) | `rag/llm.py: embed_texts` | **text-embedding-3-small** (1536 d) | — | none measured; MiniLM-class local embeddings considered in `docs/rag.md` | $0.02/1M tokens: the whole news index (1 639 chunks) costs ≈ $0.01, a query < $0.000001; quality was good enough that the entity prefilter, not the embedding, decides top-1 (MRR 0.947, EVALS §3) | text-embedding-3-large (needs `vector(3072)` migration + reindex) or a local `bge-small` — any change means a full reindex (dims differ) |
| **reranker** (cross-encoder, local) | `rag/retrieve.py` | **flashrank ms-marco-MiniLM-L-12-v2** (ONNX, 22 MB, CPU) | — | none, hybrid, dense (EVALS §3) | the only local model: +4 pp recall@8 vs dense, fresher top-1, and its score separates answerable from unanswerable retrieval queries; costs ≈ 0.65 s CPU per query, $0 | `NoopReranker` (already wired, automatic if flashrank fails to import); TinyBERT-L-2 for latency; Cohere Rerank as a hosted option |
| **KB answerer** (strategy questions with citations) | `rag/kb/answer.py` | **gpt-4o-mini** (`RAG_LLM_MODEL`) | 0 / default / fixed | — (evals in `docs/strategy_kb.md`) | same model and same verbatim-quote validator as extraction; 10 answers ≈ $0.004 | as extraction |
| **"Почему" narration** (added 23.09; route card on «К дедлайну»: code-built facts → 2–4 Russian sentences) | `core/why_narrate.py: _llm_complete` | **gpt-4o** (`WHY_LLM_MODEL`) — the only role not on 4o-mini | 0.3 / default / 280 (`max_completion_tokens`) | **none — not measured in an A/B.** The reason in `config.py` ("русский у mini был водянистый") is a developer's impression, not an eval | the facts are computed by code (`core/why_facts.py`, no LLM); `validate_why_text` admits only numbers and names from the facts, ≤ 4 sentences, no stock phrases; one retry, then a deterministic fallback text; disk cache per route (`.cache/why/`). **Cost per call (estimated from tokens, not billed):** ≈ 2.1k input (586 system + 1.2k for four few-shot pairs + 0.2–0.4k facts, `o200k_base`) + ≈ 80 output → **≈ $0.006** at $2.50 / $10.00 (≤ $0.008 at the 280-token cap); the same call on gpt-4o-mini ≈ $0.0004 (≈ 17× cheaper). ≤ 3 calls on the first view of a squad's routes, then cached; in the local cache 43 of 55 texts came from the LLM and 12 from the fallback | **candidate for a check: gpt-4o-mini vs gpt-4o on the same payloads** — validator pass rate, fallback share, blind human read of the Russian text; switching is `WHY_LLM_MODEL=gpt-4o-mini`, no code change |

Two other roles added on 23.09 run on gpt-4o-mini and were not measured separately either: the
«Сравнение» explanation (`app/compare.py`, T = 0.2, 900 tokens, facts built by code, shown next to
the code-built facts table) and the club-news digest (`rag/team_news.py`, `RAG_LLM_MODEL`, T = 0,
same verbatim-quote validator as extraction).

**One shipped default changed as a result:** explain temperature 0.2 → 0 (§4.2), applied on 17.09
in the consistency pass (`AGENT_EXPLAIN_TEMPERATURE` in `config.py`, read by `agent/llm.py:
explain_llm`). Every other default is supported by the data as it stands.

## 2. Candidates and prices

`client.models.list()` on the project key (17 Sep 2026) offers the gpt-4o, gpt-4.1, gpt-5 … gpt-5.6
families and the o-series. Candidates were chosen so that the comparison is at **T = 0 with the same
prompts and the same structured-output schemas**:

| model | input $/1M | output $/1M | why in the set |
|---|---|---|---|
| gpt-4o-mini | 0.15 | 0.60 | shipped everywhere; baseline |
| gpt-4.1-nano | 0.10 | 0.40 | the cheapest chat model on the key |
| gpt-4.1-mini | 0.40 | 1.60 | mid-price non-reasoning |
| gpt-5.4-mini | 0.75 | 4.50 | newest small model (Mar 2026); a probe showed it accepts `temperature=0` and spends 0 reasoning tokens at default effort |
| gpt-4.1 | 2.00 | 8.00 | "smartest non-reasoning model" (OpenAI); the strongest model that takes T = 0 |

Excluded: gpt-4o (2.50 / 10.00 — pricier than gpt-4.1 with an older cutoff), the o-series and the
full gpt-5.x models (reasoning models: `temperature`/`top_p` are not controllable — the very knobs
this document justifies — and reasoning tokens make cost and latency prompt-dependent; gpt-5.4 full
is $2.50 / $15.00). Prices: `evals/run_hparams.py: PRICES` (= `agent/llm.py: PRICES_PER_1M` +
the gpt-5.4 family from the OpenAI model pages).

## 3. Extraction: the role where the model matters most

27 golden players, `hybrid_rerank` retrieval v2, prompt v2, pre-LLM abstention on (4 rows never call
the model), T = 0, one shared retriever so every model saw identical chunks (`retrieval_agreement
= 1.0`; the corpus did not move during the run: 1 109 articles / 1 639 chunks visible at `as_of`).
File: `evals/results/20260917T120615Z_models.json`.

| metric | gpt-4o-mini | gpt-4.1-nano | gpt-4.1-mini | gpt-5.4-mini | gpt-4.1 |
|---|---|---|---|---|---|
| availability accuracy, strict | **0.852** (23/27) | 0.556 (15/27) | 0.778 (21/27) | **0.889** (24/27) | **0.852** (23/27) |
| availability accuracy, lenient (alternate label on `conflict` rows) | 0.926 | 0.556 | 0.889 | 0.963 | **1.000** |
| macro-F1 | 0.733 | 0.506 | 0.680 | **0.924** | 0.903 |
| start_probability in range | 0.926 | 0.778 | 0.889 | 0.926 | 0.889 |
| return_gw exact (n = 8) | 1.000 | 0.750 | 1.000 | 1.000 | 1.000 |
| evidence present when required (n = 23) | 0.957 | 0.478 | 0.913 | **1.000** | **1.000** |
| false evidence on `no_coverage` (n = 4) | 0 | 0 | 0 | 0 | 0 |
| validator fixes per row | 0.222 | 2.556 | 0.481 | **0.074** | **0.074** |
| faithfulness judge (gpt-4o-mini), n judged | 0.725, 20 | 0.818, 11 | 0.650, 20 | 0.800, 20 | 0.775, 20 |
| agreement with gpt-4o-mini labels | — | 15/27 | 24/27 | 22/27 | 23/27 |
| completion tokens per call | 140 | 130 | 164 | 189 | 171 |
| cost, 23 LLM calls (list price) | **$0.014** | $0.009 | $0.038 | $0.080 | $0.193 |
| cost per call | $0.00061 | $0.00040 | $0.00166 | $0.00347 | $0.00837 |
| LLM latency p50 / p95, ms | **1 648 / 2 089** | 2 942 / 6 506 | 2 038 / 2 821 | 1 845 / 2 179 | 2 719 / 7 090 |

Strict misses. gpt-4o-mini: Porro and Emersonn (two-sided `conflict` rows, alternate label accepted),
Timber (`misleading_news`: "didn't feature — we need to look after him" read as a doubt), Vicario
(`unknown`: his only official item is 29 days old and falls out of the candidate set — a retrieval
gap, EVALS §4.3). gpt-4.1: Porro, Hincapié, James, Emersonn — **all four are `conflict` rows**, so
lenient = 1.000; it resolves Timber and Vicario. gpt-5.4-mini: Gomes, Emersonn (conflict) and Doku
(`doubtful` for a player FPL lists as injured but "quite close" — a defensible reading). gpt-4.1-mini
is *worse* than gpt-4o-mini at 2.7× the price (two extra `unknown`s from dropped quotes).
**gpt-4.1-nano is unusable for this role**: 12 rows collapse to `unknown` because its quotes are
not verbatim (2.6 validator fixes per row → evidence dropped → rule (c) forces `unknown`); it also
loses two return GWs that the code would have derived, because the forced `unknown` clears them.

Reading. On the label itself the three usable models are within one row of each other — exactly the
run-to-run noise of a single model (§4.1: gpt-4o-mini at T = 0 scored 22, 23 and 23/27 in three
runs today). What the bigger models buy is **behaviour, not accuracy**: verbatim quotes (0.07 vs 0.22
fixes/row), evidence on every covered player (100 % vs 96 %), confidence better separated (0.85–0.89
vs 0.81 on known rows). The project's design makes that behaviour cheap to obtain otherwise: the
validator repairs or drops non-verbatim quotes deterministically, and the pipeline's false-evidence
rate is 0 % for every model. The one gap the small model cannot close by itself — Timber's dropped
quote — is a prompt-v3 item, not a model item.

**Decision: gpt-4o-mini stays.** Cost per full refresh of the 659-player bootstrap (upper bound,
every player calls the model): **$0.40** vs $2.29 (gpt-5.4-mini) vs $5.52 (gpt-4.1); with the golden
set's abstention share (23/27 call the model) $0.34 / $1.95 / $4.70. In the real pool most players
have no news at all, so the true batch cost is lower still. If a budget for quality appears,
**gpt-5.4-mini is the upgrade** (best F1, best evidence discipline, same latency as 4o-mini),
ideally only for the rows where it matters: `conflict`/`doubtful` players in the user's own squad.

### 3.1 Extraction hyperparameters (gpt-4o-mini)

Summary of EVALS §7.1 (`evals/results/20260917T115720Z_hparams_extraction.json`): T ∈ {0, 0.3,
0.7} with T = 0 and T = 0.7 run twice, plus top_p 0.5 at T = 0.7.

| | T 0 run 1 | T 0 run 2 | T 0.3 | T 0.7 run 1 | T 0.7 run 2 | T 0.7, top_p 0.5 |
|---|---|---|---|---|---|---|
| strict accuracy | 0.815 | 0.852 | 0.852 | 0.852 | 0.926 | 0.815 |
| lenient | 0.889 | 0.926 | 0.926 | 0.926 | 1.000 | 0.889 |
| fixes per row | 0.48 | 0.22 | 0.33 | 0.48 | 0.15 | 0.48 |
| faithfulness (n = 20) | 0.775 | 0.750 | 0.775 | 0.775 | 0.650 | 0.750 |
| cost / LLM p50 | $0.014 / 1.6 s | same | same | same | same | same |

Temperature has **no measurable effect on accuracy, fixes, faithfulness, cost or latency** at n = 27:
the spread between two runs with identical settings (22 vs 23/27 at T = 0; 23 vs 25/27 at T = 0.7)
is as large as any difference between temperatures, and the flipping rows are always the same
two-sided ones (Hincapié, Vicario, Timber). What temperature does change is **reproducibility**:

| identical between two runs (23 LLM rows) | T = 0 | T = 0.7 |
|---|---|---|
| availability label | 22/23 (0.957) | 21/23 (0.913) |
| start_probability | 21/23 (0.913) | 20/23 (0.870) |
| cited chunk ids | 21/23 (0.913) | 15/23 (0.652) |
| summary text | 12/23 (0.522) | 0/23 (0.000) |
| whole output | 11/23 (0.478) | 0/23 (0.000) |

**T = 0 is therefore the right setting, but it is not determinism**: even at T = 0 one label in 23
flipped and half the summaries were reworded. Anything that must be reproducible (the label, the
quote set) has to be pinned by code (validator, deterministic return-GW, abstention), which is what
the pipeline does. `top_p` is left at the API default: at T = 0.7 lowering it to 0.5 changed nothing
that the noise floor could not explain (n = 1 run), and at T = 0 it is irrelevant. `max_completion_tokens
= 800` never truncated a structured signal (23/23 `finish_reason = stop` in every arm; 130–190
completion tokens used).

## 4. Explain: the role where the model matters least

Six fixed queries (status, transfer, captain, plan, compare, lineup) were run through the live agent
once; the `ExplainRequest` (facts JSON + evidence + caveats) of each was captured and replayed with
the production prompt for every variant, so **all variants and all models saw byte-identical facts**.
Files: `20260917T115722Z_hparams_explain.json` (grid, run 1), `20260917T120327Z_hparams_explain_rep2.json`
(same facts, grid run 2), `20260917T120432Z_hparams_explain_judge-gpt-4.1-mini.json` (run 1 re-judged),
`20260917T120615Z_models.json` (models at the production cell).

### 4.1 Model comparison at the production cell (T = 0.2, max_tokens 1400; 6 answers each)

| metric | gpt-4o-mini | gpt-4.1-nano | gpt-4.1-mini | gpt-5.4-mini | gpt-4.1 |
|---|---|---|---|---|---|
| truncation | 0 | 0 | 0 | 0 | 0 |
| validator passed first time | **5/6** | 5/6 | 4/6 | 5/6 | 5/6 |
| violations (names / numbers / placeholder citations) | 0 / 0 / 1 | 0 / 1 / 0 | 1 / 2 / 0 | 0 / 1 / 0 | 0 / 2 / 0 |
| format complete (Verdict, Why, Caveats) | 1.00 | 0.67 | 1.00 | 1.00 | 1.00 |
| answer length, words / completion tokens | 192 / 477 | 186 / 367 | 285 / 609 | 245 / 594 | 262 / 645 |
| faithfulness judge (gpt-4o-mini; noisy, §5) | 0.58 | 0.75 | 0.58 | 0.67 | 0.67 |
| cost per answer | **$0.0007** | $0.0004 | $0.0022 | $0.0049 | $0.0111 |
| latency p50 / p95 | **3.5 / 5.4 s** | 2.5 / 3.4 s | 5.0 / 6.9 s | 4.2 / 6.0 s | 6.5 / 11.3 s |

The stronger models are not more faithful here — they are more *productive*: gpt-4.1 and gpt-4.1-mini
both wrote the same two derived numbers (`4.60`, `3.30`) into the lineup answer, gpt-5.4-mini wrote
`4.49`; none of these exist in the facts (the prompt forbids computing). gpt-4o-mini's only violation
was a placeholder citation (`[FACTS]`) on the same lineup query. The validator catches all of these
and triggers one regeneration, so the *user-visible* quality is the same across models; what differs
is price (15×) and latency (2×). gpt-4.1-nano is cheap and fast but dropped a required section in 2 of
6 answers. **Decision: gpt-4o-mini stays.**

### 4.2 Temperature × max_tokens (gpt-4o-mini, 6 queries × 10 cells × 2 runs)

Pooled over both runs. "Completed" = `finish_reason = stop`; validator metrics are computed over
completed answers only (a truncated answer is a failure regardless of what it says).

| max_tokens | answers | truncated | validator pass (completed) | violations / answer | completion tokens (completed) | judge gpt-4.1-mini | p50 |
|---|---|---|---|---|---|---|---|
| 300 | 36 | **100 %** | – | – | – | – | 2.8 s |
| 600 | 36 | **25 %** (transfer ×4, lineup ×5) | 0.93 | 0.07 | 431 | 0.89 | 4.3 s |
| 1000 | 36 | 0 % | 0.83 | 0.17 | 480 | 0.92 | 4.1 s |
| 1400 (production) | 12 | 0 % | 0.75 | 0.25 | 470 | 0.83 | 3.6 s |

Completed answers use 328–736 completion tokens (median ≈ 440) for 150–350 words plus a table; the
two intents with a table of routes or a full XI need 570–740. **max_tokens = 300 is unusable, 600
truncates a quarter of the answers, 1000 is the minimum, 1400 (the shipped value) gives ≈ 2× the
median as headroom** at no cost (tokens are billed as used). In production a truncated structured
answer raises in the SDK and the explain node falls back to a JSON dump of the facts — a hard
failure, which is why the truncation rate, not the length, is the deciding metric.

| temperature (cells with max_tokens ≥ 1000, no truncation confound) | answers | validator pass | violations / answer | placeholder citations | unknown numbers | words | judge gpt-4.1-mini | judge gpt-4o-mini | p50 |
|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 12 | **0.92** | **0.08** | 1 | 0 | 191 | 0.92 | 0.58 | 3.8 s |
| 0.2 (shipped until 17.09) | 24 | 0.79 | 0.21 | 5 | 0 | 191 | 0.88 | 0.60 | 4.0 s |
| 0.7 | 12 | 0.75 | 0.25 | 2 | 1 | 194 | 0.92 | 0.58 | 4.1 s |

Length, judge score, cost and latency are flat across temperature. The only metric that moves is the
deterministic validator, and it moves the expected way: fewer first-pass violations at T = 0. The
effect is small in absolute terms (1 violation in 12 answers vs 5 in 24) and comes almost entirely
from one query — the lineup answer produces a placeholder citation (`[source not available]`,
`[FACTS]`) in 7 of its 9 completed answers at any temperature, while captain and plan pass 14/14. So
this is first a **prompt problem** (lineup facts carry an "issue" note the model wants to cite; v2
candidate: "issues from FACTS need no citation") and only second a temperature one.
**Recommendation: T = 0 for explain** — nothing in the answer needs sampling diversity, and every
regeneration avoided saves 3–4 s and $0.0007. **Shipped 17.09 (consistency pass):** the default is
now `AGENT_EXPLAIN_TEMPERATURE = 0` (`config.py`; `explain_llm` reads it when no temperature is
passed). A three-scenario check at T = 0 (transfer, captain, Russian strategy question) passed the
validator first time in 3/3 with the same verdicts as the demo; the agent demos and the prompt A/B
in `docs/agent_demo_output_v2.md` / `docs/agent_prompt_ab.md` were recorded before the switch, at 0.2.

What these tables do **not** show: whether a T = 0.7 answer reads better (no human preference
labels), whether the placeholder-citation habit is reproducible under a fixed prompt (T = 0 still
reworded half of the extraction summaries between runs), and anything about long answers — the
prompt caps them at 350 words.

## 5. Judge: the model choice inside the evals

The extraction judge (`evals/judge.py`, gpt-4o-mini, T = 0) reads a two-sentence summary against
≤ 6 quotes — a task of a few hundred tokens — and its verdicts have been consistent across A/B #1,
#2 and today's runs (0.65–0.82). The explain judge has to read a ≈ 3k-token facts JSON, and there
gpt-4o-mini fails as a reader: on the first explain run it marked "probability of starting 0.64",
"expected minutes 30" and "medium rotation risk" as unsupported although all three are literally in
the facts. Handing it the list of numbers already verified by code (the hybrid judge in
`run_hparams.py`) did not stop it. Re-judging the same 38 completed answers with gpt-4.1-mini
(`--rejudge`, $0.047):

| judge | supported / partial / unsupported | mean score | flagged claims per answer | agreement between the two |
|---|---|---|---|---|
| gpt-4o-mini | 12 / 26 / 0 | 0.66 | 1.37 | 53 % |
| gpt-4.1-mini | 30 / 8 / 0 | 0.89 | 0.47 | |

The remaining gpt-4.1-mini flags are real "beyond the facts" expansions rather than fabrications:
writing "Sunderland" for a fixture the facts give as a code, calling FSI 1 "very easy" (the label
comes from the prompt's legend, not the facts), and statements about Salah in the compare query
(the facts only carry him as an unresolved mention). The validator cannot see any of these (club
names are whitelisted), so the judge is the only instrument for them.
**Decision: gpt-4.1-mini for answer-vs-facts judging, gpt-4o-mini for summary-vs-quotes judging;
absolute judge scores are still uncalibrated against humans and must be read as relative.**

## 6. Vision: `detail` as a hyperparameter

From `docs/vision.md` (9 real calls, gpt-4o-mini, T = 0, prompt v2): `detail=high` costs 36 835
image tokens (6 tiles × 5 667 + 2 833) ≈ **$0.0066 per screenshot** and reads 15/15 cards, 15/15
prices, both badges on the valid sample and exactly the expected violations on the broken one;
`detail=low` costs 2 833 image tokens ≈ **$0.0015** (4.5× cheaper, same ≈ 7 s latency) but guesses
prices (1/15 correct), misses the C/V badges, distorts one name per screen and once substituted a
real *other* player ("Livramento" for Virgil) — a failure no name-resolution step can catch. The
saving of half a cent is not worth one wrong player in the optimizer's input, so **`high` is the
default** and `low` is not offered even as a draft mode. Temperature is 0 (structured output) and is
skipped automatically for reasoning models (`vision/extract.py: _supports_temperature`). The prompt
v1 → v2 change (explicit layout scratchpad) fixed the 10/14-card failure at identical cost.

## 7. Cost per query and per batch (from measured tokens, list prices)

| | gpt-4o-mini (shipped) | gpt-5.4-mini | gpt-4.1 |
|---|---|---|---|
| router call (1 388 + 64 tokens) | $0.00025 | $0.00137 | $0.00333 |
| explain call (≈ 3 000 + 477 tokens) | $0.00073 | $0.00488 | $0.01105 |
| **typical user query** = router + explain | **$0.0010** | $0.0063 | $0.0144 |
| query that also extracts one fresh signal | $0.0016 | $0.0097 | $0.0227 |
| extraction call (3 500 + 140 tokens) | $0.00061 | $0.00347 | $0.00837 |
| **batch refresh, 659 players, every player calls the model** | **$0.40** | $2.29 | $5.52 |
| batch with the golden abstention share (23/27) | $0.34 | $1.95 | $4.70 |
| 30 days × (100 queries + 1 full batch) | ≈ **$15** | ≈ $88 | ≈ $209 |

Measured against the demo footers in `docs/agent.md` ($0.0009–0.0019 per query) the per-query
estimate is consistent. Retrieval embeddings add < $0.000001 per query; a full news reindex is
≈ $0.01, the strategy-KB index ≈ $0.002; a screenshot import adds $0.0066. The evals themselves
cost ≈ $0.80 for everything in this document (extraction grid $0.10, explain grid ×2 $0.12,
re-judge $0.05, model comparison $0.52, of which gpt-4.1 alone $0.29).

## 8. Fallbacks: how they would be wired (design only)

All OpenAI calls go through one factory, `rag/llm.py: get_openai_client()` (LangSmith-wrapped when
a key is present), and every role calls `client.chat.completions.parse(model=…, response_format=<pydantic>)`.
A vendor fallback therefore needs three things, none of which exist yet:

1. **A provider switch in the factory.** `get_openai_client()` becomes `get_llm_client(provider)`
   returning an object with the same `chat.completions.parse` surface; for Anthropic / Gemini the
   OpenAI-compatible endpoints (`base_url` + key) cover chat, and `.env.example` already reserves
   `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`. Per-role model names would come from the existing settings
   (`AGENT_ROUTER_MODEL`, `AGENT_EXPLAIN_MODEL`, `RAG_LLM_MODEL`, `VISION_MODEL`) extended with a
   provider prefix (`anthropic/claude-haiku-…`).
2. **Structured-output parity.** Strict JSON schema is OpenAI-specific; the fallback path would ask for
   JSON in the prompt and validate with the same pydantic models (`SignalDraftV2`, `RouterOutput`,
   `ExplainOutput`, `ScreenshotSquadRaw`), rejecting and retrying once on `ValidationError`. The
   deterministic validators downstream (quotes, numbers, names, FPL rules) are what make this safe:
   a fallback model that quotes loosely degrades to more `unknown`s, never to fabricated evidence
   (false-evidence rate stayed 0 % for all five models measured).
3. **Trigger and order.** Per role: primary → same-vendor fallback → cross-vendor, on `APIStatusError`
   5xx / 429 after the SDK's 3 retries, or on a role-specific quality trip-wire (extraction: fixes ≥ 5
   on a covered player; explain: two failed validations). Suggested order — router: gpt-4o-mini →
   gpt-4.1-nano → Claude Haiku; extraction: gpt-4o-mini → gpt-5.4-mini → Claude Sonnet; explain:
   gpt-4o-mini → gpt-4.1-nano → Gemini Flash; vision: gpt-4o-mini → gpt-4.1-mini → Claude Sonnet;
   judge (evals): gpt-4.1-mini → gpt-4.1. Embeddings have no hot fallback: a different model means
   different dimensions and a full reindex, so the fallback is "serve BM25-only" (`hybrid` mode
   without dense) until the index is rebuilt. Reranker: `NoopReranker` is already the automatic
   fallback when flashrank cannot load.

## 9. Trade-offs and why not a local model

- **Latency budget.** An interactive answer today is ≈ 5–8 s: router 1.0 s, tools 0.4–1.3 s (MILP),
  explain 3.5 s, plus 3–4 s when a signal must be extracted. The only local model — the reranker —
  already costs 0.65 s of CPU per retrieval. A local 7–8B instruct model on this laptop's CPU would
  need tens of seconds for the 3.5k-token extraction prompt and would still have to be validated by
  the same code; on a GPU box it would move the cost from tokens to hardware for a workload of
  ≈ $15/month. Local models were therefore not benchmarked; the reranker stays local because it is
  22 MB, has no API cost, and its retrieval gain is measured (EVALS §3).
- **Small model + deterministic guardrails beats big model + trust.** The measurements show the
  guardrails doing the work: gpt-4o-mini's extra validator fixes are repairs the user never sees;
  the bigger models' extra derived numbers in explanations are caught by the same validator. Model
  upgrades buy behaviour (verbatim quotes, no dropped sections), not correctness, at 5–15× the price.
- **Same-vendor risk.** Everything runs on one OpenAI key; a vendor outage stops the product. The
  fallback design above is the mitigation; the deterministic core (xPts, MILP, FPL API) keeps working
  without any LLM (the explain node already falls back to printing the facts).
- **Same-family evaluation.** All judges are OpenAI models judging OpenAI models; §5 shows even the
  judge's *size* changes verdicts on 47 % of answers. Human calibration of the judge is the missing
  piece before any judge number is quoted as absolute.
- **n is small.** 27 players, 6 queries, 8 router questions. One flipped row is 3.7 pp; two runs at
  identical settings differ by 1–2 rows. Every decision above rests on differences that are either
  large (nano's collapse, truncation at 300/600, the judge disagreement) or on cost, which is exact.
