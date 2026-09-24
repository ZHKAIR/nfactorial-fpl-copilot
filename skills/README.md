# Skills

A **Skill** is a folder with a `SKILL.md` (YAML frontmatter `name` + `description`, then a short
procedure) that an agent loads when the user's request matches the description. It is not a
prompt the user pastes: the host (Cursor, Claude Code, Codex) discovers it, keeps it out of the
context until it is relevant, and then follows it as a checklist. Long material lives next to it
in `references/` and is read only when needed.

| skill | purpose | installed for Cursor at |
|---|---|---|
| [`fpl-transfer-analyst`](fpl-transfer-analyst/SKILL.md) | answer FPL decision questions through the `fpl-intelligence` MCP tools (14, incl. `search_strategy_kb` for cited rules / strategy answers) with fixed tool order, guardrails and one output template | `.cursor/skills/fpl-transfer-analyst/` (byte-for-byte copy; symlinks may not be followed) |

Claude Code reads the same content from `.claude/skills/<name>/SKILL.md`; Codex from
`.agents/skills/`. Copy the folder (or run the one-liner below) — the format is identical.

```bash
rsync -a --delete skills/fpl-transfer-analyst/ .cursor/skills/fpl-transfer-analyst/   # keep the copy in sync
```

## Why a Skill and not a plain system prompt (for this domain)

- **Procedural tool order.** The right answer depends on *which* deterministic tool runs and in
  which order (`diagnose_squad` before `recommend_transfers`; `analyze_player_risk` for a doubtful
  player before quoting his xPts). A Skill encodes the branch per question type; a chat prompt
  leaves it to the model each time.
- **Guardrails that survive long chats.** "Never compute sums yourself", "no bookmaker odds",
  "ask before presenting a -4 or a chip as the decision", "ambiguous name -> ask" are the same
  invariants the LangGraph agent enforces in code (`docs/agent.md`); the Skill carries them into
  any MCP host without re-implementing the graph.
- **Consistent output.** One template (verdict, options table, Why with `[source, dd.mm]`,
  Sources, Caveats) makes answers comparable week to week and checkable against tool output.
- **Progressive disclosure.** The rules digest (scoring incl. DefCon, FT banking, two chip sets
  and the GW19 expiry) sits in `references/` and costs no tokens until a rules question arrives.
- **Cited strategy answers.** Rules and "how to play" questions go through `search_strategy_kb`
  (the strategy knowledge base of `docs/strategy_kb.md`), so every rule the host states carries
  a source instead of coming from stale training data; hit / chip verdicts get one cited rule.

## How to test it

1. Open `fpl-copilot/` as the Cursor workspace root (the project ships `.cursor/mcp.json` and
   `.cursor/skills/`); enable the `fpl-intelligence` server in Cursor Settings -> MCP (it should
   list 14 tools). Postgres must be up (`docker compose up -d db`).
2. In Cursor chat ask a trigger question, e.g. *"Should I sell João Pedro? manager 895045"*,
   *"Who should I captain this week? manager 895045"*, *"Is Haaland fit for GW5?"*,
   *"Plan my transfers for the next 5 gameweeks (895045)"*, *"Plan 5 GWs with Bench Boost in
   GW7 (6856911)"* (`build_gameweek_plan(chips=...)`), *"What if I take a -4 for Saka?"*,
   *"When should I play my wildcard?"* (strategy question -> `search_strategy_kb`).
3. Check: the agent calls the tools in the Skill's order, quotes only tool numbers, cites
   `[source, dd.mm]` for news and `[source]` / `[n]` for knowledge-base chunks, states the as-of
   time and the last-finished-GW squad caveat, asks for confirmation before a hit / Wildcard,
   and asks which player is meant on an `ambiguous` reply (try *"Should I sell Gabriel?"*).
4. Without Cursor: `uv run python scripts/mcp_smoke.py` exercises the same tools over stdio, and
   `npx @modelcontextprotocol/inspector@latest --cli uv run python -m fplcopilot.mcp_server -- --method prompts/get --prompt-name pre_deadline_review --prompt-args manager_id=895045`
   prints the prompt that mirrors the Skill workflow.
