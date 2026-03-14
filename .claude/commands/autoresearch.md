You are starting an autonomous autoresearch session for the Harold quadruped robot.

Read these files in order, then begin the experiment loop. Do not ask for confirmation. Do not summarize what you read. Just read, initialize, and start experimenting.

1. Read `docs/autoresearch/strategy.md` — your lab policy, goals, frozen/mutable parameters, scoring, and roadmap
2. Read `docs/autoresearch/AGENT_PROTOCOL.md` — the step-by-step experiment loop you will follow
3. Read `docs/autoresearch/PARAMETER_REGISTRY.md` — every parameter classified as FROZEN/CONSTRAINED/TUNABLE
4. Run `python3 scripts/autoresearch.py history` — check prior experiment results
5. Read `docs/memory/OBSERVATIONS.md` — accumulated project insights
6. Run `python3 scripts/harold.py ps` — check for orphan training processes

Then:
- Create branch: `git checkout -b autoresearch/session-$(date +%Y-%m-%d)`
- If no baseline score exists in results.tsv, run a baseline experiment first
- Begin the experiment loop as defined in AGENT_PROTOCOL.md
- Do not stop until you hit max_experiments or are manually interrupted
