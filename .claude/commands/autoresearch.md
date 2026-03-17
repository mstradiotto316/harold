You are starting an autonomous autoresearch session for the Harold quadruped robot.

Read `docs/autoresearch/program.md` — this is your complete lab policy, experiment loop, parameter registry, and scoring system in one file.

Then:
1. Run `python3 scripts/autoresearch.py state` — recover session state (baseline, kept changes, strategy)
2. Run `python3 scripts/autoresearch.py synthesize` — get strategic summary of cross-session patterns
3. Run `python3 scripts/autoresearch.py history` — check prior results
4. Read `docs/memory/OBSERVATIONS.md` — accumulated insights
5. Run `python3 scripts/harold.py ps` — check for orphan processes
6. Create branch: `git checkout -b autoresearch/session-$(date +%Y-%m-%d)`
7. Run baseline if no prior score in results.tsv
8. Begin the experiment loop. Do not stop.

## CRITICAL: You must NEVER stop

- The human may be asleep. You run until **manually interrupted** or you hit a session limit.
- There is no experiment limit. There is no "good stopping point."
- If you run out of ideas: re-read results.tsv, run `detect-plateau`, run `suggest-combinations`, try a different axis.
- If context is getting large: run `/compact` — then immediately re-read `program.md` and run `autoresearch.py state` to recover. Do NOT stop after compacting.
- **Compact every 3 experiments** to avoid context exhaustion. Do not wait until "after the next one."
- After compacting, your FIRST action must be to recover state and continue the loop — not to summarize or ask the user.
- If a tool call fails, diagnose and retry. Do not stop.
- If you are unsure what to try next, that is not a reason to stop. Pick the least-tried axis and experiment.
