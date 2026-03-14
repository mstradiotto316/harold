You are starting an autonomous autoresearch session for the Harold quadruped robot.

Read `docs/autoresearch/program.md` — this is your complete lab policy, experiment loop, parameter registry, and scoring system in one file.

Then:
1. Run `python3 scripts/autoresearch.py history` — check prior results
2. Read `docs/memory/OBSERVATIONS.md` — accumulated insights
3. Run `python3 scripts/harold.py ps` — check for orphan processes
4. Create branch: `git checkout -b autoresearch/session-$(date +%Y-%m-%d)`
5. Run baseline if no prior score in results.tsv
6. Begin the experiment loop. Do not stop.
