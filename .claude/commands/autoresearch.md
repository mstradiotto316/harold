You are starting an autonomous autoresearch session for the Harold quadruped robot.

Read `docs/autoresearch/program.md` — this is your complete lab policy, experiment loop, parameter registry, and scoring system in one file.

Then:
1. Run `python3 scripts/autoresearch.py state` — recover session state (baseline, kept changes, strategy)
2. Run `python3 scripts/autoresearch.py history` — check prior results
3. Read `docs/memory/OBSERVATIONS.md` — accumulated insights
4. Run `python3 scripts/harold.py ps` — check for orphan processes
5. Create branch: `git checkout -b autoresearch/session-$(date +%Y-%m-%d)`
6. Run baseline if no prior score in results.tsv
7. Begin the experiment loop. Do not stop.
