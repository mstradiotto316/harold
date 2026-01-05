# Memory System Index

These files capture current state, priorities, experiment history, and accumulated insights.

## Read at Session Start (in order)

1. `docs/memory/CONTEXT.md` - Current project state and goals.
2. `docs/memory/NEXT_STEPS.md` - Priority queue and pending tasks.
3. `docs/memory/EXPERIMENTS.md` - Recent experiment history.
4. `docs/memory/OBSERVATIONS.md` - Curated, current insights (short on purpose).
5. `docs/memory/HARDWARE_CONSTRAINTS.md` - Sim-to-real limits and safe ranges.

## Update at Session End

- Update `docs/memory/EXPERIMENTS.md` with any experiments run.
- Update `docs/memory/OBSERVATIONS.md` with new insights.
- Update `docs/memory/NEXT_STEPS.md` to reflect completed/new tasks.
- Create/update a session log in `docs/memory/sessions/`.
- Update `docs/memory/CONTEXT.md` if project state has significantly changed.

## Notes

- Session logs in `docs/memory/sessions/` are historical records; keep them append-only.
- AGENTS should always follow the current paths listed above.
- Historical observations live in `docs/memory/archives/`; read them only when needed to avoid context bloat. Start at `docs/memory/archives/index.md`.
