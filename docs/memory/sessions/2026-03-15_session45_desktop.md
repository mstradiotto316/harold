# 2026-03-15 Session 45 (desktop)

## Summary
- Clarified the authoritative desktop and Raspberry Pi Python environments in the stable repo documentation.
- Added explicit runtime-context guidance so future agents distinguish between desktop venv issues and Isaac Sim launcher/runtime issues.
- Left unrelated in-progress simulation code changes in the worktree untouched.

## Documentation Updated
- `AGENTS.md`
- `docs/index.md`
- `docs/overview.md`
- `docs/sim/isaac_lab_extension.md`

## Memory Updates
- Added the environment/runtime-context clarification to `docs/memory/OBSERVATIONS.md`.
- Added a maintenance reminder to `docs/memory/NEXT_STEPS.md`.
- Noted the documentation clarification in `docs/memory/CONTEXT.md`.

## Experiments
- No training or hardware experiments were run in this session.

## Notes
- Desktop Isaac Lab work should use `/home/matteo/Desktop/env_isaaclab`.
- Raspberry Pi runtime work should use system `python3`.
- Plain-shell `omni.*` import failures are expected outside the Isaac Sim app/runtime context.
