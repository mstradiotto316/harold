# 2026-03-15 Session 44 (desktop)

## Summary
- Completed a full audit of the Isaac Lab simulation, training harness, and export/deployment interfaces.
- Identified correctness, realism, and tooling issues that should be treated as blockers before trusting new training results.
- Wrote a formal remediation plan to `PLAN.md`.
- Updated memory files so the next session can execute the fixes in priority order.

## Key Findings
- Flat-task velocity tracking and command telemetry mix world-frame reward logic with body-frame observations.
- Flat-task forward reward still leaks positive reward to low-upright states.
- Rough-task domain-randomization defaults overstate robustness because sampled values are not applied to simulator physics.
- Rough-task terrain sampling currently covers only the easiest terrain levels.
- Export/deployment tooling still contains legacy 50D assumptions while the active stack is 48D.
- Optional sim policy logging fails for multi-env runs due to scalar serialization of a vector time buffer.

## Artifacts
- `PLAN.md`

## Experiments
- No training or hardware experiments were run in this session.

## Notes
- Execute the plan phases before resuming new training or hardware-alignment work.
