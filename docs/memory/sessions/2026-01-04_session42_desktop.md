# 2026-01-04 Session 42 (desktop)

## Summary
- Consolidated Pi repo/runtime workflow and validated clean git sync.
- Ran three CPG-only hardware tests; Test 3 on low-friction surface is now baseline.
- Pulled hardware session CSVs to desktop for sim-to-real comparison.
- Updated docs to standardize system `python3` on the Pi and correct runtime/log paths.

## Hardware Tests (CPG-only, 10s)
- Test 1 (0.5/0.6): `/home/pi/harold/deployment/sessions/session_2026-01-04_18-40-39.csv`
- Test 2 (0.4/0.5 + scaling): `/home/pi/harold/deployment/sessions/session_2026-01-04_18-41-11.csv`
- Test 3 (low-friction, best): `/home/pi/harold/deployment/sessions/session_2026-01-04_18-43-41.csv`

## Local Artifacts
- `logs/hardware_sessions/session_2026-01-04_18-40-39.csv`
- `logs/hardware_sessions/session_2026-01-04_18-41-11.csv`
- `logs/hardware_sessions/session_2026-01-04_18-43-41.csv`

## Decisions
- Baseline for sim-to-real alignment: Test 3 (0.4 Hz, duty 0.5, stride/lift scaling + rear boost).

## Next Steps
- Follow `docs/memory/NEXT_STEPS.md` for scripted/CPG parity checks and sim-vs-hardware command/position comparisons.
