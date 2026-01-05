# 2026-01-04 Session 43 (desktop)

## Summary
- Removed mixed CPG+policy usage; CPG is open-loop in sim and deployment.
- Removed stride/calf scaling from the CPG config and sim usage (hardware generator math is canonical).
- Standardized observation size at 48D (no gait phase).
- Standardized the shared CPG kernel in `common/cpg_math.py` to prevent drift.
- Added `.gitignore` entry for `deployment/validation/sim_logs/`.
- Removed `scripts/compare_cpg_sim.py` (shared kernel replaces it).
- Untracked `deployment/validation/sim_logs/` artifacts from git.
- Updated docs to reflect open-loop CPG and alignment focus.

## Key Artifacts
- Alignment logs (now gitignored): `deployment/validation/sim_logs/`

## Results
- Command alignment: sim cmd_pos matches hardware generator at logged phase (shared kernel).
- Tracking alignment: sim |cmd-pos| error now matches hardware (calves ~0.078 rad).

## Notes
- Open-loop CPG is for diagnostics; RL training uses pure policy control.
