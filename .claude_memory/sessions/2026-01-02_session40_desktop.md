# Session 40: Scripted Gait Alignment + Pipeline Cleanup (Desktop)

**Date**: 2026-01-02
**Machine**: Desktop (training)
**Duration**: ~2.5 hours (interactive loops)

## Goals

1. Run scripted gait in sim using Session 36 parameters with video recording.
2. Test higher actuator stiffness for responsiveness.
3. Clean up experimentation pipeline/documentation to avoid future confusion.

## Summary

- Scripted gait still fails to generate forward motion in sim even with higher stiffness.
- Stiffness=1200 improves upright and episode length, but vx remains ~0 or negative.
- `height_reward` and `body_contact_penalty` metrics are missing in scripted-gait runs.
- Fixed `harold.py` to wait for a new run directory before registering aliases.
- Updated docs to standardize defaults and remove references to non-existent helper scripts.

## Experiments

### EXP-220: Scripted Gait Baseline (Session 36 params)
- **Run**: `2026-01-02_21-15-16_ppo_torch`
- **Config**: HAROLD_SCRIPTED_GAIT=1 (default actuators)
- **Result**: ep_len=179 (FAIL), upright=0.466 (FAIL), vx=-0.007 (FAIL)
- **Video**: `logs/skrl/harold_direct/2026-01-02_21-15-16_ppo_torch/videos/train`

### EXP-222: Scripted Gait + High Stiffness (1200/50)
- **Run**: `2026-01-02_21-40-31_ppo_torch`
- **Config**: stiffness=1200, damping=50, effort=2.8
- **Result**: ep_len=482 (PASS), upright=0.975 (PASS), vx=-0.001 (FAIL)
- **Video**: `logs/skrl/harold_direct/2026-01-02_21-40-31_ppo_torch/videos/train`

### EXP-223: Scripted Gait + High Stiffness/High Damping (1200/150)
- **Run**: `2026-01-02_21-58-56_ppo_torch`
- **Config**: stiffness=1200, damping=150, effort=2.8
- **Result**: ep_len=469 (PASS), upright=0.981 (PASS), vx=-0.003 (FAIL)
- **Video**: `logs/skrl/harold_direct/2026-01-02_21-58-56_ppo_torch/videos/train`

### EXP-224: Scripted Gait + Metrics Fix (1200/50)
- **Run**: `2026-01-02_23-12-56_ppo_torch`
- **Config**: stiffness=1200, damping=50, effort=2.8
- **Result**: ep_len=485 (PASS), upright=0.975 (PASS), height=0.670 (PASS), contact=0.000 (PASS), vx=-0.002 (FAIL)
- **Video**: `logs/skrl/harold_direct/2026-01-02_23-12-56_ppo_torch/videos/train`

### EXP-225: Scripted Gait + Amplitude Scale 1.5x (1200/50)
- **Run**: `2026-01-02_23-31-31_ppo_torch`
- **Config**: HAROLD_GAIT_AMP_SCALE=1.5, stiffness=1200, damping=50
- **Result**: ep_len=489 (PASS), upright=0.973 (PASS), height=0.671 (PASS), contact=0.000 (PASS), vx=0.002 (FAIL)
- **Video**: `logs/skrl/harold_direct/2026-01-02_23-31-31_ppo_torch/videos/train`

### EXP-226: Scripted Gait + Amplitude Scale 2.0x (1200/50)
- **Run**: `2026-01-02_23-47-27_ppo_torch`
- **Config**: HAROLD_GAIT_AMP_SCALE=2.0, stiffness=1200, damping=50
- **Result**: ep_len=482 (PASS), upright=0.972 (PASS), height=0.669 (PASS), contact=0.000 (PASS), vx=0.002 (FAIL)
- **Video**: `logs/skrl/harold_direct/2026-01-02_23-47-27_ppo_torch/videos/train`

## Pipeline & Documentation Changes

- `scripts/harold.py`: default `num_envs` now 8192; run registration waits for new run dir.
- `scripts/harold.py`: added `--task`, `--mode`, and `--duration` presets; status now shows config + x displacement; run listing uses robust dir pattern.
- `harold_rough/harold.py` + `harold_pushup/harold.py`: actuator env overrides added, defaults unified (effort=2.8, stiffness=400, damping=150).
- `CLAUDE.md` + `AGENTS.md`: updated defaults, added autonomous loop guidance, removed outdated helper-script references.
- `.claude_memory/OBSERVATIONS.md`: updated context-management guidance to use `harold.py`.
- `harold_flat/harold_isaac_lab_env.py`: added height/contact metrics + gait amplitude scaling env var; metrics now logged.

## Open Issues

1. Scripted gait mismatch persists even with stiffer actuators.
2. Height reward remains ~0.67 even when upright is ~0.98 (likely offset/height mismatch).

## Observability Update

- Added per-foot contact ratio/peak force, air time mean/std, slip speed, and total x displacement metrics to TensorBoard logs.

## Next Steps

1. Investigate joint offset/height calibration mismatch vs hardware.
2. Consider additional stiffness tests (600/45, 800/60) or larger gait scaling if safe.
3. Review video outputs for gait quality and compare to hardware Session 36.

## Shutdown Status

- No training processes running.
- Safe to shut down.
