# Session 34: Large Amplitude CPG for Backlash Tolerance

**Date**: 2025-12-30
**Machine**: Desktop (training)
**Duration**: ~2 hours

## Objective

Design and test a backlash-tolerant gait following Session 33's discovery that ~30° servo backlash was absorbing the entire calf swing motion (26°).

## Session Summary

### Problem Analysis

Session 33 hardware test revealed:
- Feet never lifted during walking
- Robot shuffled by pushing, not stepping
- Root cause: 30° backlash > 26° calf swing amplitude

### Solution: Large Amplitude CPG

Increased joint amplitudes to exceed backlash zone:

| Joint | Old | New | Amplitude |
|-------|-----|-----|-----------|
| swing_calf | -1.35 | -1.38 | - |
| stance_calf | -0.90 | -0.50 | 50° (was 26°) |
| swing_thigh | 0.40 | 0.25 | - |
| stance_thigh | 0.90 | 0.95 | 40° (was 29°) |

### Training Results (EXP-171)

Trained with new large amplitude configuration:

| Progress | vx (m/s) | Verdict |
|----------|----------|---------|
| 11% | 0.014 | WALKING |
| 31% | 0.010 | STANDING (dip) |
| 51% | 0.016 | WALKING |
| 71% | 0.016 | WALKING |
| 89% | 0.017 | WALKING |
| 99% | 0.019 | WALKING |
| **Final** | **0.020** | **WALKING** |

Final metrics:
- Forward velocity: 0.020 m/s (PASS)
- Height reward: 1.54 (PASS)
- Upright mean: 0.97 (PASS)
- Episode length: 473 (PASS)
- Body contact: -0.00 (PASS)

### Key Observations

1. **No mid-training regression**: Unlike previous experiments, velocity improved throughout training
2. **Simulation doesn't model backlash**: Same vx achieved with old and new trajectories
3. **Hardware test is the validation**: Only real robot will show if backlash is overcome

## Files Changed

1. `harold_isaac_lab_env_cfg.py`:
   - Updated CPGCfg trajectory parameters
   - Updated ScriptedGaitCfg trajectory parameters
   - Added Session 34 comments

2. `deployment/config/cpg.yaml`:
   - Updated trajectory parameters to match simulation
   - Added backlash-tolerance comments

3. `deployment/policy/harold_policy.onnx`:
   - Exported new trained policy

4. Memory files:
   - CONTEXT.md - Updated current state
   - NEXT_STEPS.md - Updated priorities
   - OBSERVATIONS.md - Added Session 34 observations
   - EXPERIMENTS.md - Added EXP-171

## Next Steps (for RPi session)

1. Pull changes: `git pull`
2. Test large amplitude on hardware
3. Observe if feet actually lift now

If still insufficient:
- Implement asymmetric trajectory (fast swing, slow stance)
- Consider even larger amplitude

## Commits

All changes committed and pushed for RPi to pull.
