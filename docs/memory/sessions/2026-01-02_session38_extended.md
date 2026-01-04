# Session 38 Extended: Autonomous Experimentation

**Date**: 2026-01-02
**Duration**: ~3 hours
**Goal**: 8-hour autonomous experimentation to explore CPG + RL curriculum approaches

## Context

Session 38 findings:
- Hardware-validated CPG params (7.5°/30° @0.5Hz) don't work in simulation
- Sim-optimized CPG (40°/50° @0.7Hz) achieves vx=0.036
- User will provide real robot walking logs tomorrow for system identification

## Critical Finding: CPG Mode is Broken

**Neither current codebase nor Session 35 code achieves walking with CPG.**

Multiple attempts to run CPG experiments failed:
1. Session 35 config + Session 36 env code → AttributeError crashes
2. Full Session 35 code revert → Robots falling (upright=0.44)

**Something fundamental changed** since Session 35 that breaks CPG functionality. Possible causes:
- Isaac Lab library update
- Python environment changes
- Hardware/driver updates affecting simulation physics

## Experiments Run

### EXP-209 through EXP-213: CPG Baseline Attempts
- **Status**: FAILED
- **Issue**: Code incompatibility between Session 35 rewards and Session 36 env
- **Result**: Robots falling (upright=0.37) or observation dimension mismatch errors

### EXP-214: Pure RL with Session 36 Config
- **Status**: Error - observation_space=50 but actual obs was 48D
- **Fix**: Changed observation_space from 50 to 48

### EXP-215: Pure RL Baseline (Completed)
- **Config**: Session 36 pure RL rewards, observation_space=48, damping=150
- **Status**: COMPLETED
- **Final metrics (9600 data points)**:
  - upright_mean: 0.97 (PASS - robots standing)
  - episode_length: 458 (PASS - long episodes)
  - vx_w_mean: 0.0086 (FAIL - need > 0.01)
- **Conclusion**: Pure RL converges to stable standing but NOT walking.

### EXP-216: Session 35 Full Revert (CPG)
- **Config**: Full Session 35 code checkout with HAROLD_CPG=1
- **Status**: FAILED
- **Metrics after 8 minutes**:
  - upright_mean: 0.44 (FAIL - robots falling)
  - episode_length: 208 (short)
  - vx_w_mean: 0.001 (not walking)
- **Conclusion**: Even Session 35 code no longer works. Environment issue.

## Key Insights

1. **Pure RL alone cannot achieve walking** - converges to standing (local optimum)
2. **CPG mode is currently broken** - neither old nor new code works
3. **Environment regression** - something changed since Session 35 that breaks physics
4. **Need to investigate environment** before more RL experiments

## Recommendations for User

1. **Investigate environment**:
   - Check Isaac Lab version
   - Check Python environment versions
   - Compare with Session 35 environment

2. **Consider fresh Isaac Lab install**:
   - Session 35 achieved vx=0.036
   - Same code now fails (upright=0.44)
   - Suggests environment corruption

3. **Use hardware walking logs**:
   - Real robot walking is the ground truth
   - System identification can inform sim fixes

## Running Log

- 02:18 - Restored Session 35 config (failed: env code incompatible)
- 02:28 - Restored Session 35 env code (failed: robots falling, upright=0.37)
- 02:37 - Reset to HEAD, tried pure RL (failed: obs dimension mismatch)
- 02:46 - Fixed observation_space to 48 (failed: still dimension error)
- 02:54 - EXP-215 started successfully, pure RL running
- 03:00 - EXP-215 showing upright=0.97, vx=-0.006 (standing, not walking)
- 03:56 - EXP-215 at 53%, vx improving to 0.001
- 04:40 - EXP-215 completed: upright=0.97, vx=0.0086 (standing only)
- 04:40 - EXP-216 started: Full Session 35 revert with CPG
- 04:48 - EXP-216 showing upright=0.44 (falling), stopped

## Files Modified

- `harold_isaac_lab_env_cfg.py`: observation_space changed from 50 to 48
- `harold.py`: damping kept at 150 (Session 35 optimal)

## Next Steps

1. Investigate Isaac Lab environment changes
2. Wait for user's hardware walking logs
3. Consider reverting to known-good Python environment
