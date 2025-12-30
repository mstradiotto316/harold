# Harold Next Steps

## PRIORITY 0: Sim-to-Real Alignment (Session 29 Investigation)

**Status**: Investigation complete. **Implementation required on desktop.**

### Background
Hardware deployment testing (Session 28-29) revealed critical sim-to-real gaps causing policy to output extreme values (±60-100) and unstable behavior.

### Root Causes Identified

1. **Linear velocity mismatch**: Hardware IMU uses accelerometer integration (noisy, drifts with 0.95 decay). Simulation uses perfect physics engine velocity.

2. **Observation clipping mismatch**: Deployment clips normalized observations to ±5.0. Training has NO clipping.

3. **Joint limit mismatch**: Training uses symmetric ±30°/±90° limits. Hardware has asymmetric limits (thigh: -55° to +5°, calf: -5° to +80°).

### Implementation Tasks (Execute on Desktop)

1. **Add lin_vel noise to simulation** (`DomainRandomizationCfg`):
   - Gaussian noise: std=0.05 m/s
   - Per-episode bias: std=0.02 m/s

2. **Add observation clipping to simulation**:
   - Clip post-normalization to ±5.0 (or approximate with raw clip)

3. **Align joint limits to hardware**:
   - Shoulders: ±25° (was ±30°)
   - Thighs: -55° to +5° (was ±90°) - **MAJOR CHANGE**
   - Calves: -5° to +80° (was ±90°) - **MAJOR CHANGE**

4. **Retrain and export**:
   ```bash
   HAROLD_CPG=1 HAROLD_CMD_TRACK=1 python scripts/harold.py train \
     --hypothesis "Sim-to-real: lin_vel noise, obs clip, hw joint limits" \
     --tags "sim2real,domain_rand" --iterations 2500
   ```

### Key Files to Modify
- `harold_isaac_lab_env_cfg.py` - Add noise config, update joint limits
- `harold_isaac_lab_env.py` - Implement noise application, clipping

### Full Implementation Plan
See: `.claude/plans/unified-knitting-lagoon.md`

### Deployment Fixes Already Applied (Session 28-29)
These fixes were made to deployment code on the Pi:
1. ✅ Gravity sign flip (Z-up → Z-down convention)
2. ✅ Default commands changed to 0.3 m/s (matches training)
3. ✅ Observation clipping to ±5.0 added
4. ✅ Removed pre-scaling action clip (matches training)
5. ✅ Added 3-cycle CPG warmup before policy enables

### CPG-Only Deployment (Still Works)
```bash
cd /home/pi/Desktop/harold/deployment
python run_cpg_only.py --duration 60
```

---

## PRIORITY 1: Hardware Testing with Backlash-Robust Policy

**Context from Session 28:** Backlash robustness SOLVED! 1° position noise improves walking by 35% (vx=0.023 vs baseline 0.017). Ready for hardware testing.

### Best Checkpoints for Deployment

| Policy | Config | vx | Use Case |
|--------|--------|-----|----------|
| **EXP-148** | Backlash only | 0.023 | Best for initial hardware test |
| EXP-152 | Backlash + yaw (curriculum) | 0.015 | Full controllability |

### Export Command
```bash
python policy/export_policy.py --checkpoint \
  logs/skrl/harold_direct/2025-12-29_02-06-37_ppo_torch/checkpoints/best_agent.pt
```

### Hardware Test Plan
1. Export EXP-148 checkpoint to ONNX
2. Deploy to RPi 5 using `deployment/` pipeline
3. Test forward walking on flat surface
4. Compare motion smoothness vs previous policies
5. Verify backlash robustness transfers to real hardware

---

## ✅ COMPLETED: Backlash Robustness (Session 28)

**Finding**: 1° position noise is OPTIMAL for backlash simulation.

| Noise Level | vx (m/s) | Verdict |
|-------------|----------|---------|
| 0° (baseline) | 0.017 | WALKING |
| **1° (0.0175 rad)** | **0.023** | **WALKING (+35%)** |
| 2° (0.035 rad) | 0.007 | STANDING (too much) |

**Why it works**: Noise prevents policy from overfitting to perfect position observations. Acts as regularization.

Config settings:
```python
# DomainRandomizationCfg
enable_randomization: bool = True
randomize_per_step: bool = True

# joint_position_noise
std = 0.0175  # ~1° in radians
```

---

## ✅ COMPLETED: Yaw Rate Command Tracking (Session 28)

**Implementation**: Added yaw tracking reward + modified yaw_rate_penalty.

| Config | vx (m/s) | Verdict |
|--------|----------|---------|
| Yaw only (no backlash) | 0.011 | WALKING |
| Backlash + yaw (from scratch) | 0.003 | STANDING |
| **Curriculum (backlash→yaw)** | **0.015** | **WALKING** |

**Key insight**: Curriculum learning required for combining features.

---

## PRIORITY 2: Improve Backlash + Yaw Combination

**Finding**: Each works alone, but combining from scratch fails.

### Validated Approach: Curriculum Learning
1. Train with backlash noise (vx+vy tracking) → EXP-148 (vx=0.023)
2. Fine-tune with yaw tracking enabled (~1250 iters) → EXP-152 (vx=0.015)

### Alternative Approaches to Try
- Lower yaw command range (±0.15 instead of ±0.30)
- Lower yaw tracking weight (5 instead of 10)
- Longer backlash pre-training before adding yaw

---

## Session 28 Experiments Summary

| EXP | Config | vx | Verdict | Notes |
|-----|--------|-----|---------|-------|
| 145 | Baseline | 0.017 | WALKING | Pre-backlash reference |
| 146 | 2° backlash | 0.008 | STANDING | Too much noise |
| 147 | 2° backlash | 0.007 | STANDING | Confirmed too much |
| **148** | **1° backlash** | **0.023** | **WALKING** | **OPTIMAL (+35%)** |
| 149 | 1° backlash | 0.023 | WALKING | Reproducible |
| 150 | Backlash+yaw (scratch) | 0.003 | STANDING | Combined fails |
| 151 | Yaw only | 0.011 | WALKING | Yaw works alone |
| **152** | **Curriculum** | **0.015** | **WALKING** | **Combination works!** |
| 153 | Extended curriculum | 0.009 | STANDING | Longer = worse |

---

## Architecture Note: CPG + Residual Learning

The motion is **scripted + learned**:
```
target_joints = CPG_base_trajectory + policy_output * residual_scale
```

- **CPG (scripted)**: Timing, gait coordination, base trajectory
- **Policy (learned)**: Balance, velocity tracking, backlash adaptation
- **residual_scale=0.05**: Policy can only fine-tune, not override

---

## Files Reference

| Purpose | Path |
|---------|------|
| Best backlash policy | `logs/skrl/.../2025-12-29_02-06-37_ppo_torch/checkpoints/best_agent.pt` |
| Curriculum policy | `logs/skrl/.../2025-12-29_08-29-46_ppo_torch/checkpoints/best_agent.pt` |
| Domain randomization config | `harold_isaac_lab_env_cfg.py` - DomainRandomizationCfg |
| Yaw tracking reward | `harold_isaac_lab_env.py` - command_tracking_yaw |
| Session 28 log | `.claude_memory/sessions/2025-12-29_session28_backlash.md` |
