# Harold Next Steps

## PRIORITY 0: Sim-to-Real Alignment (Session 29)

**Status**: Implementation in progress. Testing phase.

### Implementation Status

| Change | Status | Experiment |
|--------|--------|------------|
| Lin vel noise (std=0.05 m/s) | **IMPLEMENTED** | EXP-158 |
| Lin vel per-episode bias (std=0.02) | **IMPLEMENTED** | EXP-158 |
| Observation clipping (raw ±50) | **IMPLEMENTED** | EXP-158 |
| Shoulder limits (±30° → ±25°) | PENDING | EXP-159 |
| Thigh limits (±90° → -55°/+5°) | PENDING | EXP-160 |
| Calf limits (±90° → -5°/+80°) | PENDING | EXP-161 |

### Deployment Fixes Already Applied (Session 28-29)
1. ✅ Gravity sign flip (Z-up → Z-down convention)
2. ✅ Default commands changed to 0.3 m/s
3. ✅ Observation clipping to ±5.0 added
4. ✅ Removed pre-scaling action clip
5. ✅ Added 3-cycle CPG warmup before policy enables

---

## Experiment Queue (Run in Order)

### EXP-158: Lin Vel Noise + Obs Clipping (CURRENT)

**Hypothesis**: Adding lin_vel noise and obs clipping improves sim-to-real transfer

**Changes**:
- `add_lin_vel_noise: bool = True` (std=0.05 m/s + bias std=0.02)
- `clip_observations: bool = True` (raw ±50, approx normalized ±5)

**Command**:
```bash
HAROLD_CPG=1 HAROLD_CMD_TRACK=1 HAROLD_DYN_CMD=1 python scripts/harold.py train \
  --hypothesis "Sim-to-real: lin_vel noise + obs clipping" \
  --tags "sim2real" --iterations 1250
```

**Success criteria**: vx > 0.015 (WALKING)

---

### EXP-159: Add Shoulder Limit Alignment

**Hypothesis**: Tighter shoulder limits (±30° → ±25°) won't significantly affect gait

**Changes** (add to EXP-158):
- Shoulder limits: ±0.5236 rad → ±0.4363 rad

This is a minor 5° reduction, should have minimal impact.

---

### EXP-160: Add Thigh Limit Alignment (MAJOR)

**Hypothesis**: Hardware thigh limits require gait adaptation

**Changes** (add to EXP-159):
- Thigh limits: ±1.5708 rad → -0.9599 to +0.0873 rad

**WARNING**: This is a MAJOR change from 180° range to just 60°!
- The CPG gait may need retuning
- Policy will need to learn to walk with restricted thigh motion

---

### EXP-161: Add Calf Limit Alignment (MAJOR)

**Hypothesis**: Hardware calf limits require gait adaptation

**Changes** (add to EXP-160):
- Calf limits: ±1.5708 rad → -0.0873 to +1.3963 rad

**WARNING**: This is a MAJOR change from 180° range to just 85°!

---

## After Experiment Queue

1. **Export best policy to ONNX**
2. **Deploy to RPi 5**
3. **Validate sim-to-real transfer**

---

## ✅ Session 29 Findings: Domain Randomization Tests

| EXP | Config | vx | Verdict | Notes |
|-----|--------|-----|---------|-------|
| 154 | Action noise 0.5% | 0.007 | STANDING | Action noise hurts training |
| 155 | Action noise 0.2% | 0.009 | STANDING | Still hurts |
| 156 | Action delay 0-1 steps | 0.008 | STANDING | Action delays hurt too |
| 157 | External forces 1% | - | (interrupted) | Paused for sim-to-real focus |

**Key Finding**: Action-side randomization (noise/delays) hurts training when observation noise is already present. The combination is too much uncertainty.

---

## ✅ COMPLETED: Backlash Robustness (Session 28)

**Finding**: 1° position noise is OPTIMAL for backlash simulation.

| Noise Level | vx (m/s) | Verdict |
|-------------|----------|---------|
| 0° (baseline) | 0.017 | WALKING |
| **1° (0.0175 rad)** | **0.023** | **WALKING (+35%)** |
| 2° (0.035 rad) | 0.007 | STANDING (too much) |

Config:
```python
add_joint_noise: bool = True
joint_position_noise.std = 0.0175  # ~1° in radians
```

---

## Architecture: CPG + Residual Learning

```
target_joints = CPG_base_trajectory + policy_output * residual_scale
```

- **CPG (scripted)**: Timing, gait coordination, base trajectory
- **Policy (learned)**: Balance, velocity tracking, adaptation
- **residual_scale=0.05**: Policy can only fine-tune, not override

---

## Files Reference

| Purpose | Path |
|---------|------|
| Training config | `harold_isaac_lab/.../harold_flat/harold_isaac_lab_env_cfg.py` |
| Training env | `harold_isaac_lab/.../harold_flat/harold_isaac_lab_env.py` |
| Best backlash policy | `logs/skrl/.../2025-12-29_02-06-37_ppo_torch/checkpoints/best_agent.pt` |
