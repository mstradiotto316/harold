# Harold Next Steps

## 2026-01-04: Hardware CPG (Rear Legs Still Skimming)
- Increase rear-only lift/stride: try `calf_lift_scale_back=1.25` and/or `stride_scale_back=1.4` while keeping front scales unchanged.
- Optional: add a small rear-only thigh offset or shoulder bias to unload rear legs during swing.
- Keep `SERVO_SPEED`/`SERVO_ACC` unchanged.
- Keep `harold` service stopped during manual tests; it restarts gait if active.
- If ESP32 handshake fails, reflash StreamingControl on the Pi before testing.

## Session 39 Complete: Hardware Telemetry Logging

**NEW**: The robot now logs detailed telemetry at 5 Hz during hardware runs:
- **Location**: `deployment/sessions/session_YYYY-MM-DD_HH-MM-SS.csv`
- **Data**: 12 joint positions/loads/currents/temps, bus voltage, RPi metrics
- **Documentation**: See "Hardware Session Logs" section in CLAUDE.md

Use these logs to diagnose hardware issues (backlash, thermal, power problems).

---

## PRIORITY 0: Deploy Session 35 Policy with Amplitude Scaling

**Status**: Session 38 found that hardware-validated params don't work in simulation. Use Session 35 policy (vx=0.036) with amplitude scaling at deployment.

---

## NEW: Scripted Gait Alignment Follow-ups

1. **Fix missing metrics in scripted gait runs**
   - `height_reward` and `body_contact_penalty` are not logged during scripted-gait runs.
   - Confirm whether tags are emitted in scripted mode or need explicit logging.

2. **Decide next stiffness tests**
   - 1200/50 and 1200/150 improved upright but not vx.
   - Consider testing 600/45 or 800/60 for responsiveness vs stability balance.

3. **Check sim↔hardware mismatch for Session 36 gait**
   - If metrics are still failing after stiffness tuning, re-check gait amplitude scaling in sim.

4. **Investigate low height in scripted gait**
   - Height reward stays ~0.67; likely joint offset/pose mismatch.
   - Compare sim stance height vs hardware logs at same joint angles.

5. **Validate axis/phase alignment**
   - Confirm sim joint sign conventions vs firmware (thigh/calf inversion).
   - Ensure diagonal pairing matches hardware (FL+BR, FR+BL).

6. **Use new gait metrics to match video observations**
   - Run a short scripted-gait diagnostic and inspect per-foot contact/air-time/slip metrics.

---

## Session 38 Finding: Sim ≠ Hardware Parameters

Hardware-validated parameters are too conservative for simulation:
| Configuration | vx (m/s) | Result |
|---------------|----------|--------|
| Hardware params (7.5°/30° @0.5Hz) | 0.003 | STANDING |
| Intermediate (15°/40° @0.5Hz) | 0.004 | STANDING |
| Session 35 (40°/50° @0.7Hz) | 0.036 | **WALKING** |

**Solution**: Train in sim with original params, apply amplitude scaling at deployment.

**Update (2026-01-03)**: Measured servo backlash dead zone is ~10° (previous 30° estimate was incorrect). Revisit any gait amplitude or backlash assumptions accordingly.

---

## REFERENCE: Hardware-Validated Gait Parameters (Session 36 RPi)

### What Works on Real Hardware

Through iterative hardware testing, we found gait parameters that produce smooth, controlled walking:

| Parameter | Hardware Value | Sim Equivalent | Notes |
|-----------|---------------|----------------|-------|
| **Frequency** | 0.5 Hz | 0.5 Hz | 2 second cycle, slower = stable |
| **Thigh range** | 7.5° | ~0.13 rad | Small forward/back swing |
| **Calf range** | 30° (50°-80°) | ~0.52 rad | Must maintain foot lift |
| **Thigh:Calf ratio** | 1:4 | 1:4 | Balanced joint motion |
| **Servo ACC** | 150 (max) | N/A | Instant acceleration |

### Converting to Simulation (Radians)

```python
# Hardware degrees → Simulation radians (with sign inversion)
# sim_rad = -hardware_deg * (π/180)

# REFERENCE VALUES for RL reward shaping:
THIGH_AMPLITUDE_RAD = 0.13    # ±0.065 rad from center (was 0.35 rad - too big!)
CALF_AMPLITUDE_RAD = 0.52     # ±0.26 rad from center
GAIT_FREQUENCY_HZ = 0.5       # 2 second cycle
```

### Key Insight: Original Sim Parameters Were WAY Too Aggressive

| Parameter | Original Sim | Working Hardware | Reduction Factor |
|-----------|--------------|------------------|------------------|
| Thigh amplitude | ~0.35 rad (40°) | 0.13 rad (7.5°) | **5.3x smaller** |
| Frequency | 0.7-1.0 Hz | 0.5 Hz | **1.4-2x slower** |

**The RL agent should aim for policies with SMALL joint motions, not large sweeping gaits.**

---

## Session 37 Findings: Explicit Backlash Hysteresis FAILED

| Approach | Result |
|----------|--------|
| Pure RL + 30° hysteresis | vx=-0.016 (BACKWARD) |
| Pure RL + 15° hysteresis | vx=-0.005 (oscillating) |
| CPG + 15° hysteresis | vx=+0.002 (slight forward) |
| CPG baseline (no noise) | vx=+0.006 (below target) |
| CPG + joint noise (1°) | vx=+0.005 (no improvement) |

**Key Issue**: Session 37 CPG experiments (vx≈0.005) underperform Session 35 (vx=0.036).

The hysteresis model in `_apply_backlash_hysteresis()` is implemented but DISABLED because:
1. Policy can't learn to compensate from scratch
2. Gaussian noise works better as regularization
3. Need curriculum to introduce backlash gradually

Keep the code for future curriculum learning experiments.

---

## Current Configuration

```python
# Observation space
observation_space = 50  # For CPG mode (48 + 2 gait phase)

# Backlash (DISABLED)
BacklashCfg.enable_backlash = False

# Joint noise (ENABLED)
add_joint_noise = True
joint_position_noise.std = 0.0175  # 1°

# Rewards
track_lin_vel_xy_weight = 5.0
forward_motion_weight = 3.0
lin_vel_z_weight = -0.0001
```

### Training Command
```bash
# CPG mode (recommended)
python scripts/harold.py train --mode cpg --duration standard \
  --hypothesis "description"
```

---

## Recommendations for RL Experimentation

### 1. Add Joint Amplitude Penalties

Penalize large joint angle deviations to encourage small, controlled motions:

```python
# In rewards section of harold_isaac_lab_env_cfg.py
joint_amplitude_penalty = -0.1 * torch.mean(torch.abs(joint_pos - default_pos))
```

Target: Keep thigh motion under ±0.1 rad, calf under ±0.3 rad

### 2. Consider Gait Frequency Reward

If using CPG, target 0.5 Hz instead of 0.7 Hz:

```python
# In CPG config
gait_frequency = 0.5  # was 0.7
```

### 3. Reference Motion Tracking (Optional)

Use the working hardware trajectory as a reference:

```python
# Thigh target trajectory
thigh_ref = -34.4° + 3.75° * sin(2π * 0.5 * t)  # ±3.75° = 7.5° range

# Calf target trajectory
calf_ref = 65° - 15° * cos(2π * 0.5 * t)  # 50°-80° range
```

### 4. What to Avoid

- **Large thigh amplitudes** (>15°) → causes lunging
- **Fast frequencies** (>0.7 Hz) → less stable
- **Small calf ranges** (<30°) → feet don't lift (backlash absorbs motion)

---

## Pure RL Status

Pure RL from scratch plateaus at vx ≈ 0.01 m/s. The policy finds a standing local minimum.

### Options to Break the Plateau

1. **Fine-tune from CPG checkpoint** (recommended)
   - CPG policy already walks (vx=0.034)
   - Challenge: Observation space mismatch (50D CPG vs 48D pure RL)

2. **Curriculum learning**
   - Start with very slow velocity commands
   - Gradually increase target velocity

3. **Use hardware-validated parameters as constraints**
   - Add penalties for exceeding known-good joint ranges
   - Target the 0.5 Hz gait frequency

4. **Reference motion tracking**
   - Use scripted gait trajectory as soft target
   - Reward matching the known-working motion

---

## Current Best Policies

| Session | Configuration | vx (m/s) | Status |
|---------|---------------|----------|--------|
| **Session 35** | CPG, damping=150 | 0.036 | **BEST - use for hardware** |
| Session 37 | CPG + hysteresis | 0.005 | NOT ready |

**Recommendation**: Use Session 35 checkpoint for hardware deployment.

---

## Key Files

The working scripted gait is in:
- `firmware/scripted_gait_test_1/scripted_gait_test_1.ino`

Key RL config files:
- `harold_isaac_lab_env_cfg.py` - rewards, termination
- `agents/skrl_ppo_cfg.yaml` - PPO hyperparameters
- `harold.py` - robot asset, actuators

---

## Session 36 Summary (Hardware)

### What Was Accomplished

1. **Fixed servo acceleration bug** - ACC=0→150 for instant response
2. **Iteratively tuned gait parameters** through hardware testing
3. **Achieved smooth walking** on carpet and hardwood
4. **Documented what "good" looks like** for RL reference

### Key Numbers to Remember

- **Thigh: 7.5° range** (not 40°!)
- **Calf: 30° range** (50° to 80°)
- **Frequency: 0.5 Hz** (not 0.7 or 1.0)
- **ACC: 150** (max, for backlash)
