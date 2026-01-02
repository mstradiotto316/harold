# Harold Next Steps

## PRIORITY 0: Use Hardware-Validated Gait Parameters as RL Reference

**Status**: Session 36 RPi achieved smooth controlled walking with scripted gait. These parameters define what "good" looks like for RL training.

---

## CRITICAL: Hardware-Validated Gait Parameters (Session 36 RPi)

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

## Recommendations for RL Experimentation Agent

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

## Pure RL Status (Desktop Session 36)

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

## Current Configuration Files

The working scripted gait is in:
- `firmware/scripted_gait_test_1/scripted_gait_test_1.ino`

Key RL config files:
- `harold_isaac_lab_env_cfg.py` - rewards, termination
- `agents/skrl_ppo_cfg.yaml` - PPO hyperparameters
- `harold.py` - robot asset, actuators

---

## Session 36 Summary

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

### Files Modified

1. `firmware/scripted_gait_test_1/scripted_gait_test_1.ino` - working gait
2. `.claude_memory/OBSERVATIONS.md` - detailed findings
3. `.claude_memory/NEXT_STEPS.md` - this file
