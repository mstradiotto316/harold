# Harold Next Steps

## Current Status (2025-12-26, Session 22 Complete)

### 🎉 MILESTONE: Real Robot Walking Forward!

Session 22 achieved sim-to-real alignment and the real robot now walks forward with scripted gait.

**Key Discoveries:**
1. **Simulation was too stiff** - Real servos have more "give" than stiffness=1200
2. **Thigh phase was inverted** - Changed from +sin to -sin for forward walking
3. **Floor surface matters** - Hardwood vs carpet shows significant differences

---

## PRIORITY 1: Train RL with Aligned Parameters

Now that sim-to-real is aligned, train RL policies with the new settings:

```python
# harold.py actuator settings (current)
stiffness = 400.0    # Matches real servo softness
damping = 40.0       # Proportional reduction
effort_limit = 2.8   # 95% of hardware max
```

**Immediate Experiment:**
```bash
harold train --hypothesis "RL training with stiffness=400 (sim-to-real aligned)" --tags "sim2real,retrain"
```

**Expected Outcome:** RL should learn walking gaits that transfer to real hardware because:
1. Simulation servo response matches real robot
2. Scripted gait proves both sim and real can walk forward
3. Reward structure (backward_penalty=75) is validated

---

## PRIORITY 2: Domain Randomization for Floor Surfaces

Real robot showed significant variation across floor types. Add friction randomization:

```python
# Proposed friction randomization
friction_range = (0.6, 1.2)  # Simulate different floor surfaces
```

This will make trained policies robust to:
- Hardwood floors (lower friction)
- Short carpet (medium friction)
- Long carpet (higher friction, may cause foot dragging)

---

## PRIORITY 3: Improve Foot Clearance

Real robot drags feet during walking. Options:
1. **Increase swing_calf bend** - Lift feet higher during swing phase
2. **Adjust calf trajectory timing** - Ensure foot is fully lifted before thigh moves forward
3. **Add foot clearance reward** - Reward feet being above ground during swing

---

## Current Actuator Configuration

```python
# harold.py (Session 22 aligned settings)
actuators={
    "all_joints": ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        effort_limit_sim=2.8,   # 95% of 2.94 Nm hardware max
        stiffness=400.0,        # Reduced from 1200 to match real servo softness
        damping=40.0,           # Proportional reduction
    ),
},
```

---

## Current Gait Configuration

```python
# ScriptedGaitCfg (Session 22 aligned)
frequency: float = 0.5          # 0.5 Hz (2 second cycle)
swing_thigh: float = 0.40       # Thigh back during swing
stance_thigh: float = 0.90      # Thigh forward during stance
stance_calf: float = -0.90      # Extended during stance
swing_calf: float = -1.40       # Bent during swing (matched to HW 80° limit)
shoulder_amplitude: float = 0.05
duty_cycle: float = 0.6
```

---

## Hardware Script Status

`firmware/scripted_gait_test_1/scripted_gait_test_1.ino`:
- ✅ Parameters aligned with simulation
- ✅ Walking forward (confirmed on real robot)
- ✅ Auto-start (no Enter key required)
- ✅ Slow transitions (prevents servo overload)
- ✅ -sin thigh trajectory (correct walking direction)

---

## Sign Convention Reference

**Simulation → Hardware conversion:**
```
hardware_degrees = -sim_radians × (180/π)
```

| Joint | Sim Convention | Hardware Convention |
|-------|----------------|---------------------|
| Thigh | + = forward | - = forward |
| Calf | - = bent | + = bent |

---

## 5-Metric Validation

| Priority | Metric | Threshold | Best Achieved |
|----------|--------|-----------|---------------|
| 1. SANITY | episode_length | > 100 | 356 (PASS) |
| 2. Stability | upright_mean | > 0.9 | 0.93 (PASS) |
| 3. Height | height_reward | > 1.2 | 1.41 (PASS) |
| 4. Contact | body_contact | > -0.1 | -0.01 (PASS) |
| 5. **Walking** | **vx_w_mean** | **> 0.1** | **0.04** (scripted, soft settings) |

**Goal**: Achieve vx > 0.1 m/s with RL policy that transfers to real hardware.

---

## Experiment Queue

1. **EXP-NEXT**: RL training with stiffness=400
   - Hypothesis: Softer simulation enables transferable walking policy
   - Tags: sim2real, stiffness400

2. **Friction Randomization**: Add floor surface variation
   - Hypothesis: Domain randomization improves real-world robustness
   - Tags: domain_rand, friction

3. **Foot Clearance**: Improve swing phase lift
   - Hypothesis: Higher foot clearance reduces dragging
   - Tags: foot_clearance, gait_tuning

---

## Session 22 Summary

### What We Did
1. Identified that simulation (stiffness=1200) was too stiff vs real robot
2. Reduced stiffness 1200→400, damping 50→40
3. Reduced gait frequency 1.0→0.5 Hz for real servo response
4. Fixed thigh phase (+sin → -sin) to correct walking direction
5. Aligned all parameters between simulation and hardware
6. Confirmed real robot walks forward with scripted gait

### What We Learned
1. **PD stiffness is critical for sim-to-real** - Must match real servo response
2. **Simple sinusoid != complex duty-cycle trajectory** - Phase alignment matters
3. **Floor surface creates significant variation** - Need domain randomization
4. **Feet dragging is common in early gaits** - Need to improve swing phase

### Files Modified
- `harold.py`: stiffness 1200→400, damping 50→40
- `harold_isaac_lab_env_cfg.py`: frequency 1.0→0.5 Hz, swing_calf -1.55→-1.40
- `scripted_gait_test_1.ino`: Parameters aligned, -sin trajectory, auto-start
