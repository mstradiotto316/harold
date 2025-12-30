# Harold Next Steps

## PRIORITY 0: Hardware Deployment Testing (Session 30)

**Status**: Major deployment bugs FIXED. Ready for real walking tests.

### Session 30 Fixes (CRITICAL)

| Fix | Issue | Status |
|-----|-------|--------|
| Servo speed | 1600 → 2800 (feet were dragging) | ✅ FIXED |
| Default pose (obs) | Wrong pose caused ±5.0 clipping | ✅ FIXED |
| Default pose (action) | Same issue in action converter | ✅ FIXED |
| Joint limits | Hardware limits applied to RL values | ✅ FIXED |
| Voltage telemetry | Added monitoring capability | ✅ DONE |

### Hardware Test Results

| Test | Voltage | Load | Result |
|------|---------|------|--------|
| CPG @ 0.5Hz (ground) | 12.0-12.1V | 15-26% | ✅ Feet lift properly |
| CPG @ 1.0Hz (ground) | 11.8-12.1V | 20-32% | ✅ Works well |
| CPG @ 2.0Hz (ground) | 11.9-12.1V | 20-43% | ✅ Plenty of headroom |
| Policy + CPG | 12.0-12.1V | - | ✅ No longer freezes/straightens |

### Next Hardware Tests

1. **Test CPG+Policy on ground** - Compare walking quality vs CPG-only
2. **Try faster CPG frequency** - 1.0 Hz instead of 0.5 Hz
3. **Measure forward velocity** - Does robot actually move forward?

---

## Deployment Fixes Applied (Complete List)

### From Session 30
1. ✅ **Servo speed**: 1600 → 2800 (feet now lift properly)
2. ✅ **Observation default pose**: Match simulation's athletic_pose
3. ✅ **Action default pose**: Match simulation's athletic_pose
4. ✅ **Joint limits**: Disabled wrong limits (firmware handles correctly)
5. ✅ **Voltage monitoring**: Added to firmware and Python driver

### From Session 28-29
1. ✅ Gravity sign flip (Z-up → Z-down convention)
2. ✅ Default commands changed to 0.3 m/s
3. ✅ Observation clipping to ±5.0 added
4. ✅ Removed pre-scaling action clip
5. ✅ Added 3-cycle CPG warmup before policy enables

---

## Correct Default Pose (IMPORTANT)

Both observation_builder.py and action_converter.py MUST use:

```python
# Simulation's athletic_pose (NOT the old wrong values!)
default_pose = [
    0.20, -0.20, 0.20, -0.20,   # Shoulders: alternating
    0.70, 0.70, 0.70, 0.70,     # Thighs: 0.70 rad
    -1.40, -1.40, -1.40, -1.40  # Calves: -1.40 rad
]
```

---

## Simulation Experiment Queue

### EXP-158: Lin Vel Noise + Obs Clipping

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

---

## Known Issues

1. **CPG trajectory hitch**: Calf has velocity discontinuity at stance/swing transition (phase 0.60). Visible as slight jerk in motion.

2. **Minimal foot clearance**: CPG only provides ~1.5cm foot lift. Works but marginal.

---

## Files Reference

| Purpose | Path |
|---------|------|
| Firmware | `firmware/StreamingControl/HaroldStreamingControl/HaroldStreamingControl.ino` |
| Observation builder | `deployment/inference/observation_builder.py` |
| Action converter | `deployment/inference/action_converter.py` |
| CPG generator | `deployment/inference/cpg_generator.py` |
| Policy controller | `deployment/inference/harold_controller.py` |
| Voltage monitor | `scripts/voltage_monitor.py` |
