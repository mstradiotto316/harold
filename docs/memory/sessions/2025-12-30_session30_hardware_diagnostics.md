# Session 30: Hardware Diagnostics & Policy Deployment Fixes

**Date**: 2025-12-30
**Platform**: Raspberry Pi 5 (onboard robot computer)
**Focus**: Diagnosing sluggish gait + fixing policy deployment issues

## Session Summary

User reported robot seems "sluggish and weak" when running the gait routine. Through systematic debugging, we discovered:
1. Power/voltage is NOT the issue (12.1V stable)
2. Servo load was misinterpreted (display bug - divide by 10)
3. Servo speed was too slow (fixed: 1600 → 2800)
4. CPG trajectory has minimal foot lift (~1.5cm)
5. **Critical policy deployment bugs** causing policy to fail

## Key Findings

### 1. Voltage & Load Analysis

**Voltage**: Rock solid at 12.0-12.1V throughout all tests. Power supply is NOT the issue.

**Load Readings Bug**: Original script showed "200% load" but this was a display bug. The servo load register returns -1000 to +1000 (where 1000 = 100%). Fixed by dividing by 10.

**Actual Load**:
- Suspended: 8-18% (barely working)
- Ground @ 0.5Hz: 15-26% (light load)
- Ground @ 2.0Hz: 20-43% (still comfortable)

### 2. Servo Speed Fix

**Problem**: SERVO_SPEED=1600 (~140°/sec) was too slow for servos to track CPG trajectory.

**Fix**: Increased to SERVO_SPEED=2800 (~246°/sec, 91% of max).

**Result**: Feet now lift properly during swing phase instead of dragging.

### 3. CPG Trajectory Analysis

The CPG has only ~1.5cm theoretical foot lift due to:
- swing_calf: -1.40 rad (80.2°) hitting firmware limit of 80°
- Conservative thigh angles

Could improve with more aggressive swing trajectory, but firmware limits collision safety.

### 4. Policy Deployment Bugs (CRITICAL)

Found and fixed **three major bugs** preventing policy from working:

#### Bug 1: Wrong Default Pose in Observation Builder
```python
# WRONG (was using):
default_pose = [0.0, 0.0, 0.0, 0.0,        # Shoulders
                0.3, 0.3, 0.3, 0.3,        # Thighs
                -0.75, -0.75, -0.75, -0.75] # Calves

# CORRECT (simulation uses):
default_pose = [0.20, -0.20, 0.20, -0.20,   # Shoulders: alternating
                0.70, 0.70, 0.70, 0.70,     # Thighs: 0.70 rad
                -1.40, -1.40, -1.40, -1.40] # Calves: -1.40 rad
```

This caused massive observation mismatch - normalized values were hitting ±5.0 clip limits.

#### Bug 2: Wrong Default Pose in Action Converter
Same issue - action converter also had wrong default pose.

#### Bug 3: Wrong Joint Limits in Action Converter
```python
# Limits were in HARDWARE convention:
"thigh": (-55°, +5°)  # Hardware: -0.96 to +0.087 rad

# But CPG outputs RL convention:
thigh_values: 0.40 to 0.90 rad  # All POSITIVE

# Result: Thighs clamped to 0.087 rad → legs nearly straight!
```

**Fix**: Disabled redundant limits in action_converter (firmware already handles limits correctly in hardware convention).

## Files Modified

### Firmware
- `firmware/StreamingControl/HaroldStreamingControl/HaroldStreamingControl.ino`
  - Added voltage telemetry (`voltage_dV` field)
  - Increased SERVO_SPEED: 1600 → 2800

### Deployment (Python)
- `deployment/drivers/esp32_serial.py`
  - Added `voltage_dV` and `voltage_V` to Telemetry dataclass

- `deployment/inference/observation_builder.py`
  - Fixed default_pose to match simulation's athletic_pose

- `deployment/inference/action_converter.py`
  - Fixed default_pose to match simulation's athletic_pose
  - Disabled wrong joint limits (firmware handles this)

- `deployment/run_cpg_only.py`
  - Added voltage/load logging during gait

## Files Created

- `scripts/voltage_monitor.py` - Real-time voltage and load monitoring tool

## Test Results

### CPG-Only (No Policy)
- 0.5 Hz: Works, feet lift properly with SERVO_SPEED=2800
- 1.0 Hz: Works, more dynamic motion
- 1.5-2.0 Hz: Works, servos have plenty of headroom

### Policy + CPG
After fixes, policy now runs without causing legs to straighten or freeze. Shoulder positions follow CPG pattern with small policy corrections.

## Remaining Issues

1. **CPG trajectory hitch**: Calf has velocity discontinuity at stance/swing transition (phase 0.60). Could smooth this with better trajectory math.

2. **Policy behavior**: Need more testing to verify policy is actually helping vs just not hurting.

## Next Steps

1. Test policy on ground (not just suspended)
2. Compare CPG-only vs CPG+policy walking performance
3. Consider smoothing CPG calf trajectory
4. Consider training new policy with corrected deployment code

## Key Learnings

1. **Always match default poses** between simulation and deployment
2. **Joint limits must use correct convention** (RL vs hardware)
3. **Servo speed matters** for trajectory tracking
4. **Voltage monitoring** confirmed power supply is adequate
5. **Load readings need proper scaling** (÷10 for percentage)
