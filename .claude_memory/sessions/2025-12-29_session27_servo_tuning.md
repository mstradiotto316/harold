# Session 27: Servo Tuning & Hardware Diagnostics

**Date**: 2025-12-29
**Duration**: ~2 hours
**Platform**: Raspberry Pi 5 (2GB RAM)

## Summary

This session focused on diagnosing servo performance issues, validating the Pi 5 power supply, and understanding hardware limitations for sim-to-real transfer.

## Pi 5 Stress Test Results

Ran comprehensive stress tests to validate onboard power supply:

| Test | Duration | Peak Temp | Throttling | Status |
|------|----------|-----------|------------|--------|
| CPU-only (4 cores) | 30s | 71.4°C | 0x0 | PASS |
| CPU+Memory+I/O | 30s | 75.2°C | 0x0 | PASS |
| Robot Controller Sim | 30s | 60.9°C | 0x0 | PASS |

**Conclusion**: Power supply is healthy. No undervoltage detected under any load.

## Servo Dead Zone Investigation

### Initial Diagnosis
- **Symptom**: Can rotate servo a bit before it "locks up and fights back"
- Dead zone registers (26-27) were already at minimum value (1 step = 0.088°)
- This is NOT a software dead zone issue

### Root Cause: Gear Backlash
The "dead zone" is actually **mechanical gear backlash** in the ST3215 servo gearbox:
- Servos have fine positioning capability when commanded
- But there's mechanical play (~1-3°) in the gear train
- Within this play zone, the servo doesn't resist movement
- Beyond the play zone, full servo resistance kicks in

### P Coefficient Experiments
Discovered undocumented register 21 (P coefficient) controls servo stiffness:

| P Value | Result |
|---------|--------|
| 32 (factory) | Normal response, noticeable play |
| 100 | Slightly stiffer |
| 200 | **Rapid oscillations** - unstable! |

**Conclusion**: Increasing P without matching D (damping) causes instability. Reverted to factory P=32.

### Key Finding
The gear backlash is a **hardware limitation** that cannot be fixed in software. The correct approach is to **model it in simulation** so the RL policy learns to be robust to it.

## Artifacts Created/Updated

### ServoTuning.ino (Enhanced)
`firmware/CalibrationAndSetup/ServoTuning/ServoTuning.ino`

New commands added:
- `x <id> <reg>` - Read raw register value
- `X <id>` - Dump all registers (0-70) for servo
- `W <id> <reg> <val>` - Write raw register value

### Full Register Dump (Servo 1)
```
00:   3   9   0   9   3   1   0   0   1   0
10:   0 255  15  70  80  40 232   3  12  44
20:  47  32  32   0  16   0   1   1 244   1
30:   1 183  15   0  20 200  80  10 200 200
40:   1  50 255   7   0   0 220   5 232   3
50:   0   0   0   0   0   1   1   8   0   0
60:  24   0 121  32   0   1   0 255   7   1
70:   0
```

Key undocumented registers:
- **Reg 21**: P coefficient (factory=32)
- **Reg 22**: D coefficient (factory=32)
- **Reg 23**: I coefficient (factory=0)
- **Reg 24**: Punch/min PWM (factory=16)

## Committed Changes

```
1a01974 Add collision-safe limits, 100% torque, and servo tuning utility
```

Changes included:
- `TORQUE_LIMIT = 1000` (100% max torque)
- `initServoTorque()` function
- Asymmetric collision-safe joint limits
- Watchdog timeout 250ms
- ServoTuning.ino utility with raw register access

## Critical Next Step: Model Backlash in Simulation

**Priority experiment for overnight run on desktop:**

The gear backlash creates a "friction-free zone" where small position errors don't generate corrective torque. To model this in Isaac Lab:

### Approach Options

1. **Observation noise** (simplest)
   - Increase joint position noise from 0.005 rad to 0.02-0.05 rad
   - Policy learns to handle position uncertainty

2. **Action noise**
   - Add noise to commanded positions
   - Simulates imprecise positioning

3. **Lower stiffness**
   - Reduce actuator stiffness to make servo "softer"
   - Not physically accurate but has similar effect

4. **Custom backlash model** (most accurate)
   - Implement position hysteresis in environment
   - Small errors don't generate torque until threshold exceeded

### Recommended Experiment

Start with increased observation noise (option 1) as it's simplest and already implemented:

```python
# In harold_isaac_lab_env_cfg.py - DomainRandomizationCfg
joint_position_noise: GaussianNoiseCfg = GaussianNoiseCfg(
    mean=0.0,
    std=0.03,  # Increased from 0.005 to ~1.7° to match backlash
)
```

Enable domain randomization after establishing baseline walking.

## Hardware Constraints Summary

| Parameter | Value | Notes |
|-----------|-------|-------|
| Gear backlash | ~1-3° | Mechanical, unfixable |
| Dead zone registers | 1 (minimum) | Already optimized |
| P coefficient | 32 (factory) | Higher causes oscillation |
| D coefficient | 32 (factory) | Needs to match P |
| Torque limit | 1000 (100%) | Set at boot |
