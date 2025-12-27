# Hardware Constraints for Sim-to-Real Transfer

## Purpose

This document provides **REAL-WORLD CONSTRAINTS** that experimentation agents MUST consider when proposing configuration changes. Simulation parameters should remain within these bounds for successful hardware deployment.

**Read this before modifying**: `effort_limit`, `stiffness`, `damping`, joint limits, or control rates.

---

## Servo Specifications (FeeTech ST3215)

Source: `firmware/ST3215 Servo.pdf`, `firmware/docs/Communication_Protocol_User_Manual_EN.pdf`

### Torque & Power

| Parameter | Real Hardware | Simulation | Status |
|-----------|--------------|------------|--------|
| **Max Torque** | 30 kg·cm (2.94 Nm) @12V | 2.8 Nm | 95% of max |
| Idle Current | 180 mA | N/A | - |
| Stall Current | 2.7 A | N/A | - |
| Voltage | 6-12.6V (12V recommended) | N/A | - |

**Note**: Simulation uses 2.8 Nm effort limit (95% of hardware max). Session 21 discovered that stiffness (not just effort limit) was the critical factor - increased from 200 to 1200 for realistic servo tracking.

### Position & Speed

| Parameter | Real Hardware | Simulation | Status |
|-----------|--------------|------------|--------|
| Resolution | 4096 steps/360° (0.088°/step) | Continuous | OK |
| Max Speed | 45 RPM (270°/s = 4.71 rad/s) @12V | Limited by PD | Verify on hardware |
| Control Rate | Position commands @1 Mbps bus | 20 Hz policy | OK |

### Joint Limits (Mechanical)

| Joint | Min | Max | Simulation | Status |
|-------|-----|-----|------------|--------|
| Shoulders | -30° (-0.5236 rad) | +30° (+0.5236 rad) | ±0.5236 rad | Matched |
| Thighs | -90° (-1.5708 rad) | +90° (+1.5708 rad) | ±1.5708 rad | Matched |
| Calves | -90° (-1.5708 rad) | +90° (+1.5708 rad) | ±1.5708 rad | Matched |

**WARNING**: Joint limits are mechanical stops. Exceeding them in sim will not transfer to hardware - the servo will hit physical limits and potentially damage the robot.

---

## Controller (ESP32)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Control loop | ~200 Hz | 5 ms intervals |
| Servo bus | Serial1 @ 1 Mbps | TTL half-duplex |
| USB telemetry | 115200 baud | To host PC |
| Watchdog | 250 ms timeout | E-stop on timeout |

---

## IMU (MPU6050)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Type | 6-axis (3 accel + 3 gyro) | |
| I2C address | 0x68 | |
| Sample rate | Up to 200 Hz | Matches control loop |
| Noise (angular velocity) | ~0.01 rad/s | Sim uses this for DR |
| Noise (gravity projection) | ~0.05 | Sim uses this for DR |

---

## Physical Robot

| Parameter | Value | Source |
|-----------|-------|--------|
| Total mass | ~2.0 kg | Measured |
| Body dimensions | ~20×15×8 cm | CAD model |
| Ground clearance (standing) | ~18 cm | Measured |
| Spawn height (sim) | 0.24 m | harold.py config |
| Target height (walking) | 0.275 m | GaitCfg |

---

## Safe Parameter Ranges for Experiments

### DO NOT EXCEED (Hardware Limits)

| Parameter | Current | Max Safe | Reason |
|-----------|---------|----------|--------|
| `effort_limit` | 2.8 Nm | 2.94 Nm | Currently at 95% of hardware max |
| `stiffness` | 1200 | TBD | Session 21: Higher values needed for sim accuracy |
| `damping` | 50 | TBD | Session 21: Lower ratio for responsiveness |
| `action_scale` | 0.5 | 1.0 | Larger may hit joint limits with some poses |

**Session 21 Discovery**: Original stiffness=200 was FAR too low. Servos couldn't extend legs under load. Real robot does pushups fine, so sim was inaccurate. Stiffness=1200 with damping=50 matches real servo behavior. Sim-to-real testing needed to validate these values on hardware.

### MUST PRESERVE (Sim-to-Real Critical)

| Parameter | Required Value | Reason |
|-----------|---------------|--------|
| Joint limits (shoulders) | ±30° (±0.5236 rad) | Mechanical stops |
| Joint limits (thighs/calves) | ±90° (±1.5708 rad) | Mechanical stops |
| Control rate | 20 Hz policy | Matches deployment pipeline |
| Joint order | [FL, FR, BL, BR] × [shoulder, thigh, calf] | Matches firmware |
| Sign convention | Thighs/calves inverted | See `action_config.json` |

### MAY TUNE FREELY (No Hardware Impact)

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| Reward weights | Any | Pure learning signal |
| `action_filter_beta` | 0.1-0.5 | Smoothing preference |
| Termination thresholds | Any | Training curriculum |
| Domain randomization ranges | Any | Robustness tuning |
| `height_reward`, `forward_reward` etc. | Any | Reward shaping |
| `low_height_penalty`, `low_height_threshold` | Any | Training signal |

---

## Servo Communication Protocol

From `firmware/docs/Communication_Protocol_User_Manual_EN.pdf`:

- **Packet format**: `0xFF 0xFF` + ID + Length + Instruction + Parameters + Checksum
- **ID range**: 0-253 (Harold uses IDs 1-12)
- **Broadcast ID**: 254 (0xFE) for simultaneous control
- **Instructions**: PING (0x01), READ (0x02), WRITE (0x03), SYNC WRITE (0x83)
- **Feedback available**: Position, Speed, Load, Voltage, Temperature, Current

---

## Deployment Mapping

### Joint Order (RL → Firmware)

```
RL Policy Output:  [FL_sh, FR_sh, BL_sh, BR_sh, FL_th, FR_th, BL_th, BR_th, FL_ca, FR_ca, BL_ca, BR_ca]
Servo IDs:         [1,     4,     7,     10,    2,     5,     8,     11,    3,     6,     9,     12]
```

### Sign Convention

| Joint Type | RL → Hardware | Notes |
|------------|---------------|-------|
| Shoulders | +1.0 (same) | Direct mapping |
| Thighs | -1.0 (inverted) | Servo mounting |
| Calves | -1.0 (inverted) | Servo mounting |

### Position Conversion

```
servo_units = 2047 + direction * (degrees * 4096 / 360)
```

Where `direction` is +1 or -1 per the sign convention table.

---

## References

- Servo datasheet: `firmware/ST3215 Servo.pdf`
- Communication protocol: `firmware/docs/Communication_Protocol_User_Manual_EN.pdf`
- Calibration guide: `firmware/CALIBRATION_CHECKLIST.md`
- Deployment plan: `DEPLOYMENT_PLAN.md`
- Action config: `deployment_artifacts/terrain_62/action_config.json`
