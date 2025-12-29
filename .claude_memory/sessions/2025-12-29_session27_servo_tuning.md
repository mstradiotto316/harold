# Session 27: Servo Tuning & Pi Stress Testing

**Date**: 2025-12-29
**Duration**: ~1 hour
**Platform**: Raspberry Pi 5 (2GB RAM)

## Summary

This session focused on diagnosing servo performance issues and validating the Pi 5 power supply.

## Pi 5 Stress Test Results

Ran comprehensive stress tests to validate onboard power supply:

| Test | Duration | Peak Temp | Throttling | Status |
|------|----------|-----------|------------|--------|
| CPU-only (4 cores) | 30s | 71.4°C | 0x0 | PASS |
| CPU+Memory+I/O | 30s | 75.2°C | 0x0 | PASS |
| Robot Controller Sim | 30s | 60.9°C | 0x0 | PASS |

**Conclusion**: Power supply is healthy. No undervoltage detected under any load.

## Servo Issues Diagnosed

User reported two issues:

### 1. Large Dead Zone
- **Symptom**: Can rotate servo a bit before it "locks up and fights back"
- **Root Cause**: Registers 26-27 (`SMS_STS_CW_DEAD`, `SMS_STS_CCW_DEAD`) control deadband
- **Factory default**: 3-10 encoder steps (~0.26°-0.88°)
- **Fix**: Set dead zone to 1-2 via firmware

### 2. Position Drift / Slip
- **Symptom**: Servos "forget" their center, need recalibration
- **Possible causes**:
  - Torque overload (gears skipping) - mechanical, not software fixable
  - Magnetic encoder interference from IMU
  - Calibration offset not persisting in EEPROM
  - Voltage brownouts

## Artifacts Created

### ServoTuning.ino
New utility at `firmware/CalibrationAndSetup/ServoTuning/ServoTuning.ino`

Commands:
- `s` - Scan all servos (shows dead zone, torque, offset)
- `r <id>` - Read detailed parameters
- `d <id> <v>` - Set dead zone (0-10)
- `a <v>` - Set dead zone for ALL servos
- `t <id> <v>` - Set torque limit (0-1000)
- `c <id>` - Calibrate center position
- `m` - Monitor positions in real-time

### Uncommitted Firmware Changes (from previous agent)
`firmware/StreamingControl/HaroldStreamingControl/HaroldStreamingControl.ino`:
- `TORQUE_LIMIT = 1000` (100% max torque)
- `initServoTorque()` function to set torque at boot
- Asymmetric collision-safe joint limits
- Watchdog timeout reduced to 250ms

## Key Register Map (ST3215/SMS_STS)

| Register | Name | Purpose |
|----------|------|---------|
| 26 | CW_DEAD | Clockwise dead zone (steps) |
| 27 | CCW_DEAD | Counter-clockwise dead zone (steps) |
| 31-32 | OFS_L/H | Calibration offset |
| 48-49 | TORQUE_LIMIT_L/H | Torque limit (0-1000) |
| 55 | LOCK | EPROM write lock |

## Recommendations

1. **Reduce dead zone**: Set to 2 (`a 2` command) - tightest without buzzing
2. **For slip issues**:
   - Monitor with `m` command while disturbing robot
   - If positions jump = mechanical (gear skip) or magnetic interference
   - Move IMU away from servos if magnetic interference suspected
3. **Verify calibration persistence**: Power-cycle after calibration

## Next Steps

1. Flash ServoTuning.ino and run diagnostic scan
2. Reduce dead zones across all servos
3. Test if position drift continues after adjustments
4. If drift persists, investigate magnetic interference from IMU
