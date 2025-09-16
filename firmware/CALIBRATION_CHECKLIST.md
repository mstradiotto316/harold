# Harold Servo Calibration Checklist

Follow these steps *with the robot safely supported* before running the exported policy. You will need the existing Arduino sketches under `firmware/CalibrationAndSetup/`. Record all offsets and settings in your engineering log when you finish.

## 0. Prep
- **Power**: bench supply or fully charged pack. Keep an inline fuse/E-stop available.
- **Support**: suspend Harold so the legs can move freely without ground contact.
- **Connections**: ESP32 USB to host laptop, servo bus power on, verify polarity.
- **Tooling**: Arduino IDE / `arduino-cli` configured for your ESP32 board, SCServo library installed.

## 1. Bus Sanity Check (`Ping/` sketch)
1. Open `firmware/CalibrationAndSetup/Ping/Ping.ino`.
2. Flash to the ESP32, open Serial Monitor @115200 baud.
3. Confirm IDs 1–12 respond. If any are missing, diagnose wiring/ID before proceeding.
4. Log any non-responding servo IDs.

## 2. Midpoint Trim (`CalibrateMiddles/CalibrateMiddles.ino`)
1. Flash the calibrator sketch.
2. For each servo ID 1→12:
   - Use `a/d` (±100) and `s/w` (±10) commands to align the joint mechanically to its neutral position (leg vertical, knee neutral).
   - Press `p` to save the offset; the sketch recenters automatically.
   - Record the saved offset value (printout) in your log.
3. After all joints, the sketch reports completion. Power-cycle the servo bus to ensure trims are stored.

## 3. Motion Sweep (`ChainMovementTest/ChainMovementTest.ino`)
1. Flash the chain movement test.
2. Observe each joint executing centre → +45° → centre → –45° → centre.
3. Verify direction signs match expectations (e.g., positive shoulder angle abducts the leg).
4. Note any joints that stall, jitter, or hit mechanical stops before ±45°.

## 4. Radian Command Gate (`SinglePositionTest/SinglePositionTest.ino`)
1. Flash the single-position test sketch.
2. With Serial Monitor open, send the default pose command to match simulation:
   ```
   [0,0,0,0,0.3,0.3,0.3,0.3,-0.75,-0.75,-0.75,-0.75]
   ```
   The robot should settle into the nominal stance without visible lean.
3. Test a few additional targets (e.g., +0.2 rad on `fl_thigh`, –0.2 rad on `fr_thigh`) and verify motion directions.
4. Confirm the sketch reports any clamping; repeated clamp messages mean trims or limits need adjustment.

## 5. Torque / Current Limits (optional but recommended)
- While running Step 4, note the reported load/current in Serial Monitor (if firmware prints them) or query parameters using the SCServo PC tool.
- Ensure torque limit registers (`SMS_STS_TORQUE_LIMIT_L`) remain near factory value (≈1023). Adjust only if you have a reason to derate.

## 6. Document Final Defaults
Record the following for later use in runtime software:
- Final offset counts saved per servo (from Step 2).
- Any servos with reversed directions compared to expectation (should all match `DIR` array now).
- Observed resting pose alignment (body level? any sag?).
- Maximum safe range observed for thighs/calves before mechanical interference.

## When Finished
- Re-flash the ESP32 with the latest streaming-control firmware once we develop it (later plan step).
- Keep this calibration log accessible; the host runtime will assume these trims when converting policy outputs to servo targets.
