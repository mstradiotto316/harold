# Harold Next Steps

## EXP-962 hardware validation

1. Pi sync, candidate hash, finite ONNX inference and IMU readout passed. Original FL thigh and knee directions were both reversed on the stand.
2. Verify both corrected pitch directions on the stand before closed-loop policy tests. A positive FL calf pulse should open the knee. A positive FL thigh pulse should move the knee toward the tail.
3. Compare commanded and measured joints, IMU, voltage and load on a short supported policy test before choosing a floor trial.
4. If the stand checks pass, evaluate a 0.2 m/s forward command and stop on level ground with the operator present. Use the session CSV and an external displacement observation to compare to the firmware-clamped simulation (use the corrected serial mapping). Investigate yaw/lateral tracking and high-speed saturation using the verified mapping.
5. Review remaining legacy deployment diagnostic scripts for stale 50D/phase assumptions. The deprecated direct-env standing trap remains lower priority.
