# Harold Next Steps

## EXP-962 hardware validation

1. Sync the approved bench branch to the clean Pi checkout. Check the inactive service, candidate hash, ONNX input/output and IMU.
2. With Harold supported, use a small single-joint pulse and the operator's observation to verify anatomical pitch direction. Mirrored encoder round trips alone cannot distinguish an inverted physical response.
3. Compare commanded and measured joints, IMU, voltage and load on a short supported policy test before choosing a floor trial.
4. If the stand checks pass, evaluate a 0.2 m/s forward command and stop on level ground with the operator present. Use the session CSV and an external displacement observation to compare to the firmware-clamped simulation (0.181 m/s). Investigate yaw/lateral tracking and high-speed saturation using the verified mapping.
5. Review remaining legacy deployment diagnostic scripts for stale 50D/phase assumptions. The deprecated direct-env standing trap remains lower priority.
