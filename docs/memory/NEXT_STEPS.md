# Harold Next Steps

## HIGH PRIORITY — Manager-Based Walking (EXP-785 baseline)

1. **Export manager-based policy to ONNX** — Export the trained policy for hardware deployment. The existing export pipeline may need updates for the manager-based architecture (50Hz control rate, different obs/action scaling).
2. **Run autoresearch to optimize walking gait** — Use the manager-based env (`harold_mgr`) as baseline. Tune reward weights, command ranges, and training hyperparameters to improve foot clearance, step length, and gait quality.

## MEDIUM PRIORITY — Sim-to-Real Pipeline

3. **Run real-world hardware test** — Deploy the manager-based policy on the robot off the test stand. Ensure IMU recording is active. Compare sim vs real joint positions, commands, and IMU data.
4. **Sim-to-real alignment** — Compare hardware logs to simulated walking. Tune stiffness/damping if sim doesn't match real servo behavior. Add IMU noise if needed.
5. **Review legacy deployment scripts** — Audit `deployment/test_*.py` and `deployment/debug_*` scripts. Migrate to 48D/raw-observation path or retire stale 50D/phase-based scripts.

## LOW PRIORITY — Investigation

6. **Debug DirectRLEnv standing trap** — The direct-env architecture doesn't walk despite fixing action clamping and joint penalty scope. Root cause unknown. Lower priority since manager-based works.
