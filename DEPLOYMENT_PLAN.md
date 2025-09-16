# Harold Policy Deployment Plan

## Objective
Deploy the most recently trained Harold direct-terrain policy (`logs/skrl/harold_direct/terrain_62`) onto the physical quadruped while minimizing risk. Execute the following steps sequentially, validating assumptions at each stage before progressing.

## Steps

1. **Inventory Policy Artifacts**
   - Record contents of `terrain_62` (checkpoints, params, training config).
   - Select deployment checkpoint (`best_agent.pt` unless evaluation dictates otherwise).
   - Summarize observation/action spaces, default joint pose, and clamps in a README within the log folder for quick reference.

2. **Export & Validate Inference Models** *(Completed — artifacts under `deployment_artifacts/terrain_62`)*
   - Write an export script that loads the checkpoint (including running statistics) and produces TorchScript and ONNX versions.
   - Update `action_config` with confirmed defaults, scaling, and joint order.
   - Re-run a short batch of stored rollouts to verify exported outputs match PyTorch inference.

3. **Offline Command Pipeline Test** *(Completed — see `deployment_artifacts/terrain_62/offline_commands.csv`)*
   - Replayed observation logs through ONNX export and generated servo-ready commands; noted calf joints occasionally saturate at -90° which we must watch on hardware.

4. **Refresh Hardware Calibration** *(Completed — offsets captured via `CALIBRATION_CHECKLIST.md`; updated `SinglePositionTest` matches hardware signs)*
   - Run Ping → CalibrateMiddles → ChainMovementTest → SinglePositionTest; confirmed thigh/calf inversion and encoded sign map in tooling.

5. **Upgrade Robot Firmware for Streaming Control** *(In progress — firmware skeleton at `firmware/StreamingControl/HaroldStreamingControl.ino`)*
   - Flash new sketch, verify handshake (`READY?` → `ARDUINO_READY`), START/STOP commands, watchdog, and telemetry contents.
   - Integrate IMU once available; currently streams servo load/current/temp and commanded pose.

6. **Design Observation Builder on Host**
   - Implement host-side code that converts telemetry into the 48-D observation vector expected by the policy.
   - Define estimators or assumptions for any unmeasured signals (e.g., linear velocity) and document risks.

7. **Implement Host Runtime Loop**
   - Build a 20 Hz loop that acquires telemetry, runs the ONNX policy, converts actions to joint targets, and streams commands to the MCU.
   - Add emergency stop handling, watchdog monitoring, and comprehensive logging (commands, telemetry, timestamps).

8. **Bench-Top Dry Runs (No Ground Contact)**
   - Mount robot on a stand, power servos, run the host loop with zero velocity commands.
   - Verify steady stance, telemetry sanity (IMU gravity vector, joint deltas), and watchdog recovery behavior.

9. **Incremental Motion Tests on Stand**
   - Command small-amplitude sine sweeps per joint to confirm direction and limits.
   - Feed low-speed gait commands while feet remain off ground; monitor currents/loads for saturation.

10. **Contact Rehearsal with Safety Support**
    - Use a harness or boom to allow ground contact while preventing falls.
    - Run slow policy commands, evaluate estimator stability under contact, and adjust clamps/noise handling as required.

11. **Controlled Free-Standing Trials**
    - On flat indoor terrain, execute short runs starting from idle, gradually introducing forward, yaw, and lateral commands.
    - After each attempt, review logs for torque spikes, IMU anomalies, and timing issues before continuing.

12. **Field Deployment Checklist & Rollout**
    - Prepare a go/no-go checklist covering battery health, firmware hash, policy checksum, emergency stop verification, and logging setup.
    - Begin with flat outdoor terrain, then cautiously progress to rough terrain, reviewing telemetry and video after each escalation.
