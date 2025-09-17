# Harold Policy Deployment Plan

## Objective
Deploy the most recently trained Harold direct-terrain policy (`logs/skrl/harold_direct/terrain_64_2`) onto the physical quadruped while minimizing risk. Execute the following steps sequentially, validating assumptions at each stage before progressing.

## Current Status Snapshot (Marshalling Point for New Agents)
- Firmware: streaming sketch v1 is flashed and verified on hardware (handshake, START/STOP, 1 s watchdog, telemetry). IMU streaming is still TODO but firmware scaffolding exists.
- Calibration: servo trims/signs refreshed via `CALIBRATION_CHECKLIST.md`; host action config and firmware joint tables agree with training metadata.
- Policy assets: TorchScript/ONNX exports, action_config, running stats, and offline command checks (calf joints saturate near −90°; monitor on hardware) live under `deployment_artifacts/terrain_64_2/`.
- Simulation logging: `HAROLD_POLICY_LOG_DIR` enables JSONL logs (observations, commands, filtered actions) with metadata describing joint limits, action filter β=0.4, and implicit PD gains. Rough env forces a 0.4 m/s forward command during logging to produce a repeatable gait trace for replay.
- Pending immediately: capture the ~10 s single-env rollout, build the host observation builder against that dataset, then bench/inference replays before touching ground.

## Steps

1. **Inventory Policy Artifacts**
   - Record contents of `terrain_64_2` (checkpoints, params, training config).
   - Select deployment checkpoint (`best_agent.pt` unless evaluation dictates otherwise).
   - Summarize observation/action spaces, default joint pose, and clamps in a README within the log folder for quick reference.

2. **Export & Validate Inference Models** *(Completed — artifacts under `deployment_artifacts/terrain_64_2`)*
   - Write an export script that loads the checkpoint (including running statistics) and produces TorchScript and ONNX versions.
   - Update `action_config` with confirmed defaults, scaling, and joint order.
   - Re-run a short batch of stored rollouts to verify exported outputs match PyTorch inference.

3. **Offline Command Pipeline Test** *(Completed — see `deployment_artifacts/terrain_64_2/offline_commands.csv`)*
   - Replayed observation logs through ONNX export and generated servo-ready commands; noted calf joints occasionally saturate at -90° which we must watch on hardware.

4. **Refresh Hardware Calibration** *(Completed — offsets captured via `CALIBRATION_CHECKLIST.md`; updated `SinglePositionTest` matches hardware signs)*
   - Run Ping → CalibrateMiddles → ChainMovementTest → SinglePositionTest; confirmed thigh/calf inversion and encoded sign map in tooling.

5. **Upgrade Robot Firmware for Streaming Control** *(Completed — v1 flashed & bench-tested; IMU streaming still pending)*
   - Flash new sketch, verify handshake (`READY?` → `ARDUINO_READY`), START/STOP commands, watchdog, and telemetry contents. *(Done: watchdog now stable at 1 s timeout, START/STOP re-arms correctly, telemetry packets confirmed on hardware.)*
   - Integrate IMU once available; currently streams servo load/current/temp and commanded pose. *(Open sub-task tracked for Step 6 integration work.)*

6. **Design Observation Builder on Host**
   - Implement host-side code that converts telemetry into the 48-D observation vector expected by the policy.
   - Define estimators or assumptions for any unmeasured signals (e.g., linear velocity) and document risks. *(Leverage simulated datasets to de-risk IMU requirements before hardware integration.)*
   - Build a clean logging harness in Isaac Lab (replace deprecated snippets in `harold_rough`/`harold_flat` env files) that records unclipped 48-D observations, processed joint targets, and command metadata at policy rate for curated rollouts (enabled via `HAROLD_POLICY_LOG_DIR`).
   - Capture reference trajectories (obs/actions) from simulation at nominal gait speeds; store under `deployment_artifacts/terrain_64_2/sim_logs/` with README describing terrain, friction, action filter β, actuator settings, and control command (target: single env, 0.4 m/s forward, 0 yaw, 0 lateral for ~10 s ≈200 policy steps).
   - Note: While `HAROLD_POLICY_LOG_DIR` is set the rough-task env overrides sampled commands to `[0.4, 0.0, 0.0]`. Unset the variable or revert once the dataset is captured to restore random command sampling for other runs.
   - **Bench Replay (no policy inference):**
     1. Feed recorded joint targets directly to the robot via streaming firmware while sending the matching logged commands to the observation builder.
     2. Observe whether the hardware reproduces the simulated motion envelope within actuator limits (mind Isaac Lab clamps: shoulders ±0.30 rad range, thighs/calves ±0.90 rad, filtered via EMA β=0.4, and implicit-PD dynamics with stiffness 200/ damping 75).
     3. Note any divergence due to real actuator saturation or action clipping and update clamps if needed.
   - **Inference Replay Consistency:**
     1. Stream the logged observations into the ONNX policy (host inference) and compare the policy output against the recorded processed actions.
     2. Verify consistency within tolerance (accounting for Isaac Lab’s action clipping and EMA filter) before trusting host-side inference.
     3. If deviations occur, adjust observation normalization or action scaling to match training conditions.

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
