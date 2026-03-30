# Harold Observations & Insights

## 2026-03-29: BREAKTHROUGH — Harold walks in manager-based architecture

**Harold achieves near-perfect trot (gait=9.7/10) in ManagerBasedRLEnv after just 800 iterations.**

The DirectRLEnv architecture that we've been using for all experiments (sim_flat_v1, sim_flat_v2, harold_flat) has a fundamental issue that prevents walking. After exhaustive investigation (7 parallel Opus agents auditing every subsystem), the confirmed differences are:

1. **Action clamping bug (FIXED)**: Our direct env clamped actions to [-1,1] before processing. The reference ActionManager does NOT clamp. This limited joints to ±0.2 rad from default and corrupted the action smoothness penalty + last_action observation.
2. **Joint penalty scope bug (FIXED)**: Our port filtered joint_acc/joint_vel penalties to hip joints only. The Spot reference functions ignore the SceneEntityCfg joint_names filter and penalize ALL 12 joints.
3. **Unknown remaining issue**: Even after fixing both bugs, the direct env still doesn't walk (EXP-783: gait=0.14 vs reference gait=9.7). The root cause remains unidentified.

**Solution**: Created `Harold-Velocity-Flat-v0` — a manager-based environment using the proven ManagerBasedRLEnv architecture with the Spot reward structure adapted for Harold's body naming and physical scale.

Key files:
- Config: `harold_isaac_lab/.../tasks/manager_based/harold_flat/flat_env_cfg.py`
- Training: `./isaaclab.sh -p harold_isaac_lab/scripts/skrl/train.py --task Harold-Velocity-Flat-v0 --num_envs 2048 --headless`

### Manager-based Harold training metrics (step 355200, 74%):
- gait: 9.69 (near-perfect trot)
- air_time: 0.49 (feet lifting)
- base_linear_velocity: 4.26 (strong velocity tracking)
- base_angular_velocity: 4.68 (strong yaw tracking)
- reward: 349
- episode_length: 999/1000

### Reference environment documentation
Created `docs/example_environments/` with 11 technical breakdowns of every Isaac Lab quadruped locomotion example (Spot, Go1, Go2, A1, ANYmal B/C/D). The ANYmal C direct env doc is particularly relevant as it documents the `step_dt` reward multiplication pattern for direct envs.

## Pre-Breakthrough History

All direct-env observations (Sessions 46-52, 700+ experiments) are archived in
`docs/memory/archives/OBSERVATIONS_pre_manager_based.md`. Key takeaway: the DirectRLEnv
architecture has a fundamental standing trap bug that prevents walking. The manager-based
architecture solved this completely.

## Evergreen Guardrails
- Video is the primary signal for success/failure. Metrics can lie (vx from falling, ep_len from standing). Run video review after every experiment.
- Use the 5-metric protocol (episode_length, upright_mean, height_reward, body_contact_penalty, vx_w_mean) as the quantitative complement to video, not a replacement.
- "On elbows" exploit remains a risk; always check height_reward and body_contact_penalty.

## Sim-to-Real Alignment Notes
- Backlash dead zone is ~10 degrees (2026-01-03).
- Best direct-env actuator match (effort=2.8): stiffness=1200, damping=75. Manager-based uses different actuator model (Kp=40, Kd=0.5) — sim-to-real alignment will need re-validation.
- Hardware telemetry logs include `cmd_pos_*` columns for commanded vs measured comparison.
- Hardware CPG baseline: Test 3 (0.4 Hz, duty 0.5) on lower-friction surface. Logs in `logs/hardware_sessions/`.

## Hardware Session Logs
- RPi logs: `/home/pi/harold/deployment/sessions/session_YYYY-MM-DD_HH-MM-SS.csv`
- Desktop copies: `logs/hardware_sessions/session_YYYY-MM-DD_HH-MM-SS.csv`

## Archives
Historical observations are moved to `docs/memory/archives/` to keep this file short.
- `archives/OBSERVATIONS_pre_manager_based.md` — Direct-env era (Sessions 46-52, 2026-03-15 to 2026-03-28)
- `archives/OBSERVATIONS_2026-01-04_full.md` — Hardware CPG baseline
