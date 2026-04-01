# Harold Observations & Insights

## 2026-03-31: Sim-to-real transfer stack — from shuffling to natural walking

**4-phase sim-to-real overhaul transformed Harold from shuffling (EXP-785: reward=340) to deployment-ready natural walking (EXP-814: reward=508, air_time=3.63).**

Key findings:
- **EMA action filtering (Phase 1)**: Training with deployment-matched action smoothing (beta=0.2) eliminated jittery micro-movements. Critical insight: train/deploy mismatch was the root cause of shuffling.
- **Observation noise as regularizer (Phase 2)**: Adding IMU/encoder noise INCREASED reward (+5.4%), not decreased. Noise prevents overfitting to precise sensor values.
- **Air time threshold was the stride-length lever (Phase 3)**: mode_time 0.15->0.25 produced +64% air time, visibly longer strides. This single change had the most impact on gait naturalness.
- **Perturbation robustness is free (Phase 4)**: Doubling push forces produced no regression — perturbations act as yet another regularizer.
- **Each phase composed well**: No negative interactions between EMA, noise, rewards, and perturbations.

Current best config (EXP-814): EMA beta=0.2, obs noise enabled, air_time mode_time=0.25, foot_clearance target=0.07 (wt=1.5), action_smoothness=-1.0, gait vel_threshold=0.3, push ±0.2 m/s.

## 2026-03-31: Compound config + PPO tuning — 4 KEEPs, reward 487.9

**Compound config combining 6 validated improvements + PPO tuning pushed Harold to reward 487.9 (+43.2% over EXP-785).**

Key findings:
- **Compound config composes well**: All 6 validated improvements (vel=8, gait=13, air=8, smooth=-0.5, clear=1.0/0.05, orient=-4, standing=0.02) composed without interference (EXP-802 KEEP, reward 470.0)
- **learning_epochs 5->8->10**: Monotonic improvement. More gradient passes per rollout improves sample efficiency (EXP-804 KEEP, EXP-805 KEEP)
- **rollouts 24->32**: Hurt performance (-4%). Slower convergence and lower final reward (EXP-803 DISCARD)
- **discount_factor 0.99->0.995**: Major regression (-12.6%). Higher gamma slows learning significantly (EXP-806 DISCARD)
- **base_angular_velocity_weight 5->8**: Reward inflation without quality improvement. All locomotion metrics regressed despite higher total reward (EXP-807 DISCARD)
- **air_time_variance_weight -1->-0.5**: Freed reward budget, especially improving orientation (-0.073, best ever). Variance was already negligible so penalty was over-constraining (EXP-808 KEEP)

Current best config: vel=8, gait=13, air=8, smooth=-0.5, clear=1.0/0.05, orient=-4, standing=0.02, learning_epochs=10, air_time_variance=-0.5

## 2026-03-30: Reward tuning session — 5 KEEPs, reward 405.5

**Systematic reward weight tuning improved Harold from baseline EXP-785 (reward=340.8) to EXP-799 (reward=405.5, +19%).**

Key findings:
- **base_linear_velocity_weight 5->8**: +55% velocity tracking (EXP-788 KEEP)
- **gait_weight 10->13**: Recovered gait quality, reward 390 (EXP-789 KEEP)
- **air_time_weight 5->8**: +70% foot lift time (EXP-790 KEEP)
- **action_smoothness_weight -1->-0.5**: Freed reward budget (EXP-791 KEEP)
- **foot_clearance target_height 0.03->0.05**: +27% foot clearance (EXP-799 KEEP, broke 7-DISCARD plateau)
- Longer training (30 min vs 15 min) produced WORSE results — policy converges to conservative stepping
- base_motion penalty reduction caused major regression — the penalty is needed to drive forward motion
- Results are seed-robust (seed=42 and seed=123 produce comparable metrics)
- Higher vel_weight=10 trades stability for speed (ep_len dropped to 992)

Current best config: vel=8, gait=13, air_time=8, smooth=-0.5, foot_clearance target_height=0.05

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
