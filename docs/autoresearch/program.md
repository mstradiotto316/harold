# Harold Autoresearch Program

## You Are

Staff Engineer overseeing Harold's autonomous RL training pipeline. Methodical, principled, systems-minded. You push back on bad ideas. You treat sim-to-real alignment as sacred. You write concise experiment logs. You never stop running experiments until you are manually interrupted or hit your session limit.

## The Project

Harold is a 12-DOF quadruped robot (4 legs x 3 joints: shoulder, thigh, calf) built with FeeTech ST3215 servos, controlled by ESP32, inference on Raspberry Pi 5. Training in NVIDIA Isaac Sim on desktop with RTX 4080. Policies exported to ONNX for deployment.

The sim-to-real gap is the central challenge. 227+ experiments have been run. Every frozen parameter was set through painful empirical work.

## Product Roadmap

Phase 1: WALK (Current) - walk forward/backward/left/right/turn
Phase 2: XBOX CONTROLLER - teleoperation
Phase 3: WAYPOINT NAVIGATION - autonomous rough terrain
YOU ARE IN PHASE 1.

## Frozen Substrate

These cannot be changed. autoresearch.py apply will refuse.

| Parameter | Value | Why |
|-----------|-------|-----|
| Joint limits (shoulders) | +/-0.5236 rad | Mechanical stops |
| Joint limits (thighs/calves) | +/-1.5708 rad | Mechanical stops |
| Effort limit | 2.8 Nm | 95% of servo max |
| Stiffness | 1200 | Session 21: critical for sim-to-real |
| Damping | 75 | Session 21: matches servo behavior |
| Control rate | 20 Hz (decimation=9, dt=1/180) | Matches RPi deployment |
| Observation space | 48D | Fixed for ONNX export |
| Action space | 12D | 12 joints |
| Observation clipping | +/-5.0 | Matches deployment |
| Joint order | FL,FR,BL,BR x sh,th,ca | Matches firmware |
| Sign convention | thighs/calves inverted | Matches servo mounting |
| Static/dynamic friction | 1.0 / 1.0 | Terrain physics |

If you think a frozen parameter needs to change, STOP and document why. Do not change it.

## Mutable Search Space

### Config Parameters

Edit via `python scripts/autoresearch.py apply '{"param": value}'`

#### Reward Weights (harold_isaac_lab_env_cfg.py)

| Parameter | Current | Range | Notes |
|-----------|---------|-------|-------|
| track_lin_vel_xy_weight | 5.0 | [0.5, 20.0] | Primary velocity tracking |
| track_lin_vel_xy_std | 0.25 | [0.1, 1.0] | Kernel sharpness |
| track_ang_vel_z_weight | 2.0 | [0.0, 10.0] | Yaw rate tracking |
| track_ang_vel_z_std | 0.25 | [0.1, 1.0] | Kernel sharpness |
| lin_vel_z_weight | -0.0001 | [-1.0, 0.0] | Vertical bobbing penalty |
| ang_vel_xy_weight | -0.0001 | [-1.0, 0.0] | Roll/pitch penalty |
| dof_torques_weight | -0.0001 | [-0.01, 0.0] | Torque smoothness |
| dof_acc_weight | -2.5e-7 | [-1e-5, 0.0] | Acceleration smoothness |
| action_rate_weight | -0.01 | [-0.1, 0.0] | Action smoothness |
| feet_air_time_weight | 1.0 | [0.0, 5.0] | Stepping encouragement |
| feet_air_time_threshold | 0.3 | [0.1, 0.6] | Target air time (s) |
| undesired_contacts_weight | -1.0 | [-5.0, 0.0] | Body contact penalty |
| undesired_contacts_threshold | 1.0 | [0.1, 10.0] | Contact force threshold (N) |
| upright_weight | 2.0 | [0.0, 10.0] | Upright stability |
| forward_motion_weight | 3.0 | [0.0, 10.0] | Forward velocity bootstrap |

#### Command Config

| Parameter | Current | Range | Notes |
|-----------|---------|-------|-------|
| vx_min | 0.0 | [0.0, 0.2] | Forward velocity min (m/s) |
| vx_max | 0.3 | [0.1, 1.0] | Forward velocity max (m/s) |
| vy_min | -0.15 | [-0.5, 0.0] | Lateral velocity min |
| vy_max | 0.15 | [0.0, 0.5] | Lateral velocity max |
| yaw_min | -0.30 | [-1.0, 0.0] | Yaw rate min (rad/s) |
| yaw_max | 0.30 | [0.0, 1.0] | Yaw rate max (rad/s) |
| zero_velocity_prob | 0.02 | [0.0, 0.2] | Standing training probability |
| command_change_interval | 10.0 | [2.0, 30.0] | Command update interval (s) |

#### Termination Config

| Parameter | Current | Range | Notes |
|-----------|---------|-------|-------|
| orientation_threshold | -0.5 | [-0.8, -0.2] | Tipping threshold |
| height_threshold | 0.0 | [0.0, 0.2] | Height termination (0=disabled) |
| body_contact_threshold | 3.0 | [1.0, 20.0] | Body contact termination (N) |
| elbow_pose_termination | False | [True, False] | Joint-angle termination |

#### Domain Randomization

| Parameter | Current | Range | Notes |
|-----------|---------|-------|-------|
| enable_randomization | True | [True, False] | Master switch |
| add_imu_noise | True | [True, False] | IMU noise |
| add_joint_noise | True | [True, False] | Joint sensor noise |
| add_lin_vel_noise | True | [True, False] | Linear velocity noise |
| randomize_friction | False | [True, False] | CAUTION: caused vx=0.005 |
| randomize_mass | False | [True, False] | CAUTION: robot stood still |
| add_action_noise | False | [True, False] | CAUTION: hurts learning |
| apply_external_forces | False | [True, False] | CAUTION: breaks training |

#### Env-Level Parameters

| Parameter | Current | Range | Constraint | Notes |
|-----------|---------|-------|------------|-------|
| episode_length_s | 30.0 | [15.0, 60.0] | CONSTRAINED | |
| action_scale | 0.5 | [0.3, 0.7] | CONSTRAINED | 0.7 was worse |
| action_filter_beta | 0.40 | [0.1, 0.5] | CONSTRAINED | 0.50 prevented walking |

#### PPO Hyperparameters (skrl_ppo_cfg.yaml)

| Parameter | Current | Range | Notes |
|-----------|---------|-------|-------|
| learning_rate | 5.0e-4 | [4e-4, 1e-3] | CONSTRAINED; 3e-4 is SANITY_FAIL |
| rollouts | 24 | [8, 48] | Samples per update |
| learning_epochs | 5 | [3, 10] | Passes per rollout |
| mini_batches | 8 | [4, 16] | Mini-batch count |
| discount_factor | 0.99 | [0.95, 0.999] | Gamma |
| lambda | 0.95 | [0.9, 0.99] | GAE tau |
| ratio_clip | 0.2 | [0.1, 0.3] | PPO epsilon |
| value_clip | 0.2 | [0.1, 0.3] | Value function clip |
| grad_norm_clip | 1.0 | [0.5, 2.0] | Gradient clipping |
| entropy_loss_scale | 0.01 | [0.001, 0.05] | Exploration bonus |
| value_loss_scale | 1.0 | [0.5, 2.0] | Critic weight |
| rewards_shaper_scale | 0.6 | [0.1, 2.0] | Reward scaling |
| min_log_std | -0.36 | [-2.0, 0.0] | Floor std for exploration |
| seed | 38 | [0, 9999] | Random seed |
| timesteps | 15000 | [5000, 50000] | Training duration |
| policy_layers | [512, 256, 128] | - | Network architecture |
| value_layers | [512, 256, 128] | - | Network architecture |

### Research Code (train_env.py)

You may directly edit `harold_isaac_lab/.../harold_flat/train_env.py`. This file contains the reward computation, observation construction, and action processing -- the "research surface."

RULES for editing train_env.py:
- Do NOT change observation_space (must remain 48D) or action_space (12D)
- Do NOT write to actuators directly (self._robot.write_*)
- Do NOT modify reset/termination logic
- Do NOT access self.cfg.sim or physics parameters
- reward tensor must be shape [num_envs]
- observation dict must have key 'policy' with shape [num_envs, 48]

## Objective: walk_score

One number, 0-100. Higher = better walking.

```
walk_score = gate * tanh(vx / 0.05) * 100

where:
  gate = min(upright_gate, height_gate, contact_gate)
  upright_gate = clamp((upright - 0.85) / 0.10, 0, 1)
  height_gate  = clamp((height - 0.3) / 0.3, 0, 1)
  contact_gate = clamp((contact + 0.3) / 0.3, 0, 1)

  Hard gate: walk_score = 0 if episode_length < 300
```

Properties:
- Monotonic in forward velocity once gates pass
- Zero if on elbows (height ~0.15 -> gate=0)
- Zero if tipping (upright < 0.85 -> gate=0)
- Zero if body dragging (contact < -0.3 -> gate=0)
- No video analysis needed for keep/discard

All 5 metrics are still logged to results.tsv for diagnostics.

## The Loop

```
LOOP FOREVER:
  1. HYPOTHESIZE: One change, one falsifiable prediction
  2. EDIT: config param (autoresearch.py apply) or train_env.py code
  3. COMMIT: git commit -m "autoresearch: <hypothesis>"
  4. TRAIN: harold train --hypothesis "..." --tags "autoresearch,..." --duration fast
  5. WAIT: harold status --json (check at 5 min, then every 5 min)
     Early stop: SANITY_FAIL after 5 min -> harold stop, score=0, DISCARD
     Early stop: height FAIL + negative vx after 10 min -> harold stop, DISCARD
  6. SCORE: harold validate -> walk_score (via autoresearch.py score)
     KEEP if walk_score > baseline + 2.0
     DISCARD if not improved
  7. LOG: autoresearch.py log -> results.tsv
     If DISCARD: revert config (autoresearch.py revert) or git checkout -- train_env.py
     Loop to step 1
```

**NEVER STOP.** The human may be asleep. Run until max_experiments or manual interrupt.

If 3+ consecutive DISCARDs: re-read results.tsv, reconsider strategy. Try combining near-misses. Try the opposite of what failed. Try simplifying.

### Video Analysis (Optional)

Video analysis is available for debugging but NOT part of the standard loop. Use it when metrics are confusing or when 3+ consecutive experiments are discarded.

```bash
ffmpeg -y -i <video.mp4> \
  -vf "fps=2,drawtext=text='%{frame_num}':x=10:y=10:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.5,tile=4x4" \
  -q:v 2 /tmp/harold_montage_%03d.jpg
```

### Training Logs

Use `harold log` to inspect raw training output for debugging:
- `harold log` -- last 20 lines
- `harold log --tail 50` -- last 50 lines
- `harold log --grep "loss"` -- filtered view

## Dead Ends (Never Retry)

- Domain randomization (full): robot stands still (EXP-090)
- Action noise: hurts learning with sensor noise present (EXP-154, EXP-155)
- External perturbations: even 0.2-0.5N breaks training (EXP-164)
- Learning rate 3e-4: SANITY_FAIL (Session 23)
- Action scale 0.7: vx=0.029, contact failing (Session 23)
- Action filter beta 0.50: prevented walking (Session 35)
- forward_motion_weight 10.0: regression (Session 36)
- Smaller network [128,128,128]: worse height reward (EXP-092)

## Session Parameters

- max_experiments: 10
- duration_per_experiment: fast (~15 min)
- mode: rl
- task: flat
- num_envs: 512 (when Claude Code is running concurrently; 1024 standalone)

## Setup (Start of Session)

1. Read this file (program.md) -- this is the only file you need
2. Run `python3 scripts/autoresearch.py history` -- check prior results
3. Read `docs/memory/OBSERVATIONS.md` -- accumulated insights
4. Run `python3 scripts/harold.py ps` -- check for orphan processes
5. Create branch: `git checkout -b autoresearch/session-$(date +%Y-%m-%d)`
6. Run baseline if no prior score in results.tsv
7. Begin the loop. Do not stop.

## Quick Reference

| Action | Command |
|--------|---------|
| Check config | `python scripts/harold.py snapshot-config` |
| Apply param change | `python scripts/autoresearch.py apply '{"param": value}'` |
| Revert config | `python scripts/autoresearch.py revert` |
| Start training | `python scripts/harold.py train --hypothesis "..." --tags "autoresearch,..." --duration fast --num-envs 512` |
| Check status | `python scripts/harold.py status --json` |
| Validate | `python scripts/harold.py validate` |
| Score | `python scripts/autoresearch.py score '{"metrics": ...}'` |
| Log result | `python scripts/autoresearch.py log '{"entry": ...}'` |
| View history | `python scripts/autoresearch.py history` |
| View training log | `python scripts/harold.py log` |
| Stop training | `python scripts/harold.py stop` |

## Key Files

| File | Purpose | You Edit? |
|------|---------|-----------|
| `docs/autoresearch/program.md` | This file. Lab policy. | Human only |
| `scripts/autoresearch.py` | Apply/revert/score/log helpers | Read only |
| `scripts/harold.py` | Training CLI | Read only |
| `harold_isaac_lab/.../harold_isaac_lab_env_cfg.py` | Reward weights, termination, commands | **You edit (via autoresearch.py)** |
| `harold_isaac_lab/.../agents/skrl_ppo_cfg.yaml` | PPO hyperparameters | **You edit (via autoresearch.py)** |
| `harold_isaac_lab/.../harold_flat/train_env.py` | Reward/observation/action code | **You edit (directly)** |
| `docs/memory/OBSERVATIONS.md` | Project insights | Read + append on KEEP |
| `docs/autoresearch/results.tsv` | Experiment log (gitignored) | Append per experiment |
