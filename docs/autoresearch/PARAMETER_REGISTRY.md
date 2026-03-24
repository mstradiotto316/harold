# Parameter Registry

Machine-readable registry of all tunable parameters for the Harold autoresearch system.
Used by `scripts/autoresearch.py` to validate proposed changes before applying them.

## Categories

- **FROZEN**: Never change without explicit human approval. Sim-to-real critical.
- **CONSTRAINED**: Tunable within documented range only.
- **TUNABLE**: Free to modify. Pure learning signal or training-only parameters.

## Env Config (`harold_isaac_lab_env_cfg.py`)

### Reward Weights (RewardsCfg)

| Parameter | Category | Current | Range | Notes |
|-----------|----------|---------|-------|-------|
| `track_lin_vel_xy_weight` | TUNABLE | 5.0 | [0.5, 20.0] | Spot: 5.0. Primary velocity tracking |
| `track_lin_vel_xy_std` | TUNABLE | 0.15 | [0.1, 1.0] | Harold-specific (Spot: 1.0, but Harold's cmd range is 10x smaller) |
| `track_ang_vel_z_weight` | TUNABLE | 2.0 | [0.0, 10.0] | Harold-specific (Spot: 5.0, smaller yaw range) |
| `track_ang_vel_z_std` | TUNABLE | 0.25 | [0.1, 1.0] | Harold-specific |
| `base_orientation_weight` | TUNABLE | 3.0 | [0.5, 5.0] | Spot: 3.0. Tilt penalty (replaces +3.0 upright REWARD) |
| `base_motion_weight` | TUNABLE | 2.0 | [0.5, 5.0] | Spot: 2.0. Combined vz + omega_xy (replaces lin_vel_z + ang_vel_xy) |
| `action_smoothness_weight` | TUNABLE | 1.0 | [0.1, 3.0] | Spot: 1.0. L2 norm of action diff (replaces action_rate) |
| `dof_torques_weight` | TUNABLE | -5e-4 | [-0.01, 0.0] | Spot: -5e-4 (was -0.0001) |
| `dof_acc_weight` | TUNABLE | -2.5e-7 | [-1e-5, 0.0] | Kept (negligible) |
| `shoulder_joint_vel_weight` | TUNABLE | 0.01 | [0.0, 0.1] | Spot: 1e-2 on hip joints. Harold shoulders = Spot hips |
| `feet_air_time_weight` | TUNABLE | 5.0 | [0.0, 10.0] | Spot: 5.0 (was 2.0) |
| `feet_air_time_threshold` | TUNABLE | 0.3 | [0.1, 0.6] | Spot: 0.3 |
| `continuous_gait_weight` | TUNABLE | 10.0 | [1.0, 20.0] | Spot: 10.0 (was hardcoded 5.0) |
| `air_time_variance_weight` | TUNABLE | 1.0 | [0.1, 3.0] | Spot: 1.0 (was hardcoded 0.5) |
| `foot_slip_weight` | TUNABLE | 0.5 | [0.0, 2.0] | Spot: 0.5 (was hardcoded 0.1) |
| `undesired_contacts_weight` | TUNABLE | -1.0 | [-5.0, 0.0] | Body contact penalty |
| `undesired_contacts_threshold` | TUNABLE | 1.0 | [0.1, 10.0] | Contact force threshold (N) |
| `forward_motion_weight` | TUNABLE | 3.0 | [0.0, 10.0] | Harold-specific bootstrap (Spot has none). Was 7.0 |
| `joint_pos_weight` | TUNABLE | 0.7 | [0.1, 2.0] | Spot: 0.7 |
| `joint_pos_stand_still_scale` | TUNABLE | 5.0 | [1.0, 10.0] | Spot: 5.0. 5x when standing with NO command (Spot original direction) |
| `joint_pos_velocity_threshold` | TUNABLE | 0.1 | [0.05, 0.3] | Harold-specific (Spot: 0.5) |

### Command Config (CommandCfg)

| Parameter | Category | Current | Range | Notes |
|-----------|----------|---------|-------|-------|
| `vx_min` | FROZEN | 0.15 | - | Forward velocity minimum — standing must never be optimal (Session 52 fix) |
| `vx_max` | TUNABLE | 0.3 | [0.1, 1.0] | Forward velocity maximum (m/s) |
| `vy_min` | TUNABLE | -0.15 | [-0.5, 0.0] | Lateral velocity minimum (m/s) |
| `vy_max` | TUNABLE | 0.15 | [0.0, 0.5] | Lateral velocity maximum (m/s) |
| `yaw_min` | TUNABLE | -0.30 | [-1.0, 0.0] | Yaw rate minimum (rad/s) |
| `yaw_max` | TUNABLE | 0.30 | [0.0, 1.0] | Yaw rate maximum (rad/s) |
| `zero_velocity_prob` | TUNABLE | 0.02 | [0.0, 0.2] | Standing training probability |
| `dynamic_commands` | FROZEN | False | - | Fixed command per episode — Phase 1 simplicity (Session 52 fix) |
| `command_change_interval` | TUNABLE | 10.0 | [2.0, 30.0] | Command update interval (s) |

### Termination Config (TerminationCfg)

| Parameter | Category | Current | Range | Notes |
|-----------|----------|---------|-------|-------|
| `orientation_threshold` | TUNABLE | -0.6 | [-0.8, -0.2] | Tipping threshold |
| `height_threshold` | TUNABLE | 0.0 | [0.0, 0.2] | Height termination (0=disabled) |
| `body_contact_threshold` | TUNABLE | 3.0 | [1.0, 20.0] | Body contact termination (N) |
| `elbow_pose_termination` | TUNABLE | False | [True, False] | Joint-angle termination |

### Domain Randomization (DomainRandomizationCfg)

| Parameter | Category | Current | Range | Notes |
|-----------|----------|---------|-------|-------|
| `enable_randomization` | TUNABLE | True | [True, False] | Master switch |
| `add_imu_noise` | TUNABLE | True | [True, False] | IMU noise toggle |
| `add_joint_noise` | TUNABLE | True | [True, False] | Joint sensor noise |
| `add_lin_vel_noise` | TUNABLE | True | [True, False] | Linear velocity noise |
| `randomize_friction` | TUNABLE | False | [True, False] | CAUTION: caused vx=0.005 |
| `randomize_mass` | TUNABLE | False | [True, False] | CAUTION: robot stood still |
| `add_action_noise` | TUNABLE | False | [True, False] | CAUTION: hurts learning |
| `apply_external_forces` | TUNABLE | False | [True, False] | CAUTION: breaks training |

### Env-Level Parameters (HaroldIsaacLabEnvCfg)

| Parameter | Category | Current | Range | Notes |
|-----------|----------|---------|-------|-------|
| `episode_length_s` | CONSTRAINED | 30.0 | [15.0, 60.0] | Episode duration |
| `action_scale` | CONSTRAINED | 0.5 | [0.3, 0.7] | Session 23: 0.7 was worse |
| `action_filter_beta` | CONSTRAINED | 0.2 | [0.1, 0.5] | EMA filter; 0.50 prevented walking |
| `decimation` | FROZEN | 9 | - | 180Hz sim / 20Hz policy |
| `observation_space` | FROZEN | 48 | - | Fixed observation dim |
| `action_space` | FROZEN | 12 | - | Fixed action dim (12 joints) |
| `clip_observations_value` | FROZEN | 5.0 | - | Matches deployment |

## PPO Config (`skrl_ppo_cfg.yaml`)

| Parameter | Category | Current | Range | Notes |
|-----------|----------|---------|-------|-------|
| `learning_rate` | CONSTRAINED | 1.0e-3 | [4e-4, 1e-3] | Session 54: matched Spot. Was 7e-4. 3e-4 is SANITY_FAIL |
| `rollouts` | TUNABLE | 24 | [8, 48] | Samples per update |
| `learning_epochs` | TUNABLE | 5 | [3, 10] | Passes per rollout |
| `mini_batches` | TUNABLE | 4 | [1, 64] | Session 54: matched Spot. Was 32. 4096×24/4 = 24.6k samples/mini-batch |
| `discount_factor` | TUNABLE | 0.99 | [0.95, 0.999] | Gamma |
| `lambda` | TUNABLE | 0.95 | [0.9, 0.99] | GAE tau |
| `ratio_clip` | TUNABLE | 0.2 | [0.1, 0.3] | PPO epsilon |
| `value_clip` | TUNABLE | 0.2 | [0.1, 0.3] | Value function clip |
| `grad_norm_clip` | TUNABLE | 1.0 | [0.5, 2.0] | Session 54: matched Spot. Was 0.5 |
| `entropy_loss_scale` | TUNABLE | 0.0025 | [0.001, 0.05] | Session 54: matched Spot. Was 0.01 |
| `value_loss_scale` | TUNABLE | 0.5 | [0.5, 2.0] | Session 54: matched Spot. Was 1.0 |
| `rewards_shaper_scale` | TUNABLE | 1.0 | [0.1, 2.0] | Session 54: matched Spot. Was 0.6 |
| `min_log_std` | TUNABLE | -20.0 | [-20.0, 0.0] | Session 54: matched Spot. Was -0.36 (floor std≈0.70 prevented refinement) |
| `seed` | TUNABLE | 42 | [0, 9999] | Session 54: matched Spot |
| `timesteps` | TUNABLE | 10000 | [5000, 50000] | Training duration |
| `state_preprocessor` | TUNABLE | null | [null, RunningStandardScaler] | Session 54: matched Spot. Was RunningStandardScaler |
| `value_preprocessor` | TUNABLE | null | [null, RunningStandardScaler] | Session 54: matched Spot. Was RunningStandardScaler |
| `policy_layers` | TUNABLE | [512, 256, 128] | - | Network architecture |
| `value_layers` | TUNABLE | [512, 256, 128] | - | Network architecture |

## Sim & Hardware (FROZEN - Never Change)

| Parameter | Value | Reason |
|-----------|-------|--------|
| Joint limits (shoulders) | ±0.5236 rad | Mechanical stops |
| Joint limits (thighs/calves) | ±1.5708 rad | Mechanical stops |
| Effort limit | 2.8 Nm | 95% of 2.94 Nm hardware max |
| Stiffness | 1200 | Session 21: critical for sim-to-real |
| Damping | 75 | Session 21: matches servo behavior |
| Control rate | 20 Hz policy | Matches deployment pipeline |
| Sim dt | 1/180 | Matches decimation=9 for 20Hz |
| Joint order | FL,FR,BL,BR × sh,th,ca | Matches firmware |
| Sign convention | thighs/calves inverted | Matches servo mounting |
| Static/dynamic friction | 1.0 / 1.0 | Terrain physics |
