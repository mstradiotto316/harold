# Parameter Registry

Machine-readable registry of all tunable parameters for the Harold autoresearch system.
Used by `scripts/autoresearch.py` to validate proposed changes before applying them.

Config source: `harold_isaac_lab/.../tasks/manager_based/harold_flat/flat_env_cfg.py`

## Categories

- **FROZEN**: Never change without explicit human approval. Sim-to-real critical.
- **CONSTRAINED**: Tunable within documented range only.
- **TUNABLE**: Free to modify. Pure learning signal or training-only parameters.

## Env Config (`flat_env_cfg.py`)

### Reward Weights (HaroldRewardsCfg)

Uses Spot reward functions (`spot_mdp`) with Harold-adapted parameters.
Negative weights are penalties. Each weight is inside a `RewardTermCfg(weight=X.X)`.

| Parameter | Category | Current | Range | Notes |
|-----------|----------|---------|-------|-------|
| `air_time_weight` | TUNABLE | 5.0 | [0.5, 15.0] | Spot: 5.0. Foot air time reward |
| `base_angular_velocity_weight` | TUNABLE | 5.0 | [1.0, 15.0] | Spot: 5.0. Yaw tracking |
| `base_linear_velocity_weight` | TUNABLE | 5.0 | [1.0, 15.0] | Spot: 5.0. Primary velocity tracking |
| `foot_clearance_weight` | TUNABLE | 0.5 | [0.0, 3.0] | Spot: 0.5. Foot height during swing |
| `gait_weight` | TUNABLE | 10.0 | [1.0, 25.0] | Spot: 10.0. Diagonal trot pattern |
| `action_smoothness_weight` | TUNABLE | -1.0 | [-5.0, 0.0] | Spot: -1.0. Action diff penalty |
| `air_time_variance_weight` | TUNABLE | -1.0 | [-5.0, 0.0] | Spot: -1.0. Foot timing symmetry |
| `base_motion_weight` | TUNABLE | -2.0 | [-10.0, 0.0] | Spot: -2.0. Vertical + roll/pitch motion |
| `base_orientation_weight` | TUNABLE | -3.0 | [-10.0, 0.0] | Spot: -3.0. Tilt penalty |
| `foot_slip_weight` | TUNABLE | -0.5 | [-3.0, 0.0] | Spot: -0.5. Foot sliding penalty |
| `joint_acc_weight` | TUNABLE | -1.0e-4 | [-0.01, 0.0] | Spot: -1e-4. Joint acceleration penalty |
| `joint_pos_weight` | TUNABLE | -0.7 | [-3.0, 0.0] | Spot: -0.7. Joint deviation from default |
| `joint_torques_weight` | TUNABLE | -5.0e-4 | [-0.01, 0.0] | Spot: -5e-4. Torque penalty |
| `joint_vel_weight` | TUNABLE | -1.0e-2 | [-0.1, 0.0] | Spot: -1e-2. Joint velocity penalty |

### Command Config (HaroldCommandsCfg)

Velocity command ranges. Uses `UniformVelocityCommandCfg.Ranges` tuples.

| Parameter | Category | Current | Range | Notes |
|-----------|----------|---------|-------|-------|
| `lin_vel_x_min` | TUNABLE | -0.5 | [-1.0, 0.0] | Backward velocity limit (m/s) |
| `lin_vel_x_max` | TUNABLE | 1.0 | [0.3, 2.0] | Forward velocity limit (m/s) |
| `lin_vel_y_min` | TUNABLE | -0.5 | [-1.0, 0.0] | Lateral velocity min (m/s) |
| `lin_vel_y_max` | TUNABLE | 0.5 | [0.0, 1.0] | Lateral velocity max (m/s) |
| `ang_vel_z_min` | TUNABLE | -1.0 | [-2.0, 0.0] | Yaw rate min (rad/s) |
| `ang_vel_z_max` | TUNABLE | 1.0 | [0.0, 2.0] | Yaw rate max (rad/s) |

### Env-Level Parameters (HaroldFlatEnvCfg)

| Parameter | Category | Current | Range | Notes |
|-----------|----------|---------|-------|-------|
| `episode_length_s` | CONSTRAINED | 20.0 | [10.0, 40.0] | Episode duration (Spot: 20.0) |
| `decimation` | FROZEN | 10 | - | 500Hz physics / 50Hz policy |
| `observation_space` | FROZEN | 48 | - | 3+3+3+3+12+12+12 = 48D |
| `action_space` | FROZEN | 12 | - | 12 joints |

## PPO Config (`skrl_ppo_cfg.yaml`)

Identical format to Spot reference. YAML key-value pairs.

| Parameter | Category | Current | Range | Notes |
|-----------|----------|---------|-------|-------|
| `learning_rate` | CONSTRAINED | 1.0e-3 | [4e-4, 1e-3] | Spot reference: 1e-3 |
| `rollouts` | TUNABLE | 24 | [8, 48] | Samples per update |
| `learning_epochs` | TUNABLE | 5 | [3, 10] | Passes per rollout |
| `mini_batches` | TUNABLE | 4 | [1, 16] | Spot: 4. 2048x24/4 = 12.3k samples/mini-batch |
| `discount_factor` | TUNABLE | 0.99 | [0.95, 0.999] | Gamma |
| `lambda` | TUNABLE | 0.95 | [0.9, 0.99] | GAE tau |
| `ratio_clip` | TUNABLE | 0.2 | [0.1, 0.3] | PPO epsilon |
| `value_clip` | TUNABLE | 0.2 | [0.1, 0.3] | Value function clip |
| `grad_norm_clip` | TUNABLE | 1.0 | [0.5, 2.0] | Gradient clipping |
| `entropy_loss_scale` | TUNABLE | 0.0025 | [0.001, 0.05] | Exploration encouragement |
| `value_loss_scale` | TUNABLE | 0.5 | [0.5, 2.0] | Value function loss coefficient |
| `rewards_shaper_scale` | TUNABLE | 1.0 | [0.1, 2.0] | Reward scaling |
| `min_log_std` | TUNABLE | -20.0 | [-20.0, 0.0] | Policy std floor |
| `seed` | TUNABLE | 42 | [0, 9999] | Random seed |
| `timesteps` | TUNABLE | 480000 | [100000, 1000000] | Total training timesteps |
| `policy_layers` | TUNABLE | [512, 256, 128] | - | Policy network architecture |
| `value_layers` | TUNABLE | [512, 256, 128] | - | Value network architecture |

## Sim & Hardware (FROZEN - Never Change)

| Parameter | Value | Reason |
|-----------|-------|--------|
| Joint limits (shoulders) | +/-0.5236 rad | Mechanical stops |
| Joint limits (thighs/calves) | +/-1.5708 rad | Mechanical stops |
| Effort limit | 2.8 Nm | 95% of 2.94 Nm hardware max |
| Stiffness (Kp) | 40.0 | Manager-based ImplicitActuatorCfg |
| Damping (Kd) | 0.5 | Manager-based ImplicitActuatorCfg |
| Control rate | 50 Hz (decimation=10, dt=0.002) | Manager-based: 500Hz physics, 50Hz policy |
| Joint order | FL,FR,BL,BR x sh,th,ca | Matches firmware |
| Sign convention | thighs/calves inverted | Matches servo mounting |
| Static/dynamic friction | 1.0 / 1.0 | Terrain physics |
| Action scale | 0.2 | JointPositionActionCfg scale |
