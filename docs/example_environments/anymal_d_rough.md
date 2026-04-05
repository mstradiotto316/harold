# ANYmal D Rough Locomotion Environment

Technical breakdown of the Isaac Lab ANYmal D rough-terrain velocity-tracking environment.

Source: local IsaacLab checkout + [GitHub main branch](https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_d).

---

## 1. Overview

| Field | Value |
|---|---|
| Robot | ANYmal D (ANYbotics) |
| Gym ID | `Isaac-Velocity-Rough-Anymal-D-v0` |
| Play Gym ID | `Isaac-Velocity-Rough-Anymal-D-Play-v0` |
| Terrain | Procedural rough terrain (6 sub-terrain types with curriculum) |
| Base class chain | `AnymalDRoughEnvCfg` -> `LocomotionVelocityRoughEnvCfg` -> `ManagerBasedRLEnvCfg` |

The rough env is the canonical ANYmal D locomotion task. It uses a terrain generator with curriculum progression and includes a height scanner for terrain perception.

---

## 2. Robot Specs

| Parameter | Value |
|---|---|
| Mass | ~50 kg (nominal) |
| DOF | 12 (4 legs x 3 joints: HAA, HFE, KFE) |
| Leg length | ~0.55 m (upper + lower) |
| Init height | 0.6 m |
| Actuator model | LSTM actuator net (ANYdrive 3.0, re-used from ANYmal C) |
| Saturation effort | 120.0 Nm |
| Effort limit | 80.0 Nm |
| Velocity limit | 7.5 rad/s |
| PD gains | Kp=40.0, Kd=5.0 (DC motor fallback) |
| Self-collisions | Enabled |
| Solver iters | Position: 4, Velocity: 0 |
| Soft joint pos limit factor | 0.95 |

Default joint positions:

| Joint group | Init pos (rad) |
|---|---|
| `.*HAA` | 0.0 |
| `.*F_HFE` (front hip) | 0.4 |
| `.*H_HFE` (hind hip) | -0.4 |
| `.*F_KFE` (front knee) | -0.8 |
| `.*H_KFE` (hind knee) | 0.8 |

---

## 3. Observation Space

**Total dimension: 235** (48 proprioceptive + 187 height scan).

| Observation | Dim | Noise | Notes |
|---|---|---|---|
| `base_lin_vel` | 3 | U(-0.1, 0.1) | Base linear velocity in body frame |
| `base_ang_vel` | 3 | U(-0.2, 0.2) | Base angular velocity in body frame |
| `projected_gravity` | 3 | U(-0.05, 0.05) | Gravity vector in body frame |
| `velocity_commands` | 3 | None | Commanded (vx, vy, wz) |
| `joint_pos` | 12 | U(-0.01, 0.01) | Relative joint positions |
| `joint_vel` | 12 | U(-1.5, 1.5) | Relative joint velocities |
| `actions` | 12 | None | Previous actions |
| `height_scan` | 187 | U(-0.1, 0.1) | Ray-cast height map, clipped to (-1.0, 1.0) |

Height scanner configuration:
- Grid resolution: 0.1 m
- Grid size: 1.6 m x 1.0 m (17 x 11 = 187 points)
- Offset: (0, 0, 20) m above base (rays cast downward)
- Alignment: yaw-aligned (rotates with robot heading)
- Update period: decimation x dt = 0.02 s (50 Hz)

Observation settings:
- `enable_corruption = True` (additive uniform noise during training)
- `concatenate_terms = True`

---

## 4. Action Space

| Parameter | Value |
|---|---|
| Action type | Joint position targets |
| Dimension | 12 |
| Scale | 0.5 |
| Offset | Default joint positions (`use_default_offset=True`) |
| Joint names | `[".*"]` (all 12 joints) |

Actions are position offsets scaled by 0.5 and added to default standing pose.

---

## 5. Reward Structure

| Reward term | Weight | Function | Notes |
|---|---|---|---|
| `track_lin_vel_xy_exp` | **+1.0** | exp(-error^2/0.25) | Primary task: track xy velocity |
| `track_ang_vel_z_exp` | **+0.5** | exp(-error^2/0.25) | Track yaw rate |
| `feet_air_time` | **+0.125** | air time bonus | Threshold=0.5s, feet=`.*FOOT` |
| `lin_vel_z_l2` | **-2.0** | L2 vertical velocity | Penalize bouncing |
| `ang_vel_xy_l2` | **-0.05** | L2 roll/pitch rate | Penalize body rotation |
| `dof_torques_l2` | **-1.0e-5** | L2 joint torques | Energy efficiency |
| `dof_acc_l2` | **-2.5e-7** | L2 joint accelerations | Smooth motion |
| `action_rate_l2` | **-0.01** | L2 action rate | Smooth actions |
| `undesired_contacts` | **-1.0** | thigh contacts > 1.0N | Bodies: `.*THIGH` |
| `flat_orientation_l2` | **0.0** | L2 orientation error | **Disabled** for rough terrain (robot must tilt on slopes) |
| `dof_pos_limits` | **0.0** | joint limit violation | Disabled |

Note: The rough env intentionally disables `flat_orientation_l2` (weight=0.0) since the robot needs to adapt its body orientation to uneven terrain.

---

## 6. Command Ranges

| Parameter | Value |
|---|---|
| `lin_vel_x` | [-1.0, 1.0] m/s |
| `lin_vel_y` | [-1.0, 1.0] m/s |
| `ang_vel_z` | [-1.0, 1.0] rad/s |
| `heading` | [-pi, pi] rad |
| Resampling interval | 10.0 s (fixed) |
| Standing envs fraction | 0.02 (2%) |
| Heading envs fraction | 1.0 (100%) |
| Heading command | True |
| Heading control stiffness | 0.5 |

---

## 7. Key Thresholds

| Parameter | Value |
|---|---|
| Sim dt | 0.005 s (200 Hz physics) |
| Decimation | 4 (policy at 50 Hz) |
| Episode length | 20.0 s |
| Feet air time threshold | 0.5 s |
| Undesired contact threshold | 1.0 N |
| Illegal contact threshold (termination) | 1.0 N on base |
| Terrain friction (static) | 1.0 |
| Terrain friction (dynamic) | 1.0 |
| Max init terrain level | 5 |
| Num envs (training) | 4096 |
| Env spacing | 2.5 m |

---

## 8. Domain Randomization

### Startup (once per env)

| Event | Parameters |
|---|---|
| Physics material | static_friction=0.8, dynamic_friction=0.6, restitution=0.0 (64 buckets, **not** randomized -- range is degenerate) |
| Base mass | Add U(-5.0, +5.0) kg to base body |
| Base COM offset | x: U(-0.05, 0.05), y: U(-0.05, 0.05), z: U(-0.01, 0.01) m |

### Reset (every episode reset)

| Event | Parameters |
|---|---|
| External force/torque | force=(0,0), torque=(0,0) -- effectively disabled |
| Root state | pos: U(-0.5,0.5) xy, yaw: U(-pi,pi); vel: U(-0.5,0.5) all axes |
| Joint reset | position scale U(0.5, 1.5) of default, velocity=0 |

### Interval (during episode)

| Event | Parameters |
|---|---|
| Push robot | Every 10-15s, set velocity x,y: U(-0.5, 0.5) m/s |

### Terrain Curriculum

The rough env uses a terrain curriculum (`terrain_levels_vel`) that progresses robots to harder terrain as they improve. Terrain difficulty increases across rows.

---

## 9. Terrain Configuration

`ROUGH_TERRAINS_CFG` -- procedurally generated terrain grid:

| Parameter | Value |
|---|---|
| Tile size | 8.0 x 8.0 m |
| Border width | 20.0 m |
| Grid layout | 10 rows x 20 columns (200 tiles) |
| Horizontal scale | 0.1 m |
| Vertical scale | 0.005 m |
| Slope threshold | 0.75 |

### Sub-terrain types

| Type | Proportion | Key parameters |
|---|---|---|
| `pyramid_stairs` | 20% | Step height 0.05-0.23 m, width 0.3 m, platform 3.0 m |
| `pyramid_stairs_inv` (inverted) | 20% | Step height 0.05-0.23 m, width 0.3 m, platform 3.0 m |
| `boxes` (random grid) | 20% | Grid width 0.45 m, height 0.05-0.2 m, platform 2.0 m |
| `random_rough` | 20% | Noise range 0.02-0.10, step 0.02 |
| `hf_pyramid_slope` | 10% | Slope range 0.0-0.4, platform 2.0 m |
| `hf_pyramid_slope_inv` | 10% | Slope range 0.0-0.4, platform 2.0 m |

---

## 10. Training Config

### RSL-RL PPO (Rough)

| Parameter | Value |
|---|---|
| Network (actor) | [512, 256, 128] MLP, ELU |
| Network (critic) | [512, 256, 128] MLP, ELU |
| Init noise std | 1.0 |
| Obs normalization | False (actor + critic) |
| Num steps per env | 24 |
| Max iterations | 1500 |
| Learning rate | 1.0e-3 (adaptive KL schedule) |
| Desired KL | 0.01 |
| Clip param | 0.2 |
| Entropy coef | 0.005 |
| Value loss coef | 1.0 |
| Gamma | 0.99 |
| Lambda (GAE) | 0.95 |
| Mini batches | 4 |
| Learning epochs | 5 |
| Max grad norm | 1.0 |
| Save interval | 50 |

### skrl PPO (Rough)

| Parameter | Value |
|---|---|
| Network | [512, 256, 128] MLP, ELU (shared structure) |
| Rollouts | 24 |
| Timesteps | 36000 (= 1500 iterations x 24 rollout steps) |
| Learning rate | 1.0e-3 (KLAdaptiveLR, threshold=0.01) |
| Other PPO params | Same as RSL-RL above |
| Seed | 42 |

Note: The rough env uses **larger networks** ([512, 256, 128]) than flat ([128, 128, 128]) to handle the additional height scan input and terrain complexity. Training runs 5x longer (1500 vs 300 iterations).

---

## 11. Key File Paths

Relative to IsaacLab repo root:

| File | Purpose |
|---|---|
| `source/isaaclab_tasks/.../locomotion/velocity/config/anymal_d/rough_env_cfg.py` | Rough env config |
| `source/isaaclab_tasks/.../locomotion/velocity/config/anymal_d/flat_env_cfg.py` | Flat env config (child) |
| `source/isaaclab_tasks/.../locomotion/velocity/velocity_env_cfg.py` | Base env config (rewards, obs, events) |
| `source/isaaclab_tasks/.../locomotion/velocity/config/anymal_d/__init__.py` | Gym registration |
| `source/isaaclab_tasks/.../locomotion/velocity/config/anymal_d/agents/rsl_rl_ppo_cfg.py` | RSL-RL PPO config |
| `source/isaaclab_tasks/.../locomotion/velocity/config/anymal_d/agents/skrl_rough_ppo_cfg.yaml` | skrl PPO config |
| `source/isaaclab_assets/isaaclab_assets/robots/anymal.py` | Robot articulation config |
| `source/isaaclab/isaaclab/terrains/config/rough.py` | Rough terrain generator config |

---

## 12. Relevance to Harold

ANYmal D and Harold are fundamentally different-scale robots. Direct parameter transfer will not work.

| Dimension | ANYmal D | Harold | Ratio |
|---|---|---|---|
| Mass | ~50 kg | ~2 kg | 25x |
| Leg length | ~0.55 m | ~0.15 m | 3.7x |
| Actuator torque limit | 80 Nm | ~0.5 Nm | 160x |
| Standing height | 0.6 m | ~0.12 m | 5x |
| Velocity command range | +/-1.0 m/s | TBD (scaled) | -- |

### What transfers well
- **Reward structure**: The reward terms and their relative signs are a solid template. The exponential velocity tracking with std=sqrt(0.25) is well-tested.
- **Observation set**: The proprioceptive obs (48-dim) are directly applicable. Height scan is relevant if Harold ever does rough terrain.
- **PPO hyperparameters**: lr=1e-3, gamma=0.99, lam=0.95, clip=0.2, entropy=0.005 are good defaults.
- **Episode structure**: 20s episodes, 50 Hz policy, 200 Hz physics.
- **Terrain types**: The sub-terrain mixture (stairs, boxes, rough, slopes) is a proven curriculum for quadrupeds.

### What needs scaling for Harold
- **Action scale**: 0.5 rad offset is large for Harold's small servos. Harold likely needs smaller scale.
- **Reward weights**: Torque penalty (-1.0e-5) is tuned for 80 Nm actuators; Harold's ~0.5 Nm actuators need a much larger penalty coefficient to produce equivalent regularization pressure.
- **Mass randomization**: +/-5 kg is 10% of ANYmal but would be 250% of Harold. Scale to +/-0.1 kg or similar.
- **COM randomization**: +/-5 cm is small for ANYmal but enormous relative to Harold's body. Scale down proportionally.
- **Push velocity**: +/-0.5 m/s push is gentle for 50 kg but violent for 2 kg. Scale down significantly.
- **Feet air time threshold**: 0.5s assumes ANYmal's slow gait cadence. Harold's legs are much shorter and will cycle faster; threshold should be ~0.1-0.2s.
- **Height scan grid**: 1.6 x 1.0 m grid at 0.1 m resolution covers a huge area relative to Harold. Would need to be shrunk to ~0.4 x 0.3 m for Harold's scale.
- **Terrain obstacles**: Stair step heights of 0.05-0.23 m are comparable to Harold's entire leg length. Terrain generation params would need significant downscaling.
- **Network size**: [512, 256, 128] is likely overkill for Harold's simpler flat-terrain task, but reasonable if height scan is included.
