# ANYmal B -- Rough Terrain Locomotion

Technical breakdown of the Isaac Lab rough-terrain velocity-tracking environment for the ANYmal B quadruped.

---

## 1. Overview

| Field | Value |
|-------|-------|
| Robot | ANYmal B (ANYbotics) |
| Gym ID | `Isaac-Velocity-Rough-Anymal-B-v0` |
| Terrain | Procedurally generated rough (6 sub-terrain types with curriculum) |
| Base config class | `AnymalBRoughEnvCfg` (inherits `LocomotionVelocityRoughEnvCfg`) |
| Num envs (default) | 4096 |
| Episode length | 20.0 s |
| Sim dt | 0.005 s |
| Decimation | 4 (control at 50 Hz) |

---

## 2. Robot Specs

| Parameter | Value |
|-----------|-------|
| Mass | ~30 kg (base body) |
| DOF count | 12 (3 per leg: HAA, HFE, KFE) |
| Leg length | ~0.55 m (hip to foot) |
| Init height | 0.6 m |
| Actuator model | LSTM actuator net (`anydrive_3_lstm_jit.pt`) |
| Saturation effort | 120.0 Nm |
| Effort limit | 80.0 Nm |
| Velocity limit | 7.5 rad/s |
| Self-collisions | Enabled |
| Soft joint pos limit factor | 0.95 |

Default joint positions: HAA=0.0, front HFE=0.4, hind HFE=-0.4, front KFE=-0.8, hind KFE=0.8 rad.

---

## 3. Observation Space

**Includes height scan**. Total dimension: **235** (48 proprioceptive + 187 height scan).

| Term | Dim | Noise | Clip |
|------|-----|-------|------|
| `base_lin_vel` | 3 | Uniform [-0.1, 0.1] | -- |
| `base_ang_vel` | 3 | Uniform [-0.2, 0.2] | -- |
| `projected_gravity` | 3 | Uniform [-0.05, 0.05] | -- |
| `velocity_commands` | 3 | None | -- |
| `joint_pos` (relative) | 12 | Uniform [-0.01, 0.01] | -- |
| `joint_vel` (relative) | 12 | Uniform [-1.5, 1.5] | -- |
| `last_action` | 12 | None | -- |
| `height_scan` | 187 | Uniform [-0.1, 0.1] | [-1.0, 1.0] |

Height scanner: RayCaster from base at +20m Z offset, yaw-aligned grid pattern, resolution=0.1m, size=[1.6m x 1.0m] = 17x11 = 187 rays.

Observation corruption enabled during training (`enable_corruption = True`), disabled during play.

---

## 4. Action Space

| Parameter | Value |
|-----------|-------|
| Type | Joint position targets |
| Dimension | 12 |
| Scale | 0.5 |
| Offset | Default joint positions (use_default_offset=True) |
| Joint pattern | `.*` (all 12 joints) |

Actions are scaled by 0.5 and added to the default standing pose. The LSTM actuator net converts position targets to torques.

---

## 5. Reward Structure

All rewards are computed per-step. Positive = task incentives, negative = penalties.

| Reward Term | Function | Weight | Notes |
|-------------|----------|--------|-------|
| `track_lin_vel_xy_exp` | `mdp.track_lin_vel_xy_exp` | **+1.0** | std=0.5; exp(-error^2/0.25) |
| `track_ang_vel_z_exp` | `mdp.track_ang_vel_z_exp` | **+0.5** | std=0.5; exp(-error^2/0.25) |
| `feet_air_time` | `mdp.feet_air_time` | **+0.125** | Threshold 0.5s; feet matching `.*FOOT` |
| `flat_orientation_l2` | `mdp.flat_orientation_l2` | **0.0** | Disabled for rough (terrain is uneven) |
| `lin_vel_z_l2` | `mdp.lin_vel_z_l2` | **-2.0** | Penalize vertical bounce |
| `ang_vel_xy_l2` | `mdp.ang_vel_xy_l2` | **-0.05** | Penalize roll/pitch rates |
| `dof_torques_l2` | `mdp.joint_torques_l2` | **-1.0e-5** | Energy penalty |
| `dof_acc_l2` | `mdp.joint_acc_l2` | **-2.5e-7** | Smooth joint accelerations |
| `action_rate_l2` | `mdp.action_rate_l2` | **-0.01** | Smooth actions |
| `undesired_contacts` | `mdp.undesired_contacts` | **-1.0** | Thigh contacts > 1.0 N |
| `dof_pos_limits` | `mdp.joint_pos_limits` | **0.0** | Disabled |

**Key rough-vs-flat differences**: `flat_orientation_l2` disabled (robot must tilt on slopes), lower `feet_air_time` weight (0.125 vs 0.5), lower `dof_torques_l2` penalty (1e-5 vs 2.5e-5).

---

## 6. Command Ranges

| Parameter | Value |
|-----------|-------|
| lin_vel_x | [-1.0, 1.0] m/s |
| lin_vel_y | [-1.0, 1.0] m/s |
| ang_vel_z | [-1.0, 1.0] rad/s |
| heading | [-pi, pi] rad |
| Heading command | Enabled (stiffness=0.5) |
| Resampling interval | 10.0 s |
| Standing envs fraction | 2% (`rel_standing_envs=0.02`) |
| Heading envs fraction | 100% (`rel_heading_envs=1.0`) |

---

## 7. Key Thresholds

| Threshold | Value | Context |
|-----------|-------|---------|
| Feet air time target | 0.5 s | `feet_air_time` reward |
| Undesired contact force | 1.0 N | Thigh contact penalty |
| Illegal contact force | 1.0 N | Base contact -> termination |
| Max init terrain level | 5 | Curriculum starting difficulty |
| Height scan clip | [-1.0, 1.0] m | Observation clipping |
| Max depenetration vel | 1.0 m/s | Physics solver |

---

## 8. Domain Randomization

**Startup (once per env):**

| Event | Range |
|-------|-------|
| Rigid body material | static_friction=0.8, dynamic_friction=0.6, restitution=0.0 (fixed, 64 buckets) |
| Base mass randomization | +/- 5.0 kg added to base |
| Base CoM offset | x: +/-0.05m, y: +/-0.05m, z: +/-0.01m |

**Reset (each episode):**

| Event | Range |
|-------|-------|
| Base pose | x: +/-0.5m, y: +/-0.5m, yaw: +/-pi |
| Base velocity | x/y/z: +/-0.5 m/s, roll/pitch/yaw: +/-0.5 rad/s |
| Joint positions | Scaled 0.5x--1.5x from default |
| External force/torque | Force: 0.0, Torque: 0.0 (effectively disabled) |

**Interval (during episode):**

| Event | Interval | Range |
|-------|----------|-------|
| Push robot | 10--15 s | vx: +/-0.5 m/s, vy: +/-0.5 m/s |

**Terrain curriculum** enabled: terrains generated with increasing difficulty. 10 rows x 20 cols grid, each tile 8m x 8m.

---

## 9. Terrain Configuration (ROUGH_TERRAINS_CFG)

Grid: 10 rows x 20 cols, tile size 8.0m x 8.0m, border 20.0m, horizontal_scale=0.1m, vertical_scale=0.005m.

| Sub-terrain | Proportion | Key Parameters |
|-------------|-----------|----------------|
| `pyramid_stairs` | 20% | step_height 0.05--0.23m, step_width 0.3m, platform 3.0m |
| `pyramid_stairs_inv` (inverted) | 20% | step_height 0.05--0.23m, step_width 0.3m, platform 3.0m |
| `boxes` (random grid) | 20% | grid_width 0.45m, grid_height 0.05--0.2m, platform 2.0m |
| `random_rough` | 20% | noise 0.02--0.10m, noise_step 0.02m |
| `hf_pyramid_slope` | 10% | slope 0.0--0.4, platform 2.0m |
| `hf_pyramid_slope_inv` | 10% | slope 0.0--0.4, platform 2.0m |

---

## 10. Training Config

### RSL-RL PPO

| Parameter | Value |
|-----------|-------|
| Max iterations | 1500 |
| Steps per env | 24 |
| Actor network | [512, 256, 128] ELU |
| Critic network | [512, 256, 128] ELU |
| Init noise std | 1.0 |
| Learning rate | 1e-3 (adaptive KL schedule) |
| Desired KL | 0.01 |
| Gamma | 0.99 |
| Lambda (GAE) | 0.95 |
| Clip param | 0.2 |
| Entropy coef | 0.005 |
| Value loss coef | 1.0 |
| Mini-batches | 4 |
| Learning epochs | 5 |
| Max grad norm | 1.0 |
| Obs normalization | Disabled (actor + critic) |

### skrl PPO

Same hyperparameters as RSL-RL. Network: [512, 256, 128] ELU. GaussianMixin policy with `initial_log_std=0.0`, clipped to [-20, 2]. Timesteps: 36000 (= 1500 iterations x 24 steps). Seed: 42.

**Rough uses a 5x larger network and 5x more training iterations than flat.**

---

## 11. Key File Paths

Relative to IsaacLab root:

```
source/isaaclab_tasks/.../locomotion/velocity/
  velocity_env_cfg.py                          # Base LocomotionVelocityRoughEnvCfg
  config/anymal_b/
    __init__.py                                # Gym registration
    rough_env_cfg.py                           # AnymalBRoughEnvCfg
    flat_env_cfg.py                            # AnymalBFlatEnvCfg (inherits rough)
    agents/
      rsl_rl_ppo_cfg.py                        # RSL-RL PPO configs
      skrl_flat_ppo_cfg.yaml                   # skrl flat PPO config
      skrl_rough_ppo_cfg.yaml                  # skrl rough PPO config

source/isaaclab_assets/.../robots/anymal.py    # ANYMAL_B_CFG, actuator configs

source/isaaclab/.../terrains/config/rough.py   # ROUGH_TERRAINS_CFG
```

---

## 12. Relevance to Harold

| Dimension | ANYmal B | Harold | Ratio |
|-----------|----------|--------|-------|
| Mass | ~30 kg | ~2 kg | 15:1 |
| Leg length | ~0.55 m | ~0.15 m | 3.7:1 |
| DOF | 12 (3/leg) | 12 (3/leg) | 1:1 |
| Actuator | 80 Nm ANYdrive (LSTM net) | Small servos | Orders of magnitude |
| Velocity cmd range | +/-1.0 m/s | Much lower (scaled to body) | ~3-5x |

**Scaling considerations for Harold:**

- **Height scan**: 187-dim exteroceptive observation is only needed for rough terrain. Harold's current work is flat-terrain-first, so this is not immediately relevant. When Harold moves to rough, the scan grid dimensions (1.6m x 1.0m) would need scaling to Harold's footprint (~0.3m x 0.2m).
- **Network size**: [512, 256, 128] for rough is large. Harold's flat task uses smaller networks; rough terrain would likely need capacity increase but probably not to ANYmal's level given Harold's simpler dynamics.
- **Terrain curriculum**: ANYmal starts at level 5 of 10 difficulty rows. For Harold, terrain difficulty ranges (step heights 0.05--0.23m) are larger than Harold's entire leg length (0.15m). Terrain generation would need complete reparameterization.
- **Mass randomization**: +/-5 kg is ~17% of ANYmal. For Harold, equivalent would be +/-0.3 kg.
- **Velocity commands**: 1.0 m/s is moderate for ANYmal (~1.8 body-lengths/s). For Harold, 1.0 m/s would be ~6.7 body-lengths/s. Harold experiments use 0.15 m/s velocity thresholds (EXP-775+).
- **Feet air time**: 0.5 s threshold assumes ANYmal's natural stride frequency (~2 Hz). Harold's smaller legs cycle faster; threshold needs scaling down.
- **Push disturbance**: +/-0.5 m/s every 10--15 s is gentle for 30 kg. For 2 kg Harold this is a massive perturbation. Scale impulse by mass ratio.
- **Training duration**: 1500 iterations for rough terrain. Harold at 5000 iterations is already 3x longer but solving a different (arguably harder) problem -- learning to walk at all rather than refining an established gait on varied terrain.
- **Orientation penalty**: Disabled (0.0) for rough because the robot must tilt on slopes. This is a useful design pattern: separate flat and rough reward tuning.
