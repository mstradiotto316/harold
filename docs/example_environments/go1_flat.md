# Go1 Flat Locomotion Environment -- Technical Breakdown

## 1. Overview

| Field | Value |
|---|---|
| Robot | Unitree Go1 |
| Gym ID | `Isaac-Velocity-Flat-Unitree-Go1-v0` |
| Terrain | Flat plane (no procedural terrain) |
| Base class chain | `UnitreeGo1FlatEnvCfg` -> `UnitreeGo1RoughEnvCfg` -> `LocomotionVelocityRoughEnvCfg` |
| Num envs (default) | 4096 |
| Episode length | 20.0 s |
| Sim dt | 0.005 s |
| Decimation | 4 (policy at 50 Hz) |

## 2. Robot Specs

| Spec | Value |
|---|---|
| Mass | ~12 kg |
| DOF | 12 (3 per leg x 4 legs: hip, thigh, calf) |
| Joint names | `*_hip_joint`, `*_thigh_joint`, `*_calf_joint` |
| Upper leg (thigh) | ~0.20 m |
| Lower leg (calf) | ~0.20 m |
| Init height | 0.4 m above ground |
| Actuator model | **MLP ActuatorNet** (`ActuatorNetMLPCfg`) -- learned from real hardware |
| Effort limit | 23.7 Nm |
| Velocity limit | 30.0 rad/s |
| Soft joint pos limit factor | 0.9 |

Default joint positions (rad):

| Joint group | Default |
|---|---|
| `*L_hip_joint` | 0.1 |
| `*R_hip_joint` | -0.1 |
| `F[L,R]_thigh_joint` | 0.8 |
| `R[L,R]_thigh_joint` | 1.0 |
| `*_calf_joint` | -1.5 |

## 3. Observation Space

The flat env removes the height scan from the rough env. All observations are concatenated.

| Observation | Dimensions | Noise |
|---|---|---|
| `base_lin_vel` (body-frame linear velocity) | 3 | Uniform(-0.1, 0.1) |
| `base_ang_vel` (body-frame angular velocity) | 3 | Uniform(-0.2, 0.2) |
| `projected_gravity` (gravity in body frame) | 3 | Uniform(-0.05, 0.05) |
| `velocity_commands` (target vx, vy, wz) | 3 | None |
| `joint_pos` (relative to default) | 12 | Uniform(-0.01, 0.01) |
| `joint_vel` (relative) | 12 | Uniform(-1.5, 1.5) |
| `actions` (last action) | 12 | None |
| ~~`height_scan`~~ | **Removed** | -- |

**Total obs dim: 48** (3+3+3+3+12+12+12, no height scan)

Noise corruption is enabled during training (`enable_corruption = True`), disabled for play.

## 4. Action Space

| Field | Value |
|---|---|
| Type | Joint position targets (relative to default pose) |
| Joints | All 12 (`.*`) |
| Scale | **0.25** (reduced from base 0.5) |
| Use default offset | True (actions are offsets from default joint positions) |

Actions are 12-dimensional. Each action value is multiplied by 0.25 and added to the default joint position to produce the target position for the MLP actuator net.

## 5. Reward Structure

Rewards are inherited from rough, then overridden for flat. Final effective weights:

| Reward Term | Function | Weight | Notes |
|---|---|---|---|
| `track_lin_vel_xy_exp` | Gaussian tracking of commanded xy vel | **+1.5** | std = sqrt(0.25) = 0.5 |
| `track_ang_vel_z_exp` | Gaussian tracking of commanded yaw vel | **+0.75** | std = sqrt(0.25) = 0.5 |
| `feet_air_time` | Reward feet being in the air | **+0.25** | threshold=0.5s, body=`.*_foot` |
| `flat_orientation_l2` | Penalize non-flat body orientation | **-2.5** | Enabled for flat only |
| `lin_vel_z_l2` | Penalize vertical velocity | **-2.0** | Inherited from base |
| `ang_vel_xy_l2` | Penalize roll/pitch angular velocity | **-0.05** | Inherited from base |
| `dof_torques_l2` | Penalize joint torques | **-0.0002** | Reduced from base -1e-5 |
| `dof_acc_l2` | Penalize joint accelerations | **-2.5e-7** | Same as base |
| `action_rate_l2` | Penalize action changes | **-0.01** | Inherited from base |
| ~~`undesired_contacts`~~ | -- | **Disabled** | Set to None in Go1 rough |
| ~~`dof_pos_limits`~~ | -- | weight=0.0 | Present but zero weight |

**Key difference from rough**: `flat_orientation_l2` is activated at -2.5 (zero in rough) and `feet_air_time` is boosted to 0.25 (vs 0.01 in rough).

## 6. Command Ranges

| Command | Range | Unit |
|---|---|---|
| `lin_vel_x` | [-1.0, 1.0] | m/s |
| `lin_vel_y` | [-1.0, 1.0] | m/s |
| `ang_vel_z` | [-1.0, 1.0] | rad/s |
| `heading` | [-pi, pi] | rad |

| Parameter | Value |
|---|---|
| Resampling interval | 10.0 s |
| Standing envs fraction | 0.02 (2%) |
| Heading envs fraction | 1.0 (all) |
| Heading command | Enabled |
| Heading control stiffness | 0.5 |

## 7. Key Thresholds

| Threshold | Value | Context |
|---|---|---|
| Feet air time threshold | 0.5 s | Reward: feet_air_time |
| Base contact termination | 1.0 N | Illegal contact on `trunk` |
| Soft joint limit factor | 0.9 | 90% of URDF limits |

## 8. Domain Randomization

Events inherited from base, then modified for Go1:

**Startup events:**

| Event | Parameter | Value |
|---|---|---|
| Physics material | static_friction | 0.8 |
| | dynamic_friction | 0.6 |
| | restitution | 0.0 |
| | num_buckets | 64 |
| Add base mass | body | `trunk` |
| | mass range | **[-1.0, 3.0] kg** (reduced from base [-5.0, 5.0]) |
| ~~Base COM randomization~~ | -- | **Disabled** (set to None) |

**Reset events:**

| Event | Parameter | Value |
|---|---|---|
| External force/torque | body | `trunk` |
| | force/torque range | (0.0, 0.0) -- effectively zero |
| Reset base pose | x, y | [-0.5, 0.5] m |
| | yaw | [-3.14, 3.14] rad |
| | velocity | all zero (no random init velocity) |
| Reset joints | position range | **(1.0, 1.0)** -- exact default pose |

**Interval events:**

| Event | Value |
|---|---|
| ~~Push robot~~ | **Disabled** (set to None) |

## 9. Training Config (PPO)

### RSL-RL

| Parameter | Value |
|---|---|
| Max iterations | **300** |
| Steps per env | 24 |
| Actor hidden dims | **[128, 128, 128]** |
| Critic hidden dims | **[128, 128, 128]** |
| Activation | ELU |
| Learning rate | 1e-3 |
| Schedule | Adaptive (KL-based) |
| Desired KL | 0.01 |
| Gamma | 0.99 |
| Lambda (GAE) | 0.95 |
| Clip param | 0.2 |
| Entropy coef | 0.01 |
| Value loss coef | 1.0 |
| Mini batches | 4 |
| Learning epochs | 5 |
| Max grad norm | 1.0 |

### skrl

| Parameter | Value |
|---|---|
| Timesteps | **7,200** (= 300 iter x 24 rollout) |
| Network | [128, 128, 128] ELU, shared backbone |
| Policy class | GaussianMixin |
| Initial log std | 0.0 |
| Seed | 42 |
| All other PPO params | Same as RSL-RL above |

## 10. Key File Paths

All relative to IsaacLab repository root:

```
source/isaaclab_tasks/.../locomotion/velocity/
  config/go1/
    __init__.py                  # Gym registration
    rough_env_cfg.py             # UnitreeGo1RoughEnvCfg (parent)
    flat_env_cfg.py              # UnitreeGo1FlatEnvCfg
    agents/
      rsl_rl_ppo_cfg.py          # RSL-RL PPO config
      skrl_flat_ppo_cfg.yaml     # skrl flat PPO config
      skrl_rough_ppo_cfg.yaml    # skrl rough PPO config
  velocity_env_cfg.py            # LocomotionVelocityRoughEnvCfg (base)

source/isaaclab_assets/isaaclab_assets/robots/
  unitree.py                     # UNITREE_GO1_CFG, GO1_ACTUATOR_CFG

source/isaaclab/isaaclab/terrains/config/
  rough.py                       # ROUGH_TERRAINS_CFG
```

## 11. Relevance to Harold

| Dimension | Go1 | Harold | Ratio |
|---|---|---|---|
| Mass | ~12 kg | ~2 kg | 6x heavier |
| Leg length | ~0.20 m (thigh+calf ~0.40 m total) | ~0.15 m | ~2.7x longer |
| Effort limit | 23.7 Nm | Much lower | -- |
| Action scale | 0.25 | Needs tuning | -- |

**Key scaling considerations for Harold:**

- **Mass randomization**: Go1 adds [-1, +3] kg to a 12 kg body (8-25% of body mass). For Harold at 2 kg, equivalent perturbation would be ~[-0.17, +0.5] kg.
- **Action scale**: Go1 uses 0.25 (half of the base 0.5). Harold's smaller joints may need even smaller scale.
- **Velocity commands**: +/-1.0 m/s is aggressive for a 12 kg robot; Harold may need tighter ranges given shorter legs and less torque.
- **Feet air time threshold**: 0.5 s assumes Go1 gait timing. Harold's lighter mass and shorter legs produce faster natural gaits; a lower threshold (0.2-0.3 s) may be appropriate.
- **Flat orientation penalty**: Weight of -2.5 is strong. Harold's higher center-of-mass-to-footprint ratio may need different tuning.
- **No height scan**: Flat env drops it entirely -- same approach works for Harold flat training.
- **MLP actuator net**: Go1 uses a learned actuator model. Harold likely uses implicit or DC motor actuators, which changes the sim-to-real gap characteristics.
- **Network size**: [128, 128, 128] for flat is smaller than rough [512, 256, 128]. This is a good baseline for Harold flat experiments.
