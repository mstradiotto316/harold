# Go1 Rough Locomotion Environment -- Technical Breakdown

## 1. Overview

| Field | Value |
|---|---|
| Robot | Unitree Go1 |
| Gym ID | `Isaac-Velocity-Rough-Unitree-Go1-v0` |
| Terrain | Procedurally generated rough terrain (6 sub-terrain types with curriculum) |
| Base class chain | `UnitreeGo1RoughEnvCfg` -> `LocomotionVelocityRoughEnvCfg` |
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

Includes height scan for terrain perception. All observations are concatenated.

| Observation | Dimensions | Noise |
|---|---|---|
| `base_lin_vel` (body-frame linear velocity) | 3 | Uniform(-0.1, 0.1) |
| `base_ang_vel` (body-frame angular velocity) | 3 | Uniform(-0.2, 0.2) |
| `projected_gravity` (gravity in body frame) | 3 | Uniform(-0.05, 0.05) |
| `velocity_commands` (target vx, vy, wz) | 3 | None |
| `joint_pos` (relative to default) | 12 | Uniform(-0.01, 0.01) |
| `joint_vel` (relative) | 12 | Uniform(-1.5, 1.5) |
| `actions` (last action) | 12 | None |
| `height_scan` | **187** | Uniform(-0.1, 0.1), clipped to [-1.0, 1.0] |

Height scanner config:
- Prim path: `{ENV_REGEX_NS}/Robot/trunk`
- Pattern: Grid 1.6m x 1.0m at 0.1m resolution = 16 x 10 = 160 points (actual dim may vary with alignment)
- Offset: (0, 0, 20.0) with ray_alignment="yaw"
- Update period: decimation x dt = 0.02 s

**Total obs dim: ~235** (48 + height_scan)

Noise corruption is enabled during training, disabled for play.

## 4. Action Space

| Field | Value |
|---|---|
| Type | Joint position targets (relative to default pose) |
| Joints | All 12 (`.*`) |
| Scale | **0.25** (reduced from base 0.5) |
| Use default offset | True (actions are offsets from default joint positions) |

Actions are 12-dimensional. Each action value is multiplied by 0.25 and added to the default joint position to produce the target position for the MLP actuator net.

## 5. Reward Structure

All reward terms with final effective weights after Go1 overrides:

| Reward Term | Function | Weight | Notes |
|---|---|---|---|
| `track_lin_vel_xy_exp` | Gaussian tracking of commanded xy vel | **+1.5** | std = sqrt(0.25) = 0.5 |
| `track_ang_vel_z_exp` | Gaussian tracking of commanded yaw vel | **+0.75** | std = sqrt(0.25) = 0.5 |
| `feet_air_time` | Reward feet being in the air | **+0.01** | threshold=0.5s, body=`.*_foot` |
| `lin_vel_z_l2` | Penalize vertical velocity | **-2.0** | Inherited from base |
| `ang_vel_xy_l2` | Penalize roll/pitch angular velocity | **-0.05** | Inherited from base |
| `dof_torques_l2` | Penalize joint torques | **-0.0002** | Increased 20x from base -1e-5 |
| `dof_acc_l2` | Penalize joint accelerations | **-2.5e-7** | Same as base |
| `action_rate_l2` | Penalize action changes | **-0.01** | Inherited from base |
| `flat_orientation_l2` | Penalize non-flat body orientation | **0.0** | Disabled (zero weight) |
| ~~`undesired_contacts`~~ | -- | **Disabled** | Set to None (was -1.0 on `*THIGH` in base) |
| ~~`dof_pos_limits`~~ | -- | weight=0.0 | Present but zero weight |

**Key differences from base**: tracking rewards boosted (1.0->1.5 and 0.5->0.75), torque penalty increased 20x (-1e-5 -> -0.0002), feet_air_time reduced (0.125->0.01), undesired contacts disabled, feet body names changed from `.*FOOT` to `.*_foot`.

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
| Max init terrain level | 5 | Curriculum starting difficulty |

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
| | velocity | **all zero** (base uses [-0.5, 0.5] for all) |
| Reset joints | position range | **(1.0, 1.0)** -- exact default pose (base uses [0.5, 1.5]) |

**Interval events:**

| Event | Value |
|---|---|
| ~~Push robot~~ | **Disabled** (set to None) |

**Terrain randomization** (implicit via procedural generation -- see section below).

## 9. Terrain Configuration

The rough env uses `ROUGH_TERRAINS_CFG` with Go1-specific overrides:

| Parameter | Value |
|---|---|
| Tile size | 8.0 x 8.0 m |
| Grid | 10 rows x 20 cols |
| Border width | 20.0 m |
| Horizontal scale | 0.1 m |
| Vertical scale | 0.005 m |
| Curriculum | **Enabled** |

### Sub-terrain types (with Go1 overrides)

| Sub-terrain | Proportion | Key Parameters | Go1 Override |
|---|---|---|---|
| `pyramid_stairs` | 20% | step height [0.05, 0.23] m, step width 0.3 m | -- |
| `pyramid_stairs_inv` | 20% | step height [0.05, 0.23] m, step width 0.3 m | -- |
| `boxes` | 20% | grid width 0.45 m, grid height [0.05, 0.2] m | **height [0.025, 0.1] m** (halved) |
| `random_rough` | 20% | noise [0.02, 0.10], step 0.02 | **noise [0.01, 0.06], step 0.01** (reduced) |
| `hf_pyramid_slope` | 10% | slope [0.0, 0.4] | -- |
| `hf_pyramid_slope_inv` | 10% | slope [0.0, 0.4] | -- |

Terrains are scaled down because Go1 is a small robot. The `boxes` grid height is halved (max 0.1 m vs 0.2 m) and `random_rough` noise is reduced.

## 10. Training Config (PPO)

### RSL-RL

| Parameter | Value |
|---|---|
| Max iterations | **1500** |
| Steps per env | 24 |
| Save interval | 50 |
| Actor hidden dims | **[512, 256, 128]** |
| Critic hidden dims | **[512, 256, 128]** |
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
| Timesteps | **36,000** (= 1500 iter x 24 rollout) |
| Network | [512, 256, 128] ELU, shared backbone |
| Policy class | GaussianMixin |
| Initial log std | 0.0 |
| Seed | 42 |
| All other PPO params | Same as RSL-RL above |

**Rough vs flat training**: 5x more iterations (1500 vs 300), larger network ([512,256,128] vs [128,128,128]).

## 11. Key File Paths

All relative to IsaacLab repository root:

```
source/isaaclab_tasks/.../locomotion/velocity/
  config/go1/
    __init__.py                  # Gym registration
    rough_env_cfg.py             # UnitreeGo1RoughEnvCfg
    flat_env_cfg.py              # UnitreeGo1FlatEnvCfg (inherits rough)
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

## 12. Relevance to Harold

| Dimension | Go1 | Harold | Ratio |
|---|---|---|---|
| Mass | ~12 kg | ~2 kg | 6x heavier |
| Leg length | ~0.20 m (thigh+calf ~0.40 m total) | ~0.15 m | ~2.7x longer |
| Effort limit | 23.7 Nm | Much lower | -- |
| Action scale | 0.25 | Needs tuning | -- |

**Key scaling considerations for Harold:**

- **Terrain scaling is critical**: Go1 already scales down boxes (max 0.1 m) and roughness (max 0.06) from the default. Harold at ~0.15 m leg length would need even more aggressive reduction -- obstacles above 0.05 m are already significant for Harold.
- **Mass randomization**: Go1 adds [-1, +3] kg to ~12 kg (8-25%). For Harold at 2 kg, equivalent would be ~[-0.17, +0.5] kg.
- **Feet air time**: 0.01 weight in rough is very low (25x lower than flat's 0.25). This suggests rough terrain training deprioritizes gait regularity in favor of traversal. Harold may need even less gait reward during rough training given its tendency toward false-positive "walking" behaviors.
- **Height scan**: The 1.6m x 1.0m scan at 0.1m resolution is sized for Go1. Harold's smaller body and step length may benefit from a tighter, higher-resolution scan window.
- **Network size**: [512, 256, 128] for rough accounts for the larger observation space (height scan adds ~187 dims). This is a reasonable starting point for Harold rough training.
- **Torque penalty**: -0.0002 is 20x the base value. Go1's MLP actuator net models real motor dynamics including friction and delay. Harold's actuator model choice affects how aggressive this penalty should be.
- **No push robot / no COM randomization**: Go1 disables both. This is a conservative choice for a small robot -- same approach may suit Harold initially.
- **Zero reset velocity**: Go1 always resets to zero velocity (base uses random [-0.5, 0.5]). This simplifies early training at the cost of robustness.
- **Curriculum**: Terrain levels progress from easy to hard. Harold would benefit from the same curriculum but with terrain parameters scaled to its body size.
- **Training length**: 1500 iterations at 4096 envs x 24 steps = ~147M env steps total. Harold autoresearch uses 5000 iterations at 16384 envs -- substantially more data.
