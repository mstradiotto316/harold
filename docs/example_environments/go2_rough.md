# Go2 Rough Locomotion Environment

Technical breakdown of the Isaac Lab rough-terrain velocity-tracking task for the Unitree Go2 quadruped.

Source: `Isaac-Velocity-Rough-Unitree-Go2-v0` (Isaac Lab upstream)

---

## 1. Overview

| Field | Value |
|---|---|
| Robot | Unitree Go2 |
| Gym ID | `Isaac-Velocity-Rough-Unitree-Go2-v0` |
| Terrain | Procedurally generated rough terrain (6 sub-terrain types) |
| Task | Velocity tracking (lin_vel_x, lin_vel_y, ang_vel_z) with height scan |
| Base class chain | `UnitreeGo2RoughEnvCfg` -> `LocomotionVelocityRoughEnvCfg` |
| Env framework | Manager-based (`ManagerBasedRLEnv`) |

The rough env adds a height scanner, terrain curriculum, and terrain-adapted reward weights. The Go2-specific overrides scale down terrain difficulty, adjust reward weights, and modify domain randomization for the smaller robot.

---

## 2. Robot Specs

Identical to flat -- same `UNITREE_GO2_CFG` robot config.

| Parameter | Value |
|---|---|
| Mass | ~12 kg (base body) |
| Total DOF | 12 (3 per leg x 4 legs) |
| Joint names | `*_hip_joint`, `*_thigh_joint`, `*_calf_joint` |
| Leg length | ~0.30 m (thigh + calf) |
| Standing height | 0.40 m (init spawn height) |
| Actuator type | DC Motor (`DCMotorCfg`) |
| Effort limit | 23.5 Nm |
| Velocity limit | 30.0 rad/s |
| PD stiffness (Kp) | 25.0 |
| PD damping (Kd) | 0.5 |
| Friction | 0.0 (actuator friction) |
| Self-collisions | Disabled |

### Initial Joint Positions

| Joint group | Value (rad) |
|---|---|
| `*L_hip_joint` | 0.1 |
| `*R_hip_joint` | -0.1 |
| `F[L,R]_thigh_joint` | 0.8 |
| `R[L,R]_thigh_joint` | 1.0 |
| `*_calf_joint` | -1.5 |

---

## 3. Observation Space

The rough env includes a height scan, significantly expanding the observation vector.

| Term | Dim | Noise | Notes |
|---|---|---|---|
| `base_lin_vel` | 3 | Uniform(-0.1, 0.1) | Body-frame linear velocity |
| `base_ang_vel` | 3 | Uniform(-0.2, 0.2) | Body-frame angular velocity |
| `projected_gravity` | 3 | Uniform(-0.05, 0.05) | Gravity projected into body frame |
| `velocity_commands` | 3 | None | Commanded (vx, vy, wz) |
| `joint_pos` | 12 | Uniform(-0.01, 0.01) | Relative to default positions |
| `joint_vel` | 12 | Uniform(-1.5, 1.5) | Relative joint velocities |
| `actions` | 12 | None | Previous action |
| `height_scan` | 187 | Uniform(-0.1, 0.1) | Raycaster height map, clipped to [-1.0, 1.0] |

**Total observation dim: 235**

### Height Scanner Config

| Parameter | Value |
|---|---|
| Prim path | `{ENV_REGEX_NS}/Robot/base` |
| Pattern | Grid, resolution=0.1 m, size=[1.6, 1.0] m |
| Offset | (0, 0, 20) -- cast rays downward from above |
| Alignment | Yaw-only (ignores roll/pitch) |
| Update period | 0.02 s (decimation x dt) |
| Points | ~16 x 10 + edges = 187 rays |

---

## 4. Action Space

| Parameter | Value |
|---|---|
| Type | Joint position targets (`JointPositionActionCfg`) |
| Dimension | 12 |
| Scale | **0.25** (Go2 override; base class uses 0.5) |
| Offset | Default joint positions (`use_default_offset=True`) |
| Control mode | Position targets -> DC Motor PD controller |

The action is: `target = default_pos + action * 0.25`. The DC motor then applies PD control with Kp=25, Kd=0.5 to track this target.

---

## 5. Reward Structure

Weights shown are the **final effective values** after Go2 rough overrides on the base class.

| Reward term | Weight | Function | Notes |
|---|---|---|---|
| `track_lin_vel_xy_exp` | **+1.5** | exp(-error^2 / 0.25) | Primary task reward (Go2 override from 1.0) |
| `track_ang_vel_z_exp` | **+0.75** | exp(-error^2 / 0.25) | Yaw tracking (Go2 override from 0.5) |
| `feet_air_time` | **+0.01** | sum((air_time - 0.5) * first_contact) | Encourage stepping (low weight for rough) |
| `flat_orientation_l2` | **0.0** | roll^2 + pitch^2 | **Disabled** (weight=0.0 in rough, enabled only in flat) |
| `lin_vel_z_l2` | **-2.0** | vz^2 | Penalize vertical bouncing |
| `ang_vel_xy_l2` | **-0.05** | (wx^2 + wy^2) | Penalize roll/pitch angular velocity |
| `dof_torques_l2` | **-0.0002** | sum(torques^2) | Go2 override (base: -1e-5) |
| `dof_acc_l2` | **-2.5e-7** | sum(accel^2) | Joint acceleration penalty |
| `action_rate_l2` | **-0.01** | sum((a_t - a_{t-1})^2) | Smooth actions |
| ~~`undesired_contacts`~~ | -- | -- | **Removed** (set to `None` by Go2) |
| ~~`dof_pos_limits`~~ | 0.0 | -- | Defined but zero weight |

### Reward differences: Rough vs Flat

| Term | Rough | Flat |
|---|---|---|
| `feet_air_time` | 0.01 | **0.25** (25x larger) |
| `flat_orientation_l2` | 0.0 | **-2.5** (enabled) |

In rough terrain, the orientation penalty is disabled (terrain will naturally tilt the robot), and the stepping reward is minimal (the terrain itself forces foot lifting). The flat env needs explicit incentives for both.

---

## 6. Command Ranges

| Parameter | Value |
|---|---|
| `lin_vel_x` | [-1.0, 1.0] m/s |
| `lin_vel_y` | [-1.0, 1.0] m/s |
| `ang_vel_z` | [-1.0, 1.0] rad/s |
| `heading` | [-pi, pi] rad |
| Heading command | Enabled (`heading_command=True`) |
| Heading stiffness | 0.5 |
| Resampling interval | 10.0 s (fixed) |
| Standing envs fraction | 0.02 (2% get zero command) |
| Heading envs fraction | 1.0 (all envs use heading) |

---

## 7. Key Thresholds

| Threshold | Value | Usage |
|---|---|---|
| Feet air time threshold | 0.5 s | Reward target step duration |
| Velocity gate for air time reward | 0.1 m/s | No air-time reward below this |
| Episode length | 20.0 s | Max episode duration |
| Base contact termination | 1.0 N | Terminate if `base` touches ground |
| Soft joint pos limit factor | 0.9 | 90% of joint limits enforced |
| Max init terrain level | 5 | Curriculum starting difficulty |

---

## 8. Domain Randomization (Events)

### Startup (once per env)

| Event | Parameters |
|---|---|
| Physics material | Static friction: 0.8, Dynamic friction: 0.6, Restitution: 0.0 (64 buckets) |
| Add base mass | **[-1.0, 3.0] kg** (Go2 override; base: [-5.0, 5.0]) |
| ~~Base CoM offset~~ | **Removed** (`base_com = None`) |

### Reset (per episode)

| Event | Parameters |
|---|---|
| External force/torque on base | Force: [0, 0], Torque: [0, 0] (effectively disabled, but body_names set to `base`) |
| Reset base pose | x: [-0.5, 0.5], y: [-0.5, 0.5], yaw: [-pi, pi]; **zero velocity** (Go2 override; base randomizes velocity) |
| Reset joints | Position scale: **[1.0, 1.0]** (Go2 override = exact default pos; base: [0.5, 1.5]) |

### Interval

| Event | Parameters |
|---|---|
| ~~Push robot~~ | **Removed** (`push_robot = None`) |

### Notable Go2 simplifications vs base class

The Go2 config is surprisingly conservative with randomization:
- **No push disturbances** during episodes
- **No CoM randomization**
- **No initial velocity randomization** (zeroed out)
- **No joint position randomization at reset** (scale 1.0 = exactly default)
- Mass randomization range narrowed from +/-5 kg to [-1, +3] kg
- External force/torque effectively zeroed out

---

## 9. Terrain Configuration

### Base terrain (ROUGH_TERRAINS_CFG)

| Parameter | Value |
|---|---|
| Tile size | 8.0 x 8.0 m |
| Grid | 10 rows x 20 cols |
| Border width | 20.0 m |
| Horizontal scale | 0.1 m |
| Vertical scale | 0.005 m |
| Slope threshold | 0.75 |
| Curriculum | Enabled (via terrain_levels) |

### Sub-terrain types and Go2 overrides

| Sub-terrain | Proportion | Base params | Go2 override |
|---|---|---|---|
| `pyramid_stairs` | 20% | step_height: [0.05, 0.23] m, step_width: 0.3 m | -- (unchanged) |
| `pyramid_stairs_inv` | 20% | step_height: [0.05, 0.23] m | -- (unchanged) |
| `boxes` | 20% | grid_height: [0.05, 0.2] m | **grid_height: [0.025, 0.1] m** (halved) |
| `random_rough` | 20% | noise_range: [0.02, 0.10], step: 0.02 | **noise_range: [0.01, 0.06], step: 0.01** (reduced) |
| `hf_pyramid_slope` | 10% | slope_range: [0.0, 0.4] | -- (unchanged) |
| `hf_pyramid_slope_inv` | 10% | slope_range: [0.0, 0.4] | -- (unchanged) |

The Go2 overrides scale down `boxes` grid height (max 0.1 m vs 0.2 m) and `random_rough` noise (max 0.06 vs 0.10) because the Go2 is a smaller robot with shorter legs. Stairs and slopes are left at default difficulty.

---

## 10. Simulation Parameters

| Parameter | Value |
|---|---|
| Sim dt | 0.005 s (200 Hz physics) |
| Decimation | 4 (policy runs at 50 Hz) |
| Policy dt | 0.02 s |
| Render interval | 4 (matches decimation) |
| Num envs (train) | 4096 |
| Env spacing | 2.5 m |
| Ground friction | static=1.0, dynamic=1.0 (terrain material) |
| Contact sensor | All bodies (`.*`), history_length=3, track_air_time=True |
| Height scanner update | 0.02 s |
| Contact forces update | 0.005 s (every physics step) |

---

## 11. Training Config (PPO)

### RSL-RL

| Parameter | Value |
|---|---|
| Actor network | **[512, 256, 128]** (larger for rough; flat uses [128, 128, 128]) |
| Critic network | **[512, 256, 128]** |
| Activation | ELU |
| Init noise std | 1.0 |
| Obs normalization | Disabled (actor and critic) |
| Max iterations | **1500** (5x longer than flat's 300) |
| Steps per env | 24 |
| Mini batches | 4 |
| Learning epochs | 5 |
| Learning rate | 1e-3 |
| LR schedule | Adaptive (KL-based) |
| Desired KL | 0.01 |
| Gamma | 0.99 |
| Lambda (GAE) | 0.95 |
| Clip param | 0.2 |
| Entropy coef | 0.01 |
| Value loss coef | 1.0 |
| Max grad norm | 1.0 |
| Save interval | 50 iterations |

### skrl (YAML config)

| Parameter | Value |
|---|---|
| Seed | 42 |
| Policy class | GaussianMixin |
| Value class | DeterministicMixin |
| Network layers | **[512, 256, 128]** |
| Activations | ELU |
| Rollouts | 24 |
| Timesteps | **36000** (= 1500 iterations x 24 rollout steps) |
| Learning rate | 1e-3 |
| LR scheduler | KLAdaptiveLR (threshold 0.01) |
| Clip actions | False |
| Log std range | [-20, 2] |
| Initial log std | 0.0 |

### Why the larger network?

The rough env observation space is 235-dim (vs 48-dim flat) due to the 187-dim height scan. The [512, 256, 128] network has substantially more capacity to process the terrain information and learn terrain-adaptive gaits. This is standard Isaac Lab practice: bigger obs -> bigger network.

---

## 12. Terminations

| Termination | Condition |
|---|---|
| Time out | Episode exceeds 20.0 s |
| Base contact | `base` body contact force > 1.0 N |

Note: `base` body name is a Go2 override. The base class uses generic detection; Go2 specifies exactly the `base` link.

---

## 13. Curriculum

| Term | Function |
|---|---|
| `terrain_levels` | `mdp.terrain_levels_vel` -- advances terrain difficulty based on velocity tracking performance |

The terrain generator runs with `curriculum=True`, meaning rows represent increasing difficulty. Envs are promoted/demoted across terrain rows based on how well they track commanded velocities.

---

## 14. Key File Paths

All paths relative to IsaacLab root:

```
source/isaaclab_tasks/.../locomotion/velocity/
  config/go2/
    __init__.py                    # Gym registration (4 env IDs)
    rough_env_cfg.py               # UnitreeGo2RoughEnvCfg (main config)
    flat_env_cfg.py                # UnitreeGo2FlatEnvCfg (inherits rough)
    agents/
      rsl_rl_ppo_cfg.py            # RSL-RL PPO configs (rough + flat)
      skrl_rough_ppo_cfg.yaml      # skrl rough config
      skrl_flat_ppo_cfg.yaml       # skrl flat config
  velocity_env_cfg.py              # LocomotionVelocityRoughEnvCfg (base class)
  mdp/
    rewards.py                     # feet_air_time, track_lin_vel_xy_exp, etc.

source/isaaclab_assets/.../robots/
  unitree.py                       # UNITREE_GO2_CFG (DC Motor, 12 DOF)

source/isaaclab/isaaclab/terrains/config/
  rough.py                         # ROUGH_TERRAINS_CFG (6 sub-terrain types)
```

---

## 15. Relevance to Harold

### Size comparison

| Property | Go2 | Harold | Ratio |
|---|---|---|---|
| Body mass | ~12 kg | ~2 kg | 6x |
| Leg length | ~0.30 m | ~0.15 m | 2x |
| Standing height | 0.40 m | ~0.20 m | 2x |
| Effort limit | 23.5 Nm | ~1-2 Nm | ~15x |

### Key takeaways for Harold

1. **Terrain scaling**: Go2 already scales down terrain difficulty from the base class (box height halved, rough noise reduced by ~40%). Harold at half the leg length would need even more aggressive scaling -- box heights capped at ~0.05 m, noise range ~[0.005, 0.03].

2. **Height scan**: The 1.6 x 1.0 m grid at 0.1 m resolution is oversized for Harold. A proportionally scaled grid would be ~0.8 x 0.5 m, or the resolution could be kept at 0.1 m with a smaller footprint.

3. **Network capacity**: The [512, 256, 128] rough network is justified by the 235-dim obs space. If Harold uses a similar height scan, the same network size is reasonable. Without height scan, [128, 128, 128] is sufficient.

4. **Training duration**: 1500 iterations for rough terrain. Harold's rough terrain training would likely need even more iterations due to the added difficulty of being a lighter/smaller robot on relatively larger obstacles.

5. **Reward scaling for rough**: The orientation penalty is disabled (weight=0.0) in rough terrain because terrain naturally tilts the robot. The stepping reward is very low (0.01). This philosophy should carry over to Harold rough -- let the terrain shape the gait.

6. **Conservative randomization**: Go2 rough is surprisingly gentle with DR -- no pushes, no CoM shifts, no velocity randomization at reset, narrow mass range. This suggests that for a small robot like Harold, starting with zero DR on rough terrain is reasonable. Add complexity only after the base task is solved.

7. **Curriculum strategy**: Terrain curriculum is critical for rough -- it prevents the robot from facing impossible terrain early in training. Harold would benefit from the same approach, starting on easy terrain and progressing.

8. **Action scale consistency**: Both flat and rough use the same 0.25 action scale. This makes sense -- the action range is a robot property, not a terrain property.

9. **Termination on base contact**: Using only the `base` body (not thighs) for termination is lenient. Go2 also removes `undesired_contacts` for thighs entirely. This suggests letting the reward structure (not hard termination) handle body contact policy for small robots.

10. **Step duration**: 0.5 s feet air time threshold implies a ~1 Hz gait (0.5 s air + 0.5 s ground per foot). Go2's natural trot is roughly this speed. Harold's smaller legs may have a natural gait closer to 2-3 Hz, suggesting a threshold of 0.2-0.3 s.
