# Go2 Flat Locomotion Environment

Technical breakdown of the Isaac Lab flat-terrain velocity-tracking task for the Unitree Go2 quadruped.

Source: `Isaac-Velocity-Flat-Unitree-Go2-v0` (Isaac Lab upstream)

---

## 1. Overview

| Field | Value |
|---|---|
| Robot | Unitree Go2 |
| Gym ID | `Isaac-Velocity-Flat-Unitree-Go2-v0` |
| Terrain | Flat plane (no procedural terrain) |
| Task | Velocity tracking (lin_vel_x, lin_vel_y, ang_vel_z) |
| Base class chain | `UnitreeGo2FlatEnvCfg` -> `UnitreeGo2RoughEnvCfg` -> `LocomotionVelocityRoughEnvCfg` |
| Env framework | Manager-based (`ManagerBasedRLEnv`) |

The flat env inherits from the rough env and overrides: terrain to plane, removes height scan from observations, removes terrain curriculum, and adjusts two reward weights.

---

## 2. Robot Specs

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

The flat env removes the height scan from the rough env. All remaining terms are concatenated into a single vector.

| Term | Dim | Noise | Notes |
|---|---|---|---|
| `base_lin_vel` | 3 | Uniform(-0.1, 0.1) | Body-frame linear velocity |
| `base_ang_vel` | 3 | Uniform(-0.2, 0.2) | Body-frame angular velocity |
| `projected_gravity` | 3 | Uniform(-0.05, 0.05) | Gravity projected into body frame |
| `velocity_commands` | 3 | None | Commanded (vx, vy, wz) |
| `joint_pos` | 12 | Uniform(-0.01, 0.01) | Relative to default positions |
| `joint_vel` | 12 | Uniform(-1.5, 1.5) | Relative joint velocities |
| `actions` | 12 | None | Previous action |
| ~~`height_scan`~~ | ~~187~~ | -- | **Removed** (set to `None`) |

**Total observation dim: 48** (vs 235 in rough env)

Observation corruption (`enable_corruption`) is enabled during training (additive uniform noise on each term). Disabled for play/eval configs.

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

Weights shown are the **final effective values** after Go2 flat overrides.

| Reward term | Weight | Function | Notes |
|---|---|---|---|
| `track_lin_vel_xy_exp` | **+1.5** | exp(-error^2 / 0.25) | Primary task reward (Go2 override from 1.0) |
| `track_ang_vel_z_exp` | **+0.75** | exp(-error^2 / 0.25) | Yaw tracking (Go2 override from 0.5) |
| `feet_air_time` | **+0.25** | sum((air_time - 0.5) * first_contact) | Encourage stepping (flat override from 0.01) |
| `flat_orientation_l2` | **-2.5** | roll^2 + pitch^2 | Keep body level (flat override from 0.0) |
| `lin_vel_z_l2` | **-2.0** | vz^2 | Penalize vertical bouncing |
| `ang_vel_xy_l2` | **-0.05** | (wx^2 + wy^2) | Penalize roll/pitch angular velocity |
| `dof_torques_l2` | **-0.0002** | sum(torques^2) | Go2 override (base: -1e-5) |
| `dof_acc_l2` | **-2.5e-7** | sum(accel^2) | Joint acceleration penalty |
| `action_rate_l2` | **-0.01** | sum((a_t - a_{t-1})^2) | Smooth actions |
| ~~`undesired_contacts`~~ | -- | -- | **Removed** (set to `None` in Go2 rough) |
| ~~`dof_pos_limits`~~ | 0.0 | -- | Defined but zero weight |

### Key reward design notes

- The `feet_air_time` reward uses a 0.5s threshold and `.*_foot` body names (Go2 override). It is zero when commanded velocity norm < 0.1 m/s.
- `flat_orientation_l2` is -2.5 in flat (bumped up from 0.0 in rough) -- flat terrain emphasizes level body.
- `feet_air_time` is 0.25 in flat (bumped up from 0.01 in rough) -- encourages more active stepping on easy terrain.
- `undesired_contacts` (thigh contacts) is removed for Go2.

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
| Base contact termination | 1.0 N | Terminate if base touches ground |
| Soft joint pos limit factor | 0.9 | 90% of joint limits enforced |

---

## 8. Domain Randomization (Events)

### Startup (once per env)

| Event | Parameters |
|---|---|
| Physics material | Static friction: 0.8, Dynamic friction: 0.6, Restitution: 0.0 (64 buckets) |
| Add base mass | **[-1.0, 3.0] kg** (Go2 override; base class: [-5.0, 5.0]) |
| ~~Base CoM offset~~ | **Removed** (set to `None` in Go2 rough) |

### Reset (per episode)

| Event | Parameters |
|---|---|
| External force/torque on base | Force: [0, 0], Torque: [0, 0] (effectively no force) |
| Reset base pose | x: [-0.5, 0.5], y: [-0.5, 0.5], yaw: [-pi, pi]; **zero velocity** (Go2 override) |
| Reset joints | Position scale: [1.0, 1.0] (Go2 override = exact default; base: [0.5, 1.5]) |

### Interval

| Event | Parameters |
|---|---|
| ~~Push robot~~ | **Removed** (set to `None` in Go2 rough) |

---

## 9. Simulation Parameters

| Parameter | Value |
|---|---|
| Sim dt | 0.005 s (200 Hz physics) |
| Decimation | 4 (policy runs at 50 Hz) |
| Policy dt | 0.02 s |
| Render interval | 4 (matches decimation) |
| Num envs (train) | 4096 |
| Env spacing | 2.5 m |
| Ground friction | static=1.0, dynamic=1.0 (terrain material) |
| Terrain type | Plane |
| Terrain curriculum | **Disabled** |
| Height scanner | **Disabled** |

---

## 10. Training Config (PPO)

### RSL-RL

| Parameter | Value |
|---|---|
| Actor network | [128, 128, 128] (flat override; rough uses [512, 256, 128]) |
| Critic network | [128, 128, 128] |
| Activation | ELU |
| Init noise std | 1.0 |
| Obs normalization | Disabled (actor and critic) |
| Max iterations | **300** (flat override; rough: 1500) |
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
| Network layers | [128, 128, 128] |
| Activations | ELU |
| Rollouts | 24 |
| Timesteps | 7200 (= 300 iterations x 24 rollout steps) |
| Learning rate | 1e-3 |
| LR scheduler | KLAdaptiveLR (threshold 0.01) |
| Clip actions | False |
| Log std range | [-20, 2] |
| Initial log std | 0.0 |

---

## 11. Key File Paths

All paths relative to IsaacLab root:

```
source/isaaclab_tasks/.../locomotion/velocity/
  config/go2/
    __init__.py                    # Gym registration
    flat_env_cfg.py                # UnitreeGo2FlatEnvCfg (inherits rough)
    rough_env_cfg.py               # UnitreeGo2RoughEnvCfg (inherits base)
    agents/
      rsl_rl_ppo_cfg.py            # RSL-RL PPO configs
      skrl_flat_ppo_cfg.yaml       # skrl flat config
      skrl_rough_ppo_cfg.yaml      # skrl rough config
  velocity_env_cfg.py              # LocomotionVelocityRoughEnvCfg (base class)
  mdp/
    rewards.py                     # Reward function implementations

source/isaaclab_assets/.../robots/
  unitree.py                       # UNITREE_GO2_CFG robot definition

source/isaaclab/isaaclab/terrains/config/
  rough.py                         # ROUGH_TERRAINS_CFG (not used in flat)
```

---

## 12. Relevance to Harold

### Size comparison

| Property | Go2 | Harold | Ratio |
|---|---|---|---|
| Body mass | ~12 kg | ~2 kg | 6x |
| Leg length | ~0.30 m | ~0.15 m | 2x |
| Standing height | 0.40 m | ~0.20 m | 2x |
| Effort limit | 23.5 Nm | ~1-2 Nm | ~15x |

### Key takeaways for Harold

1. **Action scale**: Go2 uses 0.25 (already reduced from the 0.5 base class default). Harold may need even smaller scaling given its tiny servos.

2. **Reward magnitudes**: Go2's `dof_torques_l2` weight is -0.0002 -- much larger than the base -1e-5. This was increased for Go2 specifically. Harold's torques are ~15x smaller, so the torque penalty weight needs careful scaling.

3. **Mass randomization**: Go2 adds [-1.0, +3.0] kg to a ~12 kg body (8-25% perturbation). For Harold at ~2 kg, proportional randomization would be [-0.16, +0.5] kg.

4. **Velocity commands**: [-1.0, 1.0] m/s is aggressive for a 12 kg robot. For Harold's shorter legs, proportionally scaling by leg length ratio gives ~[-0.5, 0.5] m/s.

5. **Network size**: Flat uses [128, 128, 128] -- already small. Harold's 48-dim obs space (assuming same structure minus height scan) fits this well.

6. **Feet air time threshold**: 0.5 s is tuned for Go2's gait frequency. Harold's lighter legs swing faster; a lower threshold (0.3-0.4 s) may be more appropriate.

7. **No push disturbance**: Go2 flat disables push events. This simplifies training but reduces robustness.

8. **Episode length**: 20 s episodes with 50 Hz policy = 1000 steps/episode max.

9. **Training length**: Only 300 iterations for flat (vs 1500 rough). Flat terrain converges fast -- Harold flat experiments should also be short.

10. **Observation noise**: The noise magnitudes are absolute, not relative to the robot's scale. Joint vel noise of +/-1.5 rad/s may be proportionally larger for Harold's smaller/faster joints -- consider reducing.
