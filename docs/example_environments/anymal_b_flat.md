# ANYmal B -- Flat Terrain Locomotion

Technical breakdown of the Isaac Lab flat-terrain velocity-tracking environment for the ANYmal B quadruped.

---

## 1. Overview

| Field | Value |
|-------|-------|
| Robot | ANYmal B (ANYbotics) |
| Gym ID | `Isaac-Velocity-Flat-Anymal-B-v0` |
| Terrain | Flat plane (no procedural generation) |
| Base config class | `AnymalBFlatEnvCfg` (inherits `AnymalBRoughEnvCfg` -> `LocomotionVelocityRoughEnvCfg`) |
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

**No height scan** -- removed for flat terrain. Total dimension: **48**.

| Term | Dim | Noise |
|------|-----|-------|
| `base_lin_vel` | 3 | Uniform [-0.1, 0.1] |
| `base_ang_vel` | 3 | Uniform [-0.2, 0.2] |
| `projected_gravity` | 3 | Uniform [-0.05, 0.05] |
| `velocity_commands` | 3 | None |
| `joint_pos` (relative) | 12 | Uniform [-0.01, 0.01] |
| `joint_vel` (relative) | 12 | Uniform [-1.5, 1.5] |
| `last_action` | 12 | None |

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
| `feet_air_time` | `mdp.feet_air_time` | **+0.5** | Threshold 0.5s; feet matching `.*FOOT` (**4x rough**) |
| `flat_orientation_l2` | `mdp.flat_orientation_l2` | **-5.0** | L2 of projected gravity error (**enabled for flat, 0.0 in rough**) |
| `lin_vel_z_l2` | `mdp.lin_vel_z_l2` | **-2.0** | Penalize vertical bounce |
| `ang_vel_xy_l2` | `mdp.ang_vel_xy_l2` | **-0.05** | Penalize roll/pitch rates |
| `dof_torques_l2` | `mdp.joint_torques_l2` | **-2.5e-5** | (**2.5x rough**) |
| `dof_acc_l2` | `mdp.joint_acc_l2` | **-2.5e-7** | Same as rough |
| `action_rate_l2` | `mdp.action_rate_l2` | **-0.01** | Smooth actions |
| `undesired_contacts` | `mdp.undesired_contacts` | **-1.0** | Thigh contacts > 1.0 N |
| `dof_pos_limits` | `mdp.joint_pos_limits` | 0.0 | Disabled |

**Key flat-vs-rough differences**: `flat_orientation_l2` activated at -5.0, `dof_torques_l2` raised to -2.5e-5, `feet_air_time` raised to +0.5.

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

**No terrain curriculum** for flat.

---

## 9. Training Config

### RSL-RL PPO

| Parameter | Value |
|-----------|-------|
| Max iterations | 300 |
| Steps per env | 24 |
| Actor network | [128, 128, 128] ELU |
| Critic network | [128, 128, 128] ELU |
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

Same hyperparameters as RSL-RL. Network: [128, 128, 128] ELU. GaussianMixin policy with `initial_log_std=0.0`, clipped to [-20, 2]. Timesteps: 7200 (= 300 iterations x 24 steps). Seed: 42.

---

## 10. Key File Paths

Relative to IsaacLab root:

```
source/isaaclab_tasks/.../locomotion/velocity/
  velocity_env_cfg.py                          # Base LocomotionVelocityRoughEnvCfg
  config/anymal_b/
    __init__.py                                # Gym registration
    rough_env_cfg.py                           # AnymalBRoughEnvCfg (parent)
    flat_env_cfg.py                            # AnymalBFlatEnvCfg
    agents/
      rsl_rl_ppo_cfg.py                        # RSL-RL PPO configs
      skrl_flat_ppo_cfg.yaml                   # skrl flat PPO config
      skrl_rough_ppo_cfg.yaml                  # skrl rough PPO config

source/isaaclab_assets/.../robots/anymal.py    # ANYMAL_B_CFG, actuator configs

source/isaaclab/.../terrains/config/rough.py   # ROUGH_TERRAINS_CFG (not used for flat)
```

---

## 11. Relevance to Harold

| Dimension | ANYmal B | Harold | Ratio |
|-----------|----------|--------|-------|
| Mass | ~30 kg | ~2 kg | 15:1 |
| Leg length | ~0.55 m | ~0.15 m | 3.7:1 |
| DOF | 12 (3/leg) | 12 (3/leg) | 1:1 |
| Actuator | 80 Nm ANYdrive (LSTM net) | Small servos | Orders of magnitude |
| Velocity cmd range | +/-1.0 m/s | Much lower (scaled to body) | ~3-5x |

**Scaling considerations for Harold:**

- **Mass randomization**: +/-5 kg is ~17% of ANYmal base mass. For Harold at ~2 kg, equivalent would be +/-0.3 kg.
- **Action scale**: 0.5 rad offset from default works for ANYmal's large joints; Harold may need different scaling given smaller servo range.
- **Velocity commands**: 1.0 m/s is moderate for ANYmal (~1.8 body-lengths/s). For Harold, 1.0 m/s would be ~6.7 body-lengths/s -- likely too fast. Harold experiments use lower thresholds (0.15 m/s range explored in EXP-775+).
- **Feet air time threshold**: 0.5 s assumes ANYmal's natural gait frequency. Harold's smaller legs cycle faster; threshold needs reduction.
- **Push velocity**: +/-0.5 m/s push is gentle for 30 kg ANYmal but would launch 2 kg Harold. Scale down proportionally.
- **Flat orientation penalty**: -5.0 weight is a strong signal for flat terrain. Harold's sim_flat_v2 baseline can adopt similar emphasis.
- **Network size**: [128, 128, 128] for flat is notably smaller than rough [512, 256, 128]. Harold's flat-terrain task may benefit from similar small networks.
- **Training duration**: Only 300 iterations for flat (vs 1500 rough). Flat terrain converges faster -- Harold autoresearch uses ~5000 iterations but on a harder problem (first-time walking).
