# Isaac Lab -- Unitree A1 Flat Locomotion Environment

Technical reference extracted from source files (Isaac Lab, commit ~2026-03).

---

## 1. Overview

| Field | Value |
|-------|-------|
| Robot | Unitree A1 (quadruped) |
| Gym ID | `Isaac-Velocity-Flat-Unitree-A1-v0` |
| Task | Velocity tracking on flat terrain |
| Terrain | `plane` (no procedural generation, no curriculum) |
| Base class chain | `LocomotionVelocityRoughEnvCfg` -> `UnitreeA1RoughEnvCfg` -> `UnitreeA1FlatEnvCfg` |

## 2. Robot Specs

| Parameter | Value |
|-----------|-------|
| Mass | ~12 kg (real robot spec) |
| DOF | 12 (3 per leg: hip, thigh, calf) |
| Joint names | `{FL,FR,RL,RR}_{hip,thigh,calf}_joint` |
| Actuator model | `DCMotorCfg` |
| Effort limit | 33.5 Nm |
| Saturation effort | 33.5 Nm |
| Velocity limit | 21.0 rad/s |
| PD gains | Kp = 25.0, Kd = 0.5 |
| Init height | 0.42 m |
| Init joint pos | hip L/R = +/-0.1, front thigh = 0.8, rear thigh = 1.0, calf = -1.5 (rad) |
| Soft joint limit factor | 0.9 |

## 3. Observation Space (Flat)

Height scan is **removed** for the flat variant. Observation corruption is enabled during training (additive uniform noise).

| Term | Dim | Noise range |
|------|-----|-------------|
| `base_lin_vel` | 3 | [-0.1, 0.1] |
| `base_ang_vel` | 3 | [-0.2, 0.2] |
| `projected_gravity` | 3 | [-0.05, 0.05] |
| `velocity_commands` | 3 | none |
| `joint_pos_rel` | 12 | [-0.01, 0.01] |
| `joint_vel_rel` | 12 | [-1.5, 1.5] |
| `last_action` | 12 | none |
| **Total** | **48** | |

## 4. Action Space

| Parameter | Value |
|-----------|-------|
| Type | Joint position targets (offsets from default pose) |
| Dim | 12 |
| Scale | **0.25** (rough_env_cfg override; base class default is 0.5) |
| Use default offset | True (actions are relative to init joint positions) |

Control frequency: sim dt = 0.005 s, decimation = 4, so policy runs at **50 Hz**.

## 5. Reward Structure

Effective weights after the full inheritance chain (`base` -> `rough_a1` -> `flat_a1`):

| Reward term | Function | Weight | Notes |
|-------------|----------|--------|-------|
| `track_lin_vel_xy_exp` | exp(-error^2 / 0.25) | **+1.5** | Primary task reward (rough_a1 override from 1.0) |
| `track_ang_vel_z_exp` | exp(-error^2 / 0.25) | **+0.75** | Yaw tracking (rough_a1 override from 0.5) |
| `feet_air_time` | air time bonus for `.*_foot` | **+0.25** | Flat override (rough_a1 had 0.01, base had 0.125). Threshold = 0.5 s |
| `flat_orientation_l2` | L2 on projected gravity deviation | **-2.5** | Flat override (base had 0.0) -- penalizes tilt |
| `lin_vel_z_l2` | L2 on vertical velocity | **-2.0** | Inherited from base |
| `ang_vel_xy_l2` | L2 on roll/pitch angular vel | **-0.05** | Inherited from base |
| `dof_torques_l2` | L2 on joint torques | **-0.0002** | rough_a1 override (base had -1e-5) |
| `dof_acc_l2` | L2 on joint accelerations | **-2.5e-7** | Inherited from base (same in rough_a1) |
| `action_rate_l2` | L2 on action changes | **-0.01** | Inherited from base |
| `undesired_contacts` | -- | **disabled** | Set to `None` by rough_a1 (base penalized thigh contacts) |
| `dof_pos_limits` | -- | 0.0 | Inherited, effectively disabled |

## 6. Command Ranges

| Parameter | Value |
|-----------|-------|
| `lin_vel_x` | [-1.0, 1.0] m/s |
| `lin_vel_y` | [-1.0, 1.0] m/s |
| `ang_vel_z` | [-1.0, 1.0] rad/s |
| `heading` | [-pi, pi] rad |
| Heading command | True (heading_control_stiffness = 0.5) |
| Resampling interval | 10.0 s |
| Rel standing envs | 2% |
| Rel heading envs | 100% |

## 7. Key Thresholds

| Threshold | Value | Context |
|-----------|-------|---------|
| Feet air time threshold | 0.5 s | Minimum air time to earn `feet_air_time` bonus |
| Illegal contact force | 1.0 N | Terminates if trunk contacts ground above this |
| Episode length | 20.0 s | Maximum episode duration |
| Soft joint limit factor | 0.9 | Joints limited to 90% of physical range |

## 8. Domain Randomization

### Startup events (applied once)
| Event | Parameter | A1 Value |
|-------|-----------|----------|
| Rigid body material | static/dynamic friction | 0.8 / 0.6 (64 buckets) |
| Add base mass | mass range on `trunk` | **[-1.0, +3.0] kg** (rough_a1 override; base was [-5, 5]) |
| Base COM shift | -- | **disabled** (`None` in rough_a1) |

### Reset events (each episode reset)
| Event | Parameter | A1 Value |
|-------|-----------|----------|
| External force/torque on `trunk` | force/torque | (0, 0) -- effectively zero |
| Reset base pose | x, y, yaw | [-0.5, 0.5] m, [-pi, pi] rad |
| Reset base velocity | all axes | **0.0** (rough_a1 override; base randomized to +/-0.5) |
| Reset joint positions | scale range | **(1.0, 1.0)** -- exact default pose (rough_a1 override; base was [0.5, 1.5]) |

### Interval events
| Event | Parameter | A1 Value |
|-------|-----------|----------|
| Push robot | -- | **disabled** (`None` in rough_a1) |

Note: For the **play** variant (`UnitreeA1FlatEnvCfg_PLAY`), observation corruption and external forces are also disabled.

## 9. Training Config

### skrl PPO (flat)
| Parameter | Value |
|-----------|-------|
| Network | MLP [128, 128, 128] (shared architecture, ELU) |
| Separate policy/value | False |
| Rollout steps | 24 |
| Learning epochs | 5 |
| Mini batches | 4 |
| Discount (gamma) | 0.99 |
| GAE lambda | 0.95 |
| Learning rate | 1e-3 (KL-adaptive, threshold 0.01) |
| Clip ratio | 0.2 |
| Value clip | 0.2 |
| Entropy coeff | 0.01 |
| Grad norm clip | 1.0 |
| Total timesteps | 7200 (rollout steps, not env steps) |
| Seed | 42 |

### rsl_rl PPO (flat)
| Parameter | Value |
|-----------|-------|
| Network | MLP [128, 128, 128] (actor + critic, ELU) |
| Max iterations | 300 |
| Steps per env | 24 |
| Init noise std | 1.0 |
| Obs normalization | False (actor and critic) |
| LR schedule | adaptive (desired_kl = 0.01) |
| Other PPO params | Same as skrl (gamma=0.99, lam=0.95, clip=0.2, entropy=0.01) |

### Scene
| Parameter | Value |
|-----------|-------|
| Num envs (train) | 4096 |
| Env spacing | 2.5 m |
| Sim dt | 0.005 s |
| Decimation | 4 (policy at 50 Hz) |

## 10. Key File Paths

All paths relative to IsaacLab repo root:

```
source/isaaclab_tasks/.../locomotion/velocity/
  velocity_env_cfg.py                    # Base LocomotionVelocityRoughEnvCfg (rewards, obs, events)
  config/a1/
    rough_env_cfg.py                     # UnitreeA1RoughEnvCfg (A1-specific overrides)
    flat_env_cfg.py                      # UnitreeA1FlatEnvCfg (flat terrain overrides)
    __init__.py                          # Gym registration
    agents/
      skrl_flat_ppo_cfg.yaml             # skrl PPO hyperparams (flat)
      skrl_rough_ppo_cfg.yaml            # skrl PPO hyperparams (rough)
      rsl_rl_ppo_cfg.py                  # rsl_rl PPO configs (both flat + rough)
      sb3_ppo_cfg.yaml                   # Stable Baselines3 config

source/isaaclab_assets/.../robots/
  unitree.py                             # UNITREE_A1_CFG (actuators, init state, USD path)
```

## 11. Relevance to Harold

| Dimension | A1 | Harold | Ratio |
|-----------|----|--------|-------|
| Mass | ~12 kg | ~2 kg | 6x heavier |
| Leg length | ~0.25 m (thigh+calf) | ~0.15 m | 1.7x longer |
| DOF | 12 | 12 | Same topology |
| Actuator effort | 33.5 Nm | TBD | Much stronger |

**Scaling considerations:**

- **Action scale**: A1 uses 0.25. Harold may need a similar or smaller scale given shorter lever arms.
- **Velocity commands**: A1 commands up to 1.0 m/s. Harold's shorter legs mean max achievable velocity is lower; command ranges of ~0.3-0.5 m/s are more realistic.
- **Mass randomization**: A1 adds [-1, +3] kg to a 12 kg robot (up to +25%). Harold at 2 kg should use proportionally smaller ranges (e.g., [-0.2, +0.5] kg).
- **Reward weights**: The `dof_torques_l2` penalty (-0.0002) is tuned for A1's 33.5 Nm motors. Harold's weaker actuators produce smaller torques, so this penalty may need to be smaller or the weight rebalanced.
- **Feet air time threshold**: 0.5 s is tuned for A1's gait cycle. Harold's lighter body and shorter legs may have a faster natural gait frequency; a lower threshold (0.2-0.3 s) could be more appropriate.
- **Flat orientation penalty**: The -2.5 weight on `flat_orientation_l2` is significant and likely transferable, since keeping the body level is equally important for a small robot.
- **PD gains**: A1 uses Kp=25, Kd=0.5 with DCMotor model. These are absolute values and must be retuned for Harold's actuator characteristics.
