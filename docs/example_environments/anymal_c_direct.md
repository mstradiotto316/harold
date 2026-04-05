# ANYmal C Direct Environment -- Technical Breakdown

Reference implementation of a **direct** (non-manager-based) quadruped locomotion environment in Isaac Lab.
This is the same architecture Harold uses.

---

## 1. Overview

| Field | Value |
|---|---|
| Robot | ANYmal C (ANYbotics) |
| Gym IDs | `Isaac-Velocity-Flat-Anymal-C-Direct-v0`, `Isaac-Velocity-Rough-Anymal-C-Direct-v0` |
| Terrain | Flat plane (flat variant) / procedural rough (rough variant) |
| Architecture | **DirectRLEnv** -- all observations, rewards, resets, and actions are computed in a single Python class with explicit tensor math. No manager infrastructure. |
| Base class | `isaaclab.envs.DirectRLEnv` |
| Episode length | 20 s |
| Sim dt | 1/200 s (0.005 s) |
| Decimation | 4 (policy runs at 50 Hz) |
| Num envs (default) | 4096, spacing 4.0 m |

---

## 2. Robot Specs

| Field | Value |
|---|---|
| Mass | ~50 kg (ANYmal C nominal) |
| DOF | 12 (3 per leg: HAA, HFE, KFE) |
| Init height | 0.6 m |
| Actuator model | `ActuatorNetLSTMCfg` (learned LSTM actuator net for ANYdrive 3.0) |
| Effort limit | 80 N*m (saturation 120 N*m) |
| Velocity limit | 7.5 rad/s |
| Joint stiffness / damping | N/A (LSTM net replaces PD) |
| Default joint pos | HAA=0.0, front HFE=0.4, hind HFE=-0.4, front KFE=-0.8, hind KFE=0.8 |
| Self-collisions | Enabled |
| Soft joint pos limit factor | 0.95 |

---

## 3. Observation Space

**Flat variant: 48 dims. Rough variant: 235 dims** (48 + 187 height scan).

Observations are concatenated in this exact order (see `_get_observations`):

| # | Observation | Dims | Notes |
|---|---|---|---|
| 1 | `root_lin_vel_b` | 3 | Base linear velocity in body frame |
| 2 | `root_ang_vel_b` | 3 | Base angular velocity in body frame |
| 3 | `projected_gravity_b` | 3 | Gravity vector projected into body frame |
| 4 | `commands` | 3 | (vx, vy, wz) velocity commands |
| 5 | `joint_pos - default_joint_pos` | 12 | Joint position deviation from default |
| 6 | `joint_vel` | 12 | Joint velocities |
| 7 | `height_data` | 187 | **Rough only.** Height scan from RayCaster, clipped to [-1, 1] |
| 8 | `actions` | 12 | Previous actions (set from `_actions` at the start of `_get_observations`) |

**No observation noise** is applied in the direct variant. The manager-based variant adds uniform noise to most observations; the direct env does not.

---

## 4. Action Space

| Field | Value |
|---|---|
| Dims | 12 (one per joint) |
| Control mode | Joint position targets |
| Scaling | `action_scale = 0.5` |
| Processing | `processed_actions = 0.5 * actions + default_joint_pos` |
| Application | `set_joint_position_target(processed_actions)` |

Actions are **offsets from the default joint configuration**, scaled by 0.5 radians. The LSTM actuator net then converts position targets into joint torques internally.

---

## 5. Reward Structure

All 10 reward terms, with exact computation from the code. Every term is multiplied by `self.step_dt` (= decimation * sim_dt = 0.02 s at 50 Hz policy).

### 5.1 Tracking Rewards (positive)

| Term | Weight (flat) | Computation |
|---|---|---|
| **track_lin_vel_xy_exp** | **+1.0** | `exp(-sum((cmd_xy - vel_xy)^2) / 0.25)` -- Gaussian kernel on XY velocity error, sigma^2 = 0.25 |
| **track_ang_vel_z_exp** | **+0.5** | `exp(-(cmd_wz - wz)^2 / 0.25)` -- Gaussian kernel on yaw rate error, sigma^2 = 0.25 |
| **feet_air_time** | **+0.5** | `sum((last_air_time - 0.5) * first_contact)` gated by `norm(cmd_xy) > 0.1`. Rewards feet that stay in the air for ~0.5 s before touching down. Only active when the robot is commanded to move. |

### 5.2 Penalties (negative)

| Term | Weight (flat) | Computation |
|---|---|---|
| **lin_vel_z_l2** | **-2.0** | `vel_z^2` -- Penalizes vertical bouncing |
| **ang_vel_xy_l2** | **-0.05** | `sum(ang_vel_xy^2)` -- Penalizes roll/pitch angular velocity |
| **dof_torques_l2** | **-2.5e-5** | `sum(applied_torque^2)` -- Penalizes high joint torques |
| **dof_acc_l2** | **-2.5e-7** | `sum(joint_acc^2)` -- Penalizes joint acceleration |
| **action_rate_l2** | **-0.01** | `sum((action_t - action_{t-1})^2)` -- Penalizes jerky actions |
| **undesired_contacts** | **-1.0** | Count of `*THIGH` bodies with contact force > 1.0 N (from 3-step history). Penalizes thigh-ground contact. |
| **flat_orientation_l2** | **-5.0** (flat) / **0.0** (rough) | `sum(projected_gravity_xy^2)` -- Penalizes body tilt. Disabled on rough terrain. |

### 5.3 Reward Aggregation

```python
reward = sum(all_reward_terms)  # simple summation, no clipping
```

Each term is: `raw_value * weight * step_dt`. The `step_dt` factor normalizes rewards to be per-second rather than per-step.

### 5.4 Reward Scale Comparison: Direct vs Manager-Based

The direct and manager-based ANYmal C envs use **almost** the same reward scales, but there are two notable differences:

| Term | Direct (flat) | Manager-based (rough) |
|---|---|---|
| dof_torques_l2 | -2.5e-5 | -1.0e-5 |
| feet_air_time | +0.5 | +0.125 |
| flat_orientation_l2 | -5.0 | 0.0 (disabled) |

The manager-based version also has `dof_pos_limits` (weight 0.0, disabled) which is absent from the direct version.

---

## 6. Command Ranges

Commands are sampled **uniformly** in `_reset_idx`:

```python
self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
```

| Command | Range | Units |
|---|---|---|
| vx (forward) | [-1.0, 1.0] | m/s |
| vy (lateral) | [-1.0, 1.0] | m/s |
| wz (yaw rate) | [-1.0, 1.0] | rad/s |

Commands are sampled once per episode reset and **never resampled** mid-episode. The manager-based variant resamples every 10 s and includes heading commands; the direct variant does not.

---

## 7. Key Thresholds

| Parameter | Value | Where Used |
|---|---|---|
| Feet air time target | 0.5 s | `(last_air_time - 0.5) * first_contact` in feet_air_time reward |
| Velocity gate for gait reward | 0.1 m/s | `norm(cmd_xy) > 0.1` -- gait reward is zero when standing |
| Undesired contact force threshold | 1.0 N | Thigh contacts above this count as undesired |
| Base contact termination threshold | 1.0 N | Any base contact force > 1 N triggers episode death |
| Contact sensor history length | 3 steps | Used for undesired contact and termination checks |
| Contact sensor update period | 0.005 s | Matches sim dt |

---

## 8. Domain Randomization

The direct ANYmal C env has **minimal** randomization -- only two terms at startup, no runtime perturbations:

| Event | Mode | Details |
|---|---|---|
| **physics_material** | startup | Friction: static=0.8, dynamic=0.6, restitution=0.0 (deterministic, not random ranges). 64 buckets. |
| **add_base_mass** | startup | Base mass +/- 5.0 kg (uniform). |

**Not present** (but present in manager-based):
- No base COM randomization
- No external force/torque perturbations
- No periodic push events
- No initial pose/velocity randomization (reset goes to exact default state + terrain offset)
- No joint position randomization on reset
- No observation noise

This is a significant difference from the manager-based version which has 6 randomization events.

---

## 9. Training Config (skrl PPO)

### Flat Variant

| Parameter | Value |
|---|---|
| Seed | 42 |
| Network | [128, 128, 128] MLP, ELU, shared backbone (separate=False) |
| Policy class | GaussianMixin (continuous) |
| initial_log_std | 0.0 |
| Rollout length | 24 steps |
| Learning epochs | 5 |
| Mini-batches | 4 |
| Discount (gamma) | 0.99 |
| GAE lambda | 0.95 |
| Learning rate | 1e-3, KL-adaptive (threshold 0.01) |
| Clip ratio | 0.2 (both policy and value) |
| Entropy coeff | 0.005 |
| Value loss coeff | 1.0 |
| Grad norm clip | 1.0 |
| Reward shaper scale | 0.6 |
| Timesteps | 36,000 (= 36000/24 = 1500 updates at 4096 envs) |
| Value preprocessor | RunningStandardScaler |

### Rough Variant

Identical to flat except:
- Network: [512, 256, 128] (larger for height scan input)
- All other hyperparameters are the same

---

## 10. Key File Paths

All paths relative to IsaacLab repo root:

```
source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c/
    __init__.py              # Gym registration
    anymal_c_env.py          # Environment class (AnymalCEnv)
    anymal_c_env_cfg.py      # Config classes (AnymalCFlatEnvCfg, AnymalCRoughEnvCfg)
    agents/
        skrl_flat_ppo_cfg.yaml
        skrl_rough_ppo_cfg.yaml
        rsl_rl_ppo_cfg.py
        rl_games_flat_ppo_cfg.yaml
        rl_games_rough_ppo_cfg.yaml

source/isaaclab_assets/isaaclab_assets/robots/anymal.py   # Robot ArticulationCfg

# Manager-based counterpart (for comparison):
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/
    velocity_env_cfg.py      # Base config with all reward/obs/event managers
    config/anymal_c/
        rough_env_cfg.py     # ANYmal C manager-based rough config
```

---

## 11. Direct vs Manager-Based Architecture

This is the most important section for Harold development. The ANYmal C environment exists in **both** architectures in the Isaac Lab codebase, making it the canonical comparison.

### How Direct Implements Rewards

In the direct env, rewards are computed in a single `_get_rewards()` method as raw tensor operations:

```python
def _get_rewards(self) -> torch.Tensor:
    lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
    lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
    ...
    rewards = {
        "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
        ...
    }
    reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
    return reward
```

Key properties:
- **Weights are multiplied inline** with `self.cfg.<name>_reward_scale * self.step_dt`
- **No manager dispatch** -- no `RewTerm` wrappers, no `func` references, no `params` dict
- **No automatic `step_dt` multiplication** -- the direct env must do it manually (the manager-based `RewardManager` does it automatically)
- **Reward functions are not reusable** -- they are hardcoded in the env class, not in a shared `mdp` module

### How Manager-Based Implements Rewards

In the manager-based env, rewards are declared as config objects:

```python
track_lin_vel_xy_exp = RewTerm(
    func=mdp.track_lin_vel_xy_exp, weight=1.0,
    params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
)
```

Key properties:
- **Weights are declared in config**, applied by `RewardManager`
- **`step_dt` multiplication is automatic** -- the `RewardManager` multiplies every term by `dt`
- **Functions live in `mdp` module** -- reusable across envs
- **Noise, corruption, and clipping** are handled by `ObservationManager`

### Critical Differences for Porting

| Aspect | Manager-Based | Direct |
|---|---|---|
| `step_dt` multiplication | Automatic (by RewardManager) | **Manual** -- must multiply every reward term by `self.step_dt` |
| Observation noise | Configured via `Unoise` on each `ObsTerm` | Must be added manually (ANYmal C direct does NOT add any) |
| Command resampling | Automatic by `CommandManager` (every 10 s) | Manual in `_reset_idx` only (once per episode) |
| Standing envs | `rel_standing_envs=0.02` (2% get zero commands) | Not implemented |
| Heading command | Supported via `heading_command=True` | Not implemented (only vx, vy, wz) |
| Reset randomization | 5 events (pose, velocity, joints, push, external force) | 0 runtime events (only default state + terrain offset) |
| Reward function signature | `func(env, ...) -> Tensor` | Inline tensor math in `_get_rewards()` |
| Adding/removing rewards | Change config dict | Edit Python code |

### The `step_dt` Trap

The most dangerous porting bug: in manager-based configs, reward weights are "per second" because `RewardManager` multiplies by `dt`. When porting to direct, you must **manually** multiply each reward term by `self.step_dt`. If you forget, rewards will be 50x larger than intended (at 50 Hz policy rate). If you multiply twice, they will be 50x smaller.

The ANYmal C direct env does this correctly:
```python
"track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
```

---

## 12. Relevance to Harold

Harold's `sim_flat_v2` was ported from the **manager-based Spot config** to a **direct architecture**. The ANYmal C direct env is the official reference for how to do this correctly.

### What Harold Should Verify Against This Reference

1. **`step_dt` multiplication**: Every reward term in Harold's `_get_rewards()` must include `* self.step_dt`. Cross-check against ANYmal C's pattern.

2. **Reward computation correctness**: The tracking rewards use `exp(-error / sigma^2)` with sigma^2=0.25. Verify Harold uses the same kernel and does not accidentally use a different sigma or forget the negative sign.

3. **Feet air time gating**: ANYmal C gates the gait reward with `norm(cmd_xy) > 0.1` so standing robots are not rewarded for lifting feet. Harold's velocity threshold experiments (EXP-775+) are exploring this same concept.

4. **Undesired contacts**: ANYmal C checks `*THIGH` bodies using contact force history (max over 3 timesteps > 1.0 N). Verify Harold's contact body selection and threshold match the intended anatomy.

5. **Termination**: ANYmal C terminates only on base contact (> 1 N) and timeout. No other termination conditions. This is simpler than many locomotion envs.

6. **Missing randomization**: The direct ANYmal C env has almost no domain randomization compared to its manager-based counterpart. If Harold's direct port also dropped randomization events during porting, this could explain training instability or sim-to-real gaps.

7. **Command resampling**: ANYmal C direct samples commands once per episode. The manager-based version resamples every 10 s. If Harold needs command resampling, it must be implemented manually.

8. **Observation noise**: The direct env has none. If Harold needs noise for sim-to-real transfer, it must add it explicitly in `_get_observations()`.

### Architectural Lesson

The direct architecture trades configurability for transparency. Every computation is visible in one file, there is no hidden `step_dt` multiplication, no manager dispatch overhead, and no risk of config inheritance bugs. The cost is that changes require editing Python code rather than swapping config values. For an experimental project like Harold, this transparency is a net positive -- reward bugs are immediately visible in the source.
