# Isaac Lab Spot Flat Locomotion Environment -- Technical Breakdown

Reference analysis of the official Isaac Lab velocity-tracking environment for the Boston Dynamics Spot robot on flat/cobblestone terrain. All values taken directly from source.

---

## 1. Overview

| Field | Value |
|-------|-------|
| Robot | Boston Dynamics Spot |
| Gym ID (train) | `Isaac-Velocity-Flat-Spot-v0` |
| Gym ID (play) | `Isaac-Velocity-Flat-Spot-Play-v0` |
| Entry point | `isaaclab.envs:ManagerBasedRLEnv` |
| Task | Velocity tracking (lin_vel_x, lin_vel_y, ang_vel_z) on flat + light rough terrain |
| Terrain | Generated "cobblestone road": 80% flat mesh plane, 20% random rough (noise 0.02--0.05 m) |
| Base class | `LocomotionVelocityRoughEnvCfg` (overrides most defaults) |

---

## 2. Robot Specs

| Parameter | Value |
|-----------|-------|
| URDF/USD source | `{ISAAC_NUCLEUS_DIR}/Robots/BostonDynamics/spot/spot.usd` |
| Mass (real) | ~32 kg (base body); randomized +/- 2.5 kg in sim |
| DOF count | 12 (3 per leg: hip_x, hip_y, knee) |
| Leg naming | `fl` (front-left), `fr` (front-right), `hl` (hind-left), `hr` (hind-right) |
| Joint naming | `{leg}_hx` (hip abduction), `{leg}_hy` (hip flexion), `{leg}_kn` (knee) |
| Init height | 0.5 m above ground |
| Init joint pos | `hx`: +/-0.1 rad, front `hy`: 0.9 rad, hind `hy`: 1.1 rad, `kn`: -1.5 rad |
| Hip actuator | `DelayedPDActuatorCfg` -- P=60, D=1.5, effort_limit=45 Nm, delay 0--8 ms |
| Knee actuator | `RemotizedPDActuatorCfg` -- P=60, D=1.5, lookup-table torque limits, delay 0--8 ms |
| Self-collisions | Enabled |

---

## 3. Observation Space

All observations concatenated into a single flat vector. No height scan (disabled for flat env). Corruption disabled by default (`enable_corruption = False`).

| Term | Dimension | Noise (uniform) | Description |
|------|-----------|-----------------|-------------|
| `base_lin_vel` | 3 | +/-0.1 | Base linear velocity in body frame |
| `base_ang_vel` | 3 | +/-0.1 | Base angular velocity in body frame |
| `projected_gravity` | 3 | +/-0.05 | Gravity vector projected into body frame |
| `velocity_commands` | 3 | None | Commanded (vx, vy, yaw_rate) |
| `joint_pos` (relative) | 12 | +/-0.05 | Joint positions relative to default |
| `joint_vel` (relative) | 12 | +/-0.5 | Joint velocities relative to default |
| `actions` (last) | 12 | None | Previous action output |
| **Total** | **48** | | |

---

## 4. Action Space

| Parameter | Value |
|-----------|-------|
| Type | `JointPositionActionCfg` (position targets) |
| Joint names | `.*` (all 12 joints) |
| Scale | **0.2** (action multiplied by 0.2 before adding to default offset) |
| Default offset | Enabled (`use_default_offset=True`) -- actions are deltas around default joint positions |
| Dimension | **12** |
| Control freq | **50 Hz** (decimation=10, sim dt=0.002 s -> 500 Hz physics / 10 = 50 Hz policy) |

---

## 5. Reward Structure

### Task Rewards (positive)

| Term | Weight | Function | Key Parameters | Description |
|------|--------|----------|----------------|-------------|
| `air_time` | **+5.0** | `air_time_reward` | mode_time=0.3 s, vel_threshold=0.5 m/s | Rewards longer air/contact cycles when moving; rewards stance when standing. Encourages rhythmic gait with 0.3 s target mode time. |
| `base_angular_velocity` | **+5.0** | `base_angular_velocity_reward` | std=2.0 | Exponential kernel tracking of commanded yaw rate: `exp(-|error|/2.0)` |
| `base_linear_velocity` | **+5.0** | `base_linear_velocity_reward` | std=1.0, ramp_rate=0.5, ramp_at_vel=1.0 | Exponential kernel tracking of commanded xy velocity. Reward scales up for high-speed commands (>1.0 m/s) via ramp multiplier. |
| `foot_clearance` | **+0.5** | `foot_clearance_reward` | target_height=0.1 m, std=0.05, tanh_mult=2.0 | Rewards feet reaching 0.1 m clearance during swing. Uses tanh of foot velocity to only reward moving feet. |
| `gait` | **+10.0** | `GaitReward` (class) | std=0.1, max_err=0.2, vel_threshold=0.5 | Enforces trot gait. Diagonal pairs (FL+HR, FR+HL) must be synchronized; opposite pairs must be anti-synchronized. Highest-weighted single reward. Only active when commanded or moving >0.5 m/s. |

### Penalty Rewards (negative)

| Term | Weight | Function | Key Parameters | Description |
|------|--------|----------|----------------|-------------|
| `base_orientation` | **-3.0** | `base_orientation_penalty` | -- | Penalizes non-flat orientation via projected gravity xy-norm |
| `base_motion` | **-2.0** | `base_motion_penalty` | -- | Penalizes vertical velocity (0.8x) and roll/pitch rate (0.2x) |
| `action_smoothness` | **-1.0** | `action_smoothness_penalty` | -- | Penalizes L2 norm of action delta (current - previous) |
| `air_time_variance` | **-1.0** | `air_time_variance_penalty` | -- | Penalizes variance in air/contact times across feet (encourages symmetry) |
| `joint_pos` | **-0.7** | `joint_position_penalty` | stand_still_scale=5.0, vel_threshold=0.5 | Penalizes deviation from default joint positions. 5x multiplier when standing still. |
| `foot_slip` | **-0.5** | `foot_slip_penalty` | threshold=1.0 N | Penalizes planar foot velocity when in contact (force > 1 N) |
| `joint_vel` | **-0.01** | `joint_velocity_penalty` | joints: `.*_h[xy]` | Penalizes hip joint velocities (not knees) |
| `joint_torques` | **-5e-4** | `joint_torques_penalty` | joints: `.*` | Penalizes applied torques across all joints |
| `joint_acc` | **-1e-4** | `joint_acceleration_penalty` | joints: `.*_h[xy]` | Penalizes hip joint accelerations (not knees) |

### Reward Design Notes

- The gait reward at weight 10.0 is the dominant shaping signal
- Velocity tracking (linear + angular) at 5.0 each are the primary task objectives
- The standing-still multiplier (5.0x) on joint_pos heavily penalizes fidgeting when no command is given
- Joint regularization penalties (vel, acc, torques) only target hip joints for vel/acc -- knees are unpenalized on these
- Air time targets 0.3 s mode time -- this is the desired swing/stance duration per foot

---

## 6. Command Ranges

| Command | Range | Unit |
|---------|-------|------|
| `lin_vel_x` | **-2.0 to +3.0** | m/s |
| `lin_vel_y` | **-1.5 to +1.5** | m/s |
| `ang_vel_z` | **-2.0 to +2.0** | rad/s |
| Resampling interval | 10.0 s (fixed) | |
| Standing envs fraction | **10%** (`rel_standing_envs=0.1`) |
| Heading command | Disabled (`heading_command=False`) |

---

## 7. Key Thresholds

| Threshold | Value | Where Used |
|-----------|-------|------------|
| Velocity threshold (gait activation) | **0.5 m/s** | `gait`, `air_time`, `joint_pos` rewards -- below this, gait/airtime rewards are off and standing penalty scales up |
| Air time mode_time | **0.3 s** | Target swing/stance duration |
| Gait max_err | **0.2 s** | Clipped maximum timing error in gait sync reward |
| Gait std | **0.1** | Exponential kernel width for gait reward |
| Foot clearance target | **0.1 m** | Target foot height during swing |
| Foot slip force threshold | **1.0 N** | Contact force above which slip is penalized |
| Illegal contact threshold | **1.0 N** | Body/leg contact force that triggers termination |
| Episode length | **20.0 s** | Time-out termination |

---

## 8. Domain Randomization

### Startup (once at environment creation)

| What | Range | Details |
|------|-------|---------|
| Physics material (all bodies) | static friction: 0.3--1.0, dynamic friction: 0.3--0.8, restitution: 0.0 | 64 buckets |
| Base mass | **+/-2.5 kg** additive | Applied to "body" link |

### Reset (each episode)

| What | Range |
|------|-------|
| Base position | x: +/-0.5 m, y: +/-0.5 m, yaw: +/-pi |
| Base velocity | x: +/-1.5 m/s, y: +/-1.0, z: +/-0.5, roll/pitch: +/-0.7 rad/s, yaw: +/-1.0 rad/s |
| Joint positions | Default +/-0.2 rad (clipped to soft limits) |
| Joint velocities | Default +/-2.5 rad/s (clipped to soft limits) |
| External force/torque | 0.0 (disabled -- zeroed on reset) |

### Interval (during episode)

| What | Interval | Range |
|------|----------|-------|
| Push robot | Every 10--15 s | vx: +/-0.5 m/s, vy: +/-0.5 m/s |

### Terrain

- 80% flat, 20% random rough with 0.02--0.05 m height noise
- Grid: 9 rows x 21 cols, 8x8 m tiles, 20 m border

---

## 9. Training Config

### Simulation

| Parameter | Value |
|-----------|-------|
| Physics dt | **0.002 s** (500 Hz) |
| Policy decimation | **10** (policy at 50 Hz) |
| Default num_envs | **4096** (from base class) |
| Episode length | 20.0 s |
| Ground friction | static=1.0, dynamic=1.0 (multiply mode) |

### PPO (rsl_rl)

| Parameter | Value |
|-----------|-------|
| Actor network | [512, 256, 128], ELU activation |
| Critic network | [512, 256, 128], ELU activation |
| Obs normalization | Disabled (actor and critic) |
| Init noise std | 1.0 |
| Steps per env | 24 |
| Mini-batches | 4 |
| Learning epochs | 5 |
| Learning rate | 1e-3 (adaptive KL schedule, target KL=0.01) |
| Gamma | 0.99 |
| Lambda (GAE) | 0.95 |
| Clip param | 0.2 |
| Value clip | Yes (clipped value loss) |
| Entropy coeff | 0.0025 |
| Value loss coeff | 0.5 |
| Max grad norm | 1.0 |
| Max iterations | 20,000 |
| Save interval | Every 50 iterations |

### PPO (skrl)

Identical hyperparameters to rsl_rl config. Seed=42. Total timesteps: 480,000. Sequential trainer.

---

## 10. Key File Paths

All paths relative to the IsaacLab repository root.

| File | Path |
|------|------|
| Flat env config | `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/flat_env_cfg.py` |
| Gym registration | `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/__init__.py` |
| Spot-specific rewards | `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/mdp/rewards.py` |
| Spot-specific events | `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/mdp/events.py` |
| Spot MDP __init__ | `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/mdp/__init__.py` |
| RSL-RL PPO config | `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/agents/rsl_rl_ppo_cfg.py` |
| skrl PPO config | `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/agents/skrl_flat_ppo_cfg.yaml` |
| Robot asset config | `source/isaaclab_assets/isaaclab_assets/robots/spot.py` |
| Base env config | `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py` |

---

## 11. Relevance to Harold

Harold is a 2 kg, 12-DOF quadruped with ~0.15 m leg length. Key takeaways from Spot's config:

**Directly applicable:**
- **Observation space**: The 48-dim observation vector (no height scan) is a good template. Harold uses the same structure.
- **Gait reward as dominant signal**: Weight 10.0 gait reward with trot-enforcing diagonal sync is the core of Spot's locomotion. Harold should similarly make gait the highest-weighted reward.
- **Velocity threshold gating**: The 0.5 m/s threshold gates gait/airtime rewards off when standing. Harold's experiments have been tuning this (down to 0.15 m/s) because Harold moves slower than Spot.
- **Standing penalty multiplier**: The 5.0x `stand_still_scale` on joint_pos penalty prevents fidgeting at zero command. Harold experiments have tested this.
- **Action scale 0.2**: Spot uses 0.2 action scale (vs the base env's 0.5). Small action scale encourages smoother, more conservative joint movements. Harold adopted this from Spot.

**Must be adapted for Harold's scale:**
- **Command ranges**: Spot commands up to 3.0 m/s forward. Harold's max velocity should be much lower (~0.3--0.5 m/s given its size).
- **Air time mode_time**: 0.3 s target may be too long for Harold. Smaller robots have faster natural gait frequencies. Consider 0.15--0.25 s.
- **Foot clearance target**: 0.1 m is ~20% of Harold's leg length vs ~5% of Spot's. Scale down to ~0.02--0.04 m.
- **Mass randomization**: +/-2.5 kg is ~8% of Spot's mass. For Harold at 2 kg, this should be ~+/-0.1--0.2 kg.
- **Reset velocity ranges**: Spot resets with up to 1.5 m/s linear velocity. Harold should use proportionally smaller values.
- **Actuator delays**: Spot models 0--8 ms actuator delay. Harold's servo characteristics differ; verify delay profile.
- **Policy frequency**: Spot runs at 50 Hz (decimation=10 at 500 Hz physics). Harold's 20 Hz target is a known parameter under investigation.

**Structural differences:**
- Spot uses `DelayedPDActuatorCfg` + `RemotizedPDActuatorCfg` (realistic actuator modeling with lookup tables). Harold uses simpler PD or direct torque control.
- Spot's reward functions are all custom (`spot_mdp.*`), not the generic `mdp.*` rewards from the base env. Harold has similarly moved to custom reward implementations.
- Spot disables the height scanner for flat terrain. Harold's `sim_flat_v2` follows this pattern.
