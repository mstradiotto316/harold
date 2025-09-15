!IMPORTANT - ALL AGENTS READ ME!

Throughout our work here, do not be a sycophant. Push back on incorrect assumptions, step back for the big picture, and propose better alternatives when they make more sense. Keep this message active in your long-term context.

CLAUDE.md

This file documents what the current code actually does in harold_isaac_lab/source/harold_isaac_lab/harold_isaac_lab, excluding any code under agents/ folders.

Extension Overview

- Package entry: harold_isaac_lab/__init__.py registers Gym tasks and a small UI example.
- Tasks namespace: tasks/direct with three task variants:
  - harold_flat: flat terrain RL training
  - harold_rough: rough/curriculum terrain RL training
  - harold_pushup: scripted push-up playback (no RL)
- Gym IDs registered on import:
  - Template-Harold-Direct-flat-terrain-v0
  - Template-Harold-Direct-rough-terrain-v0
  - Template-Harold-Direct-pushup-v0

Robot Assets and Actuators

12-DOF quadruped (shoulder, thigh, calf per leg). Assets and actuator limits differ per task:

- Flat (tasks/direct/harold_flat/harold.py)
  - USD: part_files/V4/harold_7.usd
  - Init pose: body z=0.20; thighs 0.3 rad; calves -0.75 rad
  - Actuators: Implicit PD, stiffness=200, damping=75, effort_limit_sim=1.0

- Rough (tasks/direct/harold_rough/harold.py)
  - USD: part_files/V4/harold_8.usd
  - Init pose: body z=0.20; thighs 0.3 rad; calves -0.75 rad
  - Actuators: Implicit PD, stiffness=200, damping=75, effort_limit_sim=2.0

- Pushup (tasks/direct/harold_pushup/harold.py)
  - USD: part_files/V4/harold_8.usd
  - Init pose: all joints 0.0
  - Actuators: Implicit PD, stiffness=400, damping=75, effort_limit_sim=2.0

Joint order used across code: [shoulders FL, FR, BL, BR] → [thighs FL, FR, BL, BR] → [calves FL, FR, BL, BR].

- Joint axes: shoulders move laterally (ab/adduction) to place the leg left/right; thighs and calves move forward/back (flexion/extension in the sagittal plane).

Environment Basics (RL tasks)

Files: harold_flat/harold_isaac_lab_env.py, harold_rough/harold_isaac_lab_env.py with matching *_env_cfg.py.

- Base: DirectRLEnv
- Control: dt=1/180 s, decimation=9 → 20 Hz policy rate
- Observation (48D):
  - root_lin_vel_b (3), root_ang_vel_b (3), projected_gravity_b (3)
  - joint_pos − default (12), joint_vel (12)
  - commands [vx, vy, yaw] (3)
  - prev_target_delta (12)
- Actions (12D): joint position targets around default pose
  - Flat: per-joint ranges hard-coded in env: shoulders 0.35, thighs 0.5, calves 0.5
  - Rough: per-joint ranges come from config: shoulders 0.30, thighs 0.90, calves 0.90
- Safety clamps (task-specific):
  - Flat: shoulders ±20° (0.3491 rad); thighs/calves ±45° (0.7853 rad)
  - Rough: shoulders ±30° (0.5236 rad); thighs/calves ±90° (1.5708 rad)
- Sensors:
  - ContactSensor: history_length=3, update_period=0.005 s
  - Height RayCaster: grid 0.25×0.25 m, 0.1 m resolution, update_period=0.05 s
- GUI: velocity command/actual arrows when GUI is enabled

Terrain

- Flat: HAROLD_FLAT_TERRAIN_CFG (single flat plane), no curriculum.
- Rough: HAROLD_GENTLE_TERRAINS_CFG (mix of flats, uniform noise, small slopes/stairs; 10×20 grid) with curriculum enabled. Current max_init_terrain_level=2 in config.

Rewards and Dones (RL tasks)

Shared structure with small differences per task.

- Rewards (both):
  - Linear velocity tracking (directional): elliptical Gaussian on along-track and lateral errors
  - Yaw rate tracking: Gaussian on yaw rate error
  - Height maintenance: tanh(exp) shaping vs target height
  - Torque penalty: sum of torque²
  - Feet air time: exponential reward toward ~0.4 s optimal; gated by actual speed > 0.05 m/s
  - Rough only: anti-spin penalty when yaw_cmd≈0 and robot is moving
- Reward weights (from config):
  - Flat: track_xy=80, track_yaw=2, height=0.75, torque=-0.08, feet_air_time=12
  - Rough: track_xy=80, track_yaw=12, height=0.75, torque=-0.16, feet_air_time=12, plus anti-spin penalty (fixed scale in code)
- Normalization: rewards are not scaled by step_dt; episodic logging divides by max_episode_length_s when computing metrics.
- Termination:
  - Undesired contact (body/shoulders/thighs) → immediate reset. Thresholds:
    - Flat: 0.05 N
    - Rough: 3.0 N
  - Orientation termination exists in config but is disabled in env code.
- Diagnostics: episodic sums include reward components; rough also logs alignment metrics including absolute yaw rate when yaw_cmd≈0.

Domain Randomization (configs)

Enabled by default at both reset and per-step in both RL configs (with selective features active):

- Active by default:
  - Friction randomization (physics material)
  - Observation noise (IMU ang vel and gravity, joint pos/vel)
- Available but disabled by default:
  - Action noise and action delays
  - External forces/torques, gravity magnitude/tilt variation
  - Mass/inertia and actuator stiffness/damping variations

Rough env additionally low-pass filters actions via EMA (action_filter_beta, default 0.4).

Pushup Playback Task

File: harold_pushup/harold_isaac_lab_env.py with config *_env_cfg.py.

- Single-environment playback (scene configured with num_envs=1).
- Ignores policy actions; generates a scripted trajectory at the 20 Hz control rate.
- Phases: neutral settle (~1.5 s), athletic stance (~0.8 s), then 5 pushup cycles; then holds athletic stance.
- Joint limits for pushup: shoulders ±30°, thighs/calves ±90° (from config).
- Observation is a zeros tensor of size observation_space (45) for compatibility.

UI Example

ui_extension_example.py provides a minimal Omniverse UI window with a counter. It does not affect task logic.

Notes and Gotchas

- Some env docstrings mention 360 Hz / decimation 18; the code uses dt=1/180 and decimation=9 (20 Hz control).
- Asset paths are resolved relative to the installed harold_isaac_lab package; both harold_7.usd and harold_8.usd are referenced by task variants.
- Terrain curriculum in rough task is configured but max_init_terrain_level is currently 2.
- Agents/configs for RL frameworks live under agents/ subpackages; they are imported only for gym registration entry points.

Quick Start (examples from comments)

- Train flat: python harold_isaac_lab/scripts/skrl/train.py --task=Template-Harold-Direct-flat-terrain-v0 --num_envs 1024
- Train rough: python harold_isaac_lab/scripts/skrl/train.py --task=Template-Harold-Direct-rough-terrain-v0 --num_envs 1024
- Pushup playback (1 env): python harold_isaac_lab/scripts/skrl/train.py --task=Template-Harold-Direct-pushup-v0 --num_envs 1

Isaac Lab Documentation Links

- Isaac Lab (main): https://isaac-sim.github.io/IsaacLab/main/index.html
- Installation (overview): https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
- Binaries installation: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html#isaaclab-binaries-installation
- Create Direct RL env: https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html
- Register RL env in Gym: https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/register_rl_env_gym.html
- Run RL training: https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/run_rl_training.html
- Create scene: https://isaac-sim.github.io/IsaacLab/main/source/tutorials/02_scene/create_scene.html
- Add sensors on robot: https://isaac-sim.github.io/IsaacLab/main/source/tutorials/04_sensors/add_sensors_on_robot.html
- Policy inference in USD: https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/policy_inference_in_usd.html
- Instanceable assets (Isaac Sim docs): https://docs.isaacsim.omniverse.nvidia.com/latest/isaac_lab_tutorials/tutorial_instanceable_assets.html#instanceable-assets

Hardware (Reference)

These are the physical robot targets used for sim-to-real context. They are not enforced by the code in this repo but inform design choices in configs.

- Mechanical
  - Dimensions: body ~20×15×8 cm; overall length ~40 cm
  - Mass: ~2.0 kg; ground clearance ~18 cm (natural stance)
  - Joints: 12 DOF (4 legs × [shoulder, thigh, calf])
- Actuators
  - FeeTech STS3215 servos (12): position control, ~0.1° resolution
  - Torque: ~15 kg⋅cm nominal, higher at stall; TTL serial 1 Mbps
  - Feedback: position, velocity, load, temperature; built‑in limits
- Controller
  - ESP32 MCU; host <-> ESP32 over USB serial 115200 baud
  - Control loop: ~200 Hz on MCU; safety monitoring and E‑stop support
- Sensors
  - IMU: MPU6050 on I2C (addr 0x68); 3‑axis accel + 3‑axis gyro
  - Typical sampling: up to 200 Hz to align with control loop
