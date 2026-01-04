# Simulation Reference (Isaac Lab)

This file is a technical reference for the Isaac Lab simulation setup. Use `AGENTS.md` for workflows and `docs/memory/` for current state.

## Task variants and Gym IDs

- Tasks live under `harold_isaac_lab/source/harold_isaac_lab/harold_isaac_lab/tasks/direct/`.
- Variants:
  - `harold_flat` (flat terrain RL training)
  - `harold_rough` (rough/curriculum terrain RL training)
  - `harold_pushup` (scripted push-up playback, no RL)
- Gym IDs are registered on import (see `harold_isaac_lab/__init__.py`):
  - `Template-Harold-Direct-flat-terrain-v0`
  - `Template-Harold-Direct-rough-terrain-v0`
  - `Template-Harold-Direct-pushup-v0`

## Key files

- Flat task config: `harold_isaac_lab/source/harold_isaac_lab/harold_isaac_lab/tasks/direct/harold_flat/harold_isaac_lab_env_cfg.py`
- Rough task config: `harold_isaac_lab/source/harold_isaac_lab/harold_isaac_lab/tasks/direct/harold_rough/harold_isaac_lab_env_cfg.py`
- Pushup task config: `harold_isaac_lab/source/harold_isaac_lab/harold_isaac_lab/tasks/direct/harold_pushup/harold_isaac_lab_env_cfg.py`
- Robot asset definitions: `harold_isaac_lab/source/harold_isaac_lab/harold_isaac_lab/tasks/direct/*/harold.py`
- USD asset: `part_files/V4/harold_8.usd`

## Robot assets and actuators

- USD asset is shared across task variants.
- Default joint pose comes from `deployment/config/stance.yaml` (task-specific overrides live in each `harold.py`).
- Actuator defaults are configured per task in each `harold.py` and can be overridden via:
  - `HAROLD_ACTUATOR_STIFFNESS`
  - `HAROLD_ACTUATOR_DAMPING`
  - `HAROLD_ACTUATOR_EFFORT_LIMIT`

## Joint order and axes

- Joint order is consistent across code and deployment:
  - shoulders [FL, FR, BL, BR]
  - thighs [FL, FR, BL, BR]
  - calves [FL, FR, BL, BR]
- Shoulder joints move laterally (ab/adduction).
- Thigh and calf joints move forward/back (sagittal flexion/extension).

## Environment basics

- Environments inherit from `DirectRLEnv`.
- Control rate, simulation rate, and decimation are defined in each `*_env_cfg.py`.
- Observation contents include root velocities, projected gravity, joint positions/velocities, command inputs, and previous targets. Observation size varies by mode (RL vs CPG/scripted).
- Actions are joint position targets around the default pose; scaling and clamps are defined in the env config.
- Sensors (contact, height scanning) are configured in the env config.

## Terrain

- Flat task uses a single flat plane terrain config.
- Rough task uses a mixed terrain curriculum; curriculum settings live in the rough env config.

## Rewards and terminations

- Reward terms include velocity tracking, yaw tracking, height maintenance, torque penalties, and foot air-time shaping.
- Termination conditions include undesired contacts and orientation checks.
- Reward weights, thresholds, and curriculum schedules are configured in each `*_env_cfg.py` and should be treated as experiment-time parameters.

For validation metrics and thresholds, see `docs/memory/OBSERVABILITY.md`.

## Domain randomization

- Randomization toggles and ranges (friction, observation noise, action noise, delays, mass/inertia, actuator variations, external forces) are defined in each env config.
- Treat ranges as experiment-time parameters; refer to the config for current values.

## Scripted playback (pushup)

- `harold_pushup` ignores policy actions and generates a scripted trajectory at the control rate.
- Phase timing and joint limits are defined in the pushup env config.

## UI example

- Minimal Omniverse UI example lives at `harold_isaac_lab/source/harold_isaac_lab/harold_isaac_lab/ui_extension_example.py`.

## Notes and gotchas

- Some env docstrings drift over time; treat `*_env_cfg.py` as source of truth.
- Asset paths resolve relative to the installed `harold_isaac_lab` package.
- Rough-terrain curriculum details (levels, schedules) are configured in the rough env config.

## Quick start (manual)

- Training and monitoring should use `python scripts/harold.py` (see `AGENTS.md`).
- Manual training entry points are under `harold_isaac_lab/scripts/skrl/` and should be avoided for agent workflows.

## Isaac Lab documentation

- Main: https://isaac-sim.github.io/IsaacLab/main/index.html
- Create Direct RL env: https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html
- Register RL env in Gym: https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/register_rl_env_gym.html
- Run RL training: https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/run_rl_training.html
- Add sensors: https://isaac-sim.github.io/IsaacLab/main/source/tutorials/04_sensors/add_sensors_on_robot.html
- Policy inference in USD: https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/policy_inference_in_usd.html
