# Harold Quadruped Robot

This repository contains the code for the Harold quadruped robot, a small legged robot platform for reinforcement learning research and experimentation.

## Where to start

- `AGENTS.md` is the primary agent quickstart and workflow guide.
- `docs/index.md` maps documentation by role (desktop sim, hardware, firmware).
- `docs/memory/` contains the current state, priorities, and experiment history.

## Repository structure (high level)

- `AGENTS.md` - Agent workflows and operating rules.
- `docs/` - All documentation (reference, hardware, simulation, memory).
- `harold_isaac_lab/` - Isaac Lab extension (tasks, configs, envs).
- `scripts/` - Harold CLI entry points for training/monitoring.
- `deployment/` - RPi inference runtime and configs.
- `deployment_artifacts/` - Exported policies and deployment artifacts.
- `firmware/` - ESP32 firmware and support tooling.
- `policy/` - Auxiliary policy tooling (see `docs/index.md`).

## Quick start (desktop)

```bash
source ~/Desktop/env_isaaclab/bin/activate
cd /home/matteo/Desktop/code_projects/harold
python scripts/harold.py train
```

For full workflow guidance, see `AGENTS.md`.

## Hardware usage

See:
- `docs/reference/hardware_reference.md`
- `docs/hardware/rpi_deployment.md`
- `docs/hardware/calibration_checklist.md`

## TODOs and planning

Project priorities are tracked in `docs/memory/NEXT_STEPS.md`.
