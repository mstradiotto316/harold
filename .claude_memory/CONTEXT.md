# Harold Project Context

## Memory System
This file is part of the Harold memory system. The entry point is `/CLAUDE.md` which Claude Code reads automatically at session start. That file directs agents here.

**Memory Protocol**: Read all `.claude_memory/` files at session start, update them at session end.

---

## Goal
Train a stable forward walking gait for the Harold quadruped robot.

## Project Overview
- **Robot**: 12-DOF quadruped (4 legs × 3 joints: shoulder, thigh, calf)
- **Framework**: Isaac Lab (NVIDIA Isaac Sim) + SKRL (PPO)
- **Control Rate**: 20 Hz policy, 180 Hz simulation
- **Hardware**: Real robot exists with ESP32 controller, FeeTech STS3215 servos

## Critical Files
| Purpose | Path |
|---------|------|
| Training script | `harold_isaac_lab/scripts/skrl/train.py` |
| Flat task config | `harold_isaac_lab/source/.../tasks/direct/harold_flat/harold_isaac_lab_env_cfg.py` |
| Robot asset | `harold_isaac_lab/source/.../tasks/direct/harold_flat/harold.py` |
| USD model | `part_files/V4/harold_8.usd` |
| Best checkpoint | `logs/skrl/harold_direct/terrain_62/checkpoints/best_agent.pt` |

## Current State (2025-12-22)
- 40+ experiments completed (EXP-001 to EXP-012 in current numbering)
- terrain_62 checkpoint: stable standing but no walking
- Key discovery: **Elbow pose is a stable attractor** - robot falls forward and stays there
- Focus: Preventing elbow exploit to achieve proper standing, then walking
- EXP-012 currently running: low height penalty with world Z position

### Key Finding: The Elbow Exploit
The robot consistently finds elbow pose as a local minimum:
1. Falls forward early in training
2. Back stays elevated → passes upright check (0.93)
3. Body touches ground → slight contact penalty
4. Height too low → fails height_reward (1.7-1.9 vs 2.0 threshold)

Approaches tried and failed:
- Height termination: Scanner initialization issues
- Joint-angle termination: Robot adapts around thresholds
- Low height penalty: Robot still finds low poses

### Harold CLI Observability System (COMPLETE)
The `harold.py` script provides hypothesis-driven experimentation:
- `harold train --hypothesis "..." --tags "..."` - Start with metadata
- `harold compare EXP-034 EXP-035` - Side-by-side comparison
- `harold note EXP-034 "observation"` - Add notes to experiments
- `harold status` - State-only reporting (no prescriptive suggestions)

## System Specs
- **GPU**: NVIDIA GeForce RTX 4080 (16GB)
- **CPU**: Intel Core i7-8700K @ 3.70GHz (6 cores, 12 threads)
- **RAM**: 32 GB
- **Simulation boot time**: ~14 seconds

## Active Hypotheses
1. Current reward structure may not incentivize stable forward locomotion
2. Terrain curriculum may need tuning for gait stability
3. Action filtering (EMA beta=0.4) smooths but may limit agility

## Key Parameters
- Observation: 48D (velocities, gravity, joint states, commands, prev actions)
- Action: 12D (joint position deltas, normalized [-1, 1])
- Reward weights: forward_vel=80, lateral_vel=80, height=1, torque=-0.005, air_time=8
- Termination: body contact, orientation loss (gravity_z > -0.5)
