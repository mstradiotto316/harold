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

## Current State (2025-12-22 ~14:30)
- 15 experiments completed (EXP-001 to EXP-015)
- **BREAKTHROUGH in EXP-015**: Spawn pose fix worked!
- **Height reward: 3.43** (first time above 2.0 threshold)
- **Robot is STANDING properly** - ready for forward motion
- **EXP-015 still running**: ~31% complete, run ID `2025-12-22_13-32-42_ppo_torch`
- **Next**: Add forward reward in EXP-016 when EXP-015 completes

### Root Cause Analysis (NEW)
The problem is NOT reward engineering. The actual issues are:
1. **Contact detection gap** (FIXED): Elbow contact was below 10N threshold
2. **Spawn pose bias** (TO TEST): Shoulders at ±0.20 may cause forward lean
3. **Spawn height too low** (TO TEST): 0.24m is only 0.06m above elbow contact

### Experiment Results Summary
| EXP | Approach | Height | Outcome |
|-----|----------|--------|---------|
| 008 | Height reward only | 1.70 | Elbow pose |
| 012 | Height penalty | 1.83 | Peak 1.96, regressed |
| 014 | Contact 3N | **1.88** | Improved, still failing |

### Fast Iteration Protocol (NEW)
- **Short runs**: 1000 iterations (~15-30 min) instead of 2 hours
- **Early stopping**: If height stuck, stop and adjust config
- **Extend if improving**: Only continue promising experiments

### Harold CLI Observability System (COMPLETE)
```bash
harold train --hypothesis "..." --tags "..."  # Start with metadata
harold status                                  # Check progress
harold validate                               # Final metrics
harold compare EXP-014 EXP-015               # Side-by-side
```

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
