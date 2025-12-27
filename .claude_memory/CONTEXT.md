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
| Hardware gait script | `firmware/scripted_gait_test_1/scripted_gait_test_1.ino` |

## Current State (2025-12-27, Session 23 Complete)

**Session 23: Found stiffness=600 is better than 400 for RL learning**

### Session 23 Key Finding

| Stiffness | Height | vx (final) | vx (peak) | Verdict |
|-----------|--------|------------|-----------|---------|
| 400 | 0.74 | 0.024 | 0.024 | FAILING |
| 600 | 1.54 | 0.027 | 0.047 | STANDING |

**Stiffness=400 (sim-to-real aligned) was too soft for RL**:
- Robot couldn't maintain height (0.74 vs 1.2 threshold)
- Learning was unstable with short episodes

**Stiffness=600 is significantly better**:
- Height: 0.74 → 1.54 (+108%)
- Peak vx: 0.024 → 0.047 (+96%)
- Robot stands properly, learns better

### Current Actuator Configuration (harold.py)

| Parameter | Value | Reason |
|-----------|-------|--------|
| stiffness | 600 | Session 23: Better than 400 for RL learning |
| damping | 45 | Proportional to stiffness |
| effort_limit | 2.8 | 95% of hardware max |

### Scripted Gait Parameters (Aligned Sim ↔ Hardware)

```python
# Simulation (ScriptedGaitCfg)
frequency: 0.5 Hz
swing_thigh: 0.40 rad    # Thigh back during swing
stance_thigh: 0.90 rad   # Thigh forward during stance
stance_calf: -0.90 rad   # Extended during stance
swing_calf: -1.40 rad    # Bent during swing (matched to HW CALF_MAX=80°)

# Hardware conversion: hardware_deg = -sim_rad × (180/π)
```

### Critical Discovery: Floor Surface Matters

Real robot gait performance varies significantly by floor type:
- Hardwood vs short carpet vs long carpet show different behaviors
- This must be addressed in training via domain randomization

### Simulation Performance

With stiffness=400, frequency=0.5 Hz:
- **vx = +0.04 m/s** forward (down from 0.14 m/s with stiff settings)
- Height = 0.163 m
- Softer, more realistic servo response

---

## Previous Session Summary (Session 21)

**Root Cause Found**: PD stiffness=200 was far too low. Increased to 1200 enabled walking.

| Setting | Old | New | Result |
|---------|-----|-----|--------|
| stiffness | 200 | 1200 | Legs now extend under load |
| damping | 75 | 50 | More responsive tracking |
| effort_limit | 2.0 | 2.8 | 95% of hardware max |

**Scripted Gait Results** (Session 21, stiff settings):
- vx = +0.141 m/s (141% of 0.1 m/s target!)
- Height = 0.174 m (proper standing)

---

## Approach Status

| Approach | Status |
|----------|--------|
| **Sim-to-real alignment** | ✅ **COMPLETE** - Parameters matched |
| **Real robot scripted gait** | ✅ **WALKING FORWARD** |
| Stiffness tuning for RL | ⏳ **IN PROGRESS** - 600 better than 400, testing 800 next |
| Scripted gait validation | ✅ Success in sim and hardware |
| Contact-based gait reward | ✅ Works (diagonal gait reward) |
| Explicit backward penalty | ✅ Works (75 is optimal) |
| **Continue stiffness sweep** | 🔲 **PRIORITY 1** - Test 800, 1000 |
| **Domain randomization (floor friction)** | 🔲 **PRIORITY 2** |

---

## Harold CLI Observability System (COMPLETE)
```bash
# BEFORE starting: always check for orphan processes
harold ps                                     # List processes (shows [ORPHAN] if any)
harold stop                                   # Kill all and cleanup (if needed)

# Experiment workflow
harold train --hypothesis "..." --tags "..."  # Start with metadata
harold status                                 # Check progress (shows it/s, envs)
harold validate                               # Final metrics
harold compare EXP-014 EXP-015               # Side-by-side
```

## System Specs
- **GPU**: NVIDIA GeForce RTX 4080 (16GB)
- **CPU**: Intel Core i7-8700K @ 3.70GHz (6 cores, 12 threads)
- **RAM**: 64 GB DDR4 (upgraded 2025-12-25)
- **Simulation boot time**: ~14 seconds

## Training Configuration
- **Target duration**: 30-60 minutes per experiment (fast iteration > long runs)
- **Environment count**: 8192 (recommended), up to 16384 for max throughput
- **Video recording**: MANDATORY - never disable
- **Memory watchdog**: Auto-starts, kills at RAM>95% or Swap>70%

## Key Parameters
- Observation: 48D (velocities, gravity, joint states, commands, prev actions)
- Action: 12D (joint position deltas, normalized [-1, 1])
- Termination: body contact, orientation loss (gravity_z > -0.5)
