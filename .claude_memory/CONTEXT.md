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

## Current State (2025-12-26, Session 21 Complete)

**🎉 BREAKTHROUGH: Scripted gait achieves vx=+0.141 m/s (141% of target!)**

### Session 21 Critical Discovery

**The "Phase 1 failed" conclusion was WRONG.** Harold CAN walk with open-loop scripted control.

**Root Cause**: PD stiffness=200 was far too low for Isaac Lab's implicit actuator model. Servos couldn't extend legs under load, causing all walking attempts to fail.

**Solution**: Increased stiffness from 200 → 1200, damping from 75 → 50

| Setting | Old | New | Result |
|---------|-----|-----|--------|
| stiffness | 200 | 1200 | Legs now extend under load |
| damping | 75 | 50 | More responsive tracking |
| effort_limit | 2.0 | 2.8 | 95% of hardware max |

**Scripted Gait Results** (Phase 1 validation):
- **vx = +0.141 m/s** (141% of 0.1 m/s target!)
- Height = 0.174 m (proper standing)
- Diagonal trot pattern working

### Previous Best Results
- **Best RL stable (EXP-097)**: vx=0.034 m/s with backward_penalty=75
- **Peak observed (EXP-059)**: vx=0.076 at 89% training (regressed)
- ~100 experiments completed across 20 sessions

### Approach Status
| Approach | Status |
|----------|--------|
| **PD gain tuning (stiffness)** | ✅ **CRITICAL FIX** - stiffness 200→1200 |
| **Scripted gait validation** | ✅ **SUCCESS** - vx=0.141 m/s achieved |
| Contact-based gait reward | ✅ Works (vx=0.036 with old PD gains) |
| Explicit backward penalty | ✅ Works (75 is optimal) |
| Reward weight tuning | ❌ Exhausted (0.034 max with old PD) |
| PPO/curriculum tuning | ❌ Not the bottleneck |
| **Re-train RL with new PD gains** | 🔲 **PRIORITY 1** |
| **CPG-based action space** | 🔲 Priority 2 (may not be needed now) |

### Best Configuration (EXP-056)
```python
learning_rate = 5.0e-4
progress_forward_pos = 40.0
progress_forward_neg = 10.0
height_reward = 15.0
upright_reward = 10.0
diagonal_gait_reward = 5.0  # NEW - forward-gated
entropy_loss_scale = 0.01
```

### Harold CLI Observability System (COMPLETE)
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

**Process Safety**: The harness blocks concurrent training, but orphan processes can exist after crashes. Always run `harold ps` before starting new experiments.

## System Specs
- **GPU**: NVIDIA GeForce RTX 4080 (16GB)
- **CPU**: Intel Core i7-8700K @ 3.70GHz (6 cores, 12 threads)
- **RAM**: 64 GB DDR4 (upgraded 2025-12-25)
- **Simulation boot time**: ~14 seconds

## Training Configuration
- **Target duration**: 30-60 minutes per experiment (fast iteration > long runs)
- **Environment count**: 8192 (recommended), up to 16384 for max throughput
- **Video recording**: MANDATORY - never disable
- **Memory watchdog**: Auto-starts, kills at RAM>95% or Swap>70% (safety net, rarely triggers with 64GB RAM)
- **Kill detection**: `harold status` shows `KILLED_BY_WATCHDOG` if triggered

## Active Hypotheses
1. Current reward structure may not incentivize stable forward locomotion
2. Terrain curriculum may need tuning for gait stability
3. Action filtering (EMA beta=0.4) smooths but may limit agility

## Key Parameters
- Observation: 48D (velocities, gravity, joint states, commands, prev actions)
- Action: 12D (joint position deltas, normalized [-1, 1])
- Reward weights: forward_vel=80, lateral_vel=80, height=1, torque=-0.005, air_time=8
- Termination: body contact, orientation loss (gravity_z > -0.5)
