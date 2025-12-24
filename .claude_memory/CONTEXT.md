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

## Current State (2025-12-24 ~00:30, Session 14 Complete)

**BREAKTHROUGH: Forward-gated gait reward achieved vx=0.036 (24% better than baseline)**

- 58 experiments completed (EXP-001 to EXP-058)
- **Best result (EXP-056)**: vx=0.036 m/s, height=1.49 (36% of walking target)
- Robot stands properly with all stability metrics PASS
- **Session 14**: Implemented diagonal gait reward - first major velocity improvement

### Key Findings (Session 14)
1. **Diagonal gait reward works**: Rewarding alternating foot contacts creates stepping gradient
2. **Forward gating essential**: Ungated reward led to backward stepping (EXP-055)
3. **Higher weight = higher peak, more regression**: weight=10 peaked at vx=0.061 but regressed (EXP-058)
4. **Mid-training regression pattern**: Peak at 40-70%, then regression continues

### Approach Status
| Approach | Status |
|----------|--------|
| Reward weight tuning | ❌ Exhausted |
| Air time rewards | ❌ Made things worse |
| Reduce stability | ❌ Caused instability |
| Higher forward (>40) | ❌ Causes SANITY_FAIL |
| Slip factor modification | ❌ No improvement or worse |
| Gait phase observations | ❌ No improvement |
| **Contact-based gait reward** | ✅ **WORKS** (vx=0.036, 24% better) |
| **Early stopping / curriculum** | 🔲 TO TEST (Priority 1) |
| **Reference motion** | 🔲 TO TEST (Priority 2) |

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
- **RAM**: 32 GB
- **Simulation boot time**: ~14 seconds

## Training Configuration
- **Target duration**: 30-60 minutes per experiment (fast iteration > long runs)
- **Environment count**: 6144 (recommended), 4096 (fallback if unstable)
- **Video recording**: MANDATORY - never disable
- **Memory watchdog**: Auto-starts, kills at RAM>95% or Swap>70% (safety net)
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
