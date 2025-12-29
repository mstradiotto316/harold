# Harold Project Context

## Memory System
This file is part of the Harold memory system. The entry point is `/CLAUDE.md` which Claude Code reads automatically at session start. That file directs agents here.

**Memory Protocol**: Read all `.claude_memory/` files at session start, update them at session end.

---

## Goal
Train a controllable walking gait for the Harold quadruped robot that can follow velocity commands.

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

## Current State (2025-12-29, Session 28 Complete)

### BREAKTHROUGH: Gear Backlash Robustness SOLVED

Session 28 achieved **35% better forward velocity** by adding 1° position noise to simulate gear backlash:
- **Baseline**: vx=0.017 m/s
- **1° backlash**: vx=0.023 m/s (35% improvement!)

The noise acts as beneficial regularization, improving generalization.

### Session 28 Key Results

| Config | vx (m/s) | Verdict |
|--------|----------|---------|
| Baseline (no noise) | 0.017 | WALKING |
| 2° backlash | 0.007 | STANDING (too much) |
| **1° backlash** | **0.023** | **WALKING** (+35%) |
| Yaw only (no backlash) | 0.011 | WALKING |
| 1° backlash + yaw | 0.003 | STANDING (needs work) |

### Best Configuration (Session 28)

| Parameter | Value | Notes |
|-----------|-------|-------|
| stiffness | 400 | Sim-to-real aligned |
| damping | 30 | Proportional to stiffness |
| CPG mode | ENABLED | `HAROLD_CPG=1` |
| Command tracking | ENABLED | `HAROLD_CMD_TRACK=1` |
| Dynamic commands | ENABLED | `HAROLD_DYN_CMD=1` |
| vx_range | 0.10-0.45 | Optimal range |
| vy_range | -0.15-0.15 | Lateral commands |
| yaw_range | -0.30-0.30 | **NEW** - turn commands |
| joint_position_noise | 0.0175 rad (~1°) | **OPTIMAL** for backlash |

### Key Changes in Session 28

1. **Backlash robustness via observation noise**:
   - Enabled `randomize_per_step: True` for sensor noise
   - Set `joint_position_noise.std = 0.0175` rad (~1°)
   - This simulates ~1-3° gear backlash in ST3215 servos

2. **Added yaw rate tracking**:
   - Added `command_tracking_yaw` reward (exponential kernel)
   - Modified `yaw_rate_penalty` to track deviation from commanded yaw
   - Enabled yaw command range: [-0.30, 0.30] rad/s

3. **Finding: Backlash + yaw combination needs work**:
   - Each feature works independently (WALKING)
   - Combined, they interfere (vx=0.003, STANDING)
   - May need curriculum learning or reduced complexity

---

## Approach Status

| Approach | Status |
|----------|--------|
| **CPG + Residual Learning** | ✅ **WALKING** |
| **Command Tracking (vx)** | ✅ **WORKING** |
| **Command Tracking (vy)** | ✅ **WORKING** (Session 27) |
| **Command Tracking (yaw)** | ✅ **WORKING** (standalone) |
| **Backlash robustness** | ✅ **SOLVED** (1° = 35% better) |
| **Sim-to-real alignment** | ✅ **COMPLETE** |
| Backlash + yaw combined | 🔲 Needs curriculum learning |
| Hardware RL testing | 🔲 **NEXT PRIORITY** |

---

## Environment Variables

| Variable | Effect |
|----------|--------|
| `HAROLD_CPG=1` | Enable CPG base trajectory |
| `HAROLD_CMD_TRACK=1` | Enable command tracking reward |
| `HAROLD_VAR_CMD=1` | Enable variable command sampling |
| `HAROLD_DYN_CMD=1` | Enable dynamic command changes (implies VAR_CMD) |
| `HAROLD_SCRIPTED_GAIT=1` | Enable scripted gait (no learning) |

---

## Deployment Pipeline

```
deployment/
├── policy/harold_policy.onnx   # Exported CPG policy (50D -> 12D)
├── inference/
│   ├── harold_controller.py    # Main 20 Hz control loop
│   ├── cpg_generator.py        # CPG trajectory (port from sim)
│   ├── observation_builder.py  # IMU + servo -> 50D obs
│   └── action_converter.py     # Policy -> servo commands
├── drivers/
│   ├── imu_reader_rpi5.py      # MPU6050 I2C driver
│   └── esp32_serial.py         # USB serial wrapper
└── config/
    ├── hardware.yaml           # Servo IDs, signs, limits
    └── cpg.yaml                # CPG params (match sim)
```

---

## Harold CLI

```bash
# Check for orphan processes before starting
python scripts/harold.py ps
python scripts/harold.py stop  # if needed

# Run with controllability
HAROLD_CPG=1 HAROLD_CMD_TRACK=1 HAROLD_DYN_CMD=1 python scripts/harold.py train \
  --hypothesis "..." --iterations 1250

# Monitor
python scripts/harold.py status
python scripts/harold.py validate
```

## System Specs
- **GPU**: NVIDIA GeForce RTX 4080 (16GB)
- **CPU**: Intel Core i7-8700K @ 3.70GHz (6 cores, 12 threads)
- **RAM**: 64 GB DDR4
- **Simulation boot time**: ~8 minutes with video (8192 envs)

## Training Configuration
- **Target duration**: 30-60 minutes per experiment
- **Environment count**: 8192 (recommended)
- **Video recording**: MANDATORY
