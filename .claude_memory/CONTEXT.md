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

## Current State (2025-12-28, Session 26 Complete)

### BREAKTHROUGH: Dynamic Command Tracking Working

The robot can now follow changing velocity commands during an episode. Key discovery: `zero_velocity_prob=0` is critical - any stop commands cause the policy to learn stopping instead of walking.

### Best Configuration (Session 26)

| Parameter | Value | Notes |
|-----------|-------|-------|
| stiffness | 400 | Sim-to-real aligned |
| damping | 30 | Proportional to stiffness |
| CPG mode | ENABLED | `HAROLD_CPG=1` |
| Command tracking | ENABLED | `HAROLD_CMD_TRACK=1` |
| Dynamic commands | ENABLED | `HAROLD_DYN_CMD=1` |
| vx_range | 0.10-0.45 | Optimal range found |
| zero_velocity_prob | 0.0 | **CRITICAL** - never command stop |
| command_change_interval | 10.0s | Changes every 10 seconds |
| tracking_weight | 10.0 | Sweet spot (20 was worse) |
| tracking_sigma | 0.1 | Sweet spot (0.15 was worse) |

### Session 26 Results

| EXP | Config | vx | Verdict |
|-----|--------|-----|---------|
| 134-138 | Various tweaks | 0.003-0.010 | STANDING |
| 139 | zero_prob=0 | 0.011 | **WALKING** |
| 140 | range 0.10-0.45 | 0.015 | **WALKING** |
| 142 | Confirmation | 0.015 | **WALKING** |
| 143 | 2500 iters | 0.016 | **WALKING** |

### Key Insights from Session 26

1. **zero_velocity_prob=0 is critical**: With 5% stop commands, policy learned to stop
2. **Optimal range 0.10-0.45**: Wider than 0.15-0.40 but not too wide (0.05-0.50)
3. **tracking_weight=10 is sweet spot**: 20 caused over-optimization
4. **tracking_sigma=0.1 is sweet spot**: 0.15 was too permissive
5. **Results are reproducible**: vx=0.015 achieved 3 times

---

## Approach Status

| Approach | Status |
|----------|--------|
| **CPG + Residual Learning** | ✅ **WALKING** |
| **Command Tracking (vx)** | ✅ **WORKING** - Dynamic commands enabled |
| **Sim-to-real alignment** | ✅ **COMPLETE** - stiffness=400 matches hardware |
| **Real robot scripted gait** | ✅ **WALKING FORWARD** |
| **RPi 5 Deployment Code** | ✅ **COMPLETE** - Full inference pipeline |
| Lateral (vy) commands | 🔲 **PRIORITY 1** - Need vy tracking reward |
| Yaw rate commands | 🔲 **PRIORITY 2** - Need yaw tracking reward |
| Hardware RL testing | 🔲 **PRIORITY 3** - Deploy controllable policy |

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
