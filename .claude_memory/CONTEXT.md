# Harold Project Context

## Memory System
This file is part of the Harold memory system. The entry point is `/CLAUDE.md` which Claude Code reads automatically at session start. That file directs agents here.

**Memory Protocol**: Read all `.claude_memory/` files at session start, update them at session end.

---

## Multi-Machine Workflow (Desktop ↔ RPi)

Harold development spans TWO machines:
- **Desktop**: Training (Isaac Lab), policy export, code development
- **RPi 5 (onboard)**: Hardware deployment, real robot control

### Transferring Work Between Machines

The git repository is present on both machines. To transfer work:

```bash
# On source machine (e.g., desktop after making changes)
git add -A
git commit -m "Description of changes"
git push

# On destination machine (e.g., RPi to receive changes)
git pull
```

### Session Handoff Protocol

When ending a session and transferring to another machine:

1. **Update memory files**: `NEXT_STEPS.md`, `OBSERVATIONS.md`, session log
2. **Commit and push**: Include all code changes and memory updates
3. **Document in NEXT_STEPS.md**: Clear instructions for the receiving agent
4. **Note the machine context**: Sessions alternate between `desktop` and `rpi` suffixes

### Current Session Files Pattern

- `.claude_memory/sessions/YYYY-MM-DD_sessionNN_desktop.md` - Desktop sessions
- `.claude_memory/sessions/YYYY-MM-DD_sessionNN_rpi.md` - RPi sessions

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

## Current State (2025-12-30, Session 32 Complete)

### Session 32: Double-Normalization Bug Fix (Desktop)

**CRITICAL BUG DISCOVERED AND FIXED**:

The deployment code was normalizing observations TWICE:
1. Manual normalization in `harold_controller.py`
2. ONNX model's internal `NormalizedPolicy` wrapper

**Fix Applied**: Removed manual normalization, pass RAW observations to ONNX.

**Validation Result**: ONNX matches PyTorch with max difference of 0.000003.

**Next Step**: Pull changes on RPi, test on hardware.

### Session 31: Deployment Stabilization (RPi)

Applied compensatory fixes (may not be needed after Session 32 fix):
- lin_vel stats override
- prev_target blending
- joint_pos blending during warmup

### Session 30: Joint Limit Alignment & CPG Optimization

**Key Achievements**:
1. **Joint limits aligned with hardware** - All limits now match hardware safe ranges
2. **CPG frequency optimized** - 0.7 Hz found optimal (sweep 0.5-0.8)
3. **Best policy exported** - vx=0.018 m/s (EXP-170)

**Hardware-Aligned Joint Limits**:
- Shoulders: ±25° (from ±30°)
- Thighs: sim [-5°, +55°] → hardware [-55°, +5°]
- Calves: sim [-80°, +5°] → hardware [-5°, +80°]

**CPG Frequency Sweep**:
| Freq | vx | Verdict |
|------|-----|---------|
| 0.5 Hz | 0.011-0.017 | WALKING |
| 0.6 Hz | 0.010 | STANDING |
| **0.7 Hz** | **0.016** | **WALKING** |
| 0.8 Hz | 0.010 | STANDING |

**Best Result**: EXP-170 (vx=0.018, WALKING) - Extended training at optimal 0.7 Hz

### Session 30 Findings

| Change | Effect |
|--------|--------|
| External perturbations | FAILED - causes falling |
| residual_scale 0.08 | STANDING - too much authority |
| swing_calf -1.35 | Works - safety margin from limit |

### Previous: Session 29 - Domain Randomization

- Action noise/delays HURT training
- Lin_vel noise + obs clipping implemented (neutral effect)

### Architecture: CPG + Residual Learning

The motion is a **combination** of scripted and learned:
```
target_joints = CPG_base_trajectory + policy_output * residual_scale
```

- **CPG (scripted)**: Provides timing, gait coordination, base trajectory
- **Policy (learned)**: Provides balance corrections, velocity tracking, adaptation
- **residual_scale=0.05**: Policy can only fine-tune, not override CPG

### Best Configuration (Session 30)

| Parameter | Value | Notes |
|-----------|-------|-------|
| stiffness | 400 | Sim-to-real aligned |
| damping | 30 | Proportional to stiffness |
| CPG frequency | **0.7 Hz** | **OPTIMAL (Session 30 sweep)** |
| CPG mode | ENABLED | `HAROLD_CPG=1` |
| Command tracking | ENABLED | `HAROLD_CMD_TRACK=1` |
| Dynamic commands | ENABLED | `HAROLD_DYN_CMD=1` |
| vx_range | 0.10-0.45 | Optimal range |
| residual_scale | 0.05 | 0.08 causes regression |
| swing_calf | -1.35 | Safety margin from limit |
| joint_position_noise | **0.0175 rad (~1°)** | **OPTIMAL for backlash** |

### Key Findings from Session 28

1. **Backlash robustness via observation noise**:
   - 2° noise: Too much → STANDING (vx=0.007)
   - **1° noise: OPTIMAL → WALKING (vx=0.023, +35%)**
   - Noise acts as regularization, preventing overfitting

2. **Yaw rate tracking**:
   - Works standalone (vx=0.011, WALKING)
   - Combined with backlash from scratch fails (vx=0.003)
   - **Curriculum learning works**: backlash first → yaw fine-tuning (vx=0.015)

3. **Curriculum learning is key**:
   - From scratch: backlash + yaw = STANDING
   - Curriculum: backlash → yaw = WALKING

---

## Approach Status

| Approach | Status |
|----------|--------|
| **CPG + Residual Learning** | ✅ **WALKING** |
| **Command Tracking (vx)** | ✅ **WORKING** |
| **Command Tracking (vy)** | ✅ **WORKING** (Session 27) |
| **Command Tracking (yaw)** | ✅ **WORKING** (standalone & curriculum) |
| **Backlash robustness** | ✅ **SOLVED** (1° = +35%) |
| **Sim-to-real alignment** | ✅ **COMPLETE** (Session 30) |
| **Joint limit alignment** | ✅ **COMPLETE** (Session 30) |
| **CPG frequency optimization** | ✅ **COMPLETE** (0.7 Hz optimal) |
| **External perturbations** | ❌ **FAILED** (causes falling) |
| **Policy exported** | ✅ **READY** (deployment/policy/) |
| Hardware deployment | 🔲 **NEXT PRIORITY** |

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

# Run with controllability + backlash robustness
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
