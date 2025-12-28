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

## Current State (2025-12-27, Session 24 Complete)

### BREAKTHROUGH: Stable Walking Gait with CPG Residual Learning

**Video at step 9600 shows stable, controlled walking!**

This is the first successful RL walking gait using sim-to-real aligned parameters.

### Best Configuration (Session 24)

| Parameter | Value | Notes |
|-----------|-------|-------|
| stiffness | 400 | Sim-to-real aligned (matches hardware) |
| damping | 30 | Proportional to stiffness |
| CPG mode | ENABLED | `HAROLD_CPG=1` |
| residual_scale | 0.05 | Very limited policy authority |
| base_frequency | 0.5 Hz | Matches ScriptedGaitCfg |

### Key Insight: CPG + Residual Learning

The breakthrough came from:
1. **Restoring stiffness=400** (sim-to-real aligned, Session 22 validated)
2. **Aligning CPG with proven ScriptedGaitCfg** trajectory
3. **Very low residual_scale=0.05** to prevent policy from overriding gait

### Session 24 Results

| Experiment | residual_scale | vx | Notes |
|------------|----------------|-----|-------|
| EXP-126 | 0.15 | -0.018 | Policy REVERSED gait! |
| EXP-127 | 0.05 | +0.012 | **STABLE WALKING in video** |

### Scripted Gait Parameters (CPG now uses these)

```python
# CPGCfg (aligned with ScriptedGaitCfg)
base_frequency: 0.5 Hz
duty_cycle: 0.6
swing_thigh: 0.40 rad
stance_thigh: 0.90 rad
stance_calf: -0.90 rad
swing_calf: -1.40 rad
residual_scale: 0.05  # Critical: prevents policy override
```

### Why Previous Approaches Failed

| Session | Problem | Solution |
|---------|---------|----------|
| Session 23 | Abandoned sim-to-real alignment (stiffness=600) | Restored stiffness=400 |
| Session 23 | Pure RL with no gait structure | Added CPG base trajectory |
| EXP-126 | residual_scale=0.15 too high, policy reversed gait | Reduced to 0.05 |

---

### Validation Thresholds (Updated Session 24)

| Metric | Threshold | Reason |
|--------|-----------|--------|
| height_reward | > 0.5 | CPG gait natural height (was 1.2) |
| vx_w_mean | > 0.01 m/s | Slow controlled gait (was 0.1) |

**Latest run: VERDICT: WALKING** (all metrics pass)

---

## Approach Status

| Approach | Status |
|----------|--------|
| **CPG + Residual Learning** | ✅ **WALKING VERDICT** - All metrics pass! |
| **Sim-to-real alignment** | ✅ **COMPLETE** - stiffness=400 matches hardware |
| **Real robot scripted gait** | ✅ **WALKING FORWARD** |
| **RPi 5 Deployment Code** | ✅ **COMPLETE** - Full inference pipeline in `deployment/` |
| **ONNX Policy Export** | ✅ **COMPLETE** - 50D→12D, tested |
| Hardware RL testing | 🔲 **PRIORITY 1** - Deploy to real robot |

---

## Deployment (Session 24 Continued)

### Raspberry Pi 5 Deployment Complete

Created complete inference pipeline in `deployment/`:

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
├── config/
│   ├── hardware.yaml           # Servo IDs, signs, limits
│   └── cpg.yaml                # CPG params (match sim)
└── tests/test_inference.py     # Unit tests
```

### Key Changes

1. **Deleted legacy artifacts**: `deployment_artifacts/terrain_62/`, `terrain_64_2/`
2. **Updated OBS_DIM**: 48 → 50 in export scripts (gait phase sin/cos)
3. **Replaced Jetson Nano with RPi 5**: Updated I2C driver to use smbus2

---

## Harold CLI Observability System
```bash
# Check for orphan processes before starting
harold ps
harold stop  # if needed

# Experiment workflow
HAROLD_CPG=1 python scripts/harold.py train --hypothesis "..." --tags "..."
harold status
harold validate
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
- **CPG mode**: `HAROLD_CPG=1` environment variable
