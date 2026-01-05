# Harold Project Context

## Memory System
This file is part of the Harold memory system. The entry point is `/AGENTS.md`. That file directs agents here.

**Memory Protocol**: Read core memory files at session start (CONTEXT, NEXT_STEPS, EXPERIMENTS, OBSERVATIONS, HARDWARE_CONSTRAINTS). Read archives only if needed.

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

- `docs/memory/sessions/YYYY-MM-DD_sessionNN_desktop.md` - Desktop sessions
- `docs/memory/sessions/YYYY-MM-DD_sessionNN_rpi.md` - RPi sessions

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

## Current State (2026-01-04, Session 41 RPi)

### Hardware CPG gait tuning (RPi)
- Implemented duty-cycle stance/swing CPG with per-leg scaling (front vs rear).
- Runtime CPG parameters (deployment/config/cpg.yaml):
  - frequency_hz=0.4, duty_cycle=0.5
  - stride_scale=0.35, calf_lift_scale=0.85
  - stride_scale_front=1.0, stride_scale_back=1.3
  - calf_lift_scale_front=1.0, calf_lift_scale_back=1.15
- Outcome: front impact reduced; rear legs progressed from dragging to skimming but still no clear air time.
- Test 3 on a lower-friction surface looked best; this is now the baseline for sim-to-real alignment.
- **Do NOT change SERVO_SPEED/SERVO_ACC**.
- Manual tests use: `python3 -m inference.harold_controller --cpg-only --max-seconds 10 --calibrate 0`
- `harold` systemd service will resume walking if active; keep inactive during manual tests.
- ESP32 handshake failures occurred after calibration firmware; resolved by re-flashing StreamingControl.

### Latest hardware session logs (RPi)
- `/home/pi/harold/deployment/sessions/session_2026-01-04_18-40-39.csv` (Test 1, 0.5/0.6)
- `/home/pi/harold/deployment/sessions/session_2026-01-04_18-41-11.csv` (Test 2, 0.4/0.5 + scaling)
- `/home/pi/harold/deployment/sessions/session_2026-01-04_18-43-41.csv` (Test 3, low-friction surface)
- Local copies: `logs/hardware_sessions/` on the desktop repo.

### Sync paths (RPi)
- Repo root (git checkout): `/home/pi/harold` (adjust if different).
- Runtime root: `<repo-root>/deployment` (this is where you `cd` for `python3 -m inference.harold_controller`).
- Keep `deployment/config/cpg.yaml` + `deployment/inference/cpg_generator.py` synced via git pull on the Pi.
- Workflow: edit on desktop → commit + push → `git status -sb` (clean) → `git pull --ff-only` on the Pi.

## Current State (2026-01-02, Session 39 Complete)

### Session 39: Hardware Telemetry Logging (RPi)

**NEW CAPABILITY**: Robot now logs detailed telemetry at 5 Hz during hardware runs.

- **Location**: `deployment/sessions/session_YYYY-MM-DD_HH-MM-SS.csv`
- **Data**: 60 columns - joint positions/loads/currents/temps, bus voltage, RPi metrics
- **Documentation**: See the "Hardware Session Logs" section in `docs/memory/OBSERVATIONS.md`

Tested end-to-end: 12.7s run logged 63 rows at 20 Hz control / 5 Hz logging.

### Session 40: Scripted Gait Alignment (Desktop)

**SCRIPTED GAIT STILL FAILS IN SIM (Session 36 params)** despite higher stiffness.

- Stiffness 1200 improves upright/episode length, but vx remains ≈0 or negative.
- `height_reward` and `body_contact_penalty` now log after metrics fix.
- Amplitude scaling 1.5x–2.0x yields only vx ≈ 0.002 m/s.
- `scripts/harold.py train` now waits for the new run directory before registering an alias.
- `scripts/harold.py` now supports `--mode` (rl/cpg/scripted), `--duration` presets, and `--task` selection; status shows config + x displacement.
- Actuator defaults are unified across tasks (effort=2.8, stiffness=400, damping=150) with env overrides.
- Added `HAROLD_GAIT_AMP_SCALE` for scripted/CPG diagnostics.
- Added per-foot contact/air-time/slip metrics + x displacement logging for observability.

### Session 38: Hardware-Validated CPG (Desktop)

**FINDING: Sim ≠ Hardware Parameters**

Tested hardware-validated parameters in simulation - they are too conservative:

| Configuration | vx (m/s) | Result |
|---------------|----------|--------|
| Hardware (7.5°/30° @0.5Hz) | 0.003 | STANDING |
| Intermediate (15°/40° @0.5Hz) | 0.004 | STANDING |
| Session 35 (40°/50° @0.7Hz) | 0.036 | **WALKING** |

**Conclusion**: Sim-to-real transfer needs amplitude SCALING at deployment, not reduced sim parameters.

**Status**: Reverted to Session 34/35 config. Use amplitude scaling when deploying to hardware.

### Session 37: Explicit Backlash Hysteresis (Desktop)

**BACKLASH HYSTERESIS MODEL FAILED**

Implemented explicit backlash hysteresis model. Policy cannot learn to compensate from scratch.

**Status**: Hysteresis DISABLED. Gaussian noise (1°) works better as regularization.

### Session 36 RPi: Hardware Gait Tuning (RPi)

**SMOOTH WALKING ACHIEVED**

Major breakthrough: iteratively tuned gait to smooth walking on carpet and hardwood.

Key changes:
- Fixed servo ACC=0→150 (max acceleration)
- Reduced thigh range from 40° to 7.5°
- Reduced calf range from 50° to 30°
- Slowed frequency from 0.7 Hz to 0.5 Hz

### Session 36 Desktop: Pure RL Experiments

**PURE RL PLATEAUS AT vx ≈ 0.01 m/s**

Pure RL from scratch cannot learn walking. Policy gets stuck in "standing local minimum".

---

### Session 35: Smooth Gait Development & Damping Optimization (Desktop)

**SMOOTH GAIT OPTIMIZED**:

Session 34 hardware test showed jerky walking. Session 35 completed damping sweep:

| Damping | vx (m/s) | Contact | Verdict |
|---------|----------|---------|---------|
| 30 (original) | 0.016 | -0.024 | WALKING (jerky) |
| 100 | 0.014 | -0.0002 | WALKING |
| 125 | 0.034 | -0.015 | WALKING |
| **150** | **0.036** | -0.014 | **WALKING (BEST)** |
| 175 | -0.024 | -0.002 | STANDING (too high) |

**Key Finding**: U-shaped velocity curve. Higher damping (125-150) gives faster walking!

**Final Configuration**:
| Parameter | Value |
|-----------|-------|
| damping | **150** |
| action_filter_beta | 0.40 |
| torque_penalty | -0.01 |
| action_rate_penalty | -0.1 |

**Next Step**: Test smooth gait policy on hardware.

### Session 34: Large Amplitude CPG for Backlash Tolerance (Desktop)

Session 33 discovered servo backlash was absorbing calf swing (initially estimated ~30°). Updated measurement (2026-01-03) indicates ~10° dead zone.
Session 34 increased calf amplitude to 50° to exceed backlash zone.

**Changes Applied**:
| Joint | Old Amplitude | New Amplitude |
|-------|--------------|---------------|
| Calf | 26° (-1.35 to -0.90) | **50°** (-1.38 to -0.50) |
| Thigh | 29° (0.40 to 0.90) | **40°** (0.25 to 0.95) |

**Training Result**: vx=0.020 m/s, WALKING (no regression)

### Session 33: Hardware Walking Test & Backlash Discovery (RPi)

**Critical Finding (updated)**: Measured servo backlash dead zone is ~10° (2026-01-03). Earlier ~30° estimate was incorrect.
- Feet never lifted with old gait
- Robot shuffled by pushing, not stepping
- Servos strong under load, issue is only at direction changes

### Session 32: Double-Normalization Bug Fix (Desktop)

Fixed critical bug where observations were normalized twice.

### Session 30: Joint Limit Alignment & CPG Optimization

1. **Joint limits aligned with hardware** - All limits match safe ranges
2. **CPG frequency optimized** - 0.7 Hz found optimal
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

### Best Configuration (Session 35)

| Parameter | Value | Notes |
|-----------|-------|-------|
| stiffness | 400 | Sim-to-real aligned |
| **damping** | **125** | **Reverted from 150 for safety** |
| action_filter_beta | 0.40 | Smooth action transitions |
| torque_penalty | -0.01 | Moderate torque penalty |
| action_rate_penalty | -0.1 | Penalizes rapid changes |
| CPG frequency | **0.7 Hz** | **OPTIMAL (Session 30 sweep)** |
| CPG mode | ENABLED | `HAROLD_CPG=1` |
| Command tracking | ENABLED | `HAROLD_CMD_TRACK=1` |
| Dynamic commands | ENABLED | `HAROLD_DYN_CMD=1` |
| vx_range | 0.10-0.45 | Optimal range |
| residual_scale | 0.05 | 0.08 causes regression |
| swing_calf | -1.38 | Session 34 large amplitude |
| stance_calf | -0.50 | Session 34 large amplitude |
| calf_spawn | -1.39 | Fixed to be within limit (-1.3963) |
| joint_position_noise | **0.0175 rad (~1°)** | **OPTIMAL for backlash** |
| episode_length_threshold | **300** | **15s minimum for stability** |

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

Prefer CLI flags (`--mode`, `--duration`, `--task`) for experiments; use env vars only for advanced overrides.

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
python scripts/harold.py train --mode cpg --duration short \
  --hypothesis "..."

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
