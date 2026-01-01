# Harold Next Steps

## PRIORITY 0: Test Acceleration Fix + Smooth Gait on Hardware

**Status**: CRITICAL firmware fix applied. Ready for hardware test.

---

## What Changed (Session 36 RPi - CRITICAL)

### Servo Acceleration Bug FIXED

**The Problem**: Firmware had `GAIT_ACC = 0` which meant **slowest** servo acceleration (soft ramp-up), NOT fastest. This was fighting against backlash mitigation.

**The Fix**: Changed `GAIT_ACC` from `0` to `150` (maximum/instant acceleration).

**Expected Impact**:
- Servos will now cross the backlash dead zone as fast as possible
- Combined with large amplitude (50° calf, 40° thigh), feet should actually lift
- This is "bang-bang" control - optimal for systems with mechanical play

### Also Added
- Explicit `st.EnableTorque(id, 1)` for all 12 servos during setup

---

## Hardware Test Instructions

### Option A: Test Scripted Gait (Firmware Only - Recommended First)

This tests the acceleration fix directly without the policy involved.

1. **Upload firmware to ESP32**:
   - Open `firmware/scripted_gait_test_1/scripted_gait_test_1.ino` in Arduino IDE
   - Select your ESP32 board and port
   - Upload

2. **Run test**:
   - Open Serial Monitor at 115200 baud
   - Robot will auto-start walking after warmup
   - Watch for "Acceleration: 150 (max=150 for instant response)" in startup output

3. **Observe**:
   - Do feet actually LIFT off the ground now?
   - Is motion snappier/more responsive than before?
   - Any issues with the instant acceleration?

### Option B: Test Policy Controller (RPi)

### Step 1: Pull Latest Changes
```bash
cd /home/pi/Desktop/harold
git pull
```

### Step 2: Run the Controller
```bash
source ~/harold_env/bin/activate  # or your venv path
python deployment/inference/harold_controller.py
```

### Step 3: Observe and Report

**Key questions to answer:**
1. Is the gait smoother than before? (Less jerky foot impacts?)
2. Does the robot still walk forward?
3. Any new issues introduced?

**Success criteria:**
- Noticeably reduced shock/jerk on foot contact
- Robot still walks forward (even if slower)
- No instability or falling

---

## What Changed (Session 35)

Session 34's large amplitude policy walked but was **extremely jerky with harsh shock absorption**. Session 35 added smoothing:

| Parameter | Before | After | Effect |
|-----------|--------|-------|--------|
| damping | 30 | **60** | Reduces oscillations at foot contact |
| action_filter_beta | 0.18 | **0.35** | Stronger EMA smoothing on actions |

Training result: vx=0.016 m/s (slightly lower than 0.020, expected with damping).

### Files Changed

1. `harold.py` - damping 30→60
2. `harold_isaac_lab_env_cfg.py` - action_filter_beta 0.18→0.35
3. `deployment/config/cpg.yaml` - action_filter_beta 0.18→0.35
4. `deployment/policy/harold_policy.onnx` - New smoothed policy

---

## If Still Too Jerky: Phase 2 Changes

Report back to desktop agent and we'll add:

1. **Stronger torque penalty**: -0.005 → -0.02
2. **Foot impact force penalty**: New reward penalizing harsh foot strikes
3. **Action rate penalty**: New reward penalizing rapid action changes

---

## If Hardware Test Succeeds

Next priorities:
1. Document successful smooth gait parameters
2. Consider further tuning (damping, filter strength)
3. Work toward controllable walking (vx, vy, yaw commands)

---

## Fallback Options (If Phase 2 Also Fails)

1. **Even higher damping** - Try damping=75
2. **Slower frequency** - Try 0.5 Hz instead of 0.7 Hz
3. **Asymmetric trajectory** - Fast swing, slow stance (see below)

---

## Reference: Asymmetric Trajectory Option

If sinusoidal trajectory still causes issues, modify `_compute_leg_trajectory()`:

```python
# Rewards (Session 36 final)
track_lin_vel_xy_weight = 5.0
forward_motion_weight = 3.0      # Best from sweep
lin_vel_z_weight = -0.0001       # 4000x smaller than Isaac Lab default
feet_air_time_weight = 1.0

# Commands
vx_min = 0.0, vx_max = 0.3       # Conservative
```

### Training Command
```bash
HAROLD_CMD_TRACK=1 HAROLD_DYN_CMD=1 python scripts/harold.py train \
  --hypothesis "description" --iterations 2500
```

### To Revert to CPG Mode (Working)
```bash
HAROLD_CPG=1 HAROLD_CMD_TRACK=1 HAROLD_DYN_CMD=1 python scripts/harold.py train ...
```

---

## Deployment Status

- **Last deployed**: Session 35 CPG policy (damping=125, vx=0.034)
- **Pure RL**: NOT ready for hardware (vx=0.01 too slow)
- **Recommended**: Continue using CPG-based policy for hardware tests

---

## Key Insight from Session 36

The policy finds a **standing local minimum** because:
1. Upright reward gives 0.97 when standing
2. Exponential velocity tracking gives ~0.9 even when vx=0
3. Forward motion bonus (3.0 * vx) is too weak to overcome equilibrium
4. All penalty terms are minimized by standing still

To break this, need either:
- Stronger incentive to move (asymmetric reward, curriculum)
- Prior knowledge of walking (fine-tune from CPG)
- Much more exploration time (longer training)
