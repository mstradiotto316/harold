# Harold Next Steps

## PRIORITY 0: Test Smooth Gait on Hardware

**Status**: Policy exported and pushed. Ready for hardware test.

---

## Hardware Test Instructions (RPi Agent)

### Step 1: Pull Latest Changes
```bash
cd /home/matteo/Desktop/code_projects/harold
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
# Current: symmetric sine wave (reverses at mid-swing)
swing_lift = torch.sin(swing_progress * math.pi)

# Proposed: asymmetric (fast lift, hold, slow lower)
if swing_progress < 0.5:
    swing_lift = swing_progress * 2  # Fast lift (0→1)
elif swing_progress < 0.6:
    swing_lift = 1.0  # Hold at apex
else:
    swing_lift = 1.0 - (swing_progress - 0.6) / 0.4  # Slow lower (1→0)
```

---

## Technical Constraints (Reference)

### Backlash Behavior (Session 33)

| Motion Type | Backlash Impact |
|-------------|-----------------|
| Holding position | None - servos strong |
| Sustained push/pull | Minimal |
| Direction reversal | ~30° lost |
| Small oscillations | Entirely absorbed |

### Current Gait Parameters

- Calf amplitude: 50° (-1.38 to -0.50)
- Thigh amplitude: 40° (0.25 to 0.95)
- Frequency: 0.7 Hz
- Duty cycle: 60% stance, 40% swing

---

## Deployment Status

- ✅ ONNX validation passing
- ✅ Controller runs stable at 20 Hz
- ✅ Warmup removed (immediate policy engagement)
- ✅ harold.service disabled (manual control only)
- ⚠️ FL shoulder (ID 2) needs recalibration
