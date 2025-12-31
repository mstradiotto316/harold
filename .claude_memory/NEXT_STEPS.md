# Harold Next Steps

## PRIORITY 0: Test Large Amplitude CPG on Hardware

**Status**: Policy exported, ready for hardware test.

### For the RPi Agent: Immediate Actions

```bash
# 1. Pull Session 34 changes
cd /home/matteo/Desktop/code_projects/harold
git pull

# 2. Run the controller with new large-amplitude gait
python deployment/inference/harold_controller.py

# 3. Observe and report:
#    - Do feet actually lift off the ground now?
#    - How much lift? (estimate in mm or degrees)
#    - Any stability issues with larger motion?
#    - Does robot walk forward better than Session 33?
```

**What changed**: Calf swing increased from 26° to 50° to exceed 30° backlash.

**Expected outcome**: Feet should lift ~20° (50° swing - 30° backlash).

**If feet still don't lift**: Try asymmetric trajectory (see Experiment 2 below).

---

### Session 34 Summary

1. ✅ Designed backlash-tolerant large amplitude trajectory (50° calf swing)
2. ✅ Trained and validated in simulation (vx=0.020 m/s, WALKING)
3. ✅ Exported policy and updated deployment config
4. 🔲 **Test on hardware** - Will feet lift with 50° swing vs 30° backlash?

### Changes Applied

| Joint | Old Amplitude | New Amplitude |
|-------|--------------|---------------|
| Calf | 26° (-1.35 to -0.90) | **50°** (-1.38 to -0.50) |
| Thigh | 29° (0.40 to 0.90) | **40°** (0.25 to 0.95) |

### Files Updated

- `harold_isaac_lab_env_cfg.py` - CPGCfg & ScriptedGaitCfg
- `deployment/config/cpg.yaml` - Hardware trajectory parameters
- `deployment/policy/harold_policy.onnx` - New trained policy

---

## If Hardware Test Still Fails: Experiment 2 - Asymmetric Trajectory

**Hypothesis**: Fast swing + slow stance avoids backlash issues better than larger sinusoid.

**Concept**:
- Swing phase (30% duty): Fast, unidirectional motion (no reversal)
- Brief pause at apex (pre-load against backlash)
- Stance phase (70% duty): Slow, sustained push

**Implementation**: Modify `_compute_leg_trajectory()` in `harold_isaac_lab_env.py`.

Current trajectory uses sinusoidal swing:
```python
swing_lift = torch.sin(swing_progress * math.pi)  # 0 -> 1 -> 0
calf_swing = stance_calf + (swing_calf - stance_calf) * swing_lift
```

This reverses direction at mid-swing (loses 30° to backlash each reversal).

Proposed asymmetric trajectory:
```python
# Swing phase: fast unidirectional motion (first 60% of swing)
# Apex: brief hold (60-70% of swing)
# Return: controlled descent (70-100% of swing)
if swing_progress < 0.6:
    # Fast lift (0 -> 1)
    calf_swing = stance_calf + (swing_calf - stance_calf) * (swing_progress / 0.6)
elif swing_progress < 0.7:
    # Hold at apex (pre-load)
    calf_swing = swing_calf
else:
    # Controlled return (1 -> 0)
    calf_swing = stance_calf + (swing_calf - stance_calf) * (1.0 - (swing_progress - 0.7) / 0.3)
```

---

## If Both Fail: Experiment 3 - Higher Position Noise

**Hypothesis**: Training with 15° position noise (half of backlash) improves transfer.

**Caution**: Session 28 showed 2° noise hurt training. May need curriculum.

**Changes**:
```python
# In DomainRandomizationCfg
joint_position_noise: 0.0175 → 0.26  # ~15° instead of ~1°
```

---

## Technical Constraints for Gait Design

### Backlash Behavior (from Session 33 hardware testing)

| Motion Type | Backlash Impact |
|-------------|-----------------|
| Holding position | None - servos strong |
| Sustained push/pull | Minimal - motion transfers |
| Direction reversal | ~30° lost |
| Small oscillations | Entirely absorbed |

### Design Rules

1. **Amplitude > 45°** for any joint motion to exceed backlash
2. **Avoid sinusoids** - they reverse direction smoothly (bad for backlash)
3. **Use asymmetric waveforms** - fast one way, slow the other
4. **Pre-load at apex** - brief pause before load-bearing phase
5. **Unidirectional phases** - complete full motion before reversing

---

## Deployment Status

- ✅ ONNX validation passing
- ✅ Controller runs stable at 20 Hz
- ✅ Warmup removed (immediate policy engagement)
- ✅ harold.service disabled (manual control only)
- ⚠️ FL shoulder (ID 2) needs recalibration

---

## Long-Term Goal Reminder

End goal is **remote-controlled walking in any direction + rotation**.

Current CPG is forward-only. After solving backlash issue, architecture needs:
- Parameterized CPG taking (vx, vy, yaw) commands, OR
- Higher policy authority (residual_scale > 0.05), OR
- Pure RL without CPG base trajectory

But first: **test large amplitude on hardware**.
