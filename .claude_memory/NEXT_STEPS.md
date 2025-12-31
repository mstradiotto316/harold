# Harold Next Steps

## PRIORITY 0: Redesign CPG for Backlash-Tolerant Gait

**Status**: Hardware test revealed ~30° servo backlash prevents foot lifting with current CPG.

### Session 33 Hardware Test Summary

1. ✅ Double-normalization fix validated
2. ✅ Deployment pipeline works (20 Hz stable)
3. ❌ **Feet never lifted** - robot shuffled forward by dragging
4. ❌ Current CPG swing amplitude (~29°) entirely absorbed by backlash

### Root Cause

Servos have ~30° mechanical backlash. This only affects **direction reversals**, not sustained motion or holding under load.

Current sinusoidal CPG reverses direction twice per cycle → motion lost to backlash at each reversal.

---

## Experiments for Desktop Agent

### Experiment 1: Larger Amplitude CPG

**Hypothesis**: Swing amplitude > 45° will exceed backlash zone and produce actual foot lift.

**Changes to test**:
```python
# In CPGCfg (harold_isaac_lab_env_cfg.py)
swing_calf: -1.50  → -1.70  # Increase swing amplitude
stance_calf: -0.90 → -0.60  # More extension during stance
# Total amplitude: ~63° (vs current ~29°)
```

### Experiment 2: Asymmetric Trajectory

**Hypothesis**: Fast swing + slow stance avoids backlash issues.

**Concept**:
- Swing phase (30% duty): Fast, unidirectional motion
- Stance phase (70% duty): Slow, sustained push
- Brief pause at transitions to pre-load against backlash

**Implementation**: Modify `_compute_leg_trajectory()` to use asymmetric waveform instead of sinusoid.

### Experiment 3: Higher Observation Noise for Backlash Robustness

**Hypothesis**: Training with 15° position noise (half of backlash) improves transfer.

**Changes**:
```python
# In DomainRandomizationCfg
joint_position_noise: 0.0175 → 0.26  # ~15° instead of ~1°
```

**Caution**: Session 28 showed 2° noise hurt training. May need curriculum.

---

## Technical Constraints for Gait Design

### Backlash Behavior (from hardware testing)

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

But first: **get feet off the ground with backlash-aware gait design**.
