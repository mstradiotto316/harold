# Session 38: Hardware-Validated CPG Parameters (Desktop)

**Date**: 2026-01-01
**Goal**: Apply hardware-validated gait parameters from Session 36 RPi to simulation

## Summary

Tested whether hardware-validated gait parameters would improve simulation training.
**RESULT: FAILED** - Hardware parameters are too conservative for simulation.

## Key Finding

**Sim-to-real transfer needs amplitude SCALING at deployment, not reduced sim parameters.**

| Configuration | vx (m/s) | Upright | Result |
|---------------|----------|---------|--------|
| Hardware-validated (7.5°/30° @0.5Hz) | 0.003 | 0.98 | STANDING |
| Intermediate (15°/40° @0.5Hz) | 0.004 | 0.98 | STANDING |
| Session 35 (40°/50° @0.7Hz) | 0.036 | 0.97 | **WALKING** |

The robot was very stable (upright=0.98) but barely moving with smaller amplitudes.

## Experiments Run

### EXP-205: Hardware-Validated CPG
- **Config**: 0.5 Hz, thigh 7.5°, calf 30° (matching hardware)
- **Result**: vx=0.003 m/s (STANDING)
- **Conclusion**: Too conservative for simulation

### EXP-206: Intermediate CPG
- **Config**: 0.5 Hz, thigh 15°, calf 40° (middle ground)
- **Result**: vx=0.004 m/s (STANDING)
- **Conclusion**: Still too conservative

## Technical Analysis

### Why Hardware Params Fail in Sim

1. **Different physics**: Sim has different contact dynamics, friction, inertia
2. **No backlash modeling**: Sim doesn't have the ~30° servo backlash that requires hardware to use faster acceleration
3. **Training dynamics**: RL needs larger motions to learn walking from scratch

### The Sim-to-Real Gap

| Parameter | Simulation | Hardware | Gap Factor |
|-----------|------------|----------|------------|
| Thigh amplitude | 40° | 7.5° | **5.3x** |
| Calf amplitude | 50° | 30° | 1.7x |
| Frequency | 0.7 Hz | 0.5 Hz | 1.4x |

### Recommended Approach for Transfer

Instead of reducing sim parameters:
1. **Train in sim with original params** (vx=0.036)
2. **Apply amplitude scaling at deployment**:
   - Thigh output: scale by 0.2 (7.5/40)
   - Calf output: scale by 0.6 (30/50)
3. **Reduce frequency at deployment**: 0.5 Hz instead of 0.7 Hz

## Configuration Reverted

Reverted to Session 34/35 parameters:
```python
# CPGCfg
base_frequency: float = 0.7  # Hz (was 0.5)
swing_thigh: float = 0.25    # (was 0.47)
stance_thigh: float = 0.95   # (was 0.73)
stance_calf: float = -0.50   # (was -0.59)
swing_calf: float = -1.38    # (was -1.29)
```

## Implications for Deployment

When deploying to hardware:
1. Use Session 35 policy checkpoint (vx=0.036 in sim)
2. Apply amplitude scaling to action outputs
3. Consider lower frequency in deployment inference

## Files Modified

- `harold_isaac_lab_env_cfg.py`: Tested hardware params, then reverted
- `docs/memory/sessions/2026-01-01_session38_desktop.md`: This log

## Lessons Learned

1. Sim and hardware require DIFFERENT amplitude settings
2. What works on hardware doesn't work in simulation (and vice versa)
3. The transfer should happen at deployment (scaling), not training (reduced params)
4. 0.5 Hz frequency is good for hardware, 0.7 Hz is better for sim learning
