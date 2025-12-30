# Harold Next Steps

## PRIORITY 0: Hardware Deployment (Session 30 Complete)

**Status**: NEW policy exported with optimized settings. Ready for testing.

### Session 30 Achievements

1. **Joint limits aligned with hardware** - All limits now match deployment/config/hardware.yaml
2. **CPG frequency optimized** - 0.7 Hz found optimal in sweep (better than 0.5 Hz)
3. **Best policy exported** - EXP-170 (vx=0.018 m/s)

### Updated Deployment Files

| File | Change |
|------|--------|
| `deployment/policy/harold_policy.onnx` | NEW - EXP-170 policy |
| `deployment/config/cpg.yaml` | `frequency_hz: 0.7`, `swing_calf: -1.35` |
| `simulation config` | Joint limits aligned with hardware |

---

## Deployment Steps

1. **Copy files to RPi 5**:
```bash
scp -r deployment/ pi@harold.local:~/harold/
```

2. **Install dependencies**:
```bash
pip install onnxruntime pyyaml smbus2
```

3. **Run controller**:
```bash
python inference/harold_controller.py
```

---

## Session 30 Experiment Summary

| EXP | Config | vx | Verdict |
|-----|--------|-----|---------|
| 160 | Shoulders ±25° | 0.016 | WALKING |
| 161 | Thighs aligned | 0.017 | WALKING |
| 162 | Calves aligned | 0.013 | WALKING |
| 163 | Extended training | 0.017 | WALKING |
| 164 | External perturbations | -0.044 | FAILING |
| 165 | swing_calf -1.35 | 0.011 | WALKING |
| 166 | residual_scale 0.08 | 0.007 | STANDING |
| 167 | Freq 0.6 Hz | 0.010 | STANDING |
| 168 | Freq 0.7 Hz | 0.016 | WALKING |
| 169 | Freq 0.8 Hz | 0.010 | STANDING |
| **170** | **0.7 Hz, extended** | **0.018** | **WALKING** |

---

## Key Findings (Session 30)

### What Works
- Joint limit alignment (doesn't hurt gait)
- CPG frequency 0.7 Hz (better than 0.5)
- swing_calf -1.35 (safety margin from limit)
- residual_scale 0.05 (optimal)

### What Fails
- External perturbations (even light forces cause falling)
- residual_scale 0.08 (too much policy authority)
- Frequency 0.6/0.8 Hz (worse than 0.7 Hz)

---

## Optimal Configuration

```python
# CPG (optimal)
base_frequency: 0.7  # Hz
swing_calf: -1.35    # Safety margin
residual_scale: 0.05

# Joint limits (simulation frame)
shoulder: ±0.4363 rad (±25°)
thigh: -0.0873 to +0.9599 rad (-5° to +55°)
calf: -1.3963 to +0.0873 rad (-80° to +5°)

# Domain randomization
joint_position_noise: 0.0175 rad (~1°)
add_lin_vel_noise: True
clip_observations: True
apply_external_forces: False
```

---

## Previous Deployment Fixes (Session 29-30)

1. ✅ Gravity sign flip (Z-up → Z-down)
2. ✅ Default commands 0.3 m/s
3. ✅ Observation clipping ±5.0
4. ✅ 3-cycle CPG warmup
5. ✅ Servo speed 2800
6. ✅ Default pose corrected
7. ✅ Voltage monitoring added

---

## Future Experiments (If Needed)

1. **Higher velocity** - Train with vx_target > 0.2 m/s
2. **Lateral motion** - Enable vy tracking
3. **Yaw tracking** - Curriculum approach
4. **Friction randomization** - For different floor surfaces
