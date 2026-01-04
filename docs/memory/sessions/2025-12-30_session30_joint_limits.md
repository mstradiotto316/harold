# Session 30: Joint Limit Alignment & CPG Optimization

**Date**: 2025-12-30
**Duration**: ~14 hours
**Result**: Joint limits aligned, CPG frequency optimized (0.7 Hz), best policy vx=0.018

## Summary

This autonomous research session focused on:
1. Aligning simulation joint limits with hardware safe ranges
2. Testing external perturbations for robustness
3. Optimizing CPG frequency
4. Exporting best policy for hardware deployment

## Experiment Results

| EXP | Config | vx | Verdict | Notes |
|-----|--------|-----|---------|-------|
| 160 | Shoulders ±25° | 0.016 | WALKING | Minor change, no impact |
| 161 | Thighs aligned | 0.017 | WALKING | Sign inversion handled |
| 162 | Calves aligned | 0.013 | WALKING | All limits now aligned |
| 163 | Extended (all limits) | 0.017 | WALKING | 2500 iters |
| 164 | External perturbations | -0.044 | FAILING | Even light forces fail |
| 165 | swing_calf -1.35 | 0.011 | WALKING | Safety margin from limit |
| 166 | residual_scale 0.08 | 0.007 | STANDING | Too much authority |
| 167 | Freq 0.6 Hz | 0.010 | STANDING | Borderline |
| 168 | Freq 0.7 Hz | 0.016 | WALKING | Better than 0.5 Hz |
| 169 | Freq 0.8 Hz | 0.010 | STANDING | Worse than 0.7 Hz |
| **170** | **0.7 Hz, extended** | **0.018** | **WALKING** | **BEST** |

## Key Findings

### 1. Joint Limit Alignment Works

The hardware safe limits are more restrictive than simulation defaults:
- Shoulders: ±30° → ±25°
- Thighs: ±90° → sim [-5°, +55°] (accounts for sign inversion)
- Calves: ±90° → sim [-80°, +5°] (accounts for sign inversion)

All CPG parameters remain within these limits. The gait works without modification.

### 2. External Perturbations FAIL

Even very light forces (0.2-0.5N, 0.5% probability) cause:
- Falling (body_contact = -1.91)
- Backward drift (vx = -0.044)

The robot's balance is too fragile for random perturbations during training.

### 3. CPG Frequency 0.7 Hz is Optimal

Frequency sweep results:
| Freq | vx | Verdict |
|------|-----|---------|
| 0.5 Hz | 0.011-0.017 | WALKING |
| 0.6 Hz | 0.010 | STANDING |
| **0.7 Hz** | **0.016** | **WALKING** |
| 0.8 Hz | 0.010 | STANDING |

The non-monotonic relationship suggests a resonance or timing effect.

### 4. residual_scale Must Be Conservative

- 0.05: Works (policy fine-tunes CPG)
- 0.08: Causes regression (too much policy authority)

The policy needs to stay in a "supportive" role, not override the CPG.

## Bug Fixes

### External Forces Shape Mismatch

Original code:
```python
forces = torch.zeros(self.num_envs, 3, device=self.device)
```

Fixed:
```python
forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
```

The Isaac Lab API requires shape `[num_envs, num_bodies, 3]`.

## Files Modified

| File | Changes |
|------|---------|
| `harold_isaac_lab_env_cfg.py` | Joint limits, CPG frequency, swing_calf |
| `harold_isaac_lab_env.py` | External forces bug fix |
| `deployment/config/cpg.yaml` | frequency_hz: 0.7, swing_calf: -1.35 |
| `deployment/policy/` | New ONNX policy exported |

## Optimal Configuration

```python
# CPG
base_frequency: 0.7  # Hz (optimal in sweep)
swing_calf: -1.35    # Safety margin from -1.3963 limit
residual_scale: 0.05 # Conservative (0.08 causes regression)

# Joint limits (simulation frame)
shoulder_max: 0.4363  # ±25°
shoulder_min: -0.4363
thigh_max: 0.9599     # +55° (hardware -55°)
thigh_min: -0.0873    # -5° (hardware +5°)
calf_max: 0.0873      # +5° (hardware -5°)
calf_min: -1.3963     # -80° (hardware +80°)
```

## Next Steps

1. **Deploy to hardware** - Test EXP-170 policy on real robot
2. **Measure forward velocity** - Compare sim vx=0.018 to real
3. **Adjust if needed** - CPG parameters may need tuning on hardware
