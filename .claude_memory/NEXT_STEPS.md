# Harold Next Steps

## PRIORITY 0: Deploy Double-Normalization Fix to Hardware

**Status**: Critical bug discovered and fixed on desktop. Ready to deploy to RPi.

### Background (Session 32)

The ONNX validation revealed a **double-normalization bug** in `harold_controller.py`:
- ONNX model includes normalization internally (via `NormalizedPolicy` wrapper in export)
- Deployment code was ALSO normalizing before passing to ONNX
- Result: Extreme policy outputs that Session 31 tried to fix with blending/overrides

The blending fixes in Session 31 were **compensating** for this bug, not fixing it.

### Files to Deploy to RPi

1. **`deployment/inference/harold_controller.py`** - Key change: pass RAW observations to ONNX
2. **`deployment/validation/validate_onnx_vs_sim.py`** - Uses raw observations
3. **`deployment/validation/sim_episode.json`** - Test data for validation (optional)

### Deployment Commands

```bash
# On desktop
cd ~/Desktop/code_projects/harold
scp deployment/inference/harold_controller.py pi@harold.local:~/Desktop/harold/deployment/inference/
scp deployment/validation/validate_onnx_vs_sim.py pi@harold.local:~/Desktop/harold/deployment/validation/
scp deployment/validation/sim_episode.json pi@harold.local:~/Desktop/harold/deployment/validation/

# On RPi - validate ONNX first
cd ~/Desktop/harold
python deployment/validation/validate_onnx_vs_sim.py

# If validation passes, run controller
python deployment/inference/harold_controller.py
```

### Expected Behavior After Fix

- Policy outputs should be bounded in a reasonable range without blending hacks
- Robot behavior should match simulation more closely
- Warmup blending may no longer be necessary

### What to Test

1. **Validation**: Run `validate_onnx_vs_sim.py` on RPi - should show near-zero error
2. **Controller without warmup**: Try `--warmup-cycles 0` to see if warmup is still needed
3. **Walking**: Does robot walk forward smoothly?

---

## Session 32 Fixes Applied (on Desktop)

| Fix | File | Description |
|-----|------|-------------|
| Remove manual normalization | `harold_controller.py:275-277` | ONNX normalizes internally |
| Use raw observations | `harold_controller.py:287-290` | Pass raw obs to ONNX, not normalized |
| Update validation | `validate_onnx_vs_sim.py:107-112` | Use obs_raw, not obs_normalized |

### Validation Results (Desktop)

```
ONNX output range: [-5.0492, 7.8849]
PyTorch output range: [-5.0492, 7.8849]
Max difference: 0.000003
✓ PASS: ONNX outputs match PyTorch outputs!
```

---

## Previous Session Summary

### Session 31 Achievements (RPi)

Fixed several deployment issues:
1. ✅ lin_vel stats override (training values were corrupted)
2. ✅ prev_target blending (10% actual, 90% training mean)
3. ✅ joint_pos blending during warmup

**NOTE**: These fixes may have been compensating for the double-normalization bug.

### Session 30 Achievements

1. Joint limits aligned with hardware
2. CPG frequency optimized to 0.7 Hz
3. Best policy exported - EXP-170 (vx=0.018 m/s)

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

## CRITICAL: ONNX Model Expects RAW Observations

The ONNX export (`policy/export_policy.py`) wraps the policy with `NormalizedPolicy`:

```python
class NormalizedPolicy(torch.nn.Module):
    def forward(self, obs: torch.Tensor):
        norm_obs = (obs - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        mean, value, log_std = self.base(norm_obs)
        return mean, value, log_std
```

**ALWAYS pass RAW observations to `harold_policy.onnx`.** Do NOT normalize before calling.

---

## Future Experiments (If Needed)

1. **Remove warmup blending** - May not be needed after fix
2. **Higher velocity** - Train with vx_target > 0.2 m/s
3. **Lateral motion** - Enable vy tracking
4. **Yaw tracking** - Curriculum approach
5. **Friction randomization** - For different floor surfaces
