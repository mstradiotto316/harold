# Session 32: ONNX Validation & Double-Normalization Fix (Desktop)

**Date**: 2025-12-30
**Location**: Desktop Computer
**Focus**: Validate ONNX export and fix critical deployment bug

## Summary

Discovered and fixed a critical **double-normalization bug** in the deployment code. This was likely the root cause of unstable policy outputs during hardware deployment.

## Critical Discovery: Double Normalization Bug

The ONNX export (`policy/export_policy.py`) wraps the policy with `NormalizedPolicy`:

```python
class NormalizedPolicy(torch.nn.Module):
    def forward(self, obs: torch.Tensor):
        norm_obs = (obs - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        mean, value, log_std = self.base(norm_obs)
        return mean, value, log_std
```

This means the ONNX model **already includes normalization internally** and expects **raw observations**.

However, the deployment code (`harold_controller.py`) was doing:
```python
obs_norm = normalize_observation(obs, self.running_mean, self.running_var)  # First normalization
outputs = self.policy.run(['mean'], {'obs': obs_norm...})  # ONNX does second normalization!
```

This double normalization caused extreme values: for an observation at the mean, the first normalization gives 0, then the second gives `(0 - mean) / std = -mean/std`, which is a large value.

## Fixes Applied

### 1. harold_controller.py - Remove Manual Normalization
- Removed `normalize_observation` call before ONNX inference
- ONNX model now receives raw observations
- Added clear comments explaining this

### 2. validate_onnx_vs_sim.py - Use Raw Observations
- Updated to pass `obs_raw` (not `obs_normalized`) to ONNX
- Added documentation about ONNX internals

### 3. Created validate_onnx_quick.py
- Fast validation without needing Isaac Lab simulation
- Confirms ONNX matches PyTorch with max diff of 0.000003

## Validation Results

```
ONNX output range: [-5.0492, 7.8849]
PyTorch output range: [-5.0492, 7.8849]
Max difference: 0.000003

PASS: ONNX outputs match PyTorch outputs!
```

## Session 31 Fixes Revisited

The Session 31 fixes on RPi (lin_vel override, prev_target blending, joint_pos blending) were actually **compensating for double normalization**:
- Overriding lin_vel stats: Tried to fix extreme normalized values caused by double normalization
- Blending with training mean: Kept normalized values closer to 0

With the double-normalization fix, these compensations may no longer be necessary.

## Next Steps (Priority 0)

1. **Deploy to RPi**: Copy updated `harold_controller.py` to robot
2. **Test on Hardware**: Re-run with raw observations to ONNX
3. **Simplify Warmup**: With correct normalization, warmup blending may not be needed

## Files Modified

- `deployment/inference/harold_controller.py` - Removed manual normalization
- `deployment/validation/validate_onnx_vs_sim.py` - Use raw observations
- `deployment/validation/validate_onnx_quick.py` - NEW: Fast validation script
- `harold_isaac_lab/scripts/skrl/record_episode.py` - NEW: Full simulation recording (unused due to boot time)

## Key Insight

**Always check if ONNX models have preprocessing baked in.** The export script should document whether normalization is included or expected externally.
