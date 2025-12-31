# Harold Next Steps

## PRIORITY 0: Observe Robot Walking Behavior

**Status**: Controller validated, ready for walking tests.

### Session 33 Completed (RPi)

1. ✅ ONNX validation passed (max error 0.000001)
2. ✅ Controller runs stable at 20 Hz
3. ✅ Warmup no longer necessary (tested `--warmup-cycles 0`)
4. ✅ Double-normalization fix confirmed working

### Next Hardware Tests

1. **Run controller and observe walking**
   ```bash
   python deployment/inference/harold_controller.py
   ```
   - Does robot walk forward?
   - Is motion smooth?
   - Any instability at startup?

2. **Test without warmup permanently**
   ```bash
   python deployment/inference/harold_controller.py --warmup-cycles 0
   ```
   - Compare behavior to warmup mode
   - If stable, consider making 0 the default

3. **Surface testing**
   - Hardwood (lower friction)
   - Short carpet (medium friction)
   - Long carpet (higher friction)

---

## Session 32-33 Summary

### Bug Fixed
- **Double-normalization**: Deployment was normalizing observations twice
- **Fix**: Pass RAW observations to ONNX (it normalizes internally)
- **Validation**: ONNX matches PyTorch with error < 0.000003

### Warmup Status
- Previously: 3 CPG cycles before policy (compensating for bug)
- Now: Optional (`--warmup-cycles 0` works fine)
- Default still 3 for safety, but can be disabled

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

## Future Experiments (If Current Policy Insufficient)

1. **Higher velocity training** - vx_target > 0.2 m/s
2. **Lateral motion** - Enable vy tracking
3. **Yaw tracking** - Curriculum approach (backlash → yaw)
4. **Friction randomization** - For different floor surfaces
5. **New policy export** - If EXP-170 underperforms on hardware

---

## Controller CLI Reference

```bash
# Basic run
python deployment/inference/harold_controller.py

# No warmup (now safe after bug fix)
python deployment/inference/harold_controller.py --warmup-cycles 0

# Skip IMU calibration
python deployment/inference/harold_controller.py --calibrate 0

# Direct policy mode (no CPG - not recommended)
python deployment/inference/harold_controller.py --no-cpg
```
