# Harold Next Steps

## PRIORITY 0: Validate ONNX Policy Against Simulation (Session 31)

**Status**: Robot shows walking-like behavior but needs validation. Need to record simulation data to compare against ONNX policy outputs.

### Background (Session 31 on RPi)

The deployment code was stabilized with several fixes:
1. **lin_vel override** - Training stats were corrupted (Z vel mean = -44 m/s impossible)
2. **prev_target blending** - 10% actual, 90% training mean to prevent feedback divergence
3. **joint_pos blending** - 30% actual during warmup, gradual transition to 100%

Robot showed walking-like motion on test stand, but suspended robot means IMU readings may not match what policy expects (no real acceleration/movement).

### Validation Task (RUN ON DESKTOP)

**Goal**: Record observations and actions from Isaac Lab simulation, then compare ONNX policy outputs against recorded actions.

#### Step 1: Create Recording Script

Create `harold_isaac_lab/scripts/skrl/record_episode.py`:

```python
"""Record observations and actions from a trained policy for ONNX validation."""

import torch
import numpy as np
import json
from pathlib import Path

# Load checkpoint and run episode
# For each timestep, record:
#   - Raw observation (50D, before normalization)
#   - Normalized observation (50D, after normalization)
#   - Policy action output (12D)
#   - Running mean/var used for normalization

# Save to JSON file for transfer to RPi
```

#### Step 2: Run Recording

```bash
cd ~/Desktop/code_projects/harold
source ~/Desktop/env_isaaclab/bin/activate

python harold_isaac_lab/scripts/skrl/record_episode.py \
    --checkpoint logs/skrl/harold_direct/2024-XX-XX_EXP-170/checkpoints/best_agent.pt \
    --output deployment/validation/sim_episode.json \
    --num_steps 200
```

#### Step 3: Transfer to RPi and Validate

```bash
# On desktop
scp deployment/validation/sim_episode.json pi@harold.local:~/Desktop/harold/deployment/validation/

# On RPi - run validation script (already exists or create)
python validate_onnx_vs_sim.py
```

#### Expected Output

For each timestep, compare:
- `sim_action` (from recording) vs `onnx_action` (from ONNX inference)
- Should match within floating point tolerance (~1e-5)

If they don't match, the discrepancy reveals:
- Observation format mismatch
- Normalization differences
- Coordinate convention issues

---

## Session 31 Fixes Applied (on RPi)

| Fix | File | Description |
|-----|------|-------------|
| lin_vel override | `harold_controller.py:131-137` | mean=0, std=0.5 (training had impossible values) |
| prev_target blend | `observation_builder.py:215-243` | 10% actual, 90% training mean |
| joint_pos blend | `harold_controller.py:255-272` | 30% during warmup → 100% after transition |

### Test Results (Simulated Loop)

| Metric | Before Fixes | After Fixes |
|--------|-------------|-------------|
| Policy output range | [-36, +25] | [-8, +14] |
| Max correction | 93° | 41° |
| Stability | Divergent | Bounded |

---

## Previous Session Summary

### Session 30 Achievements
1. Joint limits aligned with hardware
2. CPG frequency optimized to 0.7 Hz
3. Best policy exported - EXP-170 (vx=0.018 m/s)

### Session 30 Experiments

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

## Previous Deployment Fixes (Session 29-31)

1. ✅ Gravity sign flip (Z-up → Z-down)
2. ✅ Default commands 0.3 m/s
3. ✅ Observation clipping ±5.0
4. ✅ 3-cycle CPG warmup
5. ✅ Servo speed 2800
6. ✅ Default pose corrected
7. ✅ Voltage monitoring added
8. ✅ lin_vel stats override (Session 31)
9. ✅ prev_target blending (Session 31)
10. ✅ joint_pos blending (Session 31)

---

## Future Experiments (If Needed)

1. **Higher velocity** - Train with vx_target > 0.2 m/s
2. **Lateral motion** - Enable vy tracking
3. **Yaw tracking** - Curriculum approach
4. **Friction randomization** - For different floor surfaces
