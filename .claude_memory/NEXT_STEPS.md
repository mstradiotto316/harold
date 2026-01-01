# Harold Next Steps

## PRIORITY 0: Break the Pure RL Plateau

**Status**: Session 36 complete. Pure RL plateaus at vx ≈ 0.01 m/s after 13 experiments.

### What Was Tried (All Failed to Break Plateau)
| Approach | Result |
|----------|--------|
| forward_motion weight 3.0, 5.0, 10.0 | 3.0 best (vx=+0.010), others worse |
| Extended training (4167 iter) | Same plateau |
| Reduced lin_vel_z (-0.0001) | Same plateau |
| Increased feet_air_time (1.0) | Same plateau |
| Higher velocity commands (0.15-0.5) | Same plateau |

### Next Experiments to Try (Priority Order)

1. **Fine-tune from CPG checkpoint**
   - CPG policy already walks (vx=0.034 in Session 35)
   - Challenge: Observation space mismatch (50D CPG vs 48D pure RL)
   - Solution: Either use CPG mode or add gait phase back to pure RL

2. **Curriculum learning**
   ```python
   # In harold_isaac_lab_env.py
   vx_cmd = 0.05 * min(iteration / 1000, 1.0)  # Ramp up over 1000 iter
   ```

3. **Asymmetric forward reward**
   ```python
   # Penalize backwards much more than reward forwards
   forward_motion = 3.0 * max(vx, 0) - 10.0 * min(vx, 0)
   ```

4. **Much longer training** (10,000+ iter = 4+ hours)

5. **Reference motion tracking** (use CPG trajectory as reference)

---

## Current Pure RL Configuration

File: `harold_isaac_lab_env_cfg.py`

```python
# Rewards (Session 36 final)
track_lin_vel_xy_weight = 5.0
forward_motion_weight = 3.0      # Best from sweep
lin_vel_z_weight = -0.0001       # 4000x smaller than Isaac Lab default
feet_air_time_weight = 1.0

# Commands
vx_min = 0.0, vx_max = 0.3       # Conservative
```

### Training Command
```bash
HAROLD_CMD_TRACK=1 HAROLD_DYN_CMD=1 python scripts/harold.py train \
  --hypothesis "description" --iterations 2500
```

### To Revert to CPG Mode (Working)
```bash
HAROLD_CPG=1 HAROLD_CMD_TRACK=1 HAROLD_DYN_CMD=1 python scripts/harold.py train ...
```

---

## Deployment Status

- **Last deployed**: Session 35 CPG policy (damping=125, vx=0.034)
- **Pure RL**: NOT ready for hardware (vx=0.01 too slow)
- **Recommended**: Continue using CPG-based policy for hardware tests

---

## Key Insight from Session 36

The policy finds a **standing local minimum** because:
1. Upright reward gives 0.97 when standing
2. Exponential velocity tracking gives ~0.9 even when vx=0
3. Forward motion bonus (3.0 * vx) is too weak to overcome equilibrium
4. All penalty terms are minimized by standing still

To break this, need either:
- Stronger incentive to move (asymmetric reward, curriculum)
- Prior knowledge of walking (fine-tune from CPG)
- Much more exploration time (longer training)
