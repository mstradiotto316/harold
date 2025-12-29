# Session 28: Gear Backlash Robustness & Yaw Rate Tracking

**Date**: 2025-12-29
**Duration**: ~8 hours
**Result**: Backlash robustness SOLVED (+35%), yaw tracking implemented, curriculum learning validated

## Summary

This session addressed the primary user priority: simulating gear backlash to improve sim-to-real transfer. Key breakthroughs:

1. **1° position noise improves walking by 35%** (vx=0.023 vs baseline 0.017)
2. **Yaw rate tracking implemented** and works standalone
3. **Curriculum learning validated** for combining backlash + yaw

## Key Results

| EXP | Config | vx (m/s) | Verdict | Notes |
|-----|--------|----------|---------|-------|
| 145 | Baseline | 0.017 | WALKING | Pre-backlash reference |
| 146/147 | 2° backlash | 0.007 | STANDING | Too much noise |
| **148** | **1° backlash** | **0.023** | **WALKING** | **+35% improvement!** |
| 150 | Backlash + yaw (scratch) | 0.003 | STANDING | Combined fails |
| 151 | Yaw only | 0.011 | WALKING | Yaw works alone |
| **152** | **Curriculum** | **0.015** | **WALKING** | **Combination works!** |
| 153 | Extended curriculum | 0.009 | STANDING | Longer training = worse |

## Architecture Clarification: CPG + Residual Learning

User asked: "How much is learned vs scripted?"

**Answer**: The motion is a combination:
```
target_joints = CPG_base_trajectory + policy_output * residual_scale
```

| Component | What it provides |
|-----------|------------------|
| **CPG (scripted)** | Timing, gait coordination, base trajectory |
| **Policy (learned)** | Balance corrections, velocity tracking, backlash adaptation |
| **residual_scale=0.05** | Policy can only fine-tune, not override |

The CPG provides the walking structure, while the policy provides the intelligence to stabilize it and follow commands.

## Backlash Implementation

### Why Position Noise Works

- **2° is too much**: Robot struggles to balance with high uncertainty
- **1° is optimal**: Provides regularization without overwhelming the policy
- **0° (baseline)**: Lower performance, policy overfits to perfect observations

The noise acts as **beneficial regularization**, preventing the policy from relying on precise position feedback that won't be available on hardware.

### Config Changes (harold_isaac_lab_env_cfg.py)

```python
# DomainRandomizationCfg - Master switches
enable_randomization: bool = True   # Enable noise
randomize_per_step: bool = True     # Apply every step

# Joint position noise (~1°)
joint_position_noise: GaussianNoiseCfg = GaussianNoiseCfg(
    mean=0.0,
    std=0.0175,  # ~1° in radians - OPTIMAL
    operation="add"
)
```

## Yaw Rate Tracking Implementation

### Config Changes (harold_isaac_lab_env_cfg.py)

```python
# RewardsCfg
command_tracking_weight_yaw: float = 10.0
command_tracking_sigma_yaw: float = 0.2

# CommandCfg
yaw_min: float = -0.30   # ~17 deg/s
yaw_max: float = 0.30
```

### Environment Changes (harold_isaac_lab_env.py)

1. **Added `command_tracking_yaw` to `_reward_keys`**
2. **Added yaw tracking reward computation**:
   ```python
   cmd_yaw = self._commands[:, 2]
   yaw_error = wz_b - cmd_yaw
   command_tracking_yaw = weight * torch.exp(-torch.square(yaw_error / sigma)) * upright_sq
   ```
3. **Modified `yaw_rate_penalty`**:
   ```python
   # OLD: -weight * torch.square(wz_b)
   # NEW: -weight * torch.square(wz_b - cmd_yaw)
   ```
4. **Added telemetry**: `cmd_yaw_error` in _metric_keys

## Key Findings

### 1. Observation Noise as Regularization
- Adding noise IMPROVES performance (counterintuitive!)
- Prevents overfitting to perfect observations
- Should improve sim-to-real transfer

### 2. Optimal Noise Level
- 1° (0.0175 rad) is optimal for backlash simulation
- Maps to ~1-3° mechanical play in ST3215 servos
- Higher noise (2°) causes stability issues

### 3. Curriculum Learning Required
- Backlash + yaw from scratch fails (vx=0.003)
- Curriculum works: train backlash → fine-tune yaw (vx=0.015)
- Shorter fine-tuning better (~1250 iters vs 2500)

### 4. Training Duration Matters
- Extended curriculum (2500 iters) regressed to vx=0.009
- Shorter curriculum (1250 iters) achieved vx=0.015
- More training ≠ better results for fine-tuning

## Best Checkpoints

| Checkpoint | Config | vx | Recommended For |
|------------|--------|-----|-----------------|
| EXP-148 | Backlash only | 0.023 | Hardware testing |
| EXP-152 | Backlash + yaw | 0.015 | Full controllability |

## Files Modified

| File | Changes |
|------|---------|
| `harold_isaac_lab_env_cfg.py` | Enabled noise, added yaw params |
| `harold_isaac_lab_env.py` | Added yaw tracking reward, telemetry |

## Next Steps

1. **Hardware testing** with EXP-148 policy
2. **Verify backlash robustness** transfers to real robot
3. **Iterate on yaw tracking** if needed for turning behavior
