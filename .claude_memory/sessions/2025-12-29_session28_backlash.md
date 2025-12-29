# Session 28: Gear Backlash Robustness & Yaw Rate Tracking

**Date**: 2025-12-29
**Duration**: ~8 hours
**Result**: Backlash robustness SOLVED (35% improvement), yaw tracking implemented

## Summary

This session addressed the primary user priority: simulating gear backlash to improve sim-to-real transfer. The key finding is that **1° position noise improves walking by 35%**, acting as beneficial regularization.

## Key Results

| Config | vx (m/s) | Verdict | Notes |
|--------|----------|---------|-------|
| EXP-145 Baseline | 0.017 | WALKING | No noise |
| EXP-146/147 2° backlash | 0.007 | STANDING | Too much noise |
| **EXP-148 1° backlash** | **0.023** | **WALKING** | **35% better!** |
| EXP-150 Backlash + yaw | 0.003 | STANDING | Combined doesn't work |
| EXP-151 Yaw only | 0.011 | WALKING | Yaw works standalone |

## Backlash Implementation

### Config Changes (harold_isaac_lab_env_cfg.py)

1. **Enabled sensor noise** in DomainRandomizationCfg:
   ```python
   enable_randomization: bool = True   # Master switch for noise
   randomize_per_step: bool = True     # Apply noise every step
   ```

2. **Set optimal noise level**:
   ```python
   joint_position_noise: GaussianNoiseCfg = GaussianNoiseCfg(
       mean=0.0,
       std=0.0175,  # ~1° in radians - OPTIMAL
       operation="add"
   )
   ```

### Why 1° Works

- **2° is too much**: Robot struggles to balance with high uncertainty
- **1° is optimal**: Provides regularization without overwhelming the policy
- **0° (baseline)**: Lower performance, policy overfits to perfect observations

The noise acts as a form of regularization, preventing the policy from relying on precise position feedback that won't be available on hardware.

## Yaw Rate Tracking Implementation

### Config Changes (harold_isaac_lab_env_cfg.py)

1. **RewardsCfg** - Added yaw tracking parameters:
   ```python
   command_tracking_weight_yaw: float = 10.0
   command_tracking_sigma_yaw: float = 0.2
   ```

2. **CommandCfg** - Enabled yaw command range:
   ```python
   yaw_min: float = -0.30   # ~17 deg/s
   yaw_max: float = 0.30
   ```

### Environment Changes (harold_isaac_lab_env.py)

1. **Added yaw tracking reward** (after vy tracking):
   ```python
   cmd_yaw = self._commands[:, 2]
   yaw_error = wz_b - cmd_yaw
   command_tracking_yaw = cmd_weight_yaw * torch.exp(
       -torch.square(yaw_error / cmd_sigma_yaw)
   ) * upright_sq
   ```

2. **Modified yaw_rate_penalty**:
   ```python
   # OLD: -rewards_cfg.yaw_rate_penalty * torch.square(wz_b)
   # NEW: -rewards_cfg.yaw_rate_penalty * torch.square(wz_b - cmd_yaw)
   ```

3. **Added telemetry**:
   - `cmd_yaw_error` in _metric_keys

## Key Findings

### Backlash Robustness
1. **1° position noise is optimal**: Improves vx by 35% (0.017 → 0.023)
2. **Noise as regularization**: Prevents overfitting to perfect observations
3. **Ready for hardware**: Policy should handle real servo backlash better

### Yaw Tracking
1. **Standalone works**: vx=0.011 without backlash noise
2. **Combined doesn't work**: vx=0.003 with backlash + yaw
3. **May need curriculum**: Train backlash first, then add yaw

## Files Modified

| File | Changes |
|------|---------|
| `harold_isaac_lab_env_cfg.py` | Enabled noise, added yaw params |
| `harold_isaac_lab_env.py` | Added yaw tracking reward |

## Next Steps

1. **Hardware testing** with backlash-robust policy (EXP-148)
2. **Curriculum learning** for backlash + yaw combination
3. **Export best policy** for deployment
