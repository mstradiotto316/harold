# Session 27: Lateral (vy) Command Tracking

**Date**: 2025-12-28
**Duration**: ~2 hours
**Result**: vy tracking implemented, EXP-144 achieved WALKING (vx=0.017 m/s)

## Summary

This session extended the command tracking system from vx-only to include lateral (vy) velocity commands. The robot can now receive and track both forward and lateral velocity commands.

## Implementation

### Config Changes (harold_isaac_lab_env_cfg.py)

1. **RewardsCfg** - Added vy tracking parameters:
   ```python
   command_tracking_weight_vy: float = 10.0   # Same as vx
   command_tracking_sigma_vy: float = 0.1     # Same as vx
   ```

2. **CommandCfg** - Enabled vy command range:
   ```python
   vy_min: float = -0.15   # Conservative range
   vy_max: float = 0.15
   ```

### Environment Changes (harold_isaac_lab_env.py)

1. **Added vy tracking reward** in `_get_rewards()`:
   ```python
   # Get commanded vy
   cmd_vy = self._commands[:, 1]

   # Compute lateral velocity tracking error
   vy_error = vy_w - cmd_vy

   # Exponential reward: peak when actual matches command
   command_tracking_vy = cmd_weight_vy * torch.exp(-torch.square(vy_error / cmd_sigma_vy)) * upright_sq
   ```

2. **Modified lat_vel_penalty**:
   ```python
   # OLD: -rewards_cfg.lat_vel_penalty * torch.square(vy_w)
   # NEW: -rewards_cfg.lat_vel_penalty * torch.square(vy_w - cmd_vy)
   ```
   This change allows commanded lateral motion while still penalizing unintended drift.

3. **Added telemetry**:
   - Added `cmd_vy_error` to metric keys for TensorBoard tracking

## Experiment

### EXP-144: vy Command Tracking Implementation
- **Config**: CPG + CMD_TRACK + DYN_CMD, vy range [-0.15, 0.15], vx range [0.10, 0.45]
- **Duration**: ~55 min (1250 iterations)
- **Result**: **WALKING** (vx=0.017)
- **Verdict**: Success - vy tracking doesn't break forward walking

### Validation Metrics
- Episode Length: 471.15 (PASS)
- Upright Mean: 0.937 (PASS)
- Height Reward: 1.376 (PASS)
- Body Contact: -0.001 (PASS)
- Forward Velocity: 0.017 m/s (PASS)

## Bug Fix

Initial run failed with `KeyError: 'command_tracking_vy'` because the reward key wasn't added to `_reward_keys` list. Fixed by adding:
```python
# In _reward_keys list:
"command_tracking_vy",  # Session 27: Lateral command tracking
```

## Key Findings

1. **vy tracking doesn't break vx**: Forward walking maintained (vx=0.017)
2. **Conservative vy range works**: [-0.15, 0.15] m/s is reasonable
3. **Same hyperparameters work**: weight=10, sigma=0.1 work for vy like vx

## Next Steps

1. **PRIORITY 1**: Hardware backlash simulation (user priority)
2. **PRIORITY 2**: Add yaw rate command tracking
3. **PRIORITY 3**: Hardware deployment

## Files Modified

| File | Changes |
|------|---------|
| `harold_isaac_lab_env_cfg.py` | Added vy tracking params, enabled vy range |
| `harold_isaac_lab_env.py` | Added command_tracking_vy reward, modified lat_vel_penalty |
