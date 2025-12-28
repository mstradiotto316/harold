# Session 26: Controllability Optimization

**Date**: 2025-12-28
**Duration**: ~10 hours
**Result**: Dynamic commands working reliably with optimal configuration

## Summary

This session optimized the controllability system implemented in Session 25. Key breakthrough was discovering that `zero_velocity_prob=0` (never commanding stop) is critical for dynamic commands to work.

## Experiments

| EXP | Mode | Key Change | vx | Verdict |
|-----|------|------------|-----|---------|
| 134 | Dynamic | 5s intervals | 0.007 | STANDING |
| 135 | Dynamic | 10s intervals | 0.010 | STANDING |
| 136 | Dynamic | tracking_weight=20 | 0.010 | STANDING |
| 137 | Static var | tracking_weight=20 | 0.005 | STANDING |
| 138 | Dynamic | sigma=0.15 | 0.003 | STANDING |
| 139 | Dynamic | **zero_prob=0** | **0.011** | **WALKING** |
| 140 | Dynamic | range=0.10-0.45 | **0.015** | **WALKING** |
| 141 | Dynamic | range=0.05-0.50 | 0.012 | WALKING |
| 142 | Confirm | optimal settings | **0.015** | **WALKING** |

## Key Findings

### 1. zero_velocity_prob=0 is Critical
When 5% of commands were "stop" (vx=0), the policy learned to stop instead of walk. Removing stop commands entirely (zero_prob=0) was the breakthrough that enabled WALKING verdict.

### 2. Optimal Command Range: 0.10-0.45 m/s
- Too narrow (0.15-0.40): vx=0.011
- Optimal (0.10-0.45): vx=0.015
- Too wide (0.05-0.50): vx=0.012

### 3. Command Tracking Weight Sweet Spot: 10.0
- Weight=10: Works well (vx=0.015)
- Weight=20: Actually worse! Causes over-optimization, vx=0.005 for static

### 4. Sigma Sweet Spot: 0.1
- Sigma=0.1: Works well (vx=0.015)
- Sigma=0.15: Much worse! Too permissive, robot stops walking (vx=0.003)

### 5. Command Interval: 10s
- 5s intervals: Too frequent, vx=0.007
- 10s intervals: Better, vx=0.010+

## Optimal Configuration

```python
# CommandCfg
vx_min = 0.10
vx_max = 0.45
zero_velocity_prob = 0.0  # CRITICAL: Never command stop
command_change_interval = 10.0  # seconds

# RewardsCfg (command tracking)
command_tracking_enabled = True  # via HAROLD_CMD_TRACK=1
command_tracking_weight = 10.0
command_tracking_sigma = 0.1

# Environment variables
HAROLD_CPG=1 HAROLD_CMD_TRACK=1 HAROLD_DYN_CMD=1
```

## Not Yet Implemented

### Lateral (vy) Commands
Would require:
1. Add vy tracking to command_tracking reward
2. Modify lat_vel_penalty to not penalize commanded lateral motion

### Yaw Rate Commands
Would require:
1. Add yaw tracking to command_tracking reward
2. Modify yaw_rate_penalty similarly

## Files Modified

| File | Changes |
|------|---------|
| `harold_isaac_lab_env_cfg.py` | Tuned command tracking params, optimized ranges |

## Lessons Learned

1. **Zero commands break learning**: If the policy can satisfy reward by stopping, it will
2. **Tracking weight has sweet spot**: Too high causes over-optimization
3. **Sigma has sweet spot**: Too permissive lets policy avoid walking
4. **Results are reproducible**: EXP-140 and EXP-142 both achieved vx=0.015
