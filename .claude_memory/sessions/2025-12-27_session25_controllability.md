# Session 25: Controllability Implementation

**Date**: 2025-12-27
**Duration**: ~3 hours
**Result**: Implemented command tracking, variable commands, and dynamic commands

## Summary

This session focused on adding controllability to the robot - the ability to follow velocity commands and eventually stop/start on command.

## Implementations

### Phase 1: Command Tracking Reward (COMPLETE)
- Added `command_tracking_enabled`, `command_tracking_weight`, `command_tracking_sigma` to RewardsCfg
- Reward: `exp(-|vx - cmd_vx|²/σ²) * weight`
- Enabled via `HAROLD_CMD_TRACK=1`
- **Result**: EXP-129, EXP-130 both achieved WALKING with command tracking

### Phase 2: Variable Command Range (COMPLETE)
- Added `CommandCfg` dataclass with:
  - `vx_min`, `vx_max` (velocity range)
  - `vy_min`, `vy_max` (lateral - unused)
  - `yaw_min`, `yaw_max` (rotation - unused)
  - `zero_velocity_prob` (probability of stop command)
- Enabled via `HAROLD_VAR_CMD=1`

### Phase 3: Dynamic Commands (COMPLETE)
- Added `dynamic_commands`, `command_change_interval`, `command_change_prob` to CommandCfg
- Commands change every N seconds during episode
- Enabled via `HAROLD_DYN_CMD=1`

## Experiments

| EXP | Mode | Command Range | Zero % | vx (m/s) | Verdict |
|-----|------|--------------|--------|----------|---------|
| 129 | CPG + Track | 0.25-0.35 | 0% | 0.013 | WALKING |
| 130 | CPG + Track | 0.25-0.35 | 0% | 0.031 | WALKING |
| 131 | CPG + Track + Var | 0.0-0.5 | 10% | 0.006 | FAILING |
| 132 | CPG + Track + Var | 0.15-0.4 | 5% | 0.010 | WALKING |
| 133 | CPG + Track + Dyn | 0.15-0.4 | 5% | (interrupted) | - |

## Key Findings

1. **Command tracking works**: Adding command tracking reward doesn't break CPG walking
2. **Wide range is hard**: 0-0.5 m/s with 10% zero commands causes training to fail
3. **Narrow range works**: 0.15-0.4 m/s with 5% zero achieves WALKING
4. **Dynamic commands need testing**: EXP-133 was interrupted at 43 min

## Files Modified

| File | Changes |
|------|---------|
| `harold_isaac_lab_env_cfg.py` | Added CommandCfg, command tracking config |
| `harold_isaac_lab_env.py` | Command tracking reward, variable sampling, dynamic updates |

## Environment Variables

```bash
HAROLD_CPG=1          # Enable CPG mode
HAROLD_CMD_TRACK=1    # Enable command tracking reward
HAROLD_VAR_CMD=1      # Enable variable command sampling
HAROLD_DYN_CMD=1      # Enable dynamic command changes
```

## Next Steps

1. **Resume EXP-133**: Test dynamic commands to completion
2. **Widen range gradually**: Once dynamic works, try expanding to 0.1-0.5
3. **Add lateral/yaw**: After vx control works, add vy and yaw commands
4. **Hardware test**: Deploy controllable policy to real robot

## Current Best Configuration

```python
# Config
vx_min = 0.15
vx_max = 0.40
zero_velocity_prob = 0.05
command_change_interval = 5.0  # seconds

# Command
HAROLD_CPG=1 HAROLD_CMD_TRACK=1 HAROLD_VAR_CMD=1 python scripts/harold.py train
```
