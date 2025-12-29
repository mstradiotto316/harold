# Harold Next Steps

## QUICK START FOR NEXT AGENT

**Session 27: Servo Tuning & Hardware Diagnostics**

Pi 5 power supply validated (no throttling under stress). Servo dead zone and drift issues diagnosed.

### Immediate Hardware Tasks

1. **Flash ServoTuning.ino** to ESP32 and run `s` to scan all servos
2. **Reduce dead zones**: Run `a 2` to set dead zone to 2 for all servos
3. **Test servo responsiveness** after dead zone adjustment
4. **Commit pending firmware changes** (100% torque, collision-safe limits)

### Uncommitted Firmware Changes

File: `firmware/StreamingControl/HaroldStreamingControl/HaroldStreamingControl.ino`
- TORQUE_LIMIT = 1000 (100% max)
- initServoTorque() function
- Asymmetric collision-safe joint limits
- Watchdog timeout 250ms

---

## Session 26: Dynamic Commands Fully Optimized

Dynamic command tracking is working reliably. The robot can now follow changing velocity commands.

### Best Working Configuration

```bash
# Dynamic commands with optimal settings
HAROLD_CPG=1 HAROLD_CMD_TRACK=1 HAROLD_DYN_CMD=1 python scripts/harold.py train \
  --hypothesis "Your hypothesis" --iterations 1250
```

### Current Parameters (in harold_isaac_lab_env_cfg.py)

```python
# CommandCfg - Optimal settings found in Session 26
vx_min = 0.10       # Wider range works better
vx_max = 0.45
zero_velocity_prob = 0.0  # CRITICAL: Never command stop
command_change_interval = 10.0  # 10s between changes

# RewardsCfg - Command tracking
command_tracking_weight = 10.0   # Sweet spot (20 was worse)
command_tracking_sigma = 0.1    # Sweet spot (0.15 was worse)
```

---

## PRIORITY 1: Add Lateral (vy) Commands

Current implementation only tracks forward velocity (vx). To add lateral commands:

1. Add vy tracking to command_tracking reward:
   ```python
   # In _compute_rewards():
   cmd_vy = self._commands[:, 1]
   vy_error = vy_w - cmd_vy
   vy_tracking = cmd_weight * torch.exp(-torch.square(vy_error / cmd_sigma))
   ```

2. Modify lat_vel_penalty to not penalize commanded motion:
   ```python
   # Change from: -lat_vel_penalty * vy²
   # To: -lat_vel_penalty * (vy - cmd_vy)²
   ```

---

## PRIORITY 2: Add Yaw Rate Commands

Similar to lateral commands:
1. Add yaw tracking to command_tracking reward
2. Modify yaw_rate_penalty to track commanded yaw

---

## PRIORITY 3: Hardware Deployment

Deployment code is ready in `deployment/`:
- `policy/harold_policy.onnx` - Exported policy
- `inference/harold_controller.py` - 20 Hz control loop
- `drivers/` - IMU and ESP32 communication

---

## Session 26 Experiment Summary

| EXP | Config | vx | Verdict |
|-----|--------|-----|---------|
| 134 | Dynamic 5s | 0.007 | STANDING |
| 135 | Dynamic 10s | 0.010 | STANDING |
| 139 | zero_prob=0 | 0.011 | WALKING |
| 140 | range 0.10-0.45 | 0.015 | WALKING |
| 141 | range 0.05-0.50 | 0.012 | WALKING |
| 142 | Confirmation | 0.015 | WALKING |
| 143 | 2500 iters | 0.016 | WALKING |

**Key Finding**: `zero_velocity_prob=0` was the breakthrough. Without stop commands, the policy learns to walk.

**Best Result**: EXP-143 with 2500 iterations achieved vx=0.016 m/s.

---

## Environment Variables Reference

| Variable | Effect |
|----------|--------|
| `HAROLD_CPG=1` | Enable CPG base trajectory |
| `HAROLD_CMD_TRACK=1` | Enable command tracking reward |
| `HAROLD_VAR_CMD=1` | Enable variable command sampling |
| `HAROLD_DYN_CMD=1` | Enable dynamic command changes (implies VAR_CMD) |
| `HAROLD_SCRIPTED_GAIT=1` | Enable scripted gait (no learning) |

---

## Files Reference

| Purpose | Path |
|---------|------|
| Command config | `harold_isaac_lab_env_cfg.py` - CommandCfg class |
| Command tracking reward | `harold_isaac_lab_env.py` - _compute_rewards() |
| Session log | `.claude_memory/sessions/2025-12-28_session26_controllability_optimization.md` |
