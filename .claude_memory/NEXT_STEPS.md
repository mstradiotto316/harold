# Harold Next Steps

## PRIORITY 0: Test Smooth Gait Policy on Hardware

**Status**: Session 35 complete. Stable policy with damping=125 and fixed spawn pose.

### Session 35 Summary

1. ✅ Hardware test showed Session 34 policy was jerky
2. ✅ Completed damping sweep (30→60→75→100→125→150→175)
3. ✅ Fixed calf spawn bug (-1.40 → -1.39, was exceeding limit)
4. ✅ Reverted damping 150 → 125 (safer value)
5. ✅ Improved episode_length threshold (100 → 300 = 15s minimum)
6. 🔲 **Test on hardware** - Should be smoother than previous policies

### Latest Training Result (damping=125)

| Metric | Value |
|--------|-------|
| Episode length | 402 (PASS > 300) |
| Forward velocity | 0.032 m/s |
| Height reward | 1.40 |
| Upright mean | 0.965 |
| Body contact | -0.002 |
| Verdict | WALKING |

### Current Settings (Session 35 Final)

| Parameter | Value |
|-----------|-------|
| damping | 125 |
| action_filter_beta | 0.40 |
| torque_penalty | -0.01 |
| action_rate_penalty | -0.1 |
| calf_spawn | -1.39 rad |
| episode_length_threshold | 300 (15s) |
| CPG frequency | 0.7 Hz |

---

## Validation Improvements Made

1. **Episode length threshold raised**: 100 → 300 (5s → 15s)
   - Catches robots that fall repeatedly even if vx looks good
   - A robot must survive 15+ seconds to pass

2. **Spawn pose fixed**: Calf at -1.39 rad (within -1.3963 limit)

---

## Deployment Status

- ✅ ONNX validation passing (errors ~10^-6)
- ✅ Controller runs stable at 20 Hz
- ✅ harold.service disabled (manual control only)
- ✅ Stable policy exported (damping=125)
- ⚠️ FL shoulder (ID 2) needs recalibration

---

## Long-Term Goal Reminder

End goal is **remote-controlled walking in any direction + rotation**.

Current CPG is forward-only. After achieving smooth walking:
- Parameterized CPG taking (vx, vy, yaw) commands, OR
- Higher policy authority (residual_scale > 0.05), OR
- Pure RL without CPG base trajectory

But first: **test smooth gait on hardware**.
