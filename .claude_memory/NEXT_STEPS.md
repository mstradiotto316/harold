# Harold Next Steps

## PRIORITY 0: Test Smooth Gait Policy on Hardware

**Status**: Session 35 complete. Smooth gait policy exported with optimal damping=150.

### Session 35 Summary

1. ✅ Hardware test showed Session 34 policy was jerky
2. ✅ Completed damping sweep (30→60→75→100→125→150→175)
3. ✅ Found optimal damping=150 (best vx=0.036 m/s)
4. ✅ Exported smooth policy to deployment/policy/harold_policy.onnx
5. 🔲 **Test on hardware** - Should be smoother than previous policies

### Damping Sweep Results

| Damping | vx (m/s) | Contact | Verdict |
|---------|----------|---------|---------|
| 30 (original) | 0.016 | -0.024 | WALKING (jerky) |
| 75 | 0.014 | -0.005 | WALKING |
| 100 | 0.014 | -0.0002 | WALKING |
| 125 | 0.034 | -0.015 | WALKING |
| **150** | **0.036** | -0.014 | **WALKING (BEST)** |
| 175 | -0.024 | -0.002 | STANDING (too high) |

### Key Findings

- **U-shaped velocity curve**: vx drops at medium damping, rises at high damping
- **Optimal = 150**: Best forward velocity with acceptable contact
- **>175 breaks walking**: Robot prefers standing to walking

### Current Settings (Session 35 Final)

| Parameter | Value |
|-----------|-------|
| damping | 150 |
| action_filter_beta | 0.40 |
| torque_penalty | -0.01 |
| action_rate_penalty | -0.1 |
| CPG frequency | 0.7 Hz |

---

## If Hardware Still Jerky: Fine-Tune Smoothness

Possible next steps if hardware test reveals issues:

1. **Higher damping (closer to 175)**: Current 150 may not be smooth enough
2. **Stronger action filter (beta=0.35)**: Current 0.40 may allow too much high-frequency noise
3. **CPG trajectory shaping**: Asymmetric swing/stance for smoother foot placement

---

## Deployment Status

- ✅ ONNX validation passing (errors ~10^-6)
- ✅ Controller runs stable at 20 Hz
- ✅ harold.service disabled (manual control only)
- ✅ Smooth gait policy exported (damping=150)
- ⚠️ FL shoulder (ID 2) needs recalibration

---

## Long-Term Goal Reminder

End goal is **remote-controlled walking in any direction + rotation**.

Current CPG is forward-only. After achieving smooth walking:
- Parameterized CPG taking (vx, vy, yaw) commands, OR
- Higher policy authority (residual_scale > 0.05), OR
- Pure RL without CPG base trajectory

But first: **test smooth gait on hardware**.
