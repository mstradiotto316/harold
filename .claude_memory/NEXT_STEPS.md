# Harold Next Steps

## PRIORITY 1: Hardware Testing with Backlash-Robust Policy

**Context from Session 28:** Backlash robustness SOLVED! 1° position noise improves walking by 35% (vx=0.023 vs baseline 0.017). Ready for hardware testing.

### Experiment Plan

1. **Export best backlash-robust policy (EXP-148 or EXP-149)**:
   ```bash
   python policy/export_policy.py --checkpoint logs/skrl/harold_direct/2025-12-29_02-06-37_ppo_torch/checkpoints/best_agent.pt
   ```

2. **Deploy to RPi 5** using existing pipeline in `deployment/`

3. **Test on real hardware** - verify backlash robustness transfers

### Success Criteria
- Robot walks forward on hardware
- More robust motion than previous policies (less oscillation)
- Better handling of mechanical backlash

---

## ✅ COMPLETED: Backlash Robustness (Session 28)

**EXP-148 achieved vx=0.023 m/s (35% better than baseline!)**

Findings:
- 2° noise (0.035 rad): Too much - STANDING (vx=0.007)
- **1° noise (0.0175 rad): OPTIMAL - WALKING (vx=0.023)**
- The noise acts as beneficial regularization

Config settings:
```python
# DomainRandomizationCfg
enable_randomization: bool = True
randomize_per_step: bool = True

# joint_position_noise
std=0.0175  # ~1° in radians
```

---

## ✅ COMPLETED: Yaw Rate Command Tracking (Session 28)

**Standalone yaw tracking works (vx=0.011, WALKING)**

Implementation:
- Added `command_tracking_yaw` reward (exponential kernel)
- Modified `yaw_rate_penalty` to track deviation from commanded yaw
- Yaw command range: [-0.30, 0.30] rad/s (~±17 deg/s)

Config settings:
```python
# RewardsCfg
command_tracking_weight_yaw: float = 10.0
command_tracking_sigma_yaw: float = 0.2

# CommandCfg
yaw_min: float = -0.30
yaw_max: float = 0.30
```

---

## PRIORITY 2: Combine Backlash + Yaw Successfully

**Finding**: Each works alone, but combining them hurts performance (vx=0.003)

### Approaches to Try

1. **Curriculum learning**:
   - Train with vx+vy tracking + backlash first (established WALKING)
   - Then fine-tune with yaw tracking enabled

2. **Reduce yaw complexity**:
   - Lower yaw command range (±0.15 instead of ±0.30)
   - Lower yaw tracking weight (5 instead of 10)

3. **Sequential training**:
   - Use checkpoint from backlash-robust policy (EXP-148)
   - Fine-tune with yaw tracking

---

## Session 28 Results Summary

| EXP | Config | vx | Verdict |
|-----|--------|-----|---------|
| 145 | Baseline | 0.017 | WALKING |
| 146 | 2° backlash | 0.008 | STANDING |
| 147 | 2° backlash | 0.007 | STANDING |
| **148** | **1° backlash (2500 iter)** | **0.023** | **WALKING** |
| 149 | 1° backlash (2500 iter) | 0.023 | WALKING |
| 150 | 1° backlash + yaw (2500 iter) | 0.003 | STANDING |
| 151 | Yaw only (no backlash) | 0.011 | WALKING |

---

## Files Reference

| Purpose | Path |
|---------|------|
| Domain randomization config | `harold_isaac_lab_env_cfg.py` - DomainRandomizationCfg |
| Command tracking config | `harold_isaac_lab_env_cfg.py` - RewardsCfg |
| Command tracking rewards | `harold_isaac_lab_env.py` - _get_rewards() |
| Yaw tracking reward | `harold_isaac_lab_env.py` - command_tracking_yaw |
| Session 28 log | `.claude_memory/sessions/2025-12-29_session28_backlash.md` |
