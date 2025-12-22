# Harold Next Steps

## MANDATORY: Before ANY Experiment

```bash
python scripts/validate_training.py
```

If this returns exit code 2 (sanity check failed), **DO NOT PROCEED** with experiments. Fix the issue first.

See `.claude_memory/OBSERVABILITY.md` for the 5-metric protocol.

---

## Current Status (2025-12-22)

### EXP-038: IN PROGRESS (CHECK FIRST!)

**Training may still be running or recently completed!**

- **Run ID**: `2025-12-21_23-18-57_ppo_torch`
- **Approach**: Body contact termination with 15N threshold (lowered from 50N)
- **Config**: `body_contact_threshold: 15.0` in TerminationCfg

**First thing to do:**
```bash
# Check if training is still running
ps aux | grep train.py

# If completed, validate results
source ~/Desktop/env_isaaclab/bin/activate
python scripts/validate_training.py
```

**Last Known Metrics (25% through training):**
| Metric | Value | Status |
|--------|-------|--------|
| episode_length | 364.3 | PASS |
| upright | 0.9560 | PASS |
| height_reward | 1.53 | FAIL (< 2.0) |
| body_contact | -0.19 | FAIL |
| vx_w_mean | 0.016 | WARN |

**Promising Signs:**
- Body contact penalty initially improved dramatically (-0.014 → -0.0002)
- Episode min length 5-6 means termination IS triggering
- Forward velocity trending positive (from -0.016 to +0.036)

**If EXP-038 PASSED**: Great! Proceed to add forward motion rewards.

**If EXP-038 FAILED**: Try lower threshold (10N) or different approach.

---

## Session Summary (2025-12-22)

### Experiments This Session
| EXP | Approach | Result |
|-----|----------|--------|
| 034 | height_reward=25.0 | FAILED - Elbow exploit |
| 035 | Fine-tune from terrain_62 | FAILED - Policy degraded |
| 036 | Height termination | FAILED - Spawn pose too low |
| 037 | Body contact term (50N) | FAILED - Threshold too high |
| 038 | Body contact term (15N) | IN PROGRESS |

### Core Problem
Robot falls forward onto elbows, passes upright check (back elevated), but fails height check.

### Key Insight from EXP-037
Robot in elbow position generates ~22N contact force. The 50N threshold was too high to trigger termination. Lowering to 15N should catch this exploit.

---

## If EXP-038 Fails: Next Steps

### Option A: Even Lower Threshold (EXP-039)
```python
# In TerminationCfg
body_contact_threshold: float = 10.0  # Was 15.0
```

### Option B: Combined Approach (EXP-040)
Add both body contact AND height termination, but:
1. First modify spawn pose to be higher
2. Then enable height termination

### Option C: Curriculum Learning
1. Start with stability-only (no forward reward)
2. Save checkpoint at stable standing
3. Fine-tune with small forward reward

---

## 5-Metric Validation Protocol

| Priority | Metric | Threshold | Failure Mode |
|----------|--------|-----------|--------------|
| 1. SANITY | episode_length | > 100 | Robots dying immediately |
| 2. Stability | upright_mean | > 0.9 | Falling over |
| 3. Height | height_reward | > 2.0 | On elbows/collapsed |
| 4. Contact | body_contact | > -0.1 | Body on ground |
| 5. Walking | vx_w_mean | > 0.1 | Not walking |

**CRITICAL**: If #1 fails, all other metrics are invalid!

---

## Key Files

- `scripts/validate_training.py` - Run after every experiment
- `.claude_memory/OBSERVABILITY.md` - 5-metric protocol documentation
- `harold_isaac_lab/.../harold_flat/harold_isaac_lab_env.py` - Termination logic
- `harold_isaac_lab/.../harold_flat/harold_isaac_lab_env_cfg.py` - Reward/termination config
