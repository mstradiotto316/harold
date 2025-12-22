# Harold Next Steps

## MANDATORY: Before ANY Experiment

```bash
cd /home/matteo/Desktop/code_projects/harold
source ~/Desktop/env_isaaclab/bin/activate
python scripts/harold.py status    # Check current training status
python scripts/harold.py validate  # Validate latest completed run
python scripts/harold.py compare   # Compare recent experiments
```

---

## Current Status (2025-12-22 ~12:00)

### EXP-012: IN PROGRESS - Low Height Penalty (Clamped)

**Run ID**: `2025-12-22_11-02-29_ppo_torch`
**Approach**: Low height penalty with world Z position and clamp

**Config**:
- `low_height_penalty: -50.0` per meter below threshold
- `low_height_threshold: 0.20m`
- Uses world Z position (not height scanner)
- Deficit clamped to max 0.10m

**Progress at 30 min (27%):**
| Metric | Value | Status |
|--------|-------|--------|
| episode_length | 353 | PASS |
| height_reward | 1.90 | FAIL (< 2.0) |
| body_contact | -0.11 | FAIL (< -0.1) |
| reward_total | 1792 | Reasonable |

**Trend**: Height dropping (1.94 → 1.90), robot settling into elbow pose.

---

## Summary of Approaches Tried

| Approach | EXP | Result | Issue |
|----------|-----|--------|-------|
| Height termination | 003-007 | SANITY_FAIL | Height scanner init issues |
| Height reward only | 008 | Height 1.70 | Local minimum (elbow) |
| Joint-angle termination | 009-010 | Height 1.44-1.50 | Robot adapts around thresholds |
| Height penalty (unclamped) | 011 | Reward -25M | Height scanner bad values |
| Height penalty (clamped) | 012 | Height ~1.68 | IN PROGRESS |

---

## Key Insight: Elbow Pose is a Stable Attractor

The robot consistently finds the elbow pose as a stable local minimum:
1. Falls forward early in training
2. Back stays elevated → passes upright check
3. Body touches ground → slight contact penalty
4. Height too low → fails height threshold

**Why it's hard to escape**:
- Standing requires lifting body (energy cost + risk of falling)
- Elbow pose is stable and avoids most termination conditions
- Reward gradient from height_reward alone is insufficient

---

## Recommended Next Approach: Curriculum from terrain_62

**Hypothesis**: Start from a checkpoint that already knows how to stand, then add forward motion.

**Config**:
```python
checkpoint = "terrain_62/checkpoints/agent_128000.pt"
height_threshold = 0.0  # No height termination
body_contact_threshold = 5.0  # Aggressive
forward_reward = 0.0  # Stability only first
height_reward = 30.0
low_height_penalty = -50.0
low_height_threshold = 0.20
```

**Why this might work**:
- terrain_62 already knows how to stand
- Aggressive body contact will terminate if it falls
- Low height penalty creates gradient toward standing
- No forward reward initially (stability first)

---

## Code Changes Made Today

1. **Joint-angle termination** (disabled in current config):
   - `elbow_pose_termination: bool` in TerminationCfg
   - Checks front thigh/calf angles in `_get_dones()`

2. **Low height penalty** (active in current config):
   - `low_height_penalty: -50.0` in RewardsCfg
   - `low_height_threshold: 0.20` in RewardsCfg
   - Uses world Z position, clamped to 0.10m max deficit

3. **Height termination warmup** (active):
   - `height_termination_warmup_steps: 20` in TerminationCfg
   - Skips height termination for first 20 steps

---

## Key Files Modified

| File | Changes |
|------|---------|
| `harold_isaac_lab_env_cfg.py` | Added elbow_pose_termination, low_height_penalty |
| `harold_isaac_lab_env.py` | Implemented joint-angle termination, low height penalty |

---

## 5-Metric Validation Protocol

| Priority | Metric | Threshold | Failure Mode |
|----------|--------|-----------|--------------|
| 1. SANITY | episode_length | > 100 | Robots dying immediately |
| 2. Stability | upright_mean | > 0.9 | Falling over |
| 3. Height | height_reward | > 2.0 | On elbows/collapsed |
| 4. Contact | body_contact | > -0.1 | Body on ground |
| 5. Walking | vx_w_mean | > 0.1 | Not walking |
