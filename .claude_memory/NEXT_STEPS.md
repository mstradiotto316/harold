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

## Current Status (2025-12-22 ~17:50)

### EXP-021: IN PROGRESS
- **Run ID**: `2025-12-22_17-48-06_ppo_torch`
- **Config**: forward=40, standing_penalty=-5.0, height=15, upright=10
- **Duration**: ~30 min (1250 iterations)
- **Started**: 17:48

### Previous Session Progress (EXP-015 to EXP-019)

| EXP | Key Change | vx_w_mean | Height | Outcome |
|-----|------------|-----------|--------|---------|
| 015 | Spawn pose fix (0.30m, neutral) | ~0 | 4.08 | STANDING! |
| 016 | forward=15 | ~0 | ~6.0 | Standing, no motion |
| 017 | forward=25, height=15 | ~0 | ~1.5 | Height dominated |
| 018 | standing_penalty=-5.0 | **0.065** | 1.13 | Motion emerging! |
| 019 | forward=40 | **0.033** | 1.13 | Incomplete (wrong duration) |

**Key Finding**: Standing penalty (-5.0) successfully breaks standing equilibrium.

---

## Experiment Queue (Current Session)

### EXP-021: Baseline with Correct Duration ← IN PROGRESS
**Hypothesis**: Full 30-min run will reach vx > 0.1
**Config**: Same as EXP-019

### EXP-022: Stronger Standing Penalty
**If**: EXP-021 vx < 0.1
**Change**: `standing_penalty: -10.0` (was -5.0)

### EXP-023: Lower Height Reward
**If**: Height dropping or vx stagnant
**Change**: `height_reward: 10.0` (was 15.0)

### EXP-024: Higher Forward Reward
**If**: Stable but slow motion
**Change**: `progress_forward_pos: 50.0` (was 40.0)

### EXP-025-028: Iterate based on results

---

## Benchmark Configuration

| Duration | Iterations | Timesteps |
|----------|------------|-----------|
| 30 min   | 1250       | 30k       |
| 60 min   | 2500       | 60k       |
| 100 min  | 4167       | 100k      |

**harold.py defaults**: 6144 envs, 1250 iterations (30 min)

---

## Current Reward Configuration

```python
# harold_isaac_lab_env_cfg.py - RewardsCfg
progress_forward_pos: float = 40.0  # Strong forward incentive
progress_forward_neg: float = 5.0   # Moderate backward penalty
standing_penalty: float = -5.0      # Penalize near-zero velocity
upright_reward: float = 10.0        # Keep upright
height_reward: float = 15.0         # Reduced - crouch OK
```

---

## 5-Metric Validation Protocol

| Priority | Metric | Threshold | Status |
|----------|--------|-----------|--------|
| 1. SANITY | episode_length | > 100 | Must pass |
| 2. Stability | upright_mean | > 0.9 | Must pass |
| 3. Height | height_reward | > 1.0 | Acceptable (crouch OK) |
| 4. Contact | body_contact | > -0.1 | Must pass |
| 5. **Walking** | **vx_w_mean** | **> 0.1** | **PRIMARY GOAL** |

---

## Goal
Make the robot walk forward in a straight line. Forward motion is emerging (vx=0.03-0.06). Need to push over 0.1 m/s threshold.
