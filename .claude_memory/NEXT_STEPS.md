# Harold Next Steps

## QUICK START FOR NEXT AGENT

**Session 23 found stiffness=600 is better than 400 for RL learning. Continue experiments.**

### Immediate Action
```bash
# 1. Check for orphan processes
python scripts/harold.py ps

# 2. Current stiffness is 600 - continue testing or try 800
python scripts/harold.py train \
  --hypothesis "Stiffness=800 further testing sim-to-real balance" \
  --tags "stiffness800,sim2real,session24"

# 3. Monitor progress
python scripts/harold.py status
```

### Session 23 Key Finding: Stiffness=400 Was Too Soft

| Stiffness | Height | vx (final) | vx (peak) | Verdict |
|-----------|--------|------------|-----------|---------|
| 400 | 0.74 | 0.024 | 0.024 | FAILING |
| 600 | 1.54 | 0.027 | 0.047 | STANDING |

**Stiffness=600 is significantly better**:
- Height: 0.74 -> 1.54 (108% improvement!)
- Peak vx: 0.024 -> 0.047 (96% improvement!)
- Robot stands properly instead of crouching

---

## Current Training Configuration

### Actuators (harold.py) - CURRENTLY stiffness=600
```python
stiffness = 600.0    # Session 23: Better than 400, testing middle ground
damping = 45.0       # Proportional to stiffness
effort_limit = 2.8   # 95% of hardware max
```

### Rewards (harold_isaac_lab_env_cfg.py) - ALREADY OPTIMAL
```python
progress_forward_pos = 40.0      # Optimal from EXP-103
progress_forward_neg = 10.0      # Optimal from EXP-032
backward_motion_penalty = 75.0   # CRITICAL - must be exactly 75
diagonal_gait_reward = 5.0       # Forward-gated stepping reward
height_reward = 20.0             # Maintains posture
standing_penalty = -5.0          # Prevents standing still
```

---

## Experiment Queue

### PRIORITY 1: Continue Stiffness Sweep

**Hypothesis**: There's an optimal stiffness between 600 and 1200 for learning.

| Stiffness | Tested? | Result |
|-----------|---------|--------|
| 400 | Yes (Session 23) | FAILING, height=0.74, vx=0.024 |
| 600 | Yes (Session 23) | STANDING, height=1.54, vx=0.027, peak=0.047 |
| 800 | **NEXT** | ? |
| 1000 | Pending | ? |
| 1200 | Previous (stiff) | Good learning but poor sim-to-real |

**Command**:
```bash
python scripts/harold.py train \
  --hypothesis "Stiffness=800 testing sim-to-real balance" \
  --tags "stiffness800,sim2real"
```

### PRIORITY 2: Early Stopping at Peak

Session 23 observed peak vx=0.047 at 39% progress, regressing to 0.027 by end.

**Hypothesis**: Saving checkpoint at peak and stopping early could preserve better policies.

### PRIORITY 3: Increased Height Reward

With softer stiffness, height reward may need to be higher to maintain posture.

---

## Key Context for Next Agent

### Why stiffness=600?
Session 23 tested stiffness=400 (sim-to-real aligned from Session 22) but found the robot couldn't maintain height (0.74) and learned poorly. Stiffness=600 achieved height=1.54 and peak vx=0.047 (47% of target).

### Sim-to-Real Trade-off
- **Real robot** (Session 22): Has "softness" that requires lower stiffness
- **RL learning** (Session 23): Needs higher stiffness to maintain posture during learning
- **Balance point**: Somewhere between 600-1200

### Why backward_penalty=75?
Session 20 discovered this is a SHARP local optimum. Both 70 and 80 cause backward drift. Do NOT tune this value.

### The Regression Pattern
Training consistently shows peak vx at 30-50% progress, then regression. Session 23 saw:
- 39% progress: vx=0.047, height=1.51
- 100% progress: vx=0.027, height=1.54

---

## Files Reference

| Purpose | Path | Status |
|---------|------|--------|
| Actuator config | `harold.py` | stiffness=600 |
| Reward config | `harold_isaac_lab_env_cfg.py` | Optimal weights |
| PPO config | `agents/skrl_ppo_cfg.yaml` | LR=5e-4 |
| Hardware script | `firmware/scripted_gait_test_1/` | Walks forward |

---

## 5-Metric Validation

| Priority | Metric | Threshold | Meaning |
|----------|--------|-----------|---------|
| 1. SANITY | episode_length | > 100 | Not dying immediately |
| 2. Stability | upright_mean | > 0.9 | Not falling over |
| 3. Height | height_reward | > 1.2 | Standing properly |
| 4. Contact | body_contact | > -0.1 | No body on ground |
| 5. **Walking** | **vx_w_mean** | **> 0.1** | **Moving forward** |

Use `python scripts/harold.py validate` to check all metrics.
