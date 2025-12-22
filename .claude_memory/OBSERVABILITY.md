# Harold Observability Protocol

## CRITICAL: 5-Metric Hierarchical Validation

**Lessons Learned (2025-12-21)**:
1. I ran experiments for HOURS without realizing robots were dying immediately (3-timestep episodes)
2. The metrics I was analyzing (vx_w_mean, upright_mean) were meaningless
3. The root cause: Missing the most critical sanity check - **episode length**

---

## The 5-Metric Protocol (Check in Order!)

| Priority | Metric | Threshold | Failure Mode |
|----------|--------|-----------|--------------|
| **1. SANITY** | `Episode / Total timesteps (mean)` | **> 100** | Robots dying immediately (BUG!) |
| 2. Stability | `Episode_Metric/upright_mean` | > 0.9 | Falling over |
| 3. Height | `Episode_Reward/height_reward` | > 2.0 | On elbows/collapsed |
| 4. Contact | `Episode_Reward/body_contact_penalty` | > -0.1 | Body on ground |
| 5. Walking | `Episode_Metric/vx_w_mean` | > 0.1 m/s | Not walking |

**CRITICAL**: If metric #1 (episode length) fails, ALL OTHER METRICS ARE INVALID.

---

## Required: Run Validator After EVERY Experiment

```bash
source ~/Desktop/env_isaaclab/bin/activate
cd /home/matteo/Desktop/code_projects/harold

python scripts/validate_training.py           # Validate latest run
python scripts/validate_training.py <run_id>  # Validate specific run
python scripts/validate_training.py --list    # List recent runs
```

**Exit codes:**
- `0`: All metrics pass (or partial success)
- `1`: Some metrics failing
- `2`: SANITY CHECK FAILED - Do not trust other metrics

---

## Failure Mode Signatures

| State | ep_len | upright | height_rew | body_contact | vx |
|-------|--------|---------|------------|--------------|-----|
| **BUG: Dying instantly** | **< 100** | ? | ? | ? | ? |
| Standing correctly | > 300 | > 0.9 | > 2.5 | > -0.1 | ~0 |
| **On elbows (TRAP!)** | > 100 | > 0.9 | **< 1.0** | **< -0.2** | ~0 or + |
| Walking | > 300 | > 0.9 | > 2.0 | > -0.1 | > 0.1 |
| Fallen sideways | varies | < 0.7 | low | varies | ~0 |

---

## Known Bugs and How They Present

### The Height Termination Bug (2025-12-21)

**Symptom**: Episode length = 3 steps

**Root cause**: Height termination checked `root_pos_w[:, 2]` (world Z coordinate) instead of height above terrain. If terrain origin wasn't at Z=0, robots spawned below the threshold.

**Lesson**:
1. Always check episode_length FIRST
2. Test termination changes with quick sanity runs
3. Use height above terrain, not world Z

### The "On Elbows" Exploit (2025-12-20)

**Symptom**: upright_mean > 0.9, vx_w_mean > 0, but height_reward < 1.0

**Root cause**: Robot falls forward onto elbows. Back is elevated (passes upright check), moves forward (passes vx check), but not at standing height.

**Lesson**: Always check height_reward. If < 2.0, robot is not standing properly.

---

## Pre-Experiment Checklist

Before starting any experiment:
- [ ] Run `python scripts/validate_training.py` on last successful run
- [ ] Confirm episode_length > 300
- [ ] No uncommitted changes to termination logic (`_get_dones`)
- [ ] Quick sanity run (50 iterations) shows ep_len > 50

---

## Quick Analysis Script

If you need to manually check without the validator:

```bash
source ~/Desktop/env_isaaclab/bin/activate
cd /home/matteo/Desktop/code_projects/harold

python -c "
from tensorboard.backend.event_processing import event_accumulator
import os

log_base = 'logs/skrl/harold_direct'
runs = sorted([d for d in os.listdir(log_base) if os.path.isdir(os.path.join(log_base, d)) and '2025-' in d])
path = os.path.join(log_base, runs[-1])
print(f'Run: {runs[-1]}')

ea = event_accumulator.EventAccumulator(path)
ea.Reload()

def avg(key):
    try:
        s = ea.Scalars(key)
        return sum(x.value for x in s[-10:])/min(10, len(s)) if s else None
    except: return None

ep_len = avg('Episode / Total timesteps (mean)')
upright = avg('Info / Episode_Metric/upright_mean')
height = avg('Info / Episode_Reward/height_reward')
contact = avg('Info / Episode_Reward/body_contact_penalty')
vx = avg('Info / Episode_Metric/vx_w_mean')

print(f'')
print(f'1. SANITY: ep_len   = {ep_len:.1f} (need > 100, expect > 300)')
print(f'2. upright          = {upright:.4f} (need > 0.9)')
print(f'3. height_reward    = {height:.4f} (need > 2.0)')
print(f'4. body_contact     = {contact:.4f} (need > -0.1)')
print(f'5. vx_w_mean        = {vx:.4f} (need > 0.1)')

if ep_len < 100:
    print('')
    print('>>> CRITICAL: Episode length too short! DO NOT TRUST OTHER METRICS.')
"
```

---

## Last Updated

2025-12-21 - 5-Metric Hierarchical Protocol
- Added episode_length as #1 SANITY CHECK
- Created scripts/validate_training.py for automated validation
- Documented the Height Termination Bug
- Previous 4-metric protocol was insufficient
