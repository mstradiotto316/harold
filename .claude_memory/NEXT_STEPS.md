# Harold Next Steps

## Current Status (2025-12-24 ~08:00, Session 16 Complete)

### Best Configuration (EXP-056) - UNCHANGED
Forward-gated diagonal gait reward remains the best stable configuration.
- `learning_rate: 5.0e-4`
- `progress_forward_neg: 10.0`
- `progress_forward_pos: 40.0`
- `height_reward: 15.0`
- `upright_reward: 10.0`
- `diagonal_gait_reward: 5.0`
- `entropy_loss_scale: 0.01`

**Best Stable Metrics:**
- Forward velocity: **0.036 m/s** (36% of target)
- Height reward: **1.49** (standing properly)
- All stability metrics: PASS

**Peak Observed (Unstable):**
- EXP-059 achieved **vx=0.076** (76% of target!) at 89% training, then regressed

**Best Height Achieved:**
- EXP-068 achieved **height=2.20** (best ever!) but with backward drift (vx=-0.046)

---

## Session 16 Summary (6 experiments - Autonomous Overnight)

### What We Tried
| EXP | Change | vx | height | Notes |
|-----|--------|-----|--------|-------|
| 063 | Reduced clip_range (0.2→0.1) | +0.024 | 1.26 | Killed by watchdog at 90% |
| 064 | gait=10 + clip=0.1 | +0.019 | 1.50 | Peak vx=0.029 at 76%, regressed |
| 065 | Extended training (60min) | +0.018 | 1.44 | Peak vx=0.024 at 27%, regressed |
| 066 | gait=10, 4096 envs | -0.040 | 1.96 | Backward drift |
| 067 | Stronger forward gate (scale=50) | -0.018 | 1.89 | Backward drift |
| 068 | Hard forward gate (ReLU-like) | -0.046 | **2.20** | Best height, backward drift |

### Key Findings
1. **Reduced clip_range (0.1)**: Slower learning, didn't prevent regression
2. **Backward drift is a stable attractor**: Robot consistently learns to drift backward
3. **Height vs velocity trade-off**: Best height (2.20) came with worst velocity
4. **Forward gating variations don't help**: Sigmoid scale 50 and ReLU-like both failed
5. **Memory watchdog killed 3/6 experiments**: System memory pressure issues

### Current Code State (Reverted to EXP-056 Best Stable)
- `ratio_clip: 0.2` (standard)
- `diagonal_gait_reward: 5.0` (stable weight)
- `forward_gate: torch.sigmoid(vx * 20.0)` (reverted from hard gate)

---

## Session 15 Summary (4 experiments)

### What We Tried
| EXP | Change | vx | Notes |
|-----|--------|-----|-------|
| 059 | Early stopping (50% duration) | +0.027 | Peak vx=0.076, regressed |
| 060 | Very early stopping (400 iter) | +0.026 | No improvement |
| 061 | Velocity decay curriculum (decay=30) | +0.007 | Too aggressive |
| 062 | Softer decay (decay=10) | -0.001 | Robot learned to stand still |

### Key Findings
1. **Early stopping doesn't work**: Peak is proportional to training progress (80-90%), not absolute iterations
2. **Velocity decay curriculum FAILS**: Any decay prevents forward motion learning entirely
3. **Regression is fundamental**: Robot CAN achieve vx=0.076 but cannot maintain it
4. **Learning phases compress**: Shorter training compresses learning proportionally

---

## Session 14 Summary (4 experiments)

### What We Tried
| EXP | Change | vx | Notes |
|-----|--------|-----|-------|
| 055 | Diagonal gait (ungated) | +0.022 | Worse - no direction bias |
| **056** | **Forward-gated gait (5.0)** | **+0.036** | **NEW BEST** |
| 058 | Stronger gait (10.0) | +0.018 | Peak vx=0.061, regressed |

### Key Findings
1. **Diagonal gait reward WORKS**: First approach to beat baseline
2. **Forward gating essential**: Ungated led to backward stepping
3. **Higher weight = higher peak, more regression**: weight=10 peaked at vx=0.061 at 43%
4. **Mid-training regression**: All runs peak at 40-70%, then regress

---

## Session 13 Summary (5 experiments)

### What We Tried
| EXP | Change | vx | Notes |
|-----|--------|-----|-------|
| 050 | forward=100, LR=3e-4 | - | SANITY_FAIL |
| 051 | forward=50, neg=15 | - | SANITY_FAIL |
| 052 | Relax slip (0.10) | +0.024 | No improvement |
| 053 | Disable slip | -0.023 | WORSE |
| 054 | Gait phase obs | +0.022 | No improvement |

### Key Findings
1. **Forward reward ceiling**: Cannot increase beyond 40 without destabilizing
2. **Slip factor helps**: Removing it causes backward drift
3. **Gait phase didn't help**: Adding sin/cos to observations had no effect
4. **vx~0.03 wall confirmed**: Multiple approaches failed to break through

---

## Session 12 Summary (2 experiments)

### What We Tried
| EXP | Change | vx | Notes |
|-----|--------|-----|-------|
| 048 | Baseline verification | +0.010 | Confirmed env works |
| 049 | Reduced stability (5/8), high forward (60) | -0.081 | **FAILED** - episodes=76, unstable |

### Key Finding
**Stability rewards cannot be reduced.** Hypothesis that reducing stability would allow more forward exploration was disproven - it just causes the robot to fall over.

---

## Session 11 Summary (8 experiments overnight)

### What We Tried
| EXP | Change | vx | Notes |
|-----|--------|-----|-------|
| 030 | std baseline | 0.009 | Baseline comparison |
| 031 | LR=5e-4 | 0.004 | Lower LR alone not enough |
| **032** | **LR=5e-4, neg=10** | **0.029** | **BEST** |
| 033 | LR=3e-4 | -0.037 | Too conservative |
| 034 | forward=50 | -0.027 | Destabilized |
| 035 | longer training | 0.029 | Same as 032, no improvement |
| 036 | entropy=0.02 | -0.025 | No help |
| 037 | neg=15 | -0.034 | Too aggressive |

### Key Finding
**The robot hits a wall at vx~0.03 m/s.** Multiple hyperparameter variations were tested but none broke through to the 0.1 m/s target. This suggests:

1. **Reward structure limitation**: Velocity-based rewards may have insufficient gradient
2. **Local optimum**: Standing + slight motion is a stable attractor
3. **Policy architecture**: May need different network or training approach

---

## Recommended Next Approaches

**Note**: Sessions 15-16 tested PPO tuning and forward gating variations - all failed.
The robot CAN achieve vx=0.076 (76% of target) but cannot maintain it during continued training.
Backward drift is a stable attractor regardless of gating mechanism.

### ✅ Approaches That Work
- **Forward-gated diagonal gait reward** (EXP-056) - vx=0.036, 24% improvement (best stable)

### ❌ Approaches Ruled Out (Sessions 11-16)
- **Reward weight tuning** - No further gains from adjusting forward/stability balance
- **Air time rewards** (EXP-040-043) - Interfered with velocity learning
- **Reducing stability rewards** (EXP-049) - Caused falling/instability
- **Higher forward reward (>40)** (EXP-050/051) - Causes SANITY_FAIL
- **Slip factor modification** (EXP-052/053) - Makes things worse or no improvement
- **Gait phase observations** (EXP-054) - No significant improvement
- **Ungated gait reward** (EXP-055) - Led to backward stepping
- **Higher gait weight (10)** (EXP-058/064/066) - Peak vx=0.061 but regresses or drifts backward
- **Early stopping (50%)** (EXP-059) - Peak is proportional to progress, not absolute
- **Very early stopping** (EXP-060) - Learning phases compress proportionally
- **Velocity decay curriculum** (EXP-061/062) - Prevents forward motion learning entirely
- **Reduced PPO clip_range (0.1)** (EXP-063/064) - Slower learning, no regression prevention
- **Stronger forward gating (scale=50)** (EXP-067) - Robot learns backward drift
- **Hard forward gate (ReLU-like)** (EXP-068) - Best height but backward drift

### Priority 1: Checkpoint Selection (STILL TOP PRIORITY)
EXP-059 checkpoints exist near peak velocity (vx=0.076 at 89%):
- agent_13500.pt from EXP-059 is likely near peak
- Evaluate this checkpoint for deployment testing
- May capture walking behavior before regression

### Priority 2: Explicit Backward Penalty
Session 16 revealed backward drift as a stable attractor:
- Add explicit penalty for vx < 0 motion
- May break the backward drift optimum
- Weight needs careful tuning to avoid destabilization

### Priority 3: Reference Motion / Imitation Learning
Use manually designed walking gaits as reference:
- Define target joint trajectories for a walking cycle
- Reward tracking the reference motion
- Bypasses RL instability by providing explicit gait template

### Priority 4: Policy Architecture
Current MLP may not preserve gait memory:
- Add recurrent layer (LSTM/GRU) to maintain gait state
- Separate walking head from stability head
- Test larger networks (256→512 units)

---

## Current Configuration (Best Found)

```yaml
# skrl_ppo_cfg.yaml
learning_rate: 5.0e-4      # reduced from 1e-3
entropy_loss_scale: 0.01
timesteps: 15000           # ~15-20 min

# harold_isaac_lab_env_cfg.py
progress_forward_pos: 40.0
progress_forward_neg: 10.0  # increased from 5.0
standing_penalty: -5.0
height_reward: 15.0
upright_reward: 10.0
```

---

## 5-Metric Validation

| Priority | Metric | Threshold | Best Achieved |
|----------|--------|-----------|---------------|
| 1. SANITY | episode_length | > 100 | 432 (PASS) |
| 2. Stability | upright_mean | > 0.9 | 0.94 (PASS) |
| 3. Height | height_reward | > 1.2 | **1.67** (PASS) |
| 4. Contact | body_contact | > -0.1 | -0.006 (PASS) |
| 5. **Walking** | **vx_w_mean** | **> 0.1** | 0.029 (FAIL) |

**Goal**: Achieve vx > 0.1 m/s while maintaining standing

---

## Training Safety

### Memory Watchdog
Integrated into `harold train` to prevent OOM-induced system hangs:
- **RAM threshold**: 95% (kills training to prevent OOM)
- **Swap threshold**: 70% (kills training to prevent thrashing/GPU deadlock)
- **Detection**: `harold status` shows `STATUS: KILLED_BY_WATCHDOG` with memory stats
- **JSON**: `harold status --json` includes `killed_by_watchdog` field with full details
- **Disable**: `--no-watchdog` flag (not recommended)

### Environment Count
- **Recommended**: 6144 envs (best throughput)
- **Fallback**: 4096 envs if system unstable after crash
- **Never exceed**: 8192 envs

### Video Recording
**MANDATORY** on every run. Never disable `--video` flag. Video is critical for human review.

### Target Experiment Duration
**30-60 minutes per experiment.** Fast iteration is more valuable than long runs.
- 1250 iterations (~30 min) - Quick hypothesis test
- 2500 iterations (~60 min) - Standard experiment
- 4167 iterations (~100 min) - Extended, only if showing promise
