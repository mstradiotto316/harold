# Harold Next Steps

## Current Status (2025-12-25 ~04:30, Session 18 Complete)

### Session 18 Summary: Reference Implementation Analysis
**Tested techniques from AnyMal/Spot reference implementations - ALL made things WORSE**

| EXP | Technique | Result | Notes |
|-----|-----------|--------|-------|
| 083 | Exp velocity tracking (σ²=0.25) | vx=0.021 | σ² too large |
| 084 | Exp velocity tracking (σ²=0.01) | vx=0.004 | Symmetric kernel fails |
| 085 | Baseline verification | vx=0.0575 | CONFIRMED BEST |
| 088 | Bidirectional gait (mult) | vx=0.039 | Too restrictive |
| 089 | Bidirectional gait (add) | vx=0.018 | Still counterproductive |
| 090 | Domain randomization | vx=0.006 | Robot stands still |
| 092 | Smaller network [128,128,128] | vx=0.039 | Insufficient capacity |
| 094 | Higher LR (1.0e-3) | vx=-0.042 | Backward drift! |

**Key Insight**: Techniques for 30-50 kg robots don't transfer to Harold's 2 kg scale.

---

## Previous Status (2025-12-24 ~22:00, Session 17 Complete)

### Best Configuration (EXP-074/077/Verified) - CONFIRMED
Backward motion penalty breaks the backward drift attractor.
- `learning_rate: 5.0e-4`
- `progress_forward_neg: 10.0`
- `progress_forward_pos: 40.0`
- `height_reward: 15.0`
- `upright_reward: 10.0`
- `diagonal_gait_reward: 5.0`
- `backward_motion_penalty: 50.0`  # breaks backward drift
- `standing_penalty: -5.0`  # required for stability
- `entropy_loss_scale: 0.01`

**Best Stable Metrics (Reproducible):**
- Forward velocity: **0.057 m/s** (57% of target) - 58% improvement over baseline!
- Height reward: **1.25-1.50** (standing properly)
- All stability metrics: PASS

**Verification:** Latest run achieved vx=0.056, confirming reproducibility.

**Previous Best (EXP-056):**
- Forward velocity: 0.036 m/s (36% of target)

**Peak Observed (Unstable):**
- EXP-059 achieved **vx=0.076** (76% of target!) at 89% training, then regressed

---

## Session 17 Summary (9 experiments - Backward Penalty Breakthrough)

### What We Tried
| EXP | Change | vx | height | Notes |
|-----|--------|-----|--------|-------|
| 069 | backward=20, gait=5 | +0.012 | 1.37 | FIRST SUCCESS - broke backward drift! |
| 070 | backward=20, gait=10 | -0.053 | 1.57 | FAILED - gait overpowers penalty |
| 071 | backward=40, gait=5 | +0.039 | 1.40 | Improved! Higher penalty helps |
| 072 | backward=60, gait=5 | ABORT | ~5.0?! | Unstable, nonsensical height |
| **073** | **backward=50, gait=5** | **+0.056** | **1.39** | **NEW BEST! 56% of target** |
| 074 | +forward=60 | +0.045 | 1.45 | Worse than 073 |
| 075 | +gait=7 | +0.041 | 1.42 | Worse than 073 |
| 076 | 2500 iters | +0.057 | 1.34 | Same as 073 - longer training doesn't help |
| 077 | standing=0 | +0.024 | 0.60 | FAILED - body contact, instability |

### Key Findings
1. **Backward penalty WORKS**: Explicit penalty breaks backward drift attractor
2. **Optimal weight is 50**: Below this (20-40) weaker; above (60) causes instability
3. **Gait must stay low (5.0)**: Higher gait (10) overpowers backward penalty
4. **Standing penalty required**: Removing it causes body contact issues
5. **Longer training doesn't help**: 2500 iters same result as 1250
6. **55% improvement**: From 0.036 (EXP-056) to 0.056 (EXP-073)

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

**Note**: Session 17 breakthrough - backward penalty (50.0) breaks backward drift attractor!
Current best: vx=0.056 (56% of target). Need to push to 0.1 m/s.

### ✅ Approaches That Work
- **Backward motion penalty (50.0)** (EXP-073) - vx=0.056, 55% improvement over EXP-056
- **Forward-gated diagonal gait reward** (EXP-056) - vx=0.036, baseline with gait

### ❌ Approaches Ruled Out (Sessions 11-17)
- **Reward weight tuning** - No further gains from adjusting forward/stability balance
- **Air time rewards** (EXP-040-043) - Interfered with velocity learning
- **Reducing stability rewards** (EXP-049) - Caused falling/instability
- **Higher forward reward (>40)** (EXP-050/051) - Causes SANITY_FAIL
- **Slip factor modification** (EXP-052/053) - Makes things worse or no improvement
- **Gait phase observations** (EXP-054) - No significant improvement
- **Ungated gait reward** (EXP-055) - Led to backward stepping
- **Higher gait weight (7-10)** (EXP-058/070/075) - Peak vx=0.061 but regresses or drifts backward
- **Early stopping (50%)** (EXP-059) - Peak is proportional to progress, not absolute
- **Very early stopping** (EXP-060) - Learning phases compress proportionally
- **Velocity decay curriculum** (EXP-061/062) - Prevents forward motion learning entirely
- **Reduced PPO clip_range (0.1)** (EXP-063/064) - Slower learning, no regression prevention
- **Stronger forward gating (scale=50)** (EXP-067) - Robot learns backward drift
- **Hard forward gate (ReLU-like)** (EXP-068) - Best height but backward drift
- **Higher forward (60) with backward penalty** (EXP-074) - Worse than 073
- **Longer training (2500 iter)** (EXP-076) - Same result as 1250
- **Removing standing penalty** (EXP-077) - Causes body contact issues
- **Reduced height reward (10)** (EXP-078) - Caused body contact issues
- **Higher backward (60) + lower forward (30)** (EXP-079) - Unstable, vx=-0.005
- **Velocity threshold bonus** (EXP-080) - Counterproductive, vx=0.027
- **Stronger standing penalty (-10)** (EXP-081) - Made things worse, vx=0.028

### ❌ Approaches Ruled Out (Session 18 - Reference Impl)
- **Exponential velocity tracking** (EXP-083/084) - Symmetric kernel doesn't incentivize direction
- **Bidirectional gait reward** (EXP-088/089) - Both mult and add formulations counterproductive
- **Domain randomization** (EXP-090) - Robot learned to stand still to cope with noise
- **Smaller network [128,128,128]** (EXP-092) - Insufficient capacity, height degraded
- **Higher learning rate (1.0e-3)** (EXP-094) - Caused severe backward drift

**Key Insight**: AnyMal/Spot techniques don't transfer to Harold's 2 kg scale

### Priority 1: Reference Motion / Imitation Learning
Use manually designed walking gaits as reference:
- Define target joint trajectories for a walking cycle
- Reward tracking the reference motion
- Bypasses RL instability by providing explicit gait template

### Priority 2: Checkpoint Fine-Tuning
EXP-059 checkpoints exist near peak velocity (vx=0.076 at 89%):
- agent_13500.pt from EXP-059 is likely near peak
- Fine-tune from this checkpoint with current backward penalty config
- May combine peak velocity with stability

### Priority 3: Policy Architecture Changes
Current MLP may not preserve gait memory:
- Add recurrent layer (LSTM/GRU) to maintain gait state
- Separate walking head from stability head
- Test larger networks (256→512 units)

### Priority 4: Domain Randomization
Current training may be overfitting to exact physics:
- Enable friction/mass randomization
- May help find more robust walking gaits

---

## Current Configuration (EXP-073 Best)

```yaml
# skrl_ppo_cfg.yaml
learning_rate: 5.0e-4      # reduced from 1e-3
entropy_loss_scale: 0.01
timesteps: 15000           # ~30 min

# harold_isaac_lab_env_cfg.py
progress_forward_pos: 40.0
progress_forward_neg: 10.0
standing_penalty: -5.0
height_reward: 15.0
upright_reward: 10.0
diagonal_gait_reward: 5.0
backward_motion_penalty: 50.0  # NEW - breaks backward drift!
```

---

## 5-Metric Validation

| Priority | Metric | Threshold | Best Achieved |
|----------|--------|-----------|---------------|
| 1. SANITY | episode_length | > 100 | 356 (PASS) |
| 2. Stability | upright_mean | > 0.9 | 0.93 (PASS) |
| 3. Height | height_reward | > 1.2 | **1.25-1.50** (PASS) |
| 4. Contact | body_contact | > -0.1 | -0.01 (PASS) |
| 5. **Walking** | **vx_w_mean** | **> 0.1** | **0.057** (57% - BEST STABLE) |

**Goal**: Achieve vx > 0.1 m/s while maintaining standing
**Progress**: From 0.036 (Session 16) to 0.057 (Session 17) - 58% improvement!
**Next Milestone**: Break 0.07 m/s (70% of target)

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
