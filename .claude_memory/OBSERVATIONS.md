# Harold Observations & Insights

## Gait Development Observations

### What Works
- PPO with shared actor-critic (512→256→128) architecture
- Running normalization for observations and values
- KL-adaptive learning rate scheduling
- Action filtering (EMA beta=0.4) for smooth control

### What Needs Investigation
- Current reward structure may not produce stable forward walking
- Height maintenance reward (weight=1) may be too weak
- Feet air time reward may encourage hopping vs walking
- Termination conditions may be too aggressive

### Hypotheses to Test
1. ~~**H1**: Increasing forward velocity tracking weight (80→150) improves gait stability~~ **DISPROVEN** - EXP-003 showed this alone causes falling
2. **H2**: Adding explicit gait phase rewards (alternating leg pairs) helps walking - UNTESTED
3. **H3**: Curriculum on velocity commands helps learn walking before running - UNTESTED
4. **H4**: Joint velocity penalty reduces jerky motions - UNTESTED

### New Hypotheses (from EXP-003 failure)
5. **H5**: Need to INCREASE upright reward (1.8→5.0+) to maintain balance while encouraging movement
6. **H6**: Standing penalty should be SMALLER (0.5) or REMOVED - let forward reward do the work
7. **H7**: May need "recovery" reward for getting up after falling
8. **H8**: Gradual curriculum: first learn stable stance, then add forward incentive

## Technical Observations

### Simulation Performance
- Boot time: ~14s (acceptable for long training runs)
- Training speed: ~12 it/s with 1 env, scales with num_envs
- Checkpoint size: ~770KB-2.3MB per save

### Log Analysis Insights
- TensorBoard logs available for all runs
- Checkpoints saved every 2 timesteps during training
- Config (env.yaml, agent.yaml) preserved per experiment

## Failure Modes (Confirmed)

### FM-1: Standing in place (terrain_62)
- Robot maintains perfect balance but zero locomotion
- Cause: Reward structure allows standing as optimal

### FM-2: Fall and stay down (EXP-003) - NEW
- Robot falls over within first 75 steps and never recovers
- Cause: Standing penalty (2.0) breaks balance before walking is learned
- The penalty pushes robot to move, but without coordination it just falls
- Once fallen, no incentive or ability to recover

## Confirmed Failure Mode: Standing Instead of Walking
**Date**: 2025-12-19
**Policy**: terrain_62 (agent_128000.pt)
**Evidence**:
- Policy playback video shows zero forward displacement over 500 steps (25 seconds)
- Training videos show all 4096 robots clustered at spawn positions
- Robot maintains stable standing pose throughout episodes

**Root Cause Analysis**:
1. Reward structure allows high rewards from standing:
   - `upright_reward` (1.8) - satisfied by standing
   - `height_reward` (5.0) - satisfied by standing
   - `rear_support_bonus` (0.6) - actively rewards standing!
   - Low `progress_forward` relative to stability rewards
2. No penalty for zero velocity
3. No explicit gait cycle reward requiring leg movement

## Successful Patterns
Document successful strategies here:
- **Standing stability**: terrain_62 achieves stable standing (learned to balance)
- **EXP-012**: Achieved stable standing from scratch with upright=25, height=8, forward=0

---

## Session 2 Key Discoveries (2025-12-20 Overnight)

### Critical Finding: Motion Rewards Destabilize Standing
Every experiment that added motion rewards (forward velocity OR gait/leg movement)
eventually caused the robot to fall. The standing equilibrium is extremely fragile.

**Timeline of Destabilization:**
- Step 0-2500: Standing maintained (from checkpoint or stability training)
- Step 2500-5000: Usually still standing
- Step 5000-7500: Degradation begins
- Step 7500+: Robot fallen and not recovering

### What WORKS:
1. **Stability-only training** (upright=25, height=8, forward=0)
   - Robot learns to stand from scratch
   - Achieves stable quadruped stance by step 5000
   - Checkpoint: `2025-12-20_02-18-43_ppo_torch/checkpoints/agent_5000.pt`

### What DOES NOT work:
1. **High forward reward** (100, 30, 10, 3) - All destabilize
2. **Gait rewards alone** (leg_movement, feet_air_time) - Also destabilize
3. **Fine-tuning from standing checkpoint** - Motion gradients break stability
4. **Aggressive termination (-0.7)** - Makes learning harder, not better

### Hypotheses for Future Work:
1. **H9**: Need MUCH longer stability training (50k+ iterations) before any motion
2. **H10**: Curriculum with TINY increments (forward=0.1, wait 10k, add 0.1 more)
3. **H11**: Recovery reward (teach getting up) instead of preventing falls
4. **H12**: Use reference motion tracking (imitation learning) instead of reward shaping
5. **H13**: The robot's physical configuration may need tuning (joint limits, PD gains)

---

## 2025-12-20: CRITICAL - Observability Failure Discovered

### The "Fallen Forward" False Positive

**What Happened:**
I tracked `upright_mean > 0.9` and `vx_w_mean > 0.1` and concluded the robot was walking.
Video review revealed the robot was actually **fallen forward on its elbows** with back legs squatting.

**Why Metrics Were Misleading:**

| Metric | Value | Why Misleading |
|--------|-------|----------------|
| `upright_mean` | 0.93 | Back end elevated = high upright (even though fallen!) |
| `vx_w_mean` | 0.1+ | Falling forward = forward velocity |
| `height_reward` | 0.32 | **MISSED** - should be ~3.0 for proper standing |
| `body_contact_penalty` | -0.34 | **MISSED** - body touching ground |

**Root Cause:**
The robot learned to exploit the reward function by falling forward:
- forward=15.0, height=3.0 (5:1 ratio favored falling)
- Falling forward = positive velocity = forward reward
- Once on elbows, stable enough to not trigger termination (-0.75 threshold)

### Corrected Success Criteria

For Harold to be **truly walking**, ALL must be true:

| Metric | Threshold | What It Proves |
|--------|-----------|----------------|
| `upright_mean` | > 0.9 | Body not tilted |
| `height_reward` | > 1.2 | At correct standing height |
| `body_contact_penalty` | > -0.1 | No body parts on ground |
| `vx_w_mean` | > 0.1 m/s | Moving forward |
| `episode_length` | > 300 | Not falling quickly |

**Lesson Learned:**
NEVER trust `upright + vx` alone. ALWAYS check `height_reward` and `body_contact_penalty`.
The robot WILL exploit any gap in the reward function.

### New Hypothesis:
**H14**: Height reward must DOMINATE during training (height=25, forward<5) to prevent "falling forward" exploits.

---

## 2025-12-21: Context Management for Autonomous Agents

### Problem: Context Overflow from Training Logs

Previous agents experienced context overflow during long training runs. The tqdm progress bar
and simulation logs flood the terminal, consuming all available context.

### Solution: Background Execution + Compact Monitoring

Created helper scripts in `scripts/`:

1. **`run_experiment.sh [iterations]`** - Runs training in background
   - Redirects all output to `/tmp/harold_train.log`
   - Saves PID to `/tmp/harold_train.pid`
   - Returns immediately (doesn't block context)

2. **`check_training.sh`** - Compact progress check
   - Outputs only: STATUS, ELAPSED, LOG SIZE, PROGRESS %
   - No raw logs, no flooding
   - Use this instead of `tail -f`

3. **`analyze_run.py [run_name]`** - 4-metric analysis
   - Reads TensorBoard logs programmatically
   - Returns structured diagnosis + exit codes
   - Use after training completes

### Protocol for Long Training Runs

1. Start training: `./scripts/run_experiment.sh 4167`
2. Check periodically (every 30-90 min): `./scripts/check_training.sh`
3. After completion: `source ~/Desktop/env_isaaclab/bin/activate && python scripts/analyze_run.py`
4. Document results in EXPERIMENTS.md

### Key Insight: Observability over Video

As an LLM agent, I cannot reliably interpret video output. The 4-metric protocol
(upright, height_reward, body_contact, vx) provides reliable text-based verification.
NEVER rely on video to determine if the robot is walking.

---

## 2025-12-22: ROOT CAUSE ANALYSIS - Why Robot Falls Forward

### Critical Discovery: The Problem is NOT Reward Engineering

After 14 experiments trying reward shaping, we discovered the actual root causes:

### Root Cause #1: Contact Threshold Too High (PARTIALLY FIXED)
- **Issue**: `body_contact_threshold` was 10N
- **Problem**: Elbow contact is ~5N per point (below threshold = undetected)
- **Fix Applied**: Lowered to 3N in EXP-014
- **Result**: Contact detection improved, but insufficient alone

### Root Cause #2: Spawn Pose May Bias Forward Lean (TO TEST)
- **Issue**: Shoulder joints spawn at ±0.20 rad (asymmetric)
- **Problem**: May create initial forward momentum
- **Proposed Fix**: Neutral shoulders (0.0) and higher spawn (0.30m)

### Root Cause #3: Spawn Height Too Low
- **Issue**: Robot spawns at 0.24m height
- **Problem**: Only 0.06m clearance to elbow contact (~0.18m)
- **Proposed Fix**: Raise to 0.30m

### Key Insight: The Robot is NOT "Exploiting" Rewards

The robot correctly learned that elbow contact has no penalty (was below threshold).
**Fix the detection gap, not the rewards.**

---

## Process Improvements (2025-12-22)

### Problem: 2-Hour Experiments with Flat Rewards
We ran experiments for ~2 hours even when total reward was flat.

### Solution: Fast Iteration Protocol
- **Short runs**: 1000 iterations (~15-30 min)
- **Early stopping**: If height < 1.5 at 50%, stop and adjust
- **Extend only if improving**: If height > 2.5, continue training

### Problem: Only Monitoring Final Metrics
We extracted 7 metrics but TensorBoard logs 25-30 scalars.

### Solution: Monitor Learning Dynamics
- Policy loss (should decrease)
- Value loss (should decrease)
- Reward trend over time (should increase)
- Height trend during training (not just final)

---

## Updated Hypotheses

| ID | Hypothesis | Status |
|----|------------|--------|
| H14 | Height reward must dominate (25.0) | TESTED - Insufficient alone |
| H15 | Contact threshold must be <5N | TESTED (3N) - Partial success |
| H16 | Spawn pose causes forward lean | **CONFIRMED!** |
| H17 | Higher spawn height prevents elbow contact | **CONFIRMED!** |

---

## 2025-12-22: BREAKTHROUGH - Spawn Pose Fix Worked!

### EXP-015 Results (In Progress)
- **Height reward: 3.43** (first time above 2.0!)
- **Robot standing properly** for the first time in 15 experiments
- **Root cause confirmed**: Spawn pose was the issue, not rewards

### What Fixed It
1. Raised spawn height: 0.24m → 0.30m
2. Neutral shoulders: ±0.20 → 0.0

### Next Step
Add forward reward (progress_forward_pos: 2.0-5.0) to make robot walk.

---

## 2025-12-22 Evening: Reward Tuning Limits Reached

### Key Discovery: Policy Regression Pattern

All 5 experiments (EXP-021 through EXP-025) showed the same pattern:
1. **Early (0-25%)**: Robot explores, low height, near-zero velocity
2. **Mid (25-50%)**: Forward motion emerges (peak vx ~0.02-0.05)
3. **Late (50-100%)**: **Velocity regresses**, robot settles into standing

**Critical observation**: The robot LEARNS to walk mid-training but then UNLEARNS it.

### Reward Tuning Has Diminishing Returns

| Change | Effect on vx | Effect on height |
|--------|--------------|------------------|
| forward_pos: 40→50 | WORSE (-0.025) | WORSE (0.96) |
| forward_pos: 40→25 | WORSE (+0.005) | WORSE (1.43) |
| standing_penalty: -5→-10 | WORSE (-0.018) | WORSE (1.26) |
| forward_neg: 5→10 | WORSE (-0.014) | WORSE (1.30) |

**Conclusion**: The EXP-021 config (forward=40, height=15, penalty=-5) is a local optimum.
Further reward tuning cannot reach the 0.1 m/s walking threshold.

### Optimal Configuration Found

```python
progress_forward_pos: float = 40.0
progress_forward_neg: float = 5.0
standing_penalty: float = -5.0
height_reward: float = 15.0
upright_reward: float = 10.0
```

Result: vx = +0.023 m/s, height = 1.58 (23% of target)

### Hypotheses for Further Investigation

1. **H18**: Early stopping at peak velocity (~40-50% training) will preserve walking behavior
2. **H19**: Longer training (100k vs 30k timesteps) may achieve convergence to walking
3. **H20**: Curriculum learning (progressive forward reward) may prevent regression
4. **H21**: The regression is caused by PPO's policy update destabilizing learned behaviors
5. **H22**: Reference motion imitation may be necessary to achieve stable walking

### Training Duration Now Correct

- **Previous bug**: 27000 iterations was ~12 hours (not 30 min)
- **Fix**: 1250 iterations = 30k timesteps = ~30 min
- **Benchmark**: 6144 envs at 16.6 it/s = optimal throughput

### User Insight: Athletic Crouch is Acceptable

The user confirmed that height_reward ~1.5 (athletic crouch) is acceptable.
The height threshold of 2.0 was too strict for walking.
**ACTION TAKEN**: Threshold reduced from 2.0 to 1.2 in the validation code.
Focus should be on achieving vx > 0.1, not strict height requirements.

---

## 2025-12-22: Performance Optimization

### System Profile (RTX 4080 + i7-8700K)

| Resource | Usage | Capacity |
|----------|-------|----------|
| GPU VRAM | 45% (7.4 GB) | 16 GB |
| GPU Compute | 21-49% SM | 100% |
| RAM | 59% (19 GB) | 32 GB |

### Training Throughput
- **Average**: ~16.9 it/s (4096 envs)
- **Peak**: ~20.88 it/s
- **Samples/sec**: ~1.66M

### Applied Optimization
Added `--rendering_mode performance` flag to `harold.py` train command.
This optimizes Isaac Sim's rendering pipeline for headless training while preserving video output.

### Considered but Not Applied
- **Disable height scanner**: Still used for position data in env implementation
- **Reduce video interval**: Video is critical logging infrastructure

### Comprehensive Benchmark Results (2025-12-22)

| Envs | it/s | Samples/s | GPU MB | RAM GB | Notes |
|------|------|-----------|--------|--------|-------|
| 2048 | 20.9 | 1.03M | 4046 | 7 | Fast iterations, low throughput |
| 4096 | 19.6 | 1.93M | 4552 | 8 | Previous default |
| 6144 | 16.6 | 2.45M | 5053 | 8 | **RECOMMENDED** - good balance |
| 8192 | 14.7 | 2.88M | 5550 | 9 | Best throughput, use when system clean |

**Key Findings:**
- 8192 envs gives 1.5x throughput vs 4096
- All configs fit comfortably in GPU/RAM when system is clean
- Memory pressure from other apps (Chrome, Claude instances) caused swap thrashing at 8192
- **Parallel experiments are 43% as efficient** as single large runs (tested 2x2048)

**Configuration Updated:**
- Default: 6144 envs (safe balance)
- Default iterations: 4167 (~100k timesteps, ~100 min)
- Scripts: `benchmark_comprehensive.py`, `test_parallel.py`

**Time Calculation:**
- timesteps = max_iterations × rollouts (24)
- Example: 4167 × 24 = 100k timesteps
- At 16.6 it/s: 100k / 16.6 = 100 min

---

## 2025-12-23: System Crash Investigation

### ROOT CAUSE: OOM (Out of Memory) Induced GPU Driver Deadlock

**What Happened:**
After 1 day 8 hours 43 minutes of continuous training, the system experienced a
complete lockup - keyboard LEDs stopped responding, Ctrl+Alt+F3 failed to switch TTY.

**Evidence from Logs:**
```
Dec 22 16:38:05 kernel: Out of memory: Killed process 230536 (python)
  total-vm: 140,581,264kB (~134GB virtual)
  anon-rss: 24,026,584kB (~24GB resident)

systemd: Consumed 1d 8h 43min CPU time, 27.6G memory peak, 5.1G memory swap peak
```

**Crash Mechanism:**
1. Training memory grew to 27.6GB RAM + 5.1GB swap = 32.7GB (system has 32GB)
2. Kernel OOM killer triggered, attempted to kill training process
3. NVIDIA GPU driver had pending operations waiting for memory to complete
4. Deadlock: OOM killer can't reclaim memory, GPU driver waiting for memory
5. Complete system hang - even kernel interrupt handlers stopped

### Why This Happened

Isaac Lab training has a slow memory leak over very long runs. With 6144 envs,
memory starts at ~8GB but gradually grows. After 30+ hours, it can exceed system RAM.

### Prevention Measures Implemented

**1. Memory Watchdog (`scripts/memory_watchdog.py`)**
- Monitors RAM and swap usage every 10 seconds
- Only intervenes at truly dangerous levels (RAM > 95% or Swap > 70%)
- Warns at 85% RAM to give visibility without interrupting training
- Automatically started by `harold train`

**2. Updated Training Defaults**
- RAM kill threshold: 95% (only at critical levels)
- Swap kill threshold: 70% (only when thrashing imminent)
- Added `--no-watchdog` flag for advanced users

**3. Target Training Duration**
**30-60 minutes per experiment.** Fast iteration > long runs.
- 1250 iterations (~30 min) - Quick hypothesis test
- 2500 iterations (~60 min) - Standard experiment
- 4167 iterations (~100 min) - Extended, only if promising

**4. Video Recording**
MANDATORY on every training run. Never disable `--video` flag.

### Key Lessons

1. **Target 30-60 minute experiments** - fast iteration beats long runs
2. **Swap thrashing is deadly** - causes GPU driver deadlocks
3. **OOM + GPU = system hang** - kernel can't recover gracefully
4. **Memory watchdog is now default** in harold train (safety net, not expected to trigger)

### Current Memory Settings
```
swappiness = 60 (default)
overcommit_memory = 0 (heuristic)
```

Consider reducing swappiness to 10 for training workloads if hangs continue.

---

## 2025-12-23: Lower Learning Rate Shows Promise

### Key Discovery: LR=5e-4 Reduces Regression Severity

**EXP-028 Results** (LR=5e-4 vs standard 1e-3):
- Best standing achieved: height=2.11 (vs 1.58 with standard LR)
- Final vx stayed positive: +0.008 (vs going negative in other experiments)
- Peak vx: +0.023 at 68% (same as EXP-021 best)

### Velocity Progression Comparison

| Stage | EXP-021 (LR=1e-3) | EXP-028 (LR=5e-4) |
|-------|-------------------|-------------------|
| 8% | ~0.0 | -0.007 |
| 43% | ~0.0 | -0.027 |
| 68% | +0.023 (peak) | +0.023 (peak) |
| 93% | unknown | +0.018 |
| Final | +0.023 | +0.008 |

**Key Insight**: Lower LR doesn't prevent regression but makes it less severe.
The robot still regresses from peak (0.023→0.008) but doesn't go negative.

### New Hypothesis

**H23**: Combining lower LR with early stopping may preserve peak performance.
- Lower LR reaches same peak as standard LR (0.023 at 68%)
- If we stop at 60-70% of training, we might preserve peak
- EXP-031 should test: LR=5e-4 + 625 iterations (15 min)

### Training Stability Issues Post-Crash

After the OOM crash, training processes were unstable:
- 6144 envs caused initialization freezes
- Reduced to 4096 envs for stability
- Startup takes longer (~5-7 min before data appears)

**Workaround**: Use 4096 envs until system is fully stable

---

## 2025-12-23: Session 11 - Hyperparameter Grid Search Complete

### Key Finding: vx~0.03 is a Hard Wall

After 8 experiments systematically varying hyperparameters, the best result achieved is vx=0.029 m/s (EXP-032). Multiple variations all converge to or below this value:

| Change | Result |
|--------|--------|
| Lower LR (3e-4) | vx=-0.037 (too conservative) |
| Higher forward (50) | vx=-0.027 (destabilized) |
| Longer training | vx=0.029 (same as shorter) |
| Higher entropy (0.02) | vx=-0.025 (no help) |
| Higher backward penalty (15) | vx=-0.034 (too aggressive) |

### Best Configuration Found (EXP-032)

```yaml
learning_rate: 5.0e-4
progress_forward_pos: 40.0
progress_forward_neg: 10.0
standing_penalty: -5.0
height_reward: 15.0
upright_reward: 10.0
entropy_loss_scale: 0.01
```

**Result**: vx=0.029 m/s, height=1.67 (29% of 0.1 m/s target)

### Analysis: Why the Robot Can't Break Through

1. **Local Optimum**: "Standing + slight drift" is a stable attractor
2. **Reward Gradient Insufficient**: Velocity rewards at 40.0 don't overcome the stability rewards
3. **Policy Architecture Limitation**: The 512-256-128 network may not have capacity for complex gaits
4. **Missing Coordination Signal**: Velocity reward doesn't teach leg alternation

### Updated Hypotheses for Future Work

| ID | Hypothesis | Status |
|----|------------|--------|
| H23 | LR=5e-4 + early stopping preserves peak | **DISPROVEN** - Still regresses |
| H24 | Higher backward penalty prevents drift | **DISPROVEN** - neg=15 was worse |
| H25 | Higher entropy helps exploration | **DISPROVEN** - No improvement |
| **H26** | **Gait-based rewards needed** | **TO TEST** |
| **H27** | **Curriculum learning needed** | **TO TEST** |

### Process Improvements

1. **Grid search methodology**: Systematically vary one parameter at a time
2. **Convergence detection**: Longer training (30 min) gave same result as shorter (15 min)
3. **Sweet spot identification**: LR=5e-4, neg=10 is optimal; both higher and lower values are worse

### Current State of Knowledge

**What we can reliably achieve:**
- Robot standing properly (height > 1.5)
- Good stability (upright > 0.9, contact > -0.1)
- Slight forward drift (vx ~ 0.03 m/s)

**What we cannot achieve with current reward structure:**
- Walking (vx > 0.1 m/s)
- Consistent forward motion without regression

---

## Session 12 Observations (2025-12-23)

### Stability Reward Necessity (H28 - TESTED)

**Hypothesis**: Reducing stability rewards would allow more forward exploration

**Test**: EXP-049 reduced height_reward 15→8, upright_reward 10→5, increased forward 40→60

**Result**: SANITY_FAIL
- Episode length dropped from 344→76 steps
- Robot became unstable, strong backward drift (-0.08 m/s)
- The robot needs stability rewards to learn basic standing

**Conclusion**: Stability rewards cannot be reduced without causing instability. The local optimum at vx~0.03 is protected by the stability requirement.

### Updated Hypothesis Status

| ID | Hypothesis | Status |
|----|------------|--------|
| **H28** | **Reduce stability rewards** | **DISPROVEN** - Causes instability |
| H26 | Gait-based rewards needed | TO TEST |
| H27 | Curriculum learning needed | TO TEST |
| **H29** | Reference motion / imitation learning | TO TEST |

### Memory Management Observation

- 6144 envs can trigger memory watchdog (swap>70%) if previous runs left residual swap
- Recommended: Use 4096 envs when system has high swap, or clear swap before training
- harold.py now supports `--num-envs` argument for flexibility

### Approaches Ruled Out

1. **Simple reward weight tuning** - Tested extensively in Sessions 10-11, no breakthrough
2. **Air time rewards** - EXP-040-043 showed these interfere with velocity learning
3. **Reduced stability rewards** - EXP-049 caused instability

### Remaining Approaches to Test

1. **Contact-based gait reward**: Reward alternating foot contacts (breaks velocity dependency)
2. **Reference motion**: Define a target walking gait and reward tracking it
3. **Velocity curriculum**: Start with lower target vx, increase gradually
4. **Two-phase training**: Separate stability and forward training

---

## 2025-12-23: Session 13 - Slip Factor and Observation Space Experiments

### Slip Factor Findings

**Hypothesis**: Slip factor (SLIP_THR=0.03) limits walking exploration by penalizing foot movement.

**Results**:
- EXP-052 (SLIP_THR=0.10): vx=+0.024 (slightly worse than baseline 0.029)
- EXP-053 (f_slip=1.0 disabled): vx=-0.023 (MUCH worse, backward drift)

**Conclusion**: Slip factor HELPS forward motion. It creates a gradient toward proper walking by:
1. Penalizing sliding feet → encourages lifting feet
2. Rewarding clean steps → encourages proper gait
3. Without it, robot can "cheat" by sliding forward without stepping

### Gait Phase Observation Findings

**Hypothesis**: Adding sin/cos gait phase to observations helps policy coordinate leg movements.

**Results**:
- EXP-054 (48D→50D with gait phase): vx=+0.022 (no improvement)
- Mid-training peak: vx=0.031 at 86%, then regressed to 0.022

**Conclusion**: Gait phase observation alone doesn't break the standing optimum. The policy still prefers the reward-maximizing "stand still" strategy.

### Forward Reward Ceiling

**EXP-050/051 Findings**:
- forward=100 + LR=3e-4: SANITY_FAIL (ep=76)
- forward=50 + neg=15: SANITY_FAIL (ep=23)

**Conclusion**: Forward reward cannot be increased beyond 40 without destabilizing learning. The reward structure has a fundamental limit, not a tuning problem.

### Root Cause Analysis

The vx~0.03 ceiling is caused by a **local optimum** in reward space:

```
Reward(standing) ≈ height_reward + upright_reward + 0
Reward(stepping) ≈ (height_reward - Δh) + (upright_reward - Δu) + forward_reward
```

Where:
- Δh = height drop during step (~0.1-0.2 reward units)
- Δu = upright drop during step (~0.1-0.2 reward units)
- forward_reward = small velocity gain (~0.01-0.03 * 40 = 0.4-1.2 reward units)

Net reward for stepping is often negative or marginally positive → standing is safer.

### New Hypothesis: Contact-Based Gait Reward

To break the standing optimum, we need a reward that:
1. Is independent of velocity (breaks the standing=safe equation)
2. Directly rewards leg movement
3. Encourages alternating diagonal pairs (proper quadruped gait)

Proposed reward:
```python
gait_reward = sum(first_contact[diagonal_pair]) * weight
```

This rewards lifting and placing feet, regardless of forward velocity.

---

## 2025-12-24: Session 14 - BREAKTHROUGH with Diagonal Gait Reward

### Implementation Details

Added diagonal gait reward to `harold_isaac_lab_env.py`:
- **Diagonal pairs**: FL+BR (pair 0) vs FR+BL (pair 1) - standard trotting pattern
- **Tracking**: `_last_diagonal_pair` tracks which pair was last in contact
- **Switch reward**: Reward given when switching from one valid pair to another
- **Forward gating**: `sigmoid(vx * 20.0)` gates reward to only activate when moving forward

```python
# EXP-056 reward formula (forward-gated)
forward_gate = torch.sigmoid(vx * 20.0)  # ~0 when vx<0, ~1 when vx>0.1
diagonal_gait_reward = weight * valid_switch.float() * forward_gate * upright_sq
```

### Results Summary

| Experiment | Gait Weight | Gating | Final vx | Peak vx | Notes |
|------------|-------------|--------|----------|---------|-------|
| EXP-055 | 5.0 | None | +0.022 | - | Worse - stepped backward |
| EXP-056 | 5.0 | Forward | **+0.036** | - | **BEST final** |
| EXP-058 | 10.0 | Forward | +0.018 | +0.061 | Regressed from peak |

### Key Insights

**1. Forward Gating is Essential**
- Ungated gait reward (EXP-055) encouraged stepping in any direction → backward drift
- Forward gating (sigmoid at vx=0) biases stepping toward forward motion
- This creates the connection between stepping and forward velocity rewards

**2. Higher Weight = Higher Peak but More Regression**
- EXP-058 (weight=10) achieved peak vx=0.061 at 43% training (61% of 0.1 target!)
- But final result regressed to 0.018 (worse than weight=5)
- Higher reward amplifies both learning and instability

**3. Mid-Training Regression is Fundamental**
The regression pattern is consistent across all experiments:
- Peak performance at 40-70% of training
- Gradual regression as training continues
- PPO may be destabilizing learned behaviors

### Why Diagonal Gait Works

The diagonal gait reward breaks the standing optimum by:
1. **Velocity independence**: Reward is for foot contact pattern, not velocity
2. **Direct stepping gradient**: Every diagonal switch is rewarded
3. **Forward gating**: Connects stepping reward to forward motion

This creates a new reward structure:
```
Reward(standing) ≈ stability rewards + 0 (no gait switches)
Reward(stepping forward) ≈ stability rewards + gait_reward + velocity_reward
```

Now stepping forward has a clear reward advantage over standing.

### Remaining Challenge: Mid-Training Regression

EXP-058 shows the robot CAN walk at 0.061 m/s (61% of target), but the policy regresses during continued training. Possible solutions:
1. **Early stopping**: Save checkpoint at peak performance
2. **Curriculum**: Decay gait reward as velocity increases
3. **Lower learning rate**: Slower updates may prevent destabilization
4. **KL constraint**: Stricter PPO clipping may preserve good behaviors

### Updated Hypotheses

| ID | Hypothesis | Status |
|----|------------|--------|
| H26 | Gait-based rewards needed | **CONFIRMED** - vx improved 24% |
| **H30** | Forward gating essential for gait reward | **CONFIRMED** |
| **H31** | Higher gait weight causes more regression | **CONFIRMED** |
| **H32** | Early stopping can preserve peak performance | **DISPROVEN** (Session 15) |
| **H33** | Gait reward curriculum prevents regression | **DISPROVEN** (Session 15) |

---

## Session 15 Analysis: Early Stopping & Curriculum Failed

### Key Finding: Regression is Proportional to Training Progress

EXP-059 (50% duration) achieved peak vx=0.076 at 89% of its shortened run:
- This proves the robot CAN walk at 76% of target velocity
- But peak timing is relative to training progress, not absolute iterations
- Shortening training just compresses learning phases proportionally

### Velocity Decay Curriculum: Complete Failure

Both decay=30 (EXP-061) and decay=10 (EXP-062) failed:
- Any decay of gait reward as vx increases prevents forward motion learning
- Robot optimizes for standing still (maximum gait reward at vx=0)
- The curriculum creates a local optimum at vx=0

### Root Cause Analysis: PPO Forgetting

The consistent pattern across experiments suggests:
1. **Learning phase**: Robot learns stepping and forward motion (0-80%)
2. **Peak phase**: Maximum velocity achieved (80-90%)
3. **Forgetting phase**: Policy updates destroy walking behavior (90-100%)

This may be PPO-specific:
- Large policy updates overwrite successful behaviors
- Value function mismatch causes instability
- No explicit mechanism to preserve good behaviors

### New Hypotheses (Session 15)

| ID | Hypothesis | Status |
|----|------------|--------|
| **H34** | Checkpoint selection (mid-training) can capture peak | TO TEST |
| **H35** | PPO clip_range adjustment prevents forgetting | **DISPROVEN** (Session 16) |
| **H36** | Reference motion/imitation learning bypasses RL instability | TO TEST |
| **H37** | Recurrent policy (LSTM) preserves gait memory | TO TEST |

---

## Session 16 Analysis: PPO Tuning & Forward Gating Failed

### Key Discovery: Backward Drift is a Stable Attractor

Session 16's overnight experiments revealed that backward drift is a stable attractor that the robot consistently discovers, regardless of forward gating mechanism.

**Evidence:**
- EXP-066 (gait=10, 4096 envs): vx = -0.040 (backward)
- EXP-067 (sigmoid scale=50): vx = -0.018 (backward)
- EXP-068 (hard ReLU gate): vx = -0.046 (backward)

All three experiments achieved good height (1.89-2.20) while drifting backward.

### Why Forward Gating Failed

The forward gating mechanism (multiply gait reward by sigmoid(vx)) was designed to only reward stepping when moving forward. However:

1. **Gating creates zero gradient at vx=0**: When robot stands still, gait reward is ~0.5, providing no strong direction signal
2. **Backward drift has lower instability**: Stepping backward may be mechanically easier (robot's weight distribution)
3. **Local minimum**: Once backward drift starts, forward gating reduces gait reward, making recovery harder

### PPO Clip Range Analysis

Reduced clip_range (0.1 vs 0.2) results:
- EXP-063: vx=+0.024, height=1.26 (lower than baseline)
- EXP-064: Peak vx=+0.029, final vx=+0.019 (still regressed)

**Conclusion**: Smaller clip_range slows learning but doesn't prevent regression. The regression appears to be caused by fundamental reward structure, not policy update size.

### Height vs Velocity Trade-off

Session 16 achieved the best height ever (2.20 in EXP-068) but with the worst velocity (-0.046). This suggests:

1. **Two competing objectives**: Standing tall vs moving forward
2. **Current reward structure favors height**: Robot optimizes for height when forward motion is difficult
3. **Backward drift satisfies both partially**: Robot maintains height while "moving" (backward)

### Updated Hypotheses (Session 16)

| ID | Hypothesis | Status |
|----|------------|--------|
| **H35** | PPO clip_range adjustment prevents forgetting | **DISPROVEN** - Slower learning, same regression |
| **H38** | Backward drift is a stable attractor | **CONFIRMED** |
| **H39** | Hard forward gating (ReLU) prevents backward optimization | **DISPROVEN** |
| **H40** | Explicit backward penalty needed to break drift optimum | TO TEST |
| **H41** | Height/velocity trade-off is fundamental to reward structure | OBSERVED |

### Memory Watchdog Performance

3 of 6 experiments were killed by memory watchdog:
- EXP-063: Killed at ~90% progress
- EXP-064: Killed at ~97% progress
- EXP-065: Killed at ~52% progress

**Root cause**: 6144 envs + overnight run accumulates memory. Swap couldn't be cleared without sudo.

**Recommendation**: Use 4096 envs for overnight runs, or ensure system starts clean.

---

## 2025-12-25: RAM Upgrade & New Benchmarks

### Hardware Upgrade
RAM upgraded from 32GB to 64GB DDR4, eliminating memory as a bottleneck.

### Comprehensive Benchmark Results (64GB RAM)

| Envs | it/s | Samples/s | GPU MB | GPU % | RAM GB | RAM % |
|------|------|-----------|--------|-------|--------|-------|
| 4096 | 18.0 | 1.77M | 4296 | 26% | 8 | 13% |
| 6144 | 15.2 | 2.25M | 5007 | 31% | 9 | 14% |
| **8192** | **12.9** | **2.54M** | **5563** | **34%** | **9** | **14%** |
| 10000 | 11.3 | 2.71M | 5961 | 36% | 10 | 16% |
| 12000 | 10.6 | 3.05M | 6443 | 39% | 10 | 16% |
| 16384 | 8.7 | 3.43M | 7591 | 46% | 11 | 17% |

**Key Findings:**
1. **No OOM at any level** - Even 16384 envs uses only 17% of RAM
2. **GPU is now the bottleneck** - Not RAM
3. **1.94x throughput** at 16384 vs 4096 (3.43M vs 1.77M samples/s)
4. **8192 recommended** for fast iteration (12.9 it/s, ~97 min for 1250 iter)
5. **16384 available** for maximum throughput when time permits

### Time per 1250 Iterations (Standard Experiment)
| Envs | Time |
|------|------|
| 4096 | ~69 min |
| 6144 | ~82 min |
| 8192 | ~97 min |
| 16384 | ~143 min |

### Updated Recommendation
- **Default**: 8192 envs (13% more throughput than 6144, similar wall-clock time)
- **Max throughput**: 16384 envs when running fewer, longer experiments
- **Memory watchdog**: Now rarely triggers with 64GB headroom

---

## 2025-12-26: Session 19-20 - Critical Discovery

### backward_penalty=75 is a SHARP Local Optimum

Session 20 revealed that backward_motion_penalty has an extremely narrow optimal range:

| backward_penalty | vx | Direction |
|-----------------|-----|-----------|
| 50 | +0.057 | Forward (old best, Session 17) |
| 70 | **-0.085** | BACKWARD DRIFT |
| **75** | **+0.034** | **FORWARD** |
| 80 | **-0.053** | BACKWARD DRIFT |
| 100 | +0.028 | Forward but crouched |

**Key Insight**: Both 70 AND 80 cause backward drift. The optimal value is exactly 75.

This suggests the reward landscape has a very narrow "forward motion valley" surrounded by "backward drift attractors" on both sides.

### Optimal Configuration Locked (EXP-097)

After extensive testing, the optimal configuration is:
```python
backward_motion_penalty = 75.0   # CRITICAL - must be exactly 75
height_reward = 20.0             # 25 was worse (0.029 vs 0.034)
progress_forward_pos = 40.0      # 45 was worse (0.022 vs 0.034)
standing_penalty = -5.0          # Required for stability
diagonal_gait_reward = 5.0       # 3 works equally well
```

**Best Achieved**: vx=+0.034 m/s (34% of 0.1 target) with all metrics passing

### Why 75 Works

Hypothesis: backward_penalty=75 provides exactly the right "push" to break the backward drift attractor without overcorrecting into crouching behavior.

- **Too low (≤70)**: Robot drifts backward because the penalty isn't strong enough
- **Just right (75)**: Robot moves forward with proper posture
- **Too high (≥80)**: Robot overcorrects by minimizing all motion, causing secondary backward drift

### Updated Hypothesis Status

| ID | Hypothesis | Status |
|----|------------|--------|
| H40 | Explicit backward penalty needed to break drift optimum | **CONFIRMED** (Session 17) |
| **H42** | backward_penalty has a narrow optimal range (75) | **CONFIRMED** (Session 20) |
| **H43** | Higher height/forward rewards don't improve velocity | **CONFIRMED** (Session 20) |
| H44 | Curriculum learning may break the 0.034 ceiling | TO TEST |
| H45 | Fine-tuning from peak velocity checkpoints | TO TEST |

### Implications for Future Work

1. **Don't fine-tune backward_penalty** - The optimal is known (75)
2. **Different approaches needed** - Simple reward tuning has reached its limit
3. **Consider fundamentally different methods**:
   - Curriculum learning (height first, then velocity)
   - Checkpoint fine-tuning from peak runs
   - Reference motion / imitation learning
   - Policy architecture changes (LSTM for gait memory)

---

## 2025-12-26: Session 21 - Actuator Torque Analysis

### Root Cause: Insufficient Torque for Extended Leg Poses

**Discovery**: When testing scripted walking gaits, extended leg poses (calf=-0.50) caused the robot to fall forward. Analysis revealed:

1. **Torque requirement scales with leg extension**:
   - Athletic crouch (calf=-1.40): ~1.5 Nm required (OK)
   - Extended stance (calf=-0.50): ~3.0 Nm required (exceeds limit!)

2. **Original effort_limit was too conservative**:
   - Simulation used 2.0 Nm (68% of hardware max)
   - This prevented extended leg poses needed for walking

3. **Solution**: Increased effort_limit from 2.0 to 2.5 Nm (85% of hardware max)
   - Hardware max: 2.94 Nm
   - New simulation limit: 2.5 Nm
   - Maintains 15% safety margin for sim-to-real

### Static Pose Test Results

After increasing effort_limit:
- Robot CAN hold athletic crouch (calf=-1.40): ✓
- Robot CAN hold extended stance (calf=-0.50): Testing in progress

### Scripted Gait Implementation

Added scripted walking mode for physics validation:
- Environment variable: `HAROLD_SCRIPTED_GAIT=1`
- Static test mode: `HAROLD_STATIC_TEST=1`
- Diagonal trot pattern (FL+BR vs FR+BL)
- Configurable via `ScriptedGaitCfg`

Current gait parameters:
```python
frequency: 0.5 Hz      # Slow gait
swing_thigh: 1.10      # Leg forward after swing
stance_thigh: 0.40     # Leg back during stance
stance_calf: -0.50     # Extended during stance
swing_calf: -1.50      # Bent to lift foot during swing
```

### CRITICAL BREAKTHROUGH: PD Stiffness Was the Problem

**Discovery**: Even with effort_limit=2.8 Nm, calves were stuck at -1.57 (joint limit) when targeting -1.10. The real robot does pushups fine, so simulation was wrong.

**Root Cause**: Default stiffness=200 was far too low for the implicit actuator model.

**Solution**: Increased stiffness from 200 → 1200

| Stiffness | Damping | Calf Tracking | Height | vx |
|-----------|---------|---------------|--------|-----|
| 200 | 75 | Stuck at -1.57 | 0.126 m | ~0 |
| 300 | 90 | Stuck at -1.57 | 0.128 m | ~0 |
| 800 | 40 | -1.15 to -1.41 | 0.173 m | +0.057 m/s |
| **1200** | **50** | **-0.92 to -1.36** | **0.174 m** | **+0.141 m/s** |

**Final actuator settings**:
```python
effort_limit_sim = 2.8  # 95% of 2.94 Nm hardware max
stiffness = 1200.0      # High for real servo-like tracking
damping = 50.0          # Low damping ratio for responsiveness
```

### Scripted Gait SUCCESS

With correct PD gains, scripted diagonal trot achieves **vx = +0.141 m/s** (141% of 0.1 m/s target!)

Working gait parameters:
```python
frequency: 1.0 Hz
swing_thigh: 0.40       # Leg back during swing
stance_thigh: 0.90      # Leg forward during stance
stance_calf: -0.90      # Extended during stance
swing_calf: -1.55       # Bent during swing (foot lifted)
```

**Key Insight**: The previous "Phase 1 failed - Harold requires reactive balance" conclusion was WRONG. It was a simulation tuning issue, not a physics limitation.

### Updated Hypotheses (Session 21)

| ID | Hypothesis | Status |
|----|------------|--------|
| **H46** | Effort limit 2.0 Nm insufficient for walking | **CONFIRMED** |
| **H47** | Increasing effort_limit enables extended poses | **PARTIALLY** - helps but stiffness was key |
| **H48** | Scripted gait can prove physics work (open-loop) | **CONFIRMED** ✓ |
| **H49** | Stiffness=200 is too low for implicit actuators | **CONFIRMED** ✓ |
| **H50** | Stiffness=1200 enables realistic servo tracking | **CONFIRMED** ✓ |

---

## 2025-12-26: Session 22 - Sim-to-Real Alignment SUCCESS

### Real Robot Walking Forward

**Major Milestone**: Real hardware now walks forward with scripted gait!

The robot walks in a stable, controlled manner. Feet are dragging slightly but this is expected for early gait development.

### Key Discovery: Simulation Was Too Stiff

When comparing sim to real robot, the simulation appeared much more "rigid and responsive" while the real robot had more "give".

**Root Cause**: Stiffness=1200 created unrealistically stiff servo response.

**Solution**: Reduced stiffness from 1200 → 400

| Setting | Before | After |
|---------|--------|-------|
| stiffness | 1200 | 400 |
| damping | 50 | 40 |
| frequency | 1.0 Hz | 0.5 Hz |

### Key Discovery: Thigh Phase Was Inverted

Both sim and real robot were walking BACKWARD initially.

**Root Cause**: Hardware trajectory used `+sin` for thigh, but needed `-sin`.

**Analysis**: The thigh motion was 90° out of phase with foot contact. The thigh was moving forward during ground contact, pushing the robot backward.

**Solution**: Changed thigh trajectory from `+sin` to `-sin`:
```cpp
// OLD (walks backward)
*thigh = thigh_mid + thigh_amp * sin_phase;

// NEW (walks forward)
*thigh = thigh_mid - thigh_amp * sin_phase;
```

### Key Discovery: Floor Surface Matters Significantly

Real robot gait varies by floor type:
- **Hardwood**: Lower friction, cleaner steps
- **Short carpet**: Medium friction
- **Long carpet**: Higher friction, more foot dragging

**Implication**: Must add friction domain randomization to RL training.

### Updated Hypotheses (Session 22)

| ID | Hypothesis | Status |
|----|------------|--------|
| **H51** | Stiffness=1200 is too rigid for real servo | **CONFIRMED** ✓ |
| **H52** | Stiffness=400 matches real servo softness | **CONFIRMED** ✓ |
| **H53** | Thigh phase must be -sin for forward walking | **CONFIRMED** ✓ |
| **H54** | Floor friction needs domain randomization | **OBSERVED** - to test |
| **H55** | Feet dragging can be fixed with swing phase tuning | TO TEST |

### Sign Convention Clarified

**Simulation → Hardware conversion for thighs/calves:**
```
hardware_degrees = -sim_radians × (180/π)
```

This is because thighs and calves have `joint_sign = -1.0` in the hardware mapping.

### Aligned Parameters (Final)

**Simulation (ScriptedGaitCfg):**
```python
frequency: 0.5 Hz
swing_thigh: 0.40 rad
stance_thigh: 0.90 rad
stance_calf: -0.90 rad
swing_calf: -1.40 rad  # Matched to HW CALF_MAX=80°
```

**Hardware (scripted_gait_test_1.ino):**
```cpp
GAIT_FREQUENCY = 0.5f;
BASE_STANCE_THIGH = -51.6f;  // sim: +0.90 rad
BASE_SWING_THIGH = -22.9f;   // sim: +0.40 rad
BASE_STANCE_CALF = 51.6f;    // sim: -0.90 rad
BASE_SWING_CALF = 80.0f;     // sim: -1.40 rad
// Uses -sin for thigh trajectory
```

### Lessons Learned

1. **Sim-to-real requires matching servo response, not just joint limits**
2. **Simple sinusoid approximation requires careful phase alignment**
3. **Real-world surface variation is significant - domain randomization needed**
4. **Iterative testing between sim and hardware is essential**

---

## 2025-12-27: Session 23 - The 0.051 Barrier

### Discovery: Current Configuration Has Reached Its Limit

After 17+ experiments systematically testing all parameters, vx=0.051 m/s is the consistent ceiling.

### Optimal Configuration Locked
```python
stiffness = 600   # Too low (400) = can't stand; too high (800+) = regression
damping = 45
iterations = 500  # Shorter = undertrained; longer = regression
num_envs = 8192   # More = worse learning
learning_rate = 5e-4  # Lower = too slow
```

### Why 0.051 is the Ceiling

1. **PPO Regression Pattern**: Peak vx at 30-50% training, then regression
2. **Reward Structure Local Optimum**: Standing + slight drift maximizes reward
3. **Trade-offs are Zero-Sum**: Higher forward reward → less stability → worse overall

### Parameter Sensitivity Analysis

| Parameter | Direction | Effect |
|-----------|-----------|--------|
| stiffness ↑ | Better height, worse vx |
| stiffness ↓ | Worse height, marginally better vx |
| iterations ↑ | More regression |
| iterations ↓ | Undertraining |
| forward_pos ↑ | Destabilizes learning |
| height_reward ↓ | Robot crouches |
| gait_reward ↑ | SANITY_FAIL |
| action_scale ↑ | Body contact issues |
| num_envs ↑ | Overfits faster |
| learning_rate ↓ | Too slow |

### Implications

To achieve vx > 0.1 m/s, need fundamentally different approach:
1. **Algorithm**: SAC/TD3 instead of PPO
2. **Learning**: Curriculum (progressive difficulty)
3. **Imitation**: Reference motion tracking
4. **Architecture**: LSTM for gait memory

### Stiffness-Learning Trade-off

Session 23 revealed a fundamental trade-off:
- **Sim-to-real fidelity** (stiffness=400): Real robot behavior, but hard to learn
- **RL learning ability** (stiffness=1200): Easy to learn, but doesn't match real robot
- **Sweet spot** (stiffness=600): Best of both - learnable AND somewhat realistic

### Updated Hypotheses

| ID | Hypothesis | Status |
|----|------------|--------|
| H56 | Stiffness=600 is optimal for learning | **CONFIRMED** |
| H57 | 500 iterations prevents regression | **CONFIRMED** |
| H58 | Parameter tuning has reached limits | **CONFIRMED** |
| H59 | vx=0.051 is PPO ceiling with current rewards | **CONFIRMED** |
| H60 | Need different algorithm to break barrier | TO TEST |

---

## 2025-12-27: Session 24 - CPG Residual Learning Breakthrough

### CRITICAL INSIGHT: Don't Abandon Sim-to-Real Alignment

Session 23 optimized for RL metrics (vx=0.051) but used stiffness=600, which doesn't match the real robot. Session 24 corrected this by restoring stiffness=400 and using CPG residual learning.

### Discovery: Policy Can Reverse CPG Gait

With residual_scale=0.15 (EXP-126):
- CPG trajectory pushes forward (+0.014 m/s scripted alone)
- Policy learned corrections that REVERSED the gait
- Result: vx=-0.018 m/s (backward!)

**Cause**: Policy had enough authority to override the base trajectory.

**Solution**: Reduce residual_scale to 0.05 (EXP-127)
- Policy can only make small balance corrections
- Cannot override or reverse the gait
- Result: vx=+0.012 m/s, stable walking in video at step 9600

### CPG Alignment Critical

The original CPGCfg used DIFFERENT trajectory generation than ScriptedGaitCfg:

| Aspect | CPGCfg (old) | ScriptedGaitCfg (proven) |
|--------|--------------|--------------------------|
| Frequency | 1.0 Hz | 0.5 Hz |
| Method | Sinusoidal oscillation | Duty-cycle stance/swing |
| Amplitude | Around athletic pose | Explicit angles |

**Fix**: Updated CPGCfg to exactly match ScriptedGaitCfg parameters and use the same trajectory computation method (`_compute_leg_trajectory()`).

### residual_scale is Critical

| Scale | Result | Analysis |
|-------|--------|----------|
| 0.20 | Not tested | Original default |
| 0.15 | vx=-0.018 | Policy reversed gait |
| 0.05 | vx=+0.012, WALKING | Sweet spot found |

### Outdated Comments Corrected

Found and fixed multiple outdated comments claiming scripted gait "FAILED":
- ScriptedGaitCfg docstring said "Phase 1 FAILED"
- Warning message said "open-loop control doesn't work"

**Reality**: Session 21 achieved vx=+0.141 m/s (141% of target), Session 22 real robot walks forward.

### Lessons Learned

1. **Sim-to-real alignment trumps RL metrics**: Better to have a working sim-to-real transfer than a higher vx in sim only
2. **CPG provides structure, policy provides corrections**: Don't let policy override the proven gait
3. **Very low residual_scale (0.05) works**: Policy can still learn useful balance corrections
4. **Verify base trajectory independently**: Test scripted gait alone before adding policy
5. **Fix outdated documentation**: Wrong comments led to ignoring working solutions

### Configuration That Works

```python
# Actuator (sim-to-real aligned)
stiffness = 400.0
damping = 30.0

# CPG (aligned with proven ScriptedGaitCfg)
base_frequency = 0.5  # Hz
duty_cycle = 0.6
swing_thigh = 0.40
stance_thigh = 0.90
stance_calf = -0.90
swing_calf = -1.40
residual_scale = 0.05  # Critical!
```

### Next Experiments to Try

1. residual_scale=0.08 to give slightly more correction authority
2. Increase height_reward to improve standing height
3. Deploy to real hardware (parameters are aligned!)

---

## 2025-12-27: Session 24 Continued - RPi 5 Deployment Complete

### Deployment Pipeline Created

Replaced Jetson Nano with Raspberry Pi 5 (2GB RAM). Created complete inference pipeline:

```
deployment/
├── policy/harold_policy.onnx      # 753KB, 50D→12D
├── inference/
│   ├── harold_controller.py       # Main 20 Hz control loop
│   ├── cpg_generator.py           # CPG trajectory (matches sim exactly)
│   ├── observation_builder.py     # IMU + servo → 50D obs
│   └── action_converter.py        # Policy → servo commands
├── drivers/
│   ├── imu_reader_rpi5.py         # MPU6050 via smbus2
│   └── esp32_serial.py            # USB serial 115200
└── config/
    ├── cpg.yaml                   # CPG params
    └── hardware.yaml              # Servo IDs, signs, limits
```

### Key Technical Decisions

1. **USB Serial vs SPI**: Chose USB Serial for simplicity (ESP32 firmware already supports it)
2. **CPG on RPi 5**: Matches simulation exactly, avoids firmware complexity
3. **smbus2 for I2C**: RPi 5 compatible (smbus has issues on newer kernels)

### Legacy Cleanup

Deleted deprecated deployment artifacts:
- `deployment_artifacts/terrain_62/` - Old 48D policy
- `deployment_artifacts/terrain_64_2/` - Incomplete

Updated OBS_DIM from 48→50 in export scripts (gait phase sin/cos added in Session 24).

### Observation Space Layout (50D)

| Index | Component | Source |
|-------|-----------|--------|
| 0-2 | root_lin_vel_b | IMU accel integration |
| 3-5 | root_ang_vel_b | IMU gyro |
| 6-8 | projected_gravity | IMU accel (normalized) |
| 9-20 | joint_pos - default | ESP32 telemetry |
| 21-32 | joint_vel | Differentiate positions |
| 33-35 | commands | Fixed [0.1, 0, 0] |
| 36-47 | prev_target_delta | Previous output |
| 48-49 | gait_phase sin/cos | CPG phase |

### Tests Passing

- CPG Generator: Phase tracking, trajectory computation ✓
- Action Converter: Residual scaling, safety limits ✓
- ONNX policy: Skipped (onnxruntime not in Isaac Lab env, will install on RPi 5)

### Ready for Hardware

Next step is physical deployment:
1. Copy `deployment/` to RPi 5
2. Install dependencies: `pip install -r requirements.txt`
3. Connect ESP32 via USB, IMU via I2C
4. Run `python inference/harold_controller.py`

---

## 2025-12-28: Session 26 - Dynamic Command Tracking Optimized

### Key Discovery: zero_velocity_prob=0 is Critical

When 5% of commands were "stop" (vx=0), the policy learned to stop instead of walk. Removing stop commands entirely was the breakthrough that enabled WALKING verdict with dynamic commands.

**Evidence:**
- EXP-134-138 (zero_prob=5%): vx=0.003-0.010, all STANDING
- EXP-139 (zero_prob=0%): vx=0.011, **WALKING**

### Command Tracking Parameter Sweet Spots

| Parameter | Sweet Spot | Too Low | Too High |
|-----------|------------|---------|----------|
| tracking_weight | 10.0 | Not tested | 20.0 → over-optimization (vx=0.005) |
| tracking_sigma | 0.1 | Not tested | 0.15 → too permissive (vx=0.003) |
| vx_range | 0.10-0.45 | 0.15-0.40 → vx=0.011 | 0.05-0.50 → vx=0.012 |
| command_interval | 10.0s | 5.0s → vx=0.007 | Not tested |

### Why Higher Tracking Weight Failed

tracking_weight=20 caused the policy to over-optimize for matching the commanded velocity at the expense of actually walking. The robot learned to stand still (perfect velocity match at vx=0 during zero commands) rather than develop walking behavior.

### Why Higher Sigma Failed

sigma=0.15 was too permissive - the reward was high even with significant velocity error. The robot learned it could get high reward by standing tall (height=1.53) without walking (vx=0.003).

### Results Reproducibility

The optimal configuration (vx_range=0.10-0.45, zero_prob=0) achieved vx=0.015 in three separate runs:
- EXP-140: vx=0.015
- EXP-142: vx=0.015
- EXP-143 (2500 iters): vx=0.016

### Updated Hypotheses (Session 26)

| ID | Hypothesis | Status |
|----|------------|--------|
| H56 | zero_velocity_prob=0 enables dynamic commands | **CONFIRMED** |
| H57 | Optimal vx range is 0.10-0.45 | **CONFIRMED** |
| H58 | tracking_weight=10 is sweet spot | **CONFIRMED** |
| H59 | tracking_sigma=0.1 is sweet spot | **CONFIRMED** |
| H60 | Lateral (vy) commands need vy tracking reward | TO TEST |

### Implementation Notes for Future Work

To add lateral (vy) commands:
1. Add vy tracking to command_tracking reward in `_compute_rewards()`
2. Modify lat_vel_penalty to penalize (vy - cmd_vy)² instead of vy²
3. Start with small vy_range (±0.10) and expand gradually

To add yaw rate commands:
1. Add yaw tracking to command_tracking reward
2. Modify yaw_rate_penalty similarly
3. Start with small range

---

## 2025-12-29: Session 27 - Servo Tuning & Pi 5 Validation

### Pi 5 Power Supply: VALIDATED

Stress tested the onboard Pi 5 power supply:
- CPU-only 4-core stress: 71.4°C peak, throttled=0x0
- CPU+Memory+I/O stress: 75.2°C peak, throttled=0x0
- Robot controller simulation (20Hz): 60.9°C, throttled=0x0

**Conclusion**: Power supply is healthy. Safe to run 100% servo torque.

### Servo Dead Zone Issue

**Problem**: User reported large "dead areas" where servo can be rotated before responding.

**Root Cause**: Registers 26-27 (`SMS_STS_CW_DEAD`, `SMS_STS_CCW_DEAD`) control deadband.
Factory default is typically 3-10 encoder steps (~0.26°-0.88°).

**Solution**: Created `ServoTuning.ino` utility to adjust dead zones.
- `a 2` sets dead zone to 2 for all servos (~0.18°)
- `a 0` sets to minimum (may cause buzzing)

### Servo Position Drift Issue

**Problem**: Servos "slip" or lose their calibrated center over time.

**Possible Causes** (in order of likelihood):
1. **Torque overload** - Gears skipping when load exceeds holding torque (mechanical)
2. **Magnetic interference** - IMU/motors affecting magnetic encoder
3. **Calibration not persisting** - EEPROM write issues
4. **Voltage brownouts** - Unlikely given Pi stress test passed

**Diagnostic Approach**:
1. Use `m` command in ServoTuning to monitor positions
2. Manually disturb robot - if positions jump without commands = mechanical/magnetic issue
3. Check IMU proximity to servos

### ST3215 Register Reference

| Register | Name | Purpose |
|----------|------|---------|
| 26 | CW_DEAD | Clockwise dead zone (steps) |
| 27 | CCW_DEAD | Counter-clockwise dead zone (steps) |
| 31-32 | OFS_L/H | Calibration offset (EEPROM) |
| 48-49 | TORQUE_LIMIT_L/H | Torque limit 0-1000 (RAM) |
| 55 | LOCK | EPROM write protection |

### New Artifact: ServoTuning.ino

Location: `firmware/CalibrationAndSetup/ServoTuning/ServoTuning.ino`

Commands:
- `s` - Scan all servos (shows dead zone, torque, offset)
- `r <id>` - Read detailed parameters
- `d <id> <v>` - Set dead zone (0-10)
- `a <v>` - Set dead zone for ALL servos
- `t <id> <v>` - Set torque limit (0-1000)
- `c <id>` - Calibrate center position
- `m` - Monitor all positions in real-time
- `x <id> <reg>` - Read raw register value
- `X <id>` - Dump all registers (0-70)
- `W <id> <reg> <val>` - Write raw register value

### CRITICAL FINDING: "Dead Zone" is Gear Backlash

**Discovery**: The servo "dead zone" is NOT software-controllable. Dead zone registers (26-27) were already at minimum (1 step = 0.088°).

**Root Cause**: Mechanical gear backlash in ST3215 gearbox (~1-3° of play).

**Behavior**:
- Servo is commanded to position X
- Can physically move it within [X - backlash, X + backlash] without resistance
- Beyond that range, full servo resistance kicks in
- This is a "friction-free zone", not high friction

**P Coefficient Experiments**:
Discovered undocumented register 21 controls servo stiffness:

| P Value | Result |
|---------|--------|
| 32 (factory) | Normal response, noticeable play |
| 100 | Slightly stiffer |
| 200 | **Rapid oscillations** - unstable! |

**Conclusion**: Increasing P without matching D causes instability. Factory P=32 is optimal.

### Implication for Simulation

The gear backlash CANNOT be fixed in hardware. Must model in simulation for robust policies.

**Modeling approaches** (priority order):
1. **Observation noise**: Increase joint_position_noise to ~0.035 rad (~2°)
2. **Action noise**: Add noise to commanded positions
3. **Lower stiffness**: Make actuator softer (less accurate but similar effect)
4. **Custom backlash model**: Implement hysteresis in environment

**Priority Experiment**: Test observation noise approach first (simplest, already implemented in DomainRandomizationCfg).

---

## Updated Hypotheses (Session 27)

| ID | Hypothesis | Status |
|----|------------|--------|
| **H61** | Servo dead zone is software-controllable | **DISPROVEN** - registers already at minimum |
| **H62** | Dead zone is mechanical gear backlash | **CONFIRMED** |
| **H63** | Increasing P coefficient reduces backlash feel | **DISPROVEN** - causes oscillations |
| **H64** | Backlash must be modeled in simulation | **TO TEST** - priority experiment |
| **H65** | Observation noise can simulate backlash | **TO TEST** |

---

## Session 28 Key Observations (2025-12-29)

### Observation: Position Noise as Regularization

**Finding**: Adding 1° position noise to joint observations **improves** walking by 35%.

- **Mechanism**: Noise prevents policy from relying on precise position feedback
- **Result**: vx=0.023 (with noise) vs vx=0.017 (without noise)
- **Interpretation**: The policy learns more robust control strategies when uncertain

This is counterintuitive - adding noise typically hurts performance. But for sim-to-real, it acts as regularization that improves generalization.

### Observation: Optimal Noise Level

| Noise Level | Effect |
|-------------|--------|
| 0° | Baseline performance (vx=0.017) |
| 1° (0.0175 rad) | **OPTIMAL** - 35% improvement (vx=0.023) |
| 2° (0.035 rad) | Too much - regresses to standing (vx=0.007) |

### Observation: Multi-Objective Learning Challenges

**Finding**: Combining backlash noise + yaw tracking doesn't work (vx=0.003).

Each feature works independently:
- Backlash only: vx=0.023 (WALKING)
- Yaw only: vx=0.011 (WALKING)
- Combined: vx=0.003 (STANDING)

**Hypothesis**: The policy struggles to optimize multiple objectives simultaneously when also dealing with observation uncertainty. May need curriculum learning.

### New Hypotheses from Session 28

- **H-S28-1**: Curriculum learning (train backlash first, then add yaw) may work better
- **H-S28-2**: Lower yaw command range (±0.15 vs ±0.30) may help with combination
- **H-S28-3**: Position noise helps sim-to-real transfer (needs hardware validation)

### Observation: Curriculum Learning for Multi-Objective RL

**Finding**: When combining multiple features (backlash + yaw), curriculum learning outperforms from-scratch training.

| Approach | vx (m/s) | Verdict |
|----------|----------|---------|
| From scratch | 0.003 | STANDING |
| Curriculum | 0.015 | WALKING |

**Mechanism**: The policy struggles to optimize multiple objectives simultaneously. Training sequentially allows each skill to stabilize before adding complexity.

### Observation: Training Duration Trade-offs

**Finding**: For fine-tuning, shorter training (1250 iters) beats longer training (2500 iters).

- EXP-152 (1250 iters): vx=0.015 WALKING
- EXP-153 (2500 iters): vx=0.009 STANDING

**Hypothesis**: Extended fine-tuning causes the policy to "forget" the base skills while over-optimizing for the new objective.

### Observation: CPG + Residual Learning Architecture

The Harold walking gait is a combination of:
- **CPG (scripted)**: Provides timing, coordination, base trajectory
- **Policy (learned)**: Provides balance, velocity tracking, adaptation

The `residual_scale=0.05` limits policy authority to fine-tuning only. This prevents the policy from reversing or dramatically altering the proven CPG trajectory.

**Implication**: The vx=0.023 achievement comes from the policy learning to USE the CPG trajectory effectively, not from learning a gait from scratch.

---

## Session 29 Observations (2025-12-29)

### Observation: Action-Side Randomization Hurts Training

**Finding**: Adding noise to actions or control delays hurts training when observation noise is already present.

| EXP | Config | vx | Verdict |
|-----|--------|-----|---------|
| 154 | Action noise 0.5% | 0.007 | STANDING |
| 155 | Action noise 0.2% | 0.009 | STANDING |
| 156 | Action delay 0-1 steps | 0.008 | STANDING |

**Why Action Noise Hurts**:
- Corrupts the control signal the policy is trying to learn
- Combined with observation noise = too much uncertainty
- Policy can't determine cause-and-effect between actions and outcomes

**Why Observation Noise Works**:
- Forces policy to be robust to sensing uncertainty
- Still allows producing clean, correct actions
- Acts as regularization, not confusion

### Observation: Lin Vel Noise for Sim-to-Real

**Finding**: Hardware IMU computes linear velocity via accelerometer integration (noisy, drifts), while simulation uses perfect physics velocity.

**Solution Implemented**:
- `add_lin_vel_noise: True` (std=0.05 m/s)
- `lin_vel_bias_std: 0.02` (per-episode calibration drift)

**Training Result**: EXP-159 achieved vx=0.008 (similar to baseline) - neutral effect.

### Observation: Observation Clipping for Sim-to-Real

**Finding**: Deployment clips normalized observations to ±5.0, training had no clipping.

**Solution Implemented**:
- `clip_observations: True`
- `clip_observations_value: 5.0` (raw clip at ±50)

### Observation: Late-Training Regression Pattern

All domain randomization experiments showed the same pattern:
- Peak vx at 40-70% training (~0.013-0.015)
- Final regression to vx~0.008-0.009

This is a PPO training dynamics issue, not specific to domain randomization.

### New Hypotheses (Session 29)

| ID | Hypothesis | Status |
|----|------------|--------|
| **H-S29-1** | Action noise hurts when observation noise present | **CONFIRMED** |
| **H-S29-2** | Lin vel noise is neutral for training | **CONFIRMED** |
| **H-S29-3** | Observation clipping is neutral for training | **CONFIRMED** |
| **H-S29-4** | Joint limit alignment may require CPG retuning | TO TEST |

---

## Session 30 Observations (2025-12-30)

### Observation: Voltage Sag is NOT the Issue

**Initial Hypothesis**: Robot sluggishness was caused by power supply voltage sag under load.

**Testing**: Added voltage monitoring to firmware and ran gait tests.

**Result**: Voltage stays rock solid at 12.0-12.1V throughout all tests, even at 2.0 Hz gait.

**Conclusion**: Power supply is adequate. Sluggishness has other causes.

### Observation: Servo Load Scale Misinterpretation

**Bug Found**: Voltage monitor script showed "200% load" but this was a display bug.

**Root Cause**: Servo load register returns -1000 to +1000, where 1000 = 100%. Script was showing raw value as percentage.

**Fix**: Divide by 10 to get actual percentage.

**Actual Load**: 15-25% during normal gait - servos have massive headroom.

### Observation: Servo Speed Critical for Trajectory Tracking

**Problem**: Feet were dragging/sliding instead of lifting during swing phase.

**Root Cause**: `SERVO_SPEED=1600` (~140°/sec, 52% of max) was too slow for servos to reach swing positions.

**Fix**: Increased to `SERVO_SPEED=2800` (~246°/sec, 91% of max).

**Result**: Feet now lift properly during swing phase.

### Observation: Default Pose Mismatch Breaks Policy

**CRITICAL BUG**: Deployment used different default pose than simulation.

| Joint | Deployment (wrong) | Simulation (correct) |
|-------|-------------------|----------------------|
| Shoulders | 0.0 rad | ±0.20 rad (alternating) |
| Thighs | 0.3 rad | 0.70 rad |
| Calves | -0.75 rad | -1.40 rad |

**Effect**: Normalized observations hit ±5.0 clipping limits, causing policy to output garbage.

**Fix**: Updated observation_builder.py and action_converter.py to use simulation's athletic_pose.

### Observation: Joint Limits Convention Mismatch

**CRITICAL BUG**: Action converter applied hardware-convention limits to RL-convention values.

**Example**:
- Hardware thigh limit: -55° to +5° (after sign conversion)
- CPG thigh output: 0.40 to 0.90 rad (positive, RL convention)
- Result: Thighs clamped to +0.087 rad → legs nearly straight!

**Fix**: Disabled limits in action_converter (firmware already handles limits correctly after sign conversion).

### Observation: CPG Trajectory Has Velocity Discontinuity

**Finding**: Calf trajectory has acceleration discontinuity at phase=0.60 (stance→swing transition).

**Cause**: Calf is constant during stance, then suddenly starts moving with sine profile during swing.

**Effect**: Visible "hitch" in gait motion.

**Potential Fix**: Smooth the calf trajectory with continuous function (not yet implemented).

### New Hypotheses (Session 30)

| ID | Hypothesis | Status |
|----|------------|--------|
| **H-S30-1** | Servo speed limits trajectory tracking | **CONFIRMED** |
| **H-S30-2** | Default pose must match simulation exactly | **CONFIRMED** |
| **H-S30-3** | Joint limits must use correct convention | **CONFIRMED** |
| **H-S30-4** | CPG calf discontinuity causes visible hitch | **OBSERVED** |
| **H-S30-5** | Policy + CPG works after deployment fixes | **CONFIRMED** |

---

## Session 30 Observations (2025-12-30)

### Observation: Joint Limit Alignment is Compatible

**Finding**: Hardware safe limits are more restrictive than simulation defaults, but the CPG gait works within them.

Sign inversion matters:
- Thighs/calves have inverted signs between sim and hardware
- Hardware [-55°, +5°] becomes simulation [-5°, +55°]
- All CPG parameters stay within aligned limits

### Observation: External Perturbations FAIL

**Finding**: Even very light forces (0.2-0.5N, 0.5% probability) cause falling and backward drift.

This suggests the CPG + residual policy relies on smooth, predictable dynamics. Random forces disrupt the learned balance behaviors.

**Implication**: Hardware robustness must come from:
- Domain randomization (friction, mass, etc.)
- NOT from random force perturbations

### Observation: CPG Frequency Has Non-Monotonic Optimum

**Finding**: 0.7 Hz is better than both 0.5 Hz and 0.8 Hz.

| Freq | vx | Verdict |
|------|-----|---------|
| 0.5 Hz | 0.011-0.017 | WALKING |
| 0.6 Hz | 0.010 | STANDING |
| **0.7 Hz** | **0.016** | **WALKING** |
| 0.8 Hz | 0.010 | STANDING |

**Hypothesis**: There's a timing resonance between:
- CPG cycle duration
- Policy reaction time (20 Hz = 50ms)
- Physical dynamics (leg swing, balance)

### Observation: residual_scale is Sensitive

**Finding**: 0.05 works, 0.08 causes regression (vx 0.016 → 0.007).

**Mechanism**: Higher residual_scale gives policy more authority. But the CPG trajectory is carefully tuned - letting the policy override it disrupts the gait.

The policy should only provide:
- Balance corrections
- Velocity tracking adjustments
- Adaptation to uncertainty

NOT gait pattern changes.

### New Hypotheses from Session 30

| ID | Hypothesis | Status |
|----|------------|--------|
| **H-S30-1** | 0.7 Hz is optimal CPG frequency | **CONFIRMED** |
| **H-S30-2** | External perturbations are incompatible with CPG learning | **CONFIRMED** |
| **H-S30-3** | residual_scale > 0.05 allows policy to disrupt gait | **CONFIRMED** |
| **H-S30-4** | Joint limit alignment doesn't require CPG retuning | **CONFIRMED** |

### Best Configuration (Session 30)

```python
# CPG
base_frequency: 0.7 Hz  # Optimal
swing_calf: -1.35       # Safety margin
residual_scale: 0.05    # Conservative

# Domain Randomization
joint_position_noise: 0.0175 rad  # Backlash simulation
add_lin_vel_noise: True           # IMU drift
clip_observations: True           # Match deployment
apply_external_forces: False      # Causes failing
```

---

## Session 32 Observations (2025-12-30)

### CRITICAL BUG: Double-Normalization Discovered

**Finding**: The deployment code was applying observation normalization TWICE:
1. Manual normalization in `harold_controller.py` (line 276)
2. Internal normalization in ONNX model (baked in by `NormalizedPolicy` wrapper)

**Root Cause**: The ONNX export script (`policy/export_policy.py`) wraps the policy with `NormalizedPolicy`:
```python
class NormalizedPolicy(torch.nn.Module):
    def forward(self, obs: torch.Tensor):
        norm_obs = (obs - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        mean, value, log_std = self.base(norm_obs)
        return mean, value, log_std
```

But the deployment code ALSO normalized before calling ONNX:
```python
obs_norm = normalize_observation(obs, self.running_mean, self.running_var)
outputs = self.policy.run(['mean'], {'obs': obs_norm...})  # DOUBLE NORMALIZATION!
```

**Effect**: For an observation at the training mean:
1. First normalization: `(obs - mean) / std = 0`
2. Second normalization: `(0 - mean) / std = -mean/std` (potentially large!)

This caused extreme policy outputs and was likely the root cause of the "unstable behavior" that Session 31 tried to fix with blending and stat overrides.

**Fix**: Remove manual normalization, pass raw observations to ONNX:
```python
outputs = self.policy.run(['mean'], {'obs': obs.reshape(1, -1)...})  # ONNX normalizes internally
```

### Observation: ONNX Includes Normalization by Design

The export script's choice to include normalization in ONNX is actually good design:
- Deployment is simpler (no need to track running stats separately)
- Consistent behavior between training and inference
- No risk of using wrong stats

**Key Lesson**: When using an ONNX model, always verify whether preprocessing is baked in.

### Observation: Session 31 Fixes Were Compensating

The Session 31 fixes (lin_vel override, prev_target blending) were likely compensating for double-normalization:
- Overriding stats tried to fix extreme normalized values
- Blending kept values closer to 0

With the actual bug fixed, these compensations may no longer be necessary.

### New Hypotheses (Session 32)

| ID | Hypothesis | Status |
|----|------------|--------|
| **H-S32-1** | Double normalization caused unstable outputs | **CONFIRMED** |
| **H-S32-2** | ONNX + raw observations = correct behavior | **CONFIRMED** (validated on desktop) |
| **H-S32-3** | Session 31 blending compensated for double-norm | **TO VERIFY** on hardware |
| **H-S32-4** | With fix, blending may not be needed | **TO TEST** on hardware |

### Validated: ONNX Matches PyTorch

Ran quick validation comparing ONNX vs PyTorch outputs:
```
Max difference: 0.000003
Mean difference: 0.000000
✓ PASS: ONNX outputs match PyTorch outputs!
```

The ONNX export is correct; the deployment code was using it wrong.
