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
