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
| `height_reward` | > 2.0 | At correct standing height |
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
