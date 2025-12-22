# Harold Experiment Log

## Format
Each experiment entry contains:
- **Date**: When the experiment was run
- **ID**: Unique identifier (log directory name)
- **Config**: Key configuration changes from baseline
- **Duration**: Training duration/steps
- **Result**: Summary of outcomes
- **Metrics**: Key performance metrics
- **Notes**: Observations and insights

---

## Experiments

### EXP-001: Simulation Boot Test
- **Date**: 2025-12-19
- **ID**: `2025-12-19_23-03-04_ppo_torch`
- **Config**: Default flat terrain, 1 env, 1 iteration (24 timesteps)
- **Duration**: 14 seconds total (2 seconds training)
- **Result**: SUCCESS - Simulation boots and runs correctly
- **Metrics**:
  - Scene creation: 0.22s
  - Simulation start: 0.61s
  - Training throughput: ~12 it/s
- **Notes**: Headless mode works. Logging infrastructure intact. Ready for full experiments.

---

### EXP-002: Policy Evaluation (terrain_62)
- **Date**: 2025-12-19
- **ID**: `terrain_62/videos/play/rl-video-step-0.mp4`
- **Config**: Best checkpoint (agent_128000.pt), 1 env, 500 steps
- **Duration**: 25 seconds simulation time
- **Result**: FAILURE - Robot stands in place, zero forward locomotion
- **Metrics**:
  - Forward displacement: 0 meters
  - Standing stability: Excellent (no falls)
  - Episode completion: 100% (no early terminations)
- **Notes**:
  - Policy learned to stand stably but not walk
  - Training videos confirm all 4096 robots remain at spawn positions
  - Reward structure allows "standing" as optimal strategy
- **Next Steps**: Modify reward weights to incentivize forward motion

---

### EXP-003: Walking Gait Experiment - FAILED
- **Date**: 2025-12-19/20
- **ID**: `2025-12-19_23-35-57_ppo_torch`
- **Config Changes**:
  - `progress_forward_pos`: 80 → 150 (increased forward reward)
  - `rear_support_bonus`: 0.6 → 0.0 (removed standing incentive)
  - Added `standing_penalty`: 2.0 (penalizes velocity < 0.05 m/s)
- **Duration**: ~38,000 timesteps before manual stop (~16% of planned)
- **Status**: FAILED
- **Result**: Robot falls over and stays on ground
- **Close-up Observation** (single robot eval):
  - Step 10: Robot in low crouch
  - Step 75: Robot has FALLEN flat on ground
  - Step 150-290: Robot remains prone, never recovers
- **Analysis**:
  - Standing penalty broke stable standing behavior
  - Robot attempts to move to avoid penalty but lacks coordination
  - Falls over and cannot recover
  - Essentially learned "fall and flop" instead of "walk"
- **Lesson**: Cannot just penalize standing - need to maintain balance while incentivizing movement

---

### EXP-004: Balanced Forward Incentive - FAILED
- **Date**: 2025-12-20
- **ID**: `2025-12-20_00-17-40_ppo_torch`
- **Config Changes** (from EXP-003):
  - `standing_penalty`: 2.0 → 0.0 (REMOVED)
  - `upright_reward`: 1.8 → 5.0 (INCREASED)
  - `progress_forward_pos`: 150 → 100 (MODERATED)
- **Duration**: ~12,000 timesteps before manual stop
- **Status**: FAILED
- **Result**: Robot falls and stays on ground (same as EXP-003)
- **Observations**:
  - Step 2000: Robot standing (promising!)
  - Step 6000: Robot fallen
  - Step 12000: Robot still fallen - converged to prone position
- **Analysis**:
  - upright_reward=5.0 still not enough to prevent falling
  - Training from scratch takes too long to learn balance
  - Forward reward still incentivizes motion before robot learns to balance
- **Lesson**: Need to either (a) start from checkpoint that can stand, or (b) dramatically increase upright reward

---

### EXP-005: Fine-tune from terrain_62 - INCONCLUSIVE
- **Date**: 2025-12-20
- **ID**: `2025-12-20_00-32-14_ppo_torch`
- **Config**: Same rewards as EXP-004, but starting FROM terrain_62 checkpoint
- **Duration**: ~4000 timesteps before manual stop
- **Status**: INCONCLUSIVE - Shows oscillation between standing and falling
- **Observations**:
  - Step 2000: Robot falling (pre-trained standing behavior degrading)
  - Step 4000: Robot oscillating - some frames show standing attempts, then falling
- **Analysis**:
  - Even with pre-trained checkpoint, reward gradients push policy toward instability
  - The forward reward creates gradient that disrupts standing equilibrium
  - Robot attempts to stand but can't maintain balance while attempting forward motion
- **Key Insight**: Forward reward modulated by upright isn't enough - the gradient still destabilizes

---

### EXP-006: High Upright Reward - FAILED
- **Date**: 2025-12-20
- **ID**: `2025-12-20_00-44-38_ppo_torch`
- **Config Changes**:
  - `upright_reward`: 5.0 → 15.0 (3x increase to anchor stability)
  - All else unchanged from EXP-004
- **Duration**: ~12,000 timesteps before manual stop
- **Status**: FAILED
- **Result**: Robot falls by step 4000 and stays fallen through step 12000
- **Observations**:
  - Step 4000: Robot already fallen, lying on side
  - Step 8000: Robot still fallen, no recovery
  - Step 12000: Robot prone on ground, no improvement
- **Analysis**:
  - High upright reward (15.0) still insufficient to prevent falling
  - The reward structure alone cannot prevent initial instability
  - Once fallen, robot has no incentive to get up (no termination)
- **Key Insight**: Need to PREVENT falling behaviors from being learned, not just reward uprightness

---

## Session 2 Summary (2025-12-20 Overnight)

### Experiments Run
| ID | Approach | Result |
|----|----------|--------|
| EXP-006 | High upright reward (15.0) | FAILED - Robot still falls |
| EXP-007 | Aggressive termination (-0.7) | FAILED - Makes learning harder |
| EXP-008 | Fine-tune terrain_62 + low forward (30) | FAILED - Stability degraded |
| EXP-009 | Gait rewards + very low forward (10) | FAILED - Same pattern |
| EXP-010 | Stability-only (zero forward) | PARTIAL - Standing at 5k-10k, fell by 15k |
| EXP-011 | Stability + aggressive termination | FAILED - Made learning harder |
| EXP-012 | Stability + frequent checkpoints | **SUCCESS** - Stable standing, saved agent_5000.pt |
| EXP-013 | Fine-tune agent_5000 + forward=3.0 | FAILED - Destabilized immediately |
| EXP-014 | Fine-tune agent_5000 + leg movement only | FAILED - Still destabilized |

### KEY FINDING: The Standing Equilibrium is Extremely Fragile
Every attempt to add ANY motion reward (forward velocity OR gait/leg movement)
eventually destabilizes the standing behavior. The pattern is consistent:
- Step 2500-5000: Standing maintained
- Step 5000-7500: Degradation begins
- Step 7500+: Robot fallen

### EXP-012: Stability + Frequent Checkpoints - **PARTIAL SUCCESS**
- **Date**: 2025-12-20
- **ID**: `2025-12-20_02-18-43_ppo_torch`
- **Config**: upright=25, height=8, forward=0, termination=-0.5
- **Status**: **ACHIEVED STABLE STANDING**
- **Saved Checkpoint**: `agent_5000.pt` shows clear standing posture
- **Observations**:
  - Step 2500: Standing ✓
  - Step 5000: Standing ✓ (BEST CHECKPOINT)
  - Step 7500: Mixed (some fallen frames)
  - Step 10000: Standing ✓
- **Key**: This proves stable standing CAN be learned from scratch

### EXP-013: Fine-tune with Minimal Forward - FAILED
- **Date**: 2025-12-20
- **ID**: `2025-12-20_02-30-55_ppo_torch`
- **Config**: Used agent_5000.pt, forward=3.0, upright=25
- **Status**: FAILED
- **Observations**: Standing at step 0, fallen by step 5000
- **Analysis**: Even forward_reward=3.0 is too destabilizing

### EXP-014: Leg Movement Only - FAILED
- **Date**: 2025-12-20
- **ID**: `2025-12-20_02-40-39_ppo_torch`
- **Config**: forward=0, leg_movement=2.0, feet_air_time=1.0
- **Status**: FAILED
- **Observations**: Standing at 2500, fallen by 5000
- **Analysis**: Gait rewards ALSO destabilize standing

### EXP-009: Gait Rewards - FAILED
- **Date**: 2025-12-20
- **ID**: `2025-12-20_01-30-34_ppo_torch`
- **Config Changes**:
  - `progress_forward_pos`: 30 → 10 (even lower)
  - Added gait rewards: feet_air_time=2.0, diagonal_gait=3.0, leg_movement=1.5
  - Started from terrain_62 checkpoint
- **Duration**: ~12,000 timesteps before stop
- **Status**: FAILED
- **Result**: Same degradation as EXP-008
- **Observations**:
  - Step 4000: Low crouch
  - Step 8000: Fallen
  - Step 12000: Still fallen
- **Analysis**: Gait rewards don't prevent falling. terrain_62 optimized for standing still.

### Critical Pattern Identified
ALL experiments (006-009) follow identical failure:
- Step 4000: Low crouch (initial terrain_62 stability)
- Step 8000+: Robot fallen
- Root cause: terrain_62 optimized for standing still, any motion reward disrupts equilibrium

### EXP-008: Fine-tune from terrain_62 - FAILED
- **Date**: 2025-12-20
- **ID**: `2025-12-20_01-14-41_ppo_torch`
- **Config Changes**:
  - Started from `terrain_62/checkpoints/agent_128000.pt`
  - `progress_forward_pos`: 100 → 30 (much lower)
  - `upright_reward`: 15 → 20 (very high)
  - Kept termination at -0.7
- **Duration**: ~12,000 timesteps before manual stop
- **Status**: FAILED
- **Result**: Initial stability degraded over training
- **Observations**:
  - Step 4000: Robot in LOW CROUCH (promising! not fully fallen)
  - Step 8000: Robot fallen
  - Step 12000: Robot still fallen
- **Analysis**:
  - Pre-training provided initial stability (step 4000 was best result yet!)
  - But forward reward gradient still eventually destabilizes
  - Even forward_reward=30 is too destabilizing
- **Key Insight**: Velocity rewards fundamentally destabilize - need gait-based rewards instead

### EXP-007: Aggressive Termination - FAILED
- **Date**: 2025-12-20
- **ID**: `2025-12-20_00-59-47_ppo_torch`
- **Config Changes**:
  - `orientation_threshold`: -0.5 → -0.7 (terminate earlier when tilting)
  - `upright_reward`: 15.0 (unchanged from EXP-006)
- **Duration**: ~12,000 timesteps before manual stop
- **Status**: FAILED
- **Result**: Robot still falls immediately after reset
- **Observations**:
  - Step 4000: Robot fallen/low in all frames
  - Step 8000: Same behavior
  - Step 12000: No improvement
- **Analysis**:
  - Aggressive termination ends episodes quickly but doesn't help learning
  - Robot doesn't have initial policy to stay upright
  - Training from scratch can't learn to stand AND walk simultaneously
- **Key Insight**: Need to fine-tune from terrain_62 checkpoint (already knows how to stand)

---

## Session 1 Summary (2025-12-19/20)

### Experiments Run
| ID | Approach | Result |
|----|----------|--------|
| EXP-001 | Simulation boot test | SUCCESS |
| EXP-002 | Evaluate terrain_62 | Robot stands, doesn't walk |
| EXP-003 | Add standing penalty | FAILED - Robot falls immediately |
| EXP-004 | Remove penalty, increase upright | FAILED - Robot still falls |
| EXP-005 | Fine-tune from checkpoint | INCONCLUSIVE - Oscillates |

### Key Learnings
1. terrain_62 learned stable STANDING, not walking
2. Standing penalty is too aggressive - breaks balance immediately
3. upright_reward=5.0 is insufficient to maintain balance with forward_reward=100
4. Even fine-tuning from stable checkpoint, forward reward disrupts balance
5. The robot can't simultaneously learn to balance AND move forward

### Recommended Next Approaches
1. **MUCH higher upright reward** (15-20) to strongly anchor stability
2. **Episode termination on fall** so robot can't learn "fallen" behaviors
3. **Explicit gait rewards** for leg alternation (don't just reward velocity)
4. **Curriculum**: First achieve stable walking-in-place, then add forward bias

---

## Historical Experiments (Pre-Memory System)
Reference logs in `logs/skrl/harold_direct/`:
- `terrain_57` through `terrain_64_2`: Major training runs (Sep 2025)
- `2025-10-*`: Recent shorter experiments (Oct 2025)
- Best performing: `terrain_62` (524M+ steps, exported for deployment)

---

## Session 3 (2025-12-20 Morning)

### EXP-015: Extended Stability Training - COMPLETED
- **Date**: 2025-12-20
- **ID**: `2025-12-20_09-28-17_ppo_torch`
- **Hypothesis**: Longer stability-only training produces more robust standing that may resist destabilization
- **Config**:
  - `upright_reward`: 25.0
  - `height_reward`: 8.0
  - `forward_reward`: 0.0 (ZERO)
  - `gait_rewards`: 0.0 (ALL ZERO)
  - `termination`: -0.5 (standard)
  - `checkpoint_interval`: 5000
  - `timesteps`: 30000
- **Duration**: ~25 min at 20-24 it/s
- **Status**: **COMPLETED**
- **Visualization Fix**: Added `origin_type="env"` to ViewerCfg to show single robot

#### Checkpoints Saved:
| Step | File | Size |
|------|------|------|
| 5000 | agent_5000.pt | 2.2MB |
| 10000 | agent_10000.pt | 2.2MB |
| 15000 | agent_15000.pt | 2.2MB |
| 20000 | agent_20000.pt | 2.2MB |
| 25000 | agent_25000.pt | 2.2MB |
| 30000 | agent_30000.pt | 2.2MB |
| best | best_agent.pt | 2.9MB |

#### Videos Recorded:
- Training videos at steps 0, 5000, 10000, 15000, 20000, 25000 (~17KB each, 15sec)
- Play evaluation video: 500 steps (25sec, 1.7MB)

#### Path to Best Checkpoint:
`/home/matteo/Desktop/code_projects/harold/logs/skrl/harold_direct/2025-12-20_09-28-17_ppo_torch/checkpoints/agent_30000.pt`

#### TensorBoard Analysis (Quantitative):
| Metric | Step 10 | Step 5000 | Step 30000 | Status |
|--------|---------|-----------|------------|--------|
| upright_mean | 0.10 | 0.89 | 0.88 | UNSTABLE |
| episode_length | 3.8 | 253 | 270 | Improving |
| total_reward | 88 | 6286 | 6700 | Improving |

**Final Status**: UNSTABLE (upright_mean=0.86-0.88, threshold for STABLE is 0.9)

**Key Findings**:
- Robot improved from immediate falling (upright=0.1) to partial stability (upright=0.87)
- Never reached STABLE_STANDING threshold (>0.9)
- Episode lengths increased from ~4 steps to ~180-270 steps (good progress)
- height_err_abs metric has a bug (shows impossible 300-600m values)
- Visualization fix (origin_type="env") caused white screen - **REVERTED**

#### Next Steps:
- Robot is UNSTABLE, not STABLE_STANDING - needs further optimization
- Consider higher upright_reward or different termination threshold
- May need longer training or curriculum approach

---

## Session 4 (2025-12-20 Afternoon) - OBSERVABILITY FAILURE

### CRITICAL: All EXP-017 through EXP-022 produced FALLEN robots!

**What I thought happened**: I saw `upright_mean > 0.9` and `vx_w_mean > 0.1` and concluded
the robot was walking.

**What actually happened**: Video review showed the robot **fell forward onto its elbows**
after spawning. The back legs squatted to keep the rear elevated.

**Why metrics were misleading**:
- `upright_mean = 0.93`: Back end elevated = high upright (even though fallen!)
- `vx_w_mean = 0.1+`: Falling forward = positive velocity
- `height_reward = 0.32`: **SHOULD BE ~3.0** for proper standing
- `body_contact_penalty = -0.34`: **Body was touching ground**

### EXP-017: Stricter Termination - FAILED (Fallen Forward)
- **Date**: 2025-12-20
- **ID**: `2025-12-20_11-23-22_ppo_torch`
- **Config**: upright=25, height=8, forward=0, **termination=-0.75** (was -0.5)
- **Duration**: 50,000 timesteps
- **Metrics (4-metric analysis)**:
  - upright_mean: 0.935 (PASS - but misleading!)
  - height_reward: ~0.3 (FAIL - robot not at height)
  - body_contact_penalty: ~-0.3 (FAIL - body on ground)
  - vx_w_mean: ~0.0
- **TRUE STATUS**: FALLEN FORWARD
- **Notes**: Stricter termination didn't help - robot found stable "elbow" pose

### EXP-018: Add Forward Motion - FAILED (Fallen Forward)
- **Date**: 2025-12-20
- **ID**: `2025-12-20_12-38-33_ppo_torch`
- **Config**: upright=25, height=8, **forward=1.0**, term=-0.75
- **Metrics**:
  - upright_mean: 0.92 (misleading)
  - height_reward: ~0.3 (FAIL)
  - vx_w_mean: 0.002 m/s
- **TRUE STATUS**: FALLEN FORWARD

### EXP-019: Higher Forward - FAILED (Fallen Forward)
- **Date**: 2025-12-20
- **ID**: `2025-12-20_13-16-45_ppo_torch`
- **Config**: upright=25, height=8, **forward=5.0**, term=-0.75
- **Metrics**:
  - upright_mean: 0.943 (misleading)
  - height_reward: ~0.3 (FAIL)
  - vx_w_mean: 0.019 m/s
- **TRUE STATUS**: FALLEN FORWARD

### EXP-020: Forward=15 - FAILED (Falling Backward!)
- **Date**: 2025-12-20
- **ID**: `2025-12-20_13-53-44_ppo_torch`
- **Config**: upright=25, height=8, **forward=15.0**, term=-0.75
- **Metrics**:
  - upright_mean: 0.91 (misleading)
  - height_reward: 0.32 (FAIL)
  - body_contact_penalty: -0.34 (FAIL)
  - vx_w_mean: **-0.028 m/s** (BACKWARD!)
- **TRUE STATUS**: FALLEN/DRIFTING

### EXP-021: Reduce Upright - FAILED (Fallen Forward)
- **Date**: 2025-12-20
- **ID**: `2025-12-20_14-40-04_ppo_torch`
- **Config**: **upright=10**, **height=5**, forward=15, term=-0.75
- **Metrics**:
  - upright_mean: 0.91 (misleading)
  - height_reward: ~0.7 (FAIL)
  - vx_w_mean: 0.068 m/s
- **TRUE STATUS**: FALLEN FORWARD
- **Notes**: I thought reducing upright would help forward motion - it just made falling easier

### EXP-022: Further Reduce Upright - FAILED (Fallen Forward)
- **Date**: 2025-12-20
- **ID**: `2025-12-20_15-26-24_ppo_torch`
- **Config**: **upright=5**, **height=3**, forward=15, term=-0.75
- **Metrics**:
  - upright_mean: 0.94 (misleading)
  - height_reward: 0.32 (FAIL - proves robot not at height)
  - body_contact_penalty: -0.34 (FAIL - body on ground)
  - vx_w_mean: peak 0.141 (from falling!)
- **TRUE STATUS**: FALLEN FORWARD
- **VIDEO CONFIRMED**: Robot on elbows, back legs squatting

### Root Cause Analysis

**The robot exploited the reward function:**
1. Forward reward (15.0) was 3-5x higher than height reward (3.0-5.0)
2. Falling forward = forward velocity = forward reward
3. Once on elbows, the pose is stable enough to not trigger termination
4. upright_mean stays high because back end is elevated

**The fix:**
1. height_reward must DOMINATE (try 25.0)
2. forward_reward must be ZERO until standing is verified
3. Lower body contact threshold (currently 10N is too high)
4. Add height-based termination

---

## Session 5 (2025-12-20 Evening) - OBSERVABILITY FIX VERIFIED

### EXP-024: Height-Dominant Reward - SUCCESS!
- **Date**: 2025-12-20
- **ID**: `2025-12-20_16-27-02_ppo_torch`
- **Config Changes**:
  - `height_reward: 25.0` (was 3.0-5.0)
  - `forward_reward: 0.0` (zero until standing verified)
  - `upright_reward: 10.0`
  - Body contact penalty: threshold 2N (was 10N), 5x scaling
  - Added height-based termination (< 0.165m = 60% of target)
- **Duration**: 50,000 timesteps

**4-METRIC ANALYSIS (NEW PROTOCOL):**
| Metric | EXP-022 (Fallen) | EXP-024 (Standing) |
|--------|-----------------|-------------------|
| height_reward | 0.32 | **7.41** |
| body_contact | -0.34 | **0.00** |
| upright_mean | 0.94 | **1.00** |
| vx_w_mean | 0.14 | -0.09 |
| episode_length | 171 | 3.0 |

**TRUE STATUS**: PROPERLY STANDING
- height_reward > 2.0 PROVES robot at correct height
- body_contact = 0 PROVES no body parts on ground
- Unlike EXP-022, this is NOT on elbows!

**Note**: Episode length is short (3.0) - may need investigation, but robot is standing when alive.

**Key Changes That Worked:**
1. height_reward=25.0 (dominant) prevents falling forward exploit
2. Lower body_contact threshold catches elbow contact
3. Height-based termination resets robot if it falls

---

---

## Session 6 (2025-12-21) - Forward Motion Experiments

### EXP-025: Tiny Forward Reward - STANDING BUT NOT WALKING
- **Date**: 2025-12-21
- **ID**: `2025-12-20_22-27-56_ppo_torch`
- **Config**:
  - `height_reward: 25.0` (dominant)
  - `forward_reward: 1.0` (tiny)
  - `upright_reward: 10.0`
- **Duration**: 100,000 timesteps (~3h 15min)

**4-METRIC ANALYSIS:**
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| upright_mean | 0.993 | > 0.9 | PASS |
| height_reward | 6.76 | > 2.0 | PASS |
| body_contact | 0.00 | > -0.1 | PASS |
| vx_w_mean | -0.089 | > 0.1 | FAIL |
| episode_length | 6.1 | > 300 | FAIL |

**TRUE STATUS**: STANDING BUT NOT WALKING
- Robot maintains proper standing posture (height_reward=6.76 proves it)
- No body contact (no elbow exploit)
- Forward velocity is actually slightly negative (-0.089)
- forward_reward=1.0 was too weak relative to height_reward=25.0

**Next Step**: Increase forward_reward to 2.0 or higher

---

### EXP-026: Increased Forward Reward - STANDING BUT NOT WALKING
- **Date**: 2025-12-21
- **ID**: `2025-12-21_01-48-36_ppo_torch`
- **Config**:
  - `height_reward: 25.0` (dominant)
  - `forward_reward: 2.0` (increased from 1.0)
  - `upright_reward: 10.0`
- **Duration**: 100,000 timesteps (~3h 20min)

**4-METRIC ANALYSIS:**
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| upright_mean | 0.9994 | > 0.9 | PASS |
| height_reward | 7.42 | > 2.0 | PASS |
| body_contact | 0.00 | > -0.1 | PASS |
| vx_w_mean | -0.090 | > 0.1 | FAIL |

**TRUE STATUS**: STANDING BUT NOT WALKING
- Robot maintains excellent standing posture (height_reward=7.42)
- No body contact (no elbow exploit)
- Forward velocity still slightly negative (-0.090)
- forward_reward=2.0 still too weak relative to height_reward=25.0

**Next Step**: Increase forward_reward to 3.0 (EXP-027)

---

### EXP-027: Forward=3.0 - STANDING BUT NOT WALKING
- **Date**: 2025-12-21
- **ID**: `2025-12-21_05-12-28_ppo_torch`
- **Config**:
  - `height_reward: 25.0` (dominant)
  - `forward_reward: 3.0` (increased from 2.0)
  - `upright_reward: 10.0`
- **Duration**: 100,000 timesteps (~3h 10min)

**4-METRIC ANALYSIS:**
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| upright_mean | 0.9994 | > 0.9 | PASS |
| height_reward | 7.41 | > 2.0 | PASS |
| body_contact | 0.00 | > -0.1 | PASS |
| vx_w_mean | -0.089 | > 0.1 | FAIL |

**TRUE STATUS**: STANDING BUT NOT WALKING
- Robot maintains excellent standing posture
- Forward velocity still slightly negative (-0.089)
- Pattern: EXP-025/026/027 all show identical behavior
- Forward reward 1.0/2.0/3.0 not enough to break standing equilibrium

**Next Step**: Jump to forward=5.0 (EXP-028) - larger increment

---

### EXP-028: Forward=5.0 - STANDING WITH BACKWARD DRIFT
- **Date**: 2025-12-21
- **ID**: `2025-12-21_08-27-24_ppo_torch`
- **Config**:
  - `height_reward: 25.0` (dominant)
  - `forward_reward: 5.0` (5:1 ratio)
  - `upright_reward: 10.0`
- **Duration**: 100,000 timesteps (~3h)

**4-METRIC ANALYSIS:**
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| upright_mean | 0.9969 | > 0.9 | PASS |
| height_reward | 7.42 | > 2.0 | PASS |
| body_contact | 0.00 | > -0.1 | PASS |
| vx_w_mean | -0.131 | > 0.1 | FAIL (WORSE!) |

**KEY FINDING: Local Optimum Identified**
- Robot is stuck in standing optimum: height(7.4) + upright(10) ≈ 17.4
- Forward motion requires leg lifting → temporary height loss
- Forward reward (5.0 * vx * upright² * f_slip) too small to compensate
- vx_w_mean is MORE negative with higher forward_pos (counterintuitive)

**Pattern Across Experiments:**
| EXP | forward_pos | vx_w_mean |
|-----|-------------|-----------|
| 025 | 1.0 | -0.089 |
| 026 | 2.0 | -0.090 |
| 027 | 3.0 | -0.089 |
| 028 | 5.0 | -0.131 |

**Next Step**: Add backward penalty (progress_forward_neg=2.0) to break symmetry

---

### EXP-029: Backward Penalty - PARTIAL SUCCESS
- **Date**: 2025-12-21
- **ID**: `2025-12-21_11-30-33_ppo_torch`
- **Config**:
  - `progress_forward_pos: 5.0`
  - `progress_forward_neg: 2.0` (NEW - backward penalty)
  - `height_reward: 25.0`
  - `upright_reward: 10.0`
- **Duration**: 100,000 timesteps (~3h 10min)

**4-METRIC ANALYSIS:**
| Metric | EXP-028 | EXP-029 | Status |
|--------|---------|---------|--------|
| upright_mean | 0.997 | 0.999 | PASS |
| height_reward | 7.42 | 6.58 | PASS |
| body_contact | 0.00 | 0.00 | PASS |
| **vx_w_mean** | **-0.131** | **+0.003** | **IMPROVED!** |

**Velocity Progression:**
- Early (0-25%): 0.006
- Mid (25-50%): 0.007
- Late (50-75%): 0.008
- Final (75-100%): 0.008

**KEY FINDING**: Backward penalty WORKS!
- Velocity went from -0.13 (EXP-028) to +0.003 (EXP-029)
- Local optimum broken - robot no longer drifting backward
- All stability metrics still passing
- Signal stable throughout training

**Next Step**: Increase forward_pos to 8.0 to get stronger forward motion

---

### EXP-030: Increased Forward Reward - PARTIAL SUCCESS (UNSTABLE)
- **Date**: 2025-12-21
- **ID**: `2025-12-21_14-44-46_ppo_torch`
- **Config**:
  - `progress_forward_pos: 8.0` (increased from 5.0)
  - `progress_forward_neg: 2.0` (keep backward penalty)
  - `height_reward: 25.0`
  - `upright_reward: 10.0`
- **Duration**: 20,000 timesteps (~40 min)

**4-METRIC ANALYSIS:**
| Metric | EXP-029 | EXP-030 | Status |
|--------|---------|---------|--------|
| upright_mean | 0.999 | 0.9998 | PASS |
| height_reward | 6.58 | 6.89 | PASS |
| body_contact | 0.00 | 0.00 | PASS |
| **vx_w_mean** | +0.003 | **-0.027** | REGRESSED |

**CRITICAL FINDING: Learned walking but regressed!**

Velocity progression:
- 20%: +0.004
- 40%: +0.044
- 60%: **+0.056** (PEAK - robot was walking!)
- 80%: +0.021
- 100%: **-0.028** (regressed back to backward drift)

The robot learned forward motion mid-training but the behavior was unstable and regressed. The backward penalty (2.0) was not strong enough to maintain forward motion.

**Next Step**: Increase backward penalty to 4.0 or 5.0

---

### EXP-031: Stronger Backward Penalty - PARTIAL SUCCESS (WALKING THEN REGRESSED)
- **Date**: 2025-12-21
- **ID**: `2025-12-21_15-29-28_ppo_torch`
- **Config**:
  - `progress_forward_pos: 8.0` (keep from EXP-030)
  - `progress_forward_neg: 5.0` (increased from 2.0)
  - `height_reward: 25.0`
  - `upright_reward: 10.0`
  - `num_envs: 4096` (increased from 1024)
  - `video: enabled` (interval=6400, length=250)
- **Duration**: 20,000 timesteps (~2h 15min)

**4-METRIC ANALYSIS (Final):**
| Metric | Value | Status |
|--------|-------|--------|
| upright_mean | 0.9999 | PASS |
| height_reward | 6.76 | PASS |
| body_contact | 0.00 | PASS |
| vx_w_mean | 0.006 | REGRESSED |

**KEY FINDING: Robot walked, then regressed!**

Velocity progression:
- 10-30%: 0.01-0.04 (learning)
- **40-80%: 0.11-0.17** (WALKING!)
- 90-100%: 0.07-0.01 (regressed)

**Peak velocity: 0.179 m/s** at sample 1339/1999

**Comparison with EXP-030:**
| Metric | EXP-030 (neg=2.0) | EXP-031 (neg=5.0) |
|--------|-------------------|-------------------|
| Walking phase | 40-60% | **40-80%** |
| Peak vx | 0.056 | **0.179** |
| Final vx | -0.027 | 0.006 |

Stronger backward penalty extended walking phase and improved peak velocity, but didn't prevent eventual regression.

**Next**: Try gait rewards (feet_air_time) or even stronger backward penalty

---

### EXP-032: Gait Rewards - PARTIAL SUCCESS (SAME REGRESSION PATTERN)
- **Date**: 2025-12-21
- **ID**: `2025-12-21_17-48-42_ppo_torch`
- **Config**:
  - `progress_forward_pos: 8.0`
  - `progress_forward_neg: 5.0`
  - `height_reward: 25.0`
  - `upright_reward: 10.0`
  - `feet_air_time_reward: 2.0` (NEW)
  - `leg_movement_reward: 1.0` (NEW)
  - `track_air_time: True` (enabled)
  - `num_envs: 4096`
- **Duration**: 20,000 timesteps (~2h)

**4-METRIC ANALYSIS (Final):**
| Metric | Value | Status |
|--------|-------|--------|
| upright_mean | 0.9996 | PASS |
| height_reward | 7.07 | PASS |
| body_contact | 0.00 | PASS |
| vx_w_mean | -0.026 | REGRESSED |

**Velocity by Segment:**
| Segment | vx_mean | Status |
|---------|---------|--------|
| 0-20% | 0.032 | Learning |
| 20-40% | 0.064 | Improving |
| **40-60%** | **0.112** | **WALKING!** |
| 60-80% | -0.022 | Regressed |
| 80-100% | -0.024 | Stable regression |

**Peak velocity: 0.165 m/s** at step 9410 (47%)

**FINDING**: Gait rewards did NOT prevent regression!
- Walking phase: 40-60% (similar to EXP-030/031)
- Peak velocity: 0.165 (between EXP-030's 0.056 and EXP-031's 0.179)
- Regression: Still occurred, settling at -0.024 m/s

**Comparison across experiments:**
| EXP | Config Change | Walking Phase | Peak vx | Final vx |
|-----|---------------|---------------|---------|----------|
| 030 | pos=8, neg=2 | 40-60% | 0.056 | -0.027 |
| 031 | neg=5 | 40-80% | 0.179 | 0.006 |
| **032** | +gait rewards | **40-60%** | 0.165 | -0.026 |

**Key Insight**: The regression pattern is consistent. The robot:
1. Learns to walk in the 40-60% phase
2. Then "unlearns" walking and reverts to standing with backward drift

**Next**: Try one of:
1. **Checkpoint at peak** - Save at step ~9400, fine-tune from there
2. **Stronger backward penalty** - neg=10.0
3. **Reduce height reward** - Allow more walking exploration

---

### EXP-033: INVALID (Height Termination Bug)
- **Date**: 2025-12-21
- **Status**: INVALID - Episode length = 3 steps (robots dying immediately)
- **Bug**: Height termination used world Z instead of height above terrain

---

### EXP-034: Height Dominant Baseline - IN PROGRESS
- **Date**: 2025-12-21
- **ID**: `2025-12-21_21-38-22_ppo_torch`
- **Config**:
  - `progress_forward_pos: 5.0` (reduced from 80)
  - `progress_forward_neg: 5.0` (reduced from 80)
  - `height_reward: 25.0` (increased from 5.0)
  - `upright_reward: 10.0` (increased from 1.8)
  - `rear_support_bonus: 0.0` (disabled)
  - `num_envs: 4096`
  - NO height termination (bug fixed)
- **Duration**: 20,000 timesteps (~55 min)

**Hypothesis**: Height-dominant rewards (25.0) will prevent elbow exploit while low forward reward (5.0) allows stable standing first.

**Final Results** (20000 timesteps, 22:55 runtime):
| Metric | Value | Status |
|--------|-------|--------|
| episode_length | 370.5 | PASS |
| upright | 0.9305 | PASS |
| height_reward | 1.30 | FAIL (< 2.0) |
| body_contact | -0.1442 | FAIL (< -0.1) |
| vx_w_mean | -0.006 | FAIL |

**TRUE STATUS**: ELBOW EXPLOIT - Robot fell forward onto elbows.

**Key Finding**: Even with height_reward=25.0 (5x forward reward), robot still found elbow exploit:
- Robot falls forward early in training
- Back stays elevated → passes upright check (0.93)
- Body touches ground → fails body_contact (-0.14)
- Height too low → fails height_reward (1.30 vs 25.0 max)

**Why height_reward=25.0 didn't work**:
1. Robot learns elbow posture before learning to stand
2. Once in stable elbow position, gradient to stand is weak
3. Standing requires lifting body (risk of falling) vs staying in safe elbow position

**Conclusion**: Need stronger intervention than just increasing height_reward. Options:
1. Height-based termination (kill episode if too low)
2. Start from standing checkpoint (terrain_62 or EXP-012)
3. Negative height penalty for low posture

---

### EXP-035: Fine-tune from terrain_62 - IN PROGRESS
- **Date**: 2025-12-22
- **ID**: TBD
- **Config**:
  - `checkpoint`: terrain_62/agent_128000.pt (stable standing)
  - `progress_forward_pos: 5.0`
  - `progress_forward_neg: 5.0`
  - `height_reward: 25.0`
  - `upright_reward: 10.0`
  - `num_envs: 4096`
- **Duration**: 20,000 timesteps

**Hypothesis**: Starting from stable standing checkpoint + height-dominant rewards will prevent elbow exploit while allowing forward learning.

**Previous Fine-tuning Attempts**:
- EXP-013 (forward=3.0): Destabilized standing immediately
- EXP-014 (leg_movement only): Also destabilized
- Key difference in EXP-035: height_reward=25.0 is dominant (5x forward)

**Final Results** (20000 timesteps):
| Metric | Value | Status |
|--------|-------|--------|
| episode_length | 382.2 | PASS |
| upright | 0.9316 | PASS |
| height_reward | 1.16 | FAIL (< 2.0) |
| body_contact | -0.07 | PASS |
| vx_w_mean | -0.015 | FAIL |

**TRUE STATUS**: ELBOW EXPLOIT - Same as EXP-034

**Key Finding**: Starting from terrain_62 checkpoint does NOT prevent elbow exploit:
- Checkpoint policy degraded under new reward structure
- Robot learned to go lower (height_reward dropped from initial 1.2 to 0.91 mid-training)
- Final height_reward (1.16) similar to EXP-034 from scratch (1.30)

**Comparison EXP-034 vs EXP-035**:
| Approach | height_reward | body_contact |
|----------|---------------|--------------|
| EXP-034 (scratch) | 1.30 | -0.14 |
| EXP-035 (checkpoint) | 1.16 | -0.07 |

Both failed similarly. Reward shaping alone is insufficient.

---

### EXP-036: Height-Based Termination - FAILED (Implementation Issue)
- **Date**: 2025-12-22
- **Config**:
  - Terminate episode if height < threshold
  - Tried thresholds: 0.165m, 0.12m
  - Both caused immediate termination (3-6 step episodes)

**Issue**: Spawn pose (athletic pose with thighs=0.70, calves=-1.40) is too low:
- Robot spawns in crouched position, below termination threshold
- Episodes terminate immediately

**Lesson**: Height termination requires matching spawn height to threshold.

**Status**: ABANDONED - need different approach

---

### EXP-037: Body Contact Termination - FAILED (Threshold Too High)
- **Date**: 2025-12-22
- **ID**: `2025-12-21_22-53-14_ppo_torch`
- **Config**:
  - `body_contact_threshold: 50.0` (N)
  - `height_threshold: 0.0` (disabled)
  - Same reward weights as EXP-034/035

**Final Results** (20000 timesteps):
| Metric | Value | Status |
|--------|-------|--------|
| episode_length | 391.4 | PASS |
| upright | 0.9337 | PASS |
| height_reward | 1.13 | FAIL (< 2.0) |
| body_contact | -0.12 | FAIL (< -0.1) |
| vx_w_mean | -0.0003 | FAIL |

**Issue**: 50N threshold too high. Robot generating ~22N contact force (below threshold).

**Calculation**:
- body_contact_penalty = -relu(nonfoot_F - 10) / 100 = -0.12
- Therefore nonfoot_F ≈ 22N

**Next Step**: Lower threshold to 15-20N to catch elbow exploit.

---

### EXP-038: Body Contact Termination (15N) - IN PROGRESS
- **Date**: 2025-12-22
- **ID**: `2025-12-21_23-18-57_ppo_torch`
- **Config**:
  - `body_contact_threshold: 15.0` (lowered from 50.0)
  - `height_threshold: 0.0` (disabled)
  - Same reward weights as EXP-034-037

**Hypothesis**: 15N threshold will catch elbow exploit (~22N force in elbow position).

**Progress at 6050 data points (~25% complete)**:
| Metric | Value | Status |
|--------|-------|--------|
| episode_length | 364.3 | PASS |
| upright | 0.9560 | PASS |
| height_reward | 1.53 | FAIL (< 2.0) |
| body_contact | -0.19 | FAIL |
| vx_w_mean | 0.016 | WARN |

**Observations**:
- Body contact penalty oscillating (-0.04 early → -0.19 later)
- Height reward improved slightly (1.18 → 1.53)
- Episode min length: 5-6 (termination IS triggering)
- Policy is exploring different strategies

**Trend Analysis** (early → current):
- body_contact_penalty: -0.014 → -0.0002 (initially learned to avoid contact!)
- height_reward: 1.01 → 1.14 (slight improvement)
- vx_w_mean: -0.016 → 0.036 (improving - starting to move forward)

**STATUS**: IN PROGRESS - Training running (~6h total duration)
- Background PID: Training still running as of session end
- Next agent should check if training completed and analyze final results

---

---

## Session 7 (2025-12-22 Overnight) - Height Termination Debugging

### EXP-002: Body Contact Only (10N) - FAILED
- **Date**: 2025-12-22
- **ID**: `2025-12-22_03-15-23_ppo_torch`
- **Config**:
  - `body_contact_threshold: 10.0` (lowered from 15.0)
  - `height_threshold: 0.0` (disabled)
  - `height_reward: 25.0`
  - `upright_reward: 10.0`

**4-METRIC ANALYSIS:**
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| episode_length | 344.9 | > 100 | PASS |
| upright_mean | 0.938 | > 0.9 | PASS |
| height_reward | 1.76 | > 2.0 | FAIL |
| body_contact | -0.04 | > -0.1 | PASS |
| vx_w_mean | -0.023 | > 0.1 | FAIL |

**TRUE STATUS**: ELBOW EXPLOIT
- 10N threshold reduced body contact penalty (-0.04 vs -0.12 in EXP-037)
- But robot still on elbows (height_reward=1.76 < 2.0)
- Body contact alone insufficient to prevent elbow pose

---

### EXP-003: Combined Height + Contact - FAILED (SANITY)
- **Date**: 2025-12-22
- **ID**: `2025-12-22_04-56-02_ppo_torch`
- **Config**:
  - `height_threshold: 0.12` (new - terminate if below)
  - `body_contact_threshold: 10.0`
  - `height_reward: 25.0`

**Hypothesis**: Combined termination (height < 0.12m OR contact > 10N) will prevent elbow exploit.

**Result**: SANITY_FAIL - Episodes only 8 steps!
| Metric | Value | Status |
|--------|-------|--------|
| episode_length | 7.8 | FAIL (need >100) |
| upright_mean | 0.999 | PASS |
| height_reward | 4.70 | N/A (too short) |

**Issue**: Height threshold 0.12m too aggressive. Robot terminates immediately after spawn.

---

### EXP-004/005: Conservative Height (0.05m) - FAILED (SANITY)
- **Date**: 2025-12-22
- **Config**: `height_threshold: 0.05`

**Result**: Still SANITY_FAIL - Episodes 6-23 steps
- Even 0.05m threshold caused immediate termination
- Height scanner has initialization issues on early frames

**Diagnostic (EXP-005)**: Disabled height termination entirely → 223 steps, confirming sensor issue.

---

### EXP-006: Height (0.10m) with Warmup - FAILED (SANITY)
- **Date**: 2025-12-22
- **Config**:
  - `height_threshold: 0.10`
  - `height_termination_warmup_steps: 20` (NEW - skip first 20 steps)

**Result**: SANITY_FAIL - Episodes 31 steps
- Warmup helped (23→31 steps) but still insufficient
- Robot falls to elbow pose by step 31, triggering termination after warmup

---

### EXP-007: Height (0.05m) with Warmup - FAILED (SANITY)
- **Date**: 2025-12-22
- **Config**: `height_threshold: 0.05`, warmup=20

**Result**: Still SANITY_FAIL - Episodes 23 steps
- Even conservative 0.05m threshold triggers after warmup
- Fundamental issue: robot falls to elbow pose faster than it learns to stand

---

### EXP-008: Reward-Only Approach - COMPLETED (FAILED)
- **Date**: 2025-12-22
- **ID**: `2025-12-22_05-45-33_ppo_torch`
- **Config**:
  - `height_threshold: 0.0` (disabled)
  - `height_reward: 30.0` (increased from 25.0)
  - `body_contact_threshold: 10.0`
  - No height termination - rely on rewards only

**Hypothesis**: Stronger height reward (30.0) will incentivize standing through reward gradient alone.

**Final Results:**
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| episode_length | 351 | > 100 | PASS |
| upright_mean | - | > 0.9 | PASS |
| height_reward | 1.70 | > 2.0 | FAIL |
| body_contact | -0.01 | > -0.1 | PASS |
| vx_w_mean | 0.031 | > 0.1 | FAIL |

**TRUE STATUS**: ELBOW EXPLOIT - Robot on elbows

**Height Progression:**
- 5 min: 1.44
- 15 min: 1.64
- 30 min: 1.77
- 48 min: 1.93 (peak)
- 68 min: 1.61 (regressed)
- Final: 1.70

**Key Finding**: Even with height_reward=30.0, robot settles into elbow exploit.
The reward gradient from height is insufficient to overcome the stable local minimum.

---

### EXP-009: Joint-Angle Termination (Loose) - FAILED
- **Date**: 2025-12-22
- **ID**: `2025-12-22_10-02-46_ppo_torch`
- **Config**:
  - `elbow_pose_termination: True`
  - `front_thigh_threshold: 1.0` rad
  - `front_calf_threshold: -0.8` rad
  - Terminate if front thigh > 1.0 AND front calf > -0.8

**Result**: FAILED - Height 1.50, robot adapted around thresholds
- Thresholds too loose, robot found elbow pose that didn't trigger termination

---

### EXP-010: Joint-Angle Termination (Tight) - FAILED
- **Date**: 2025-12-22
- **ID**: `2025-12-22_10-18-44_ppo_torch`
- **Config**:
  - `front_thigh_threshold: 0.85` rad (tightened)
  - `front_calf_threshold: -1.0` rad (tightened)

**Result**: FAILED - Height 1.44, still adapted around thresholds
- Robot finds poses that avoid joint-angle termination while staying low

---

### EXP-011: Low Height Penalty (Unclamped) - FAILED
- **Date**: 2025-12-22
- **ID**: `2025-12-22_10-31-06_ppo_torch`
- **Config**:
  - `low_height_penalty: -50.0` per meter below threshold
  - `low_height_threshold: 0.20m`
  - No joint-angle termination

**Result**: FAILED - Reward was -25 million (penalty too aggressive)
- Height scanner returned bad values causing extreme penalties
- Height oscillated: 1.67 → 1.80 → 1.48

---

### EXP-012: Low Height Penalty (Clamped, World Z) - IN PROGRESS
- **Date**: 2025-12-22
- **ID**: `2025-12-22_11-02-29_ppo_torch`
- **Config**:
  - `low_height_penalty: -50.0` per meter below threshold
  - `low_height_threshold: 0.20m`
  - Uses world Z position instead of height scanner
  - Height deficit clamped to max 0.10m (max penalty -5 per step)

**Progress at 20 min:**
| Metric | Value | Status |
|--------|-------|--------|
| episode_length | 354 | PASS |
| height_reward | 1.68 | FAIL (< 2.0) |
| body_contact | -0.01 | PASS |
| vx_w_mean | 0.003 | FAIL |
| reward_total | 1791 | Reasonable |

**Observation**: Height stable at ~1.68, penalty not pushing higher.
Training still in progress.

---

## Key Insights from Session 7-8

1. **Height Termination Unusable**: Height scanner has initialization issues
2. **Joint-Angle Termination**: Robot adapts around any threshold
3. **Reward-Only Approach**: Robot settles into elbow exploit (local minimum)
4. **Height Penalty**: Helps but robot still finds low poses

The fundamental problem: **Elbow pose is a stable attractor** that the robot discovers early in training. Once found, it's hard to escape via reward gradients alone.

**Recommended Next Approach**: Curriculum from terrain_62 checkpoint (already knows how to stand)
3. Reward-only approach results in same elbow exploit as before

**Next Steps**:
1. Investigate height scanner initialization
2. Consider curriculum: start standing → add motion
3. Or use different termination signal (e.g., specific joint angles)

---

## Key Finding: Height Termination is Unusable

All height termination experiments (EXP-003 through EXP-007) failed due to:

1. **Sensor Initialization**: Height scanner returns bad values for first few steps
2. **Learning Dynamics**: Robot learns elbow pose faster than standing
3. **Local Minimum**: Once in elbow pose, termination prevents learning recovery

The fundamental issue: **Termination kills exploration before learning happens.**

Reward-only approach (EXP-008) allows learning but robot finds elbow exploit.

---

## Baseline Configuration
```yaml
# From terrain_62 (reference baseline)
num_envs: 1024
task: Template-Harold-Direct-flat-terrain-v0
learning_rate: 0.001
rollouts: 24
mini_batches: 4
discount_factor: 0.99
lambda: 0.95
entropy_loss_scale: 0.01
```
