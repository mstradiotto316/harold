# Harold Quadruped Robot RL Research Analysis Request

## Mission Briefing

You are analyzing the Harold quadruped robot reinforcement learning project. Your task is to provide a comprehensive analysis and suggest the next experiments to run.

**THE CORE PROBLEM**: The robot has discovered an exploit in the reward structure. Instead of learning to walk, it falls forward onto its elbows and stays there. This "elbow pose" yields acceptable rewards because:
- The back remains elevated (passes `upright_mean > 0.9`)
- Falling forward creates brief forward velocity
- Once stable on elbows, no termination is triggered
- Height reward gradient is too weak to overcome this local minimum

**Your goal**: Analyze 38+ failed experiments, identify patterns, and propose 5 ranked experiments that might finally achieve stable walking.

---

## Files to Read (in order)

Read these files to understand the project:

1. **`.claude_memory/EXPERIMENTS.md`** - Complete experiment history with metrics and outcomes
2. **`.claude_memory/OBSERVATIONS.md`** - Accumulated insights (if exists)
3. **`.claude_memory/HARDWARE_CONSTRAINTS.md`** - Physical robot limits for sim-to-real
4. **`harold_isaac_lab/source/harold_isaac_lab/harold_isaac_lab/tasks/direct/harold_flat/harold_isaac_lab_env_cfg.py`** - Current reward weights and termination config
5. **`harold_isaac_lab/source/harold_isaac_lab/harold_isaac_lab/tasks/direct/harold_flat/harold_isaac_lab_env.py`** - Environment implementation (reward calculations, observations)

---

## Analysis Questions

Answer these questions in your analysis:

### 1. Failure Pattern Analysis
- What are the common failure modes across experiments?
- Which reward/termination combinations were tried multiple times?
- Are there any untried combinations that might work?
- What patterns emerge from the 38+ experiments?

### 2. Root Cause Hypothesis
- Why does the robot consistently find the elbow exploit?
- What makes this pose a stable attractor in the reward landscape?
- What would break this local minimum?
- Is this a reward shaping problem, an exploration problem, or both?

### 3. Reward Structure Analysis
- What is the current reward balance (stability vs locomotion)?
- Is height_reward (currently 30.0) actually preventing the exploit?
- Why did backward penalty (progress_forward_neg) partially work in EXP-029/031 but walking regressed?
- What caused the "walking then regression" pattern in EXP-030/031/032?

### 4. Termination Strategy Evaluation
- Height termination failed (sensor initialization issues). What alternatives exist?
- Body contact termination (10N) is not catching elbow contact. Why?
- Joint-angle termination was tried (EXP-009/010) and robot adapted. What's missing?
- Is termination even the right approach, or is curriculum better?

### 5. Unexplored Approaches
Consider whether these have been tried:
- Reference motion / imitation learning
- Curriculum from standing checkpoint (terrain_62)
- Different observation space (add height to obs?)
- Action space modifications
- Gait phase rewards (sinusoidal targets)
- Spawn pose changes (start standing vs crouched)

---

## Key Metrics to Understand

| Metric | Threshold | What It Proves |
|--------|-----------|----------------|
| `episode_length` | > 100 | Episodes not dying immediately (SANITY CHECK) |
| `upright_mean` | > 0.9 | Robot orientation correct (CAN BE MISLEADING!) |
| `height_reward` | > 2.0 | Robot at standing height (NOT on elbows) |
| `body_contact_penalty` | > -0.1 | Body not touching ground |
| `vx_w_mean` | > 0.1 m/s | Robot walking forward |

**CRITICAL**:
- If `episode_length < 100`, all other metrics are invalid (BUG!)
- If `height_reward < 2.0`, robot is likely on elbows regardless of upright_mean!
- `upright_mean > 0.9` does NOT mean standing - elbow pose passes this check

---

## Key Experiments to Understand

| EXP | Approach | Outcome | Key Insight |
|-----|----------|---------|-------------|
| EXP-012 | Stability-only (forward=0) | Standing achieved | Proves stable standing is learnable |
| EXP-024 | Height-dominant (height=25) | Standing, no walking | Height reward prevents elbow but also motion |
| EXP-029 | Backward penalty (neg=5) | vx: -0.13 → +0.003 | Backward penalty breaks local minimum! |
| EXP-031 | Stronger backward (neg=5, pos=8) | **Peak vx=0.179** then regressed | Robot CAN walk but behavior unstable |
| EXP-032 | Added gait rewards | Same regression pattern | Gait rewards didn't stabilize walking |
| EXP-034-038 | Various termination strategies | All failed | Robot adapts to avoid termination |
| EXP-008 | height_reward=30, no termination | Elbow exploit | Reward gradient alone insufficient |

**Critical finding from EXP-031**: The robot CAN walk (peak vx = 0.179 m/s at 67% training), but the behavior is unstable and regresses to standing. Why?

---

## Current Configuration

```python
# RewardsCfg (harold_isaac_lab_env_cfg.py)
progress_forward_pos: float = 5.0   # Forward motion reward
progress_forward_neg: float = 5.0   # Backward motion penalty
upright_reward: float = 10.0        # Orientation reward
height_reward: float = 30.0         # HEIGHT DOMINANT
torque_penalty: float = -0.005      # Energy regularizer
lat_vel_penalty: float = 12.0       # Penalize sideways motion
yaw_rate_penalty: float = 1.0       # Dampen spinning
low_height_penalty: float = -50.0   # Penalty per meter below threshold
low_height_threshold: float = 0.20  # Penalize if height < 0.20m

# TerminationCfg
orientation_threshold: float = -0.5    # Terminate if gravity_z > -0.5
height_threshold: float = 0.0          # DISABLED (sensor issues)
body_contact_threshold: float = 10.0   # Terminate if contact > 10N
elbow_pose_termination: bool = False   # DISABLED (robot adapts)

# GaitCfg
frequency: float = 2.0
target_height: float = 0.275  # 27.5 cm standing height
```

---

## Hardware Constraints (MUST NOT VIOLATE)

These are physical robot limits - simulation parameters must stay within bounds:

| Parameter | Current | Max Safe | Reason |
|-----------|---------|----------|--------|
| `effort_limit` | 2.0 Nm | 2.5 Nm | Servos max 2.94 Nm @12V |
| `stiffness` | 200 | 400 | Higher causes oscillation |
| `damping` | 75 | 150 | Higher causes sluggish response |
| `action_scale` | 0.5 | 1.0 | Larger may hit joint limits |

**Must preserve:**
- Joint limits: shoulders ±30°, thighs/calves ±90° (mechanical stops)
- Control rate: 20 Hz (decimation=9 at 180Hz sim)
- Joint order: [FL, FR, BL, BR] × [shoulder, thigh, calf]

**May tune freely:**
- All reward weights
- Termination thresholds
- Domain randomization parameters
- Action filter beta

---

## Output Format Specification

Provide your analysis in this exact format:

### Executive Summary
(2-3 sentences: core problem diagnosis and your main insight)

### Root Cause Analysis
(Your hypothesis for why the robot finds the elbow exploit and why walking regresses)

### Pattern Analysis: What's Been Tried

| Category | Experiments | Result | Insight |
|----------|-------------|--------|---------|
| Height reward scaling | EXP-024, EXP-008 | ... | ... |
| Termination strategies | EXP-003-007, EXP-009-010 | ... | ... |
| Forward/backward penalty | EXP-029-031 | ... | ... |
| Gait rewards | EXP-032 | ... | ... |
| Curriculum/checkpoint | EXP-013, EXP-035 | ... | ... |

### Recommended Experiments (Ranked 1-5)

For each experiment, provide:

#### Experiment 1: [Descriptive Name]

**Hypothesis**: [What you expect to happen and why this approach differs from previous attempts]

**Config Changes** (exact diffs from current config):
```python
# In harold_isaac_lab_env_cfg.py

# Current value → New value
height_reward: float = 30.0  # → XX.X
progress_forward_pos: float = 5.0  # → X.X
# ... other changes
```

**Expected Outcome**: [What metrics should look like if successful]

**Failure Mode to Watch**: [What could go wrong and how to detect it]

**Why This Differs From Previous Attempts**: [Explain why EXP-XXX tried something similar and failed, and why your approach is different]

---

(Repeat for Experiments 2-5)

### Contingency Plan

If all 5 experiments fail, what fundamentally different approach should be considered?
- Different RL algorithm?
- Imitation learning from reference motion?
- Different robot spawn pose?
- Different action space (joint velocities vs positions)?
- Multi-stage curriculum?

---

## Robot Specifications (Reference)

- **12 DOF**: 4 legs × (shoulder, thigh, calf)
- **Joint order**: [shoulders FL, FR, BL, BR] → [thighs ...] → [calves ...]
- **Control rate**: 20 Hz policy (dt=1/180, decimation=9)
- **Observation**: 48D (lin_vel, ang_vel, gravity, joint_pos, joint_vel, commands, prev_actions)
- **Action**: 12D joint position deltas, normalized [-1, 1]
- **Actuators**: Implicit PD, stiffness=200, damping=75, effort_limit=2.0 Nm
- **Spawn height**: 0.24m
- **Target standing height**: 0.275m

---

## End of Prompt

Please provide your full analysis report following the output format above. Be specific about config changes - provide exact values, not just "increase X" or "try higher Y".
