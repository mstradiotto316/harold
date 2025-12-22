# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT: Instructions from the Operator

**Observability Limitations**: You cannot reliably interpret video output. Focus on TensorBoard metrics and text-based logs. Continue generating videos for human monitoring, but base your experiment conclusions on numerical metrics only.

**Context Management**: Training runs may produce hours of logs that cause context overflow. Use the helper scripts in `scripts/` to run training in the background and check progress compactly.

**Autonomous Research**: You are authorized for long run times. Plan multiple experiments per session to maximize efficiency. Be mindful of compute resources (single RTX 4080 GPU).

**Don't be a sycophant**: Push back on incorrect assumptions, propose better alternatives, and prioritize technical accuracy over validation.

---

## CRITICAL: Memory System Protocol

This project uses a persistent memory system for cross-session continuity.

### Session Start (ALWAYS DO THIS FIRST)
Read these files in order:
1. `.claude_memory/CONTEXT.md` - Current project state and goals
2. `.claude_memory/NEXT_STEPS.md` - Priority queue and pending tasks
3. `.claude_memory/EXPERIMENTS.md` - Recent experiment history
4. `.claude_memory/OBSERVATIONS.md` - Accumulated insights
5. `.claude_memory/HARDWARE_CONSTRAINTS.md` - Real-world limits for sim-to-real

### Session End (ALWAYS DO THIS BEFORE FINISHING)
1. Update `EXPERIMENTS.md` with any experiments run
2. Update `OBSERVATIONS.md` with new insights
3. Update `NEXT_STEPS.md` to reflect completed/new tasks
4. Create/update session log in `.claude_memory/sessions/YYYY-MM-DD_session.md`
5. Update `CONTEXT.md` if project state has significantly changed

---

## Harold CLI (Primary Interface for Agents)

The `harold` CLI is a **deep module** for hypothesis-driven experimentation. It hides TensorBoard complexity behind manifest files and comparison tools.

```bash
# Activate environment first
source ~/Desktop/env_isaaclab/bin/activate
cd /home/matteo/Desktop/code_projects/harold

# Start experiment with metadata (hypothesis-driven workflow)
python scripts/harold.py train --hypothesis "Lower threshold (10N) prevents elbow exploit" \
                               --tags "body_contact,elbow_fix"
python scripts/harold.py train --iterations 50000   # Custom duration
python scripts/harold.py train --checkpoint path    # Resume from checkpoint

# Check status (state-only reporting, no prescriptive suggestions)
python scripts/harold.py status                     # Current run with metrics
python scripts/harold.py status --json              # Machine-readable output

# Validate a run (by alias, directory name, or path)
python scripts/harold.py validate                   # Latest run
python scripts/harold.py validate EXP-034           # By experiment alias
python scripts/harold.py validate <run_id>          # By directory name

# List recent runs
python scripts/harold.py runs                       # Last 10 with status
python scripts/harold.py runs --hypothesis          # Include hypothesis for each

# Compare experiments side-by-side (essential for hypothesis-driven work)
python scripts/harold.py compare EXP-034 EXP-035    # Specific experiments
python scripts/harold.py compare                    # Last 5 experiments
python scripts/harold.py compare --tag forward_motion  # All with tag

# Add observations to experiments
python scripts/harold.py note EXP-034 "Robot walked at 40-80% then regressed"
```

### Status Output Example
```
RUN: 2025-12-22_14-30-00_ppo_torch (EXP-039)
HYPOTHESIS: Lower body contact threshold prevents elbow exploit
STATUS: RUNNING (45%, 1h 23m elapsed)
REWARD: 1234.5
SANITY: PASS (ep_len=342)
STANDING: PASS (height=6.8, contact=0.0)
WALKING: WARN (vx=0.03, need >0.1)
VERDICT: STANDING
DIAGNOSIS: Upright and stable, forward velocity 0.030 m/s
```

### Exit Codes
- `0` = All metrics pass (robot walking)
- `1` = Partial (standing but not walking)
- `2` = Failing (on elbows, fallen)
- `3` = Sanity failure (episodes too short)
- `4` = Not running / no data

### Manifest System
Each experiment gets a `manifest.json` with cached metrics, hypothesis, and notes. Global index at `logs/skrl/harold_direct/experiments_index.json` maps EXP-NNN aliases to directories.

---

## Low-Level Commands (for manual use)

```bash
# Direct training (verbose output - avoid in agents)
python harold_isaac_lab/scripts/skrl/train.py \
  --task=Template-Harold-Direct-flat-terrain-v0 \
  --num_envs 4096 --headless --video

# Play/evaluate a checkpoint
python harold_isaac_lab/scripts/skrl/play.py \
  --task=Template-Harold-Direct-flat-terrain-v0 \
  --checkpoint=<path_to_checkpoint.pt>

# TensorBoard monitoring
python3 -m tensorboard.main --logdir logs/skrl/harold_direct/ --bind_all
```

---

## 5-Metric Validation Protocol

**Run `python scripts/harold.py status` during training or `python scripts/harold.py validate` after completion.**

| Priority | Metric | Threshold | Failure Mode |
|----------|--------|-----------|--------------|
| 1. SANITY | `episode_length` | > 100 | Robots dying immediately (BUG!) |
| 2. Stability | `upright_mean` | > 0.9 | Falling over |
| 3. Height | `height_reward` | > 2.0 | On elbows/collapsed |
| 4. Contact | `body_contact_penalty` | > -0.1 | Body on ground |
| 5. Walking | `vx_w_mean` | > 0.1 m/s | Not walking forward |

**CRITICAL**: If SANITY fails, ALL other metrics are invalid.

**Answering the key questions:**
- **Is robot standing?** → STANDING: PASS means height > 2.0 and contact > -0.1
- **Is robot walking?** → WALKING: PASS means vx > 0.1 m/s (and standing)

---

## Code Architecture

### Isaac Lab Extension (`harold_isaac_lab/`)

The core training environment is an Isaac Lab extension:

```
harold_isaac_lab/
├── scripts/skrl/
│   ├── train.py         # Main training entry point
│   └── play.py          # Policy evaluation/playback
└── source/harold_isaac_lab/harold_isaac_lab/
    └── tasks/direct/
        ├── harold_flat/     # Flat terrain RL (primary task)
        │   ├── harold_isaac_lab_env.py      # Environment class
        │   ├── harold_isaac_lab_env_cfg.py  # Config (rewards, termination)
        │   ├── harold.py                    # Robot asset definition
        │   └── agents/skrl_ppo_cfg.yaml     # PPO hyperparameters
        ├── harold_rough/    # Rough terrain with curriculum
        └── harold_pushup/   # Scripted playback (no RL)
```

### Gym Task IDs
- `Template-Harold-Direct-flat-terrain-v0` - Primary training task
- `Template-Harold-Direct-rough-terrain-v0` - Rough terrain variant
- `Template-Harold-Direct-pushup-v0` - Scripted playback

### Key Configuration Files
| Purpose | Path |
|---------|------|
| Flat env config (rewards, termination) | `.../harold_flat/harold_isaac_lab_env_cfg.py` |
| PPO hyperparameters | `.../harold_flat/agents/skrl_ppo_cfg.yaml` |
| Robot asset (joints, actuators) | `.../harold_flat/harold.py` |
| USD model | `part_files/V4/harold_8.usd` |

### Robot Specifications
- **12 DOF**: 4 legs × (shoulder, thigh, calf)
- **Joint order**: [shoulders FL, FR, BL, BR] → [thighs ...] → [calves ...]
- **Control rate**: 20 Hz policy (dt=1/180, decimation=9)
- **Observation**: 48D (velocities, gravity, joint states, commands, prev_actions)
- **Action**: 12D joint position deltas, normalized [-1, 1]
- **Actuators**: Implicit PD, stiffness=200, damping=75, effort_limit=2.0

### Deployment Artifacts
```
deployment_artifacts/     # Exported ONNX policies for hardware
policy/                   # Hardware inference code
├── robot_controller.py   # Live IMU control loop
└── harold_policy.onnx    # Deployed neural network
```

---

## Known Failure Modes

### "On Elbows" Exploit
Robot falls forward onto elbows with back elevated. Passes `upright_mean > 0.9` but fails `height_reward < 2.0`. **Always check height_reward.**

### Height Termination Bug
If using height-based termination, check height above terrain, NOT world Z coordinate. Spawn pose must be above threshold.

### Context Overflow
Long training runs flood context with tqdm output. Always use `python scripts/harold.py train` which runs training in background.

---

## Research Analysis (Pro Models)

To get experiment suggestions from an external research model:
1. Copy the prompt from `.claude_memory/RESEARCH_PROMPT.md`
2. Paste into a research model with full codebase access
3. Receive analysis report with 5 ranked experiment suggestions

The prompt guides the model to analyze 38+ experiments and suggest new approaches.

---

## Hardware Constraints (Sim-to-Real)

Before modifying simulation parameters, check `.claude_memory/HARDWARE_CONSTRAINTS.md` for:
- **Safe parameter ranges** - What values won't break hardware
- **Must-preserve values** - Critical for deployment (joint limits, control rate, joint order)
- **Freely tunable** - No hardware impact (reward weights, termination thresholds)

Key hardware specs (FeeTech ST3215 servo):
- Max torque: 30 kg·cm (2.94 Nm) @12V - simulation uses conservative 2.0 Nm
- Position resolution: 0.088° (4096 steps/360°)
- Max speed: 45 RPM (4.71 rad/s)
- Joint limits: shoulders ±30°, thighs/calves ±90° (mechanical stops)

---

## Technical Reference

Full technical documentation (joint limits, reward formulas, domain randomization) is in `AGENTS.md`.

Hardware details (servos, IMU, controller) are in `README.md`.
