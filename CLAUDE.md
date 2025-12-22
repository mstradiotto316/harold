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

### Session End (ALWAYS DO THIS BEFORE FINISHING)
1. Update `EXPERIMENTS.md` with any experiments run
2. Update `OBSERVATIONS.md` with new insights
3. Update `NEXT_STEPS.md` to reflect completed/new tasks
4. Create/update session log in `.claude_memory/sessions/YYYY-MM-DD_session.md`
5. Update `CONTEXT.md` if project state has significantly changed

---

## Common Commands

```bash
# Activate environment
source ~/Desktop/env_isaaclab/bin/activate
cd /home/matteo/Desktop/code_projects/harold

# Training (basic)
python harold_isaac_lab/scripts/skrl/train.py \
  --task=Template-Harold-Direct-flat-terrain-v0 \
  --num_envs 1024

# Training headless with video recording
python harold_isaac_lab/scripts/skrl/train.py \
  --task=Template-Harold-Direct-flat-terrain-v0 \
  --num_envs 4096 --headless --video --video_length 250 --video_interval 6400

# Play/evaluate a checkpoint
python harold_isaac_lab/scripts/skrl/play.py \
  --task=Template-Harold-Direct-flat-terrain-v0 \
  --checkpoint=<path_to_checkpoint.pt>

# TensorBoard monitoring
python3 -m tensorboard.main --logdir logs/skrl/harold_direct/ --bind_all

# Background training (prevents context overflow)
./scripts/run_experiment.sh [iterations]  # Runs in background, outputs to /tmp/harold_train.log
./scripts/check_training.sh               # Compact progress check

# Validate training results (REQUIRED after every experiment)
python scripts/validate_training.py           # Validate latest run
python scripts/validate_training.py <run_id>  # Validate specific run
python scripts/validate_training.py --list    # List recent runs
```

---

## 5-Metric Validation Protocol

**Run `python scripts/validate_training.py` after EVERY experiment.**

| Priority | Metric | Threshold | Failure Mode |
|----------|--------|-----------|--------------|
| 1. SANITY | `episode_length` | > 100 | Robots dying immediately (BUG!) |
| 2. Stability | `upright_mean` | > 0.9 | Falling over |
| 3. Height | `height_reward` | > 2.0 | On elbows/collapsed |
| 4. Contact | `body_contact_penalty` | > -0.1 | Body on ground |
| 5. Walking | `vx_w_mean` | > 0.1 m/s | Not walking forward |

**CRITICAL**: If metric #1 fails, ALL other metrics are invalid.

Exit codes: `0` = pass, `1` = some failures, `2` = sanity check failed (do not proceed).

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
Long training runs flood context with tqdm output. Always use `./scripts/run_experiment.sh` for training.

---

## Technical Reference

Full technical documentation (joint limits, reward formulas, domain randomization) is in `AGENTS.md`.

Hardware details (servos, IMU, controller) are in `README.md`.
