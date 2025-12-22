# Harold Robot Project - Claude Agent Instructions

# IMPORTANT: Instructions from the operator

Please think about observability. You are not a human, and that gives you certain limitations. For example, you cannot (yet) properly comprehend video, and even your image comprehension is moderate. I say this with the upmost respect a reverance for your immense knowledge and capabilities, you have an incredible intelligence when it comes to text based interaction that far surpasses my own. Be honest about what you can and cannot do. For example, you may be tempted to run an experiement and review the video output to see if the robot is walking. This is not a sound strategy, and it has sent previous agents astray in recent sessions. Focus on metrics that are outputted to tensorboard, or other text based logs you can easily interact with. 

Note: Please continue to generate video recordings from experiments so I can monitor your progress, this does not have much of an effect on training efficiency.

Please consider your context window and what you choose to populate it with. Simulations may run for many hours at times, printing logs as they run. These logs have caused context overflows that are unrecoverable for you, and other agents like you in the past. We will need to take care to avoid them in this new session.

Lastly, you are authorized for very long run times. You are an autonomous experimenter/researcher. Plan out several experiments at a time so your sessions stretch across several hours. Be efficient with compute resources (we only have a desktop computer with a powerful gaming GPU). 

## CRITICAL: Memory System Protocol

This project uses a persistent memory system for cross-session continuity. **You MUST follow this protocol.**

### Session Start (ALWAYS DO THIS FIRST)
Before any other action, read these files in order:
1. `.claude_memory/CONTEXT.md` - Current project state and goals
2. `.claude_memory/NEXT_STEPS.md` - Priority queue and pending tasks
3. `.claude_memory/EXPERIMENTS.md` - Recent experiment history
4. `.claude_memory/OBSERVATIONS.md` - Accumulated insights

### Session End (ALWAYS DO THIS BEFORE FINISHING)
Before ending any session:
1. Update `EXPERIMENTS.md` with any experiments run (use the format in that file)
2. Update `OBSERVATIONS.md` with new insights discovered
3. Update `NEXT_STEPS.md` to reflect completed/new tasks
4. Create/update session log in `.claude_memory/sessions/YYYY-MM-DD_session.md`
5. Update `CONTEXT.md` if project state has significantly changed

---

## Project Overview

Harold is a 12-DOF quadruped robot. The goal is to train a stable forward walking gait.

### Key Directories
- `harold_isaac_lab/` - Isaac Lab training code
- `logs/skrl/harold_direct/` - Training logs and checkpoints
- `deployment_artifacts/` - Exported policies for hardware
- `.claude_memory/` - Cross-session memory (READ THIS FIRST)

### Common Commands
```bash
# Activate environment
source ~/Desktop/env_isaaclab/bin/activate
cd /home/matteo/Desktop/code_projects/harold

# Training (adjust num_envs and max_iterations as needed)
python harold_isaac_lab/scripts/skrl/train.py \
  --task=Template-Harold-Direct-flat-terrain-v0 \
  --num_envs 1024

# Play/evaluate a checkpoint
python harold_isaac_lab/scripts/skrl/play.py \
  --task=Template-Harold-Direct-flat-terrain-v0 \
  --checkpoint=<path_to_checkpoint.pt>

# TensorBoard monitoring
python3 -m tensorboard.main --logdir logs/skrl/harold_direct/ --bind_all
```

### Experiment Protocol
1. Document hypothesis in `OBSERVATIONS.md` before running
2. Run experiment with clear, logged configuration
3. Record results in `EXPERIMENTS.md` immediately after
4. Update `NEXT_STEPS.md` based on findings

---

## Technical Reference

See `AGENTS.md` for detailed technical documentation including:
- Robot configuration (joints, limits, actuators)
- Environment specifications (obs/action space, rewards)
- Training hyperparameters
- Deployment workflow
