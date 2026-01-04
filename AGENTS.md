# AGENTS.md

This file is the primary agent quickstart and coordination guide for this repository.

## IMPORTANT: Instructions from the Operator

**Video Recording is MANDATORY**: Every training run MUST include video recording (`--video` flag). Video is critical logging infrastructure for human review. Never disable it.

**Observability Limitations**: You cannot reliably interpret video output. Focus on TensorBoard metrics and text-based logs. Continue generating videos for human monitoring, but base your experiment conclusions on numerical metrics only.

**Context Management**: Training runs may produce hours of logs that cause context overflow. Use `python scripts/harold.py train` (background) and `python scripts/harold.py status` (compact metrics). Avoid tailing logs except for brief debugging.

**Autonomous Research**: You are authorized for long run times. Plan multiple experiments per session to maximize efficiency. Be mindful of compute resources (single RTX 4080 GPU).

**Don't be a sycophant**: Push back on incorrect assumptions, propose better alternatives, and prioritize technical accuracy over validation.

**Don't commit changes without asking me first.**

---

## Design + Documentation Principles (Ousterhout)

- Keep interfaces deep: `scripts/harold.py` should hide training complexity; avoid ad-hoc wrappers.
- Prevent information leakage: defaults live in one place; update AGENTS + code together.
- Avoid temporal decomposition: document by role (experiment vs hardware), not by "first do X then Y" fragments.
- Define errors out of existence: prefer tooling that handles missing/stale state without manual cleanup.

## CRITICAL: Memory System Protocol

This project uses a persistent memory system for cross-session continuity.

### Session Start (ALWAYS DO THIS FIRST)
Read these files in order:
1. `docs/memory/CONTEXT.md` - Current project state and goals
2. `docs/memory/NEXT_STEPS.md` - Priority queue and pending tasks
3. `docs/memory/EXPERIMENTS.md` - Recent experiment history
4. `docs/memory/OBSERVATIONS.md` - Accumulated insights
5. `docs/memory/HARDWARE_CONSTRAINTS.md` - Real-world limits for sim-to-real

### Session End (ALWAYS DO THIS BEFORE FINISHING)
1. Update `EXPERIMENTS.md` with any experiments run
2. Update `OBSERVATIONS.md` with new insights
3. Update `NEXT_STEPS.md` to reflect completed/new tasks
4. Create/update session log in `docs/memory/sessions/YYYY-MM-DD_session.md`
5. Update `CONTEXT.md` if project state has significantly changed

---

## Documentation Map (Role-Based)

Start with `docs/index.md` for the full map, then use the role-specific lists below.

### Desktop Isaac Lab experiments (training/analysis)
- `AGENTS.md`: Primary agent quickstart and experiment workflow.
- `docs/memory/CONTEXT.md`: Current project state and goals.
- `docs/memory/NEXT_STEPS.md`: Priority queue and pending tasks.
- `docs/memory/EXPERIMENTS.md`: Recent experiment history.
- `docs/memory/OBSERVATIONS.md`: Accumulated insights.
- `docs/memory/OBSERVABILITY.md`: Metrics and validation protocol.
- `docs/memory/REFERENCE_ANALYSIS.md`: Isaac Lab reference implementation analysis.
- `docs/reference/sim_reference.md`: Simulation task and config reference.
- `docs/sim/isaac_lab_extension.md`: Isaac Lab extension template details.
- `docs/overview.md`: Repo overview and installation notes.

### Hardware walking tests (RPi)
- `docs/hardware/rpi_deployment.md`: RPi runtime/inference pipeline.
- `docs/hardware/calibration_checklist.md`: Servo calibration and validation checklist.
- `docs/memory/HARDWARE_CONSTRAINTS.md`: Sim-to-real limits and safe ranges.
- `docs/memory/CONTEXT.md`: Current project state and goals.
- `docs/memory/NEXT_STEPS.md`: Priority queue and pending tasks.
- `docs/memory/OBSERVATIONS.md`: Accumulated hardware insights.
- `docs/reference/hardware_reference.md`: Hardware specs + joint/order conventions.

### Firmware, sensors, logging
- `docs/hardware/calibration_checklist.md`: Servo calibration and sign checks.
- `docs/hardware/rpi_deployment.md`: Control loop, logging, and safety features.
- `docs/reference/hardware_reference.md`: Joint order, axes, and hardware specs.
- `docs/hardware/servos/`: Servo datasheets and protocol references.
- `docs/memory/OBSERVATIONS.md`: Known issues and logging insights.

### Shared references
- `docs/kinematics/harold_8_kinematics.yaml`: USD-derived joint/mesh kinematics spec (review before stance or sim-to-real alignment changes).
- `deployment/config/stance.yaml`: Canonical ready stance for hardware + simulation (single source of truth).

---

## Stance Updates

- Canonical ready stance lives in `deployment/config/stance.yaml`.
- Override path with `HAROLD_STANCE_PATH=/path/to/stance.yaml` when needed.
- After changing stance, run `python scripts/sync_stance.py` and re-flash the ESP32 firmware.

## Role Quickstart

**Experimentation Agent (Desktop)**
```bash
source ~/Desktop/env_isaaclab/bin/activate
cd /home/matteo/Desktop/code_projects/harold
python scripts/harold.py ps
python scripts/harold.py train --hypothesis "…" --tags "…"
python scripts/harold.py status
```

**Hardware Agent (RPi)**

Before touching hardware:
- Read `docs/memory/CONTEXT.md` for SSH info, repo/runtime paths, and sync workflow.
- Read `docs/reference/hardware_reference.md` for the system map and ESP32 flashing.
- Read `docs/hardware/rpi_deployment.md` for the runtime pipeline.
- Read `docs/hardware/calibration_checklist.md` before calibration changes.
- Sync runtime code via git: commit + push on desktop, then on the Pi run `git status -sb` (must be clean) and `git pull --ff-only`.
- If the Pi repo is dirty, STOP and reconcile the changes on desktop before testing; do not edit code on the Pi.
- Agents must execute the SSH/tests themselves; do not ask the operator to run these steps.

RPi layout (authoritative):
- Git repo on the Pi (syncs with desktop): `/home/pi/harold`
- Runtime root used by systemd/manual runs: `/home/pi/harold/deployment`
- Hardware logs live under `deployment/logs/` and `deployment/sessions/` (gitignored, but accessible).
- Keep `/home/matteo/Desktop/code_projects/harold` and `/home/pi/harold` in lockstep via git push/pull (ff-only).

```bash
source ~/envs/harold/bin/activate
cd <runtime-root>
sudo systemctl status harold
python3 -m inference.harold_controller   # manual run (service stopped)
python3 -m inference.harold_controller --cpg-only --max-seconds 60   # scripted/CPG-only walking test
```

Hardware logs:
- Runtime logs: `<runtime-root>/logs/harold.log`
- Telemetry CSVs: `<runtime-root>/sessions/session_YYYY-MM-DD_HH-MM-SS.csv`

---

## Sim-to-Real Guardrails (Required)

- Read `docs/memory/HARDWARE_CONSTRAINTS.md` before changing simulation parameters.
- Do not change must-preserve values (joint limits, control rate, joint order, effort limits) without explicit approval.
- Use `python scripts/harold.py` for training/monitoring; avoid direct `skrl` invocations unless debugging.
- Use the default `num_envs` from `scripts/harold.py`; override only if training fails to launch. Record overrides in `docs/memory/NEXT_STEPS.md` and `docs/memory/OBSERVATIONS.md`.
- Keep sim-to-real alignment as a first-class constraint; do not trade it away for short-term sim walking.

## Hardware → Sim Feedback Loop

After any hardware test:
- Summarize findings in `docs/memory/OBSERVATIONS.md` (telemetry, failure modes, behaviors).
- Add actionable follow-ups to `docs/memory/NEXT_STEPS.md`.
- If a finding changes sim configs or reward priorities, note it in `docs/memory/CONTEXT.md` and tag the next experiment with the rationale.

---

## Harold CLI (Primary Interface for Agents)

The `harold` CLI is a **deep module** for hypothesis-driven experimentation. It hides TensorBoard complexity behind manifest files and comparison tools.

**Single Interface Rule**: For training/monitoring, use `python scripts/harold.py` only. Do not create custom runner scripts.
Defaults for env count, headless/video settings, and duration live in the CLI; avoid overriding unless debugging a specific issue.

### Before Starting an Experiment (IMPORTANT)

**The harness automatically blocks concurrent training.** If you try to start training while another is running, you'll get an error. However, orphan processes can exist after crashes or interrupted sessions.

```bash
# ALWAYS check for existing processes before starting
python scripts/harold.py ps

# If orphans exist, clean them up
python scripts/harold.py stop
```

**What are orphan processes?** Training spawns child processes. If the parent dies unexpectedly (OOM, crash, context overflow), children may keep running. `harold ps` shows `[ORPHAN]` for processes not tracked by the PID file. `harold stop` kills all training processes and cleans up PID files.

### Starting Experiments

```bash
# Activate environment first
source ~/Desktop/env_isaaclab/bin/activate
cd /home/matteo/Desktop/code_projects/harold

# Start experiment with metadata (hypothesis-driven workflow)
python scripts/harold.py train --hypothesis "Lower threshold (10N) prevents elbow exploit" \
                               --tags "body_contact,elbow_fix"
python scripts/harold.py train --duration standard # Preset duration (see scripts/harold.py)
python scripts/harold.py train --checkpoint path   # Resume from checkpoint
python scripts/harold.py train --mode cpg          # CPG residual mode
python scripts/harold.py train --mode scripted     # Scripted gait (policy ignored)
python scripts/harold.py train --task rough        # Rough terrain task
```

### Autonomous Loop (Interactive for Hours)

Use this loop for long sessions so you keep control without flooding logs:

```bash
# Start a run (defaults handle envs/headless/video)
python scripts/harold.py train --hypothesis "…" --tags "…"

# Sleep 15-30 minutes
sleep 1200

# Check status + metrics
python scripts/harold.py status

# If clearly failing, stop early and validate
python scripts/harold.py stop
python scripts/harold.py validate
```

**Early stop rules** (after 10-15 min of data):
- SANITY fails → stop
- Height/contact failing AND vx negative → stop
- Standing passes but vx low → keep running
- If vx is noisy, use `x_displacement` to confirm real forward progress

**Scripted gait runs**: stop once the first video exists and metrics are clearly failing; additional time won’t improve a scripted trajectory.

### Monitoring & Analysis

```bash
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

### Video Location (Training)

Videos are saved under:
`logs/skrl/harold_direct/<run_id>/videos/train`

### Process Management

```bash
python scripts/harold.py ps                   # List all training processes
python scripts/harold.py stop                 # Stop all training and cleanup
```

### Status Output Example
```
RUN: 2025-12-22_14-30-00_ppo_torch (EXP-039)
HYPOTHESIS: Lower body contact threshold prevents elbow exploit
CONFIG: task=flat, mode=rl, duration=short
STATUS: RUNNING (45%, 1h 23m elapsed, 6.5 it/s, 8192 envs)
REWARD: 1234.5
SANITY: PASS (ep_len=342)
STANDING: PASS (height=0.72, contact=-0.02)
WALKING: WARN (vx=0.005, need >0.01)
DISPLACEMENT: x=0.02 (|x|=0.02)
VERDICT: STANDING
DIAGNOSIS: Upright and stable, forward velocity 0.005 m/s
```

The STATUS line shows: progress %, elapsed time, iterations/second, and environment count.
Example values above are illustrative; use `harold status` for current metrics.

### Exit Codes
- `0` = All metrics pass (robot walking)
- `1` = Partial (standing but not walking)
- `2` = Failing (on elbows, fallen)
- `3` = Sanity failure (episodes too short)
- `4` = Not running / no data

### Manifest System
Each experiment gets a `manifest.json` with:
- `alias`: EXP-NNN identifier
- `hypothesis`: What you're testing
- `tags`: For filtering in `harold compare --tag`
- `training_config`: `{task, mode, duration, num_envs, iterations, gait_scale}` for status display
- `summary.final`: Cached final metrics
- `summary.verdict`: WALKING/STANDING/FAILING/etc.
- `notes`: Timestamped observations

Global index at `logs/skrl/harold_direct/experiments_index.json` maps EXP-NNN aliases to directories.

### JSON Output (`harold status --json`)
Machine-readable output includes:
- `running`, `pid`, `elapsed_seconds`: Process state
- `progress`, `current_iteration`, `total_iterations`: Training progress
- `iterations_per_second`: Current training rate
- `training_config`: `{task, mode, duration, num_envs, iterations, gait_scale}` from manifest
- `metrics`: All TensorBoard metrics
- `status`, `diagnosis`, `exit_code`: Validation result
- `killed_by_watchdog`: Memory watchdog kill info (if applicable)

---

## Low-Level Commands (manual/legacy, avoid for agents)

```bash
# Direct training (verbose output - avoid in agents)
# NOTE: --video is MANDATORY, never omit it
python harold_isaac_lab/scripts/skrl/train.py \
  --task=Template-Harold-Direct-flat-terrain-v0 \
  --num_envs <num_envs> --headless --video --video_length <frames> --video_interval <steps>

# Play/evaluate a checkpoint
python harold_isaac_lab/scripts/skrl/play.py \
  --task=Template-Harold-Direct-flat-terrain-v0 \
  --checkpoint=<path_to_checkpoint.pt>

# TensorBoard monitoring
python3 -m tensorboard.main --logdir logs/skrl/harold_direct/ --bind_all
```

---

## 5-Metric Validation Protocol

Use `docs/memory/OBSERVABILITY.md` for current thresholds and failure signatures.

Check metrics in order:
- `episode_length` (sanity)
- `upright_mean`
- `height_reward`
- `body_contact_penalty`
- `vx_w_mean`

If sanity fails, ignore the rest.

Per-foot gait diagnostics are logged under `Episode_Metric/*` (contact ratio, peak force, air time, slip, x displacement).

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
- **Control/sim rates**: Defined in env config (`*_env_cfg.py`); do not assume fixed values.
- **Observation**: Contents and size vary by mode; see env config.
- **Action**: Joint position deltas; scaling/clamps defined in env config.
- **Actuators**: Implicit PD with gains/limits defined per task (`*/harold.py`).

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
Robot falls forward onto elbows with back elevated. Passes `upright_mean > 0.9` but fails `height_reward < 0.5`. **Always check height_reward.**

### Height Termination Bug
If using height-based termination, check height above terrain, NOT world Z coordinate. Spawn pose must be above threshold.

### Context Overflow
Long training runs flood context with tqdm output. Always use `python scripts/harold.py train` which runs training in background.

---

## Simulation Settings & Memory Safety

Defaults for env count, headless/video, and duration presets live in `scripts/harold.py`. Use the CLI and avoid hard-coded overrides unless debugging.

### Memory Watchdog
Training starts a watchdog to prevent OOM-induced system hangs. Thresholds and behavior are defined in `scripts/memory_watchdog.py` and surfaced in `harold status`.

### Training Duration Presets
Use `python scripts/harold.py train --duration <preset>`; presets are defined in `scripts/harold.py`.

### Actuator Overrides (Diagnostics)

```bash
HAROLD_ACTUATOR_STIFFNESS=<stiffness> HAROLD_ACTUATOR_DAMPING=<damping> HAROLD_ACTUATOR_EFFORT_LIMIT=<limit> \
python scripts/harold.py train --hypothesis "…" --tags "actuator_sweep"
```

Use overrides for responsiveness tests; keep within `docs/memory/HARDWARE_CONSTRAINTS.md`.

### Gait Overrides (Diagnostics)

```bash
python scripts/harold.py train --mode scripted --gait-scale <scale> --hypothesis "…" --tags "gait_scale"
```

Only use for diagnostics; keep defaults aligned to hardware.

---

## Hardware Constraints (Sim-to-Real)

Use `docs/memory/HARDWARE_CONSTRAINTS.md` as the source of truth for safe ranges, must-preserve values, and hardware specs.

---

## Hardware Session Logs (RPi Telemetry)

Telemetry logging details and analysis tips live in `docs/memory/OBSERVATIONS.md` (Hardware Session Logs section).

---

## Technical Reference

Full technical references live in:
- `docs/reference/sim_reference.md`
- `docs/reference/hardware_reference.md`
