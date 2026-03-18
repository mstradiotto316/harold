# Video Annotation Skill

Automated analysis of RL training video footage from Isaac Sim using Claude's multimodal capabilities and the 1M token context window.

## Purpose

Training runs produce video recordings at `logs/skrl/harold_direct/<run_id>/videos/train/`. Each recording step produces **4 separate video files**, one per camera angle:

- `rl-video-step-<N>-side.mp4` — Sagittal plane (gait cycle, pitch, leg kinematics)
- `rl-video-step-<N>-front.mp4` — Coronal plane (roll, lateral stability, leg spread)
- `rl-video-step-<N>-top.mp4` — Dorsal/transverse plane (foot placement, yaw, heading)
- `rl-video-step-<N>-iso.mp4` — Isometric 3/4 view (overall 3D pose context)

This skill enables Claude to extract frames, load them into context, and produce structured annotations covering gait quality, stability, failure modes, and training progress.

## How It Works

1. **Extract frames** from the `.mp4` using `ffmpeg`
2. **Load frames** into the context window as images (via the Read tool)
3. **Analyze** the full sequence for locomotion quality metrics
4. **Output** a structured annotation report

## Frame Sampling Strategy

Videos are typically recorded at 20fps, 960x540, H.264 — one file per camera angle (4 cameras total). At ~1,200 tokens per frame, a 12-second clip from all 4 cameras (4 × ~251 frames) would consume ~1.2M tokens at native rate. Sampling recommendations:

| Scenario | Sample Rate | Frames (12s clip) | Token Budget |
|----------|-------------|-------------------|--------------|
| Quick check | 2fps | ~24 | ~38K |
| Standard analysis | 5fps | ~60 | ~96K |
| Detailed gait analysis | 10fps | ~120 | ~192K |
| Full frame-by-frame | 20fps (native) | ~251 | ~400K |

**Recommendation:** 2-5fps is sufficient for most locomotion analysis. Consecutive frames at 20fps are nearly identical. Reserve full rate for investigating rapid state changes (falls, collisions, episode resets).

### Efficiency Techniques

- **Contact sheets**: Use `ffmpeg` to tile frames into grids (e.g., 4x4 montages), reducing the number of image reads while preserving temporal coverage.
- **Frame stamping**: Burn frame numbers/timestamps into images before loading with `ffmpeg drawtext` for easier reference.
- **Adaptive sampling**: Extract at 2fps first, then re-extract at full rate around interesting events (falls, gait transitions).

### ffmpeg Commands

```bash
# Preferred: use the harold CLI to extract frames from all cameras
python scripts/harold.py frames --json

# Manual extraction from a single camera view
ffmpeg -y -i rl-video-step-3200-side.mp4 -vf "fps=2" -q:v 2 /tmp/frames/side/frame_%04d.jpg

# Extract with frame number overlay
ffmpeg -y -i rl-video-step-3200-side.mp4 -vf "fps=5,drawtext=text='%{frame_num}':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5" -q:v 2 /tmp/frames/frame_%04d.jpg

# Create 4x4 contact sheet montages (16 frames per image)
ffmpeg -y -i rl-video-step-3200-side.mp4 -vf "fps=2,tile=4x4" -q:v 2 /tmp/montage_%03d.jpg
```

## Annotation Schema

The skill produces analysis covering these categories:

### 1. Stability Assessment
- Does the robot stay upright for the full episode?
- Any stumbles, falls, or near-falls?
- Episode termination reason (timeout vs. fall vs. recording cutoff)

### 2. Gait Classification
- **Gait type**: trot, walk, gallop, crawl, bound, or unclassified
- **Gait regularity**: periodic vs. irregular vs. chaotic
- **Stride period**: estimated frames per full gait cycle

### 3. Body Posture
- **Pitch**: forward lean, backward lean, or level
- **Roll**: lateral tilt or level
- **Height**: consistent, bouncing, or sinking over time
- **Yaw drift**: heading deviation from straight-line travel

### 4. Leg Kinematics
- Front vs. rear leg amplitude balance
- Left vs. right symmetry
- Ground clearance during swing phase
- Foot slip indicators (feet dragging)

### 5. Training Progress Indicators
- Comparison to expected behavior at this training step
- Signs of reward hacking or degenerate policies
- Qualitative "grade" (failing / standing / walking / running)

### 6. Notable Events (with frame numbers)
- Falls or recovery attempts
- Gait transitions
- Episode resets
- Contact with environment boundaries

## Example Output Format

```
## Video Analysis: rl-video-step-3200 (4 cameras)
Run: 2026-01-02_23-47-27_ppo_torch | Step: 3200

### Summary
Stable trot gait maintained for full 12.5s episode. No falls.
Grade: WALKING (early-stage)

### Stability: PASS
- Upright throughout, no stumbles
- Episode ends at recording cutoff (not a fall)

### Gait: TROT
- Diagonal pair coordination (FL+RR, FR+RL)
- Stride period: ~10-12 frames (0.5-0.6s)
- Regularity: periodic, slightly stiff

### Posture
- Pitch: slight forward lean (consistent)
- Roll: minimal oscillation (normal for trot)
- Height: stable, no drift
- Yaw: mild rightward drift

### Legs
- Front legs: aggressive swing, good clearance
- Rear legs: shorter stroke, stiffer
- L/R symmetry: minor asymmetry
- No foot dragging observed

### Events
- None (steady-state walking throughout)

### Training Assessment
At step 3200, maintaining a stable trot without falling is a positive
signal. Areas to watch as training continues: gait smoothness,
front/rear balance, heading control.
```

## Visual Conventions

Training videos include built-in visual annotations to aid analysis:

### HUD Overlay (side and iso views only)
A text strip in the bottom-left corner shows:
- **F:** Frame number within this recording clip (0-249)
- **S:** Training step at which this recording started
- **R:** Episode reset count — how many times env 0 has reset during this clip

### Reset Flash
When an episode reset occurs (termination or timeout), all camera views display a **red border with a "RESET" label** for that frame. This makes resets unambiguous even when the robot hasn't moved far from its spawn position (e.g., during early training death loops).

When analyzing videos, count the "RESET" frames to determine how many episodes occurred within the clip. A high reset count in a short clip indicates frequent termination (unstable policy). The R counter in the HUD also tracks this.

### Axis Orientation Widget
Each camera view displays a small axis indicator in the bottom-right corner showing how the world coordinate axes project into that view. The color convention is:
- **Red:** +X (forward direction of travel)
- **Green:** +Y (leftward)
- **Blue:** +Z (upward)

Use this to determine which direction the robot is moving, leaning, or drifting in each camera view.

### Scene Colors
- **Robot:** Orange (high contrast)
- **Ground:** Dark grey
- **Background:** Neighboring environment robots are hidden — only the primary robot (env 0) is visible

## Video File Locations

Training videos follow this path convention:
```
logs/skrl/harold_direct/<run_id>/videos/train/rl-video-step-<N>-<camera>.mp4
```

Where:
- `<run_id>` is a timestamped directory (e.g., `2026-01-02_23-47-27_ppo_torch`) or a named run (e.g., `terrain_57`)
- `<N>` is the training step at which the video was recorded
- `<camera>` is one of: `side`, `front`, `top`, `iso`
- Videos are recorded at intervals configured by `--video_interval` (default: every 3200 steps)

## Comparing Across Training Steps

A powerful use case is analyzing videos from multiple training steps in the same run to observe policy evolution:

```
rl-video-step-0-side.mp4       # Random policy (expect chaos)
rl-video-step-6400-side.mp4    # Early training
rl-video-step-32000-side.mp4   # Mid training
rl-video-step-96000-side.mp4   # Late training
# (also -front.mp4, -top.mp4, -iso.mp4 for each step)
```

For cross-step comparison, extract 1-2 keyframes from each video and load them side by side, annotating the progression.

## Relationship to Existing Observability

This skill complements (does not replace) the existing metrics pipeline:

| Source | What It Tells You |
|--------|-------------------|
| `harold status` / TensorBoard | Quantitative metrics (reward, vx, height, contact) |
| Video annotation | Qualitative behavior (gait style, posture, failure mode) |
| Hardware telemetry | Real-world servo/IMU data |

Metrics can tell you *that* the robot is walking at 0.3 m/s. Video annotation tells you *how* -- whether the gait is natural, whether legs are dragging, whether the body is wobbling.
