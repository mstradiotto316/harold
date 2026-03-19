# CLAUDE.md

Project-level instructions for Claude Code sessions in this repository.

## Quick Start

- Primary agent guide: `AGENTS.md`
- Documentation index: `docs/index.md`
- Memory system: `docs/memory/CONTEXT.md` (read first at session start)

## Key Rules

- **Video is mandatory** for all training runs (`--video` flag).
- **Don't commit without asking first** — EXCEPT during `/autoresearch` sessions, where commits are part of the autonomous loop.
- Use `python scripts/harold.py` for all training/monitoring -- no ad-hoc scripts.
- Read `docs/memory/HARDWARE_CONSTRAINTS.md` before changing sim parameters.

## Known Issues

- **Coordinate frame bug** (PATCHED): The USD asset has a baked-in axis flip that misaligns body-frame +X with the robot's visual forward direction. Currently patched with a 180° Z rotation on the spawn quaternion. See `COORDINATE_FRAME_BUG.md` for full details before touching robot orientation, body-frame velocities, or the USD asset.

## Autoresearch (Autonomous Experimentation)

Harold has an autoresearch system for running RL experiments autonomously, inspired by Karpathy's autoresearch.

- **Program doc** (single source of truth): `docs/autoresearch/program.md`
- **Helper script**: `scripts/autoresearch.py` (apply/revert/score/log/backfill)
- **Results log**: `docs/autoresearch/results.tsv` (gitignored, append-only)

To start an autoresearch session, run `/autoresearch` or read `program.md` and follow the loop.

## Skills

### Video Annotation (`docs/skills/video_annotation.md`)

Analyze RL training videos frame-by-frame using multimodal vision. Extract frames with `ffmpeg`, load into context, and produce structured annotations covering gait type, stability, body posture, leg kinematics, and training progress.

- Videos live at: `logs/skrl/harold_direct/<run_id>/videos/train/rl-video-step-<N>.mp4`
- Typical format: 20fps, 1280x720, H.264, ~12s per clip
- Sample at 2-5fps for standard analysis (20fps is wasteful for most cases)
- See the full skill doc for ffmpeg commands, annotation schema, and output format
