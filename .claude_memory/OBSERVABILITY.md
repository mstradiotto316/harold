# Harold Observability Protocol

## Design Philosophy

This observability system follows Ousterhout's "A Philosophy of Software Design":
- **Deep Module**: `harold.py` hides TensorBoard complexity behind simple commands
- **Information Hiding**: Agents never see event file paths or metric tag names
- **Define Errors Out of Existence**: Missing manifests auto-generated; partial runs return useful data
- **State-Only Reporting**: Reports facts without prescriptive suggestions

---

## Harold CLI Commands

```bash
source ~/Desktop/env_isaaclab/bin/activate
cd /home/matteo/Desktop/code_projects/harold

# BEFORE starting: always check for orphan processes
python scripts/harold.py ps                  # List processes (shows [ORPHAN] if any)
python scripts/harold.py stop                # Kill all and cleanup (if needed)

# Start experiment with metadata (hypothesis-driven workflow)
python scripts/harold.py train --hypothesis "Lower threshold (10N) prevents elbow exploit" \
                               --tags "body_contact,elbow_fix"

# Check current status (shows progress %, elapsed time, it/s, envs)
python scripts/harold.py status              # Current run with metrics
python scripts/harold.py status --json       # Machine-readable output

# Validate a run (by alias, directory name, or path)
python scripts/harold.py validate            # Latest run
python scripts/harold.py validate EXP-034    # By experiment alias
python scripts/harold.py validate <run_id>   # By directory name

# List recent runs
python scripts/harold.py runs                # Last 10 with status
python scripts/harold.py runs --hypothesis   # Include hypothesis for each

# Compare experiments side-by-side (essential for hypothesis-driven work)
python scripts/harold.py compare EXP-034 EXP-035  # Specific experiments
python scripts/harold.py compare                  # Last 5 experiments
python scripts/harold.py compare --tag forward_motion  # All with tag

# Add observations to experiments
python scripts/harold.py note EXP-034 "Robot walked at 40-80% then regressed"
```

**Process Safety**: The harness blocks concurrent training, but orphans can exist after crashes.

**Exit codes:**
- `0`: All metrics pass (robot walking)
- `1`: Partial (standing but not walking)
- `2`: Failing (on elbows, fallen)
- `3`: Sanity failure (episodes too short)
- `4`: Not running / no data

---

## 5-Metric Protocol (Check in Order!)

| Priority | Metric | Threshold | Failure Mode |
|----------|--------|-----------|--------------|
| **1. SANITY** | `episode_length` | **> 100** | Robots dying immediately (BUG!) |
| 2. Stability | `upright_mean` | > 0.9 | Falling over |
| 3. Height | `height_reward` | > 1.2 | On elbows/collapsed |
| 4. Contact | `body_contact` | > -0.1 | Body on ground |
| 5. Walking | `vx_w_mean` | > 0.1 m/s | Not walking |

**CRITICAL**: If metric #1 (episode length) fails, ALL OTHER METRICS ARE INVALID.

Note: Height threshold was reduced from 2.0 to 1.2 based on observations that 2.0 was too strict for walking.

---

## Manifest System

Each experiment gets a `manifest.json` file (auto-generated on first access):

```json
{
  "id": "2025-12-21_21-38-22_ppo_torch",
  "alias": "EXP-034",
  "hypothesis": "Height-dominant rewards will prevent elbow exploit",
  "tags": ["height_reward", "baseline"],
  "started_at": "2025-12-21T21:38:22Z",
  "status": "completed",
  "notes": [
    {"timestamp": "2025-12-22T01:00:00Z", "text": "Robot fell forward onto elbows"}
  ],
  "summary": {
    "final": {"episode_length": 370.5, "vx_w_mean": -0.006, ...},
    "verdict": "FAILING"
  }
}
```

Global index at `logs/skrl/harold_direct/experiments_index.json` maps aliases to directories.

---

## Failure Mode Signatures

| State | ep_len | upright | height_rew | body_contact | vx |
|-------|--------|---------|------------|--------------|-----|
| **BUG: Dying instantly** | **< 100** | ? | ? | ? | ? |
| Standing correctly | > 300 | > 0.9 | > 1.2 | > -0.1 | ~0 |
| **On elbows (TRAP!)** | > 100 | > 0.9 | **< 1.0** | **< -0.2** | ~0 or + |
| Walking | > 300 | > 0.9 | > 1.2 | > -0.1 | > 0.1 |
| Fallen sideways | varies | < 0.7 | low | varies | ~0 |

---

## Hypothesis-Driven Workflow

```bash
# 1. Form hypothesis and run experiment
harold train --hypothesis "Lower body contact threshold (10N) prevents elbow exploit" \
             --tags "body_contact,elbow_fix"

# 2. Check progress mid-training
harold status

# 3. After completion, add observations
harold note EXP-039 "Episodes terminated correctly but robot still falls forward initially"

# 4. Compare with previous attempts
harold compare EXP-037 EXP-038 EXP-039

# 5. Interpret results and form next hypothesis...
```

---

## Known Failure Patterns

### The Height Termination Bug (2025-12-21)

**Symptom**: Episode length = 3 steps

**Root cause**: Height termination checked `root_pos_w[:, 2]` (world Z) instead of height above terrain.

**Lesson**: Always check episode_length FIRST before analyzing other metrics.

### The "On Elbows" Exploit (2025-12-20)

**Symptom**: upright_mean > 0.9, vx_w_mean > 0, but height_reward < 1.0

**Root cause**: Robot falls forward onto elbows. Back is elevated (passes upright), but not at standing height.

**Lesson**: Always check height_reward. If < 1.2, robot is not standing properly.

---

## Last Updated

2025-12-23 - CLI Enhancement & Documentation Update
- Added `harold ps` and `harold stop` for process management
- Added orphan process detection and cleanup
- STATUS line now shows it/s and environment count
- JSON output includes progress, current_iteration, iterations_per_second
- Training config (num_envs, iterations) stored in manifest
- Height threshold updated from 2.0 to 1.2

2025-12-22 - Observability System Enhancement
- Added hypothesis and tag support to experiments
- Added `harold compare` for side-by-side experiment comparison
- Added `harold note` for experiment observations
- Converted to state-only reporting (removed NEXT: suggestions)
- Added manifest.json caching for fast metric access
- Added experiment aliases (EXP-001, EXP-002, ...)
