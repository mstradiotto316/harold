# Autoresearch Agent Protocol

Step-by-step instructions for Claude Code when running autonomous experiment sessions.
Inspired by Karpathy's autoresearch, adapted for robotics RL.

**Key principle**: The human edits `strategy.md`, the agent edits config files.

---

## Session Initialization (Once)

1. Read `docs/autoresearch/strategy.md` for goals and constraints
2. Read `docs/autoresearch/PARAMETER_REGISTRY.md` for what can be changed
3. Run `python scripts/autoresearch.py history` to see prior results
4. Read `docs/memory/OBSERVATIONS.md` for accumulated insights
5. Run `python scripts/harold.py ps` to check for orphan processes
6. Create a branch: `git checkout -b autoresearch/session-YYYY-MM-DD`
7. If no baseline score exists in results.tsv, run a baseline experiment first with the current config unchanged

---

## Each Experiment Iteration

### Step 1: Hypothesize

Based on strategy.md goals and prior results, propose ONE parameter change.

- Identify the highest-priority unmet goal from strategy.md
- Review what has been tried (from results.tsv)
- Write a specific, falsifiable hypothesis: "Changing X from A to B will improve Y because Z"

### Step 2: Validate the Change

```bash
python scripts/autoresearch.py load-registry
```

Check that the parameter is not FROZEN and the proposed value is within range.
Also check strategy.md "Do Not Change" list.

### Step 3: Apply Config Change

```bash
python scripts/autoresearch.py apply '{"parameter_name": new_value}'
```

Then commit:
```bash
git add <modified files>
git commit -m "autoresearch: <hypothesis summary>"
```

### Step 4: Train

```bash
python scripts/harold.py train \
  --hypothesis "<hypothesis text>" \
  --tags "autoresearch,<specific_tags>" \
  --duration <from strategy.md>
```

### Step 5: Wait and Monitor

Check at intervals using:
```bash
python scripts/harold.py status --json
```

**Monitoring schedule:**
- short (30 min): check at 10 min, then every 5 min
- standard (60 min): check at 15 min, then every 10 min

**Early stop rules:**
- SANITY_FAIL after 10 min of data → `harold stop`, score=0, DISCARD
- Height FAIL + negative vx after 15 min → `harold stop`, DISCARD
- Watchdog killed the process → score=0, DISCARD

### Step 6: Evaluate (Quantitative)

```bash
python scripts/harold.py validate
python scripts/harold.py status --json
```

Extract the 5 metrics: episode_length, upright_mean, height_reward, body_contact, vx_w_mean.
Record the verdict: WALKING / STANDING / FAILING / SANITY_FAIL.

Compute quantitative score:
```bash
python scripts/autoresearch.py score '{"episode_length": 450, "upright_mean": 0.96, "height_reward": 0.65, "body_contact": -0.01, "vx_w_mean": 0.018}'
```

### Step 7: Evaluate (Qualitative - Video)

Find the latest training video:
```bash
ls logs/skrl/harold_direct/<run_id>/videos/train/ | sort | tail -1
```

Extract montage frames (16 frames per image at 2fps):
```bash
ffmpeg -y -i <video.mp4> \
  -vf "fps=2,drawtext=text='%{frame_num}':x=10:y=10:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.5,tile=4x4" \
  -q:v 2 /tmp/harold_montage_%03d.jpg
```

Load 1-3 montage images and analyze per `docs/skills/video_annotation.md`.

Map to qualitative score (0-100):
- 80-100: Stable trot/walk, regular stepping, good clearance
- 60-80: Walking with minor issues (asymmetry, slight drag)
- 30-60: Standing with occasional stepping attempts
- 10-30: Degenerate (elbows, shuffling, frozen)
- 0-10: Falling, chaotic, no useful behavior

### Step 8: Compute Composite Score

```
composite = 0.7 * quantitative_score + 0.3 * qualitative_score
```

If video analysis was skipped (extraction failed), use quantitative_score only.

### Step 9: Keep or Discard

**KEEP** if `composite > baseline_composite + 2.0`:
1. Config change stays in git (commit from Step 3 is preserved)
2. Update baseline score to the new composite
3. Commit: `git commit --allow-empty -m "autoresearch: KEEP EXP-NNN (score=XX.X, VERDICT)"`

**DISCARD** if score did not improve:
1. Revert config: `python scripts/autoresearch.py revert`
2. Stage and commit: `git add -A && git commit -m "autoresearch: DISCARD EXP-NNN (score=XX.X) - revert config"`

**Edge case**: If score is within 2 points of baseline but video clearly shows better behavior (e.g., stepping where before there was none), the agent may override with explicit justification.

### Step 10: Log Result

```bash
python scripts/autoresearch.py log '{
  "exp_alias": "EXP-228",
  "timestamp": "2026-03-15T01:30:00Z",
  "hypothesis": "Increase feet_air_time_weight to 2.0",
  "changed_params": "feet_air_time_weight:1.0->2.0",
  "duration_min": "32",
  "verdict": "STANDING",
  "vx": "0.008",
  "upright": "0.97",
  "height": "0.65",
  "contact": "-0.01",
  "ep_len": "450",
  "quant_score": "42.3",
  "qual_score": "35.0",
  "composite": "40.1",
  "decision": "DISCARD",
  "gait_type": "standing",
  "video_notes": "No visible stepping, robot stands still"
}'
```

### Step 11: Update Memory (If KEEP)

If the KEEP result revealed a novel insight, update `docs/memory/OBSERVATIONS.md`.
Do NOT update memory for DISCARD results (they are negative results, logged in results.tsv).

### Step 12: Loop or End

- If `experiments_run < max_experiments` from strategy.md, go to Step 1
- If 3 consecutive DISCARDs, pause and reconsider strategy direction
- Otherwise, proceed to Session End

---

## Session End

1. Write session summary:
   ```
   docs/memory/sessions/YYYY-MM-DD_autoresearch.md
   ```
   Include: number of experiments, kept/discarded ratio, best score, key findings.

2. Update `docs/memory/EXPERIMENTS.md` with a summary block.

3. Commit: `git commit -m "autoresearch: session summary (N experiments, M kept)"`

---

## NEVER STOP Rules (from Karpathy)

Once the experiment loop begins, do NOT pause to ask the human if you should continue.
The human may be away or asleep. You are autonomous.

Exceptions:
- Strategy.md says to stop after N experiments → respect the limit
- 3+ consecutive DISCARDs → reassess (but don't ask, just adjust strategy)
- Infrastructure failure (GPU crash, disk full) → stop and document

---

## Quick Reference

| Action | Command |
|--------|---------|
| Check config | `python scripts/harold.py snapshot-config` |
| Check registry | `python scripts/autoresearch.py load-registry` |
| Apply change | `python scripts/autoresearch.py apply '{"param": value}'` |
| Revert change | `python scripts/autoresearch.py revert` |
| Start training | `python scripts/harold.py train --hypothesis "..." --tags "autoresearch,..."` |
| Check status | `python scripts/harold.py status --json` |
| Validate | `python scripts/harold.py validate` |
| Score | `python scripts/autoresearch.py score '{"metrics": ...}'` |
| Log result | `python scripts/autoresearch.py log '{"entry": ...}'` |
| View history | `python scripts/autoresearch.py history` |
