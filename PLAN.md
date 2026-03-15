# Simulation Audit Remediation Plan

Date: 2026-03-15
Scope: Isaac Lab simulation, training harness, and sim-to-real export/deployment interfaces.

## Objective

Fix the correctness, realism, and tooling issues identified in the March 2026 audit before trusting new training results or using new policies for hardware work.

## Execution Rules

- Preserve the single-interface workflow: use `python scripts/harold.py` for training and monitoring.
- Keep video recording enabled for every training run.
- Do not change must-preserve sim-to-real constraints in `docs/memory/HARDWARE_CONSTRAINTS.md` without explicit approval.
- Treat flat-task correctness fixes and export/deployment consistency as blockers for future policy work.

## Phase 1: Correctness Blockers In The Flat Task

### 1. Align reward and telemetry frames with the observation frame

Issue:
- The flat task observes body-frame velocity but computes velocity tracking reward and command-error metrics in world frame.

Changes:
- In `harold_isaac_lab/.../harold_flat/train_env.py`, switch flat-task velocity tracking and command-error telemetry to body-frame `root_lin_vel_b`.
- Audit any remaining world-frame use in the flat task and keep world-frame metrics only when they are explicitly diagnostic, not part of the learning signal.
- Update comments to state the chosen frame unambiguously.

Acceptance:
- Observation, reward, and command telemetry all use the same frame convention.
- A targeted test or assertion confirms body-frame commands are compared to body-frame velocity.

### 2. Remove forward reward leakage for fallen states

Issue:
- `forward_motion` still gives positive reward to low-upright states because upright is clamped to a minimum of `0.5`.

Changes:
- Replace the current gate with a stricter one that drives reward to zero or negative for clearly bad posture.
- Consider gating on a combination of upright, terrain-relative height, and undesired body contact so the reward cannot be farmed by diving forward.
- Keep the logic simple enough to reason about from logged metrics.

Acceptance:
- Fallen or elbow-contact states cannot receive positive forward reward.
- A regression test covers at least one low-upright case and one healthy-upright case.

### 3. Fix observation clipping semantics

Issue:
- The flat config says clipping matches deployment `clip_obs=5.0`, but the env currently clips raw observations to `[-50, 50]`.

Changes:
- Decide on one of two valid designs and document it:
- Option A: remove env-side raw clipping entirely and rely on the skrl running-stat preprocessor during training.
- Option B: keep explicit env-side clipping, but rename/document it as raw-observation clipping rather than normalized clipping.
- Align deployment comments and validation tooling with the chosen design.

Acceptance:
- Code, comments, and deployment assumptions match.
- No remaining references claim normalized clipping is happening when it is not.

### 4. Reset the EMA action filter on episode reset

Issue:
- `_actions_smooth` is not reset, so each new episode inherits filtered action state from the prior episode.

Changes:
- Clear `_actions_smooth` in flat and rough `_reset_idx`.
- Verify any other stateful action buffers are reset consistently.

Acceptance:
- First-step targets after reset are not contaminated by the previous episode.

### 5. Fix multi-env policy logging

Issue:
- Optional policy logging casts the vector `_time` tensor to `float`, which fails for multi-env runs.

Changes:
- Log a scalar timestamp explicitly, e.g. env 0 time or per-entry environment-specific time.
- Make flat and rough logging behavior consistent.
- Keep the JSONL schema stable or version it if it changes.

Acceptance:
- `HAROLD_POLICY_LOG_DIR` works for `num_envs > 1`.
- A smoke run writes valid JSONL without runtime exceptions.

### 6. Fix flat-task termination telemetry

Issue:
- Flat-task reset logging reports all terminated episodes under `Episode_Termination/orientation`, even when height/contact/elbow logic caused the reset.

Changes:
- Split reset logging into explicit buckets for orientation, height, body contact, elbow pose, and timeout.
- Reuse the same boolean masks used by `_get_dones` so reporting matches actual reset reasons.

Acceptance:
- Termination counters reflect real reset causes.
- `harold status` and TensorBoard can be interpreted without guessing.

## Phase 2: Rough-Task Realism And Coverage

### 7. Make rough-task domain randomization real, or disable misleading defaults

Issue:
- Rough-task config enables reset-time randomization by default, but sampled friction/stiffness/damping/mass values are not applied to the simulator.

Changes:
- Either implement actual Isaac/PhysX property mutation for the supported randomization dimensions, or default those toggles to `False` until true application exists.
- Keep sensor noise separate from physics randomization so the active randomization path is obvious.
- Add logging for the effective randomized values that are actually applied.

Acceptance:
- If a randomization flag is `True`, simulator properties measurably change.
- If dynamic property mutation is not yet feasible, defaults no longer claim robustness that is not present.

### 8. Restore intended rough-terrain coverage

Issue:
- Rough terrain currently samples only levels `0` and `1`.

Changes:
- Decide whether the task should sample the full configured terrain range immediately or follow a curriculum schedule.
- Fix `max_init_terrain_level` usage and any off-by-one assumptions.
- Implement actual progression if curriculum is intended, or disable the curriculum claim if not.

Acceptance:
- Terrain sampling matches the documented design.
- A simple diagnostic log or histogram shows environments visiting the intended terrain levels.

### 9. Remove stale or misleading config comments

Issue:
- Several comments and docstrings no longer match behavior or defaults.

Changes:
- Audit flat, rough, pushup, and export/deployment comments touched by the fixes above.
- Update comments only where they materially improve operator understanding.

Acceptance:
- No known comment remains in conflict with current code for the audited paths.

## Phase 3: Export And Deployment Consistency

### 10. Migrate the export path fully to 48D

Issue:
- The current training/controller stack is 48D, but exporter and validation tools still hardcode 50D.

Changes:
- Update `policy/export_policy.py` and the related validation scripts to use 48D.
- Prefer deriving observation dimension from checkpoint stats or model metadata instead of hardcoding it again.
- Regenerate `deployment/policy/policy_metadata.json` from a current 48D checkpoint.

Acceptance:
- Exported TorchScript/ONNX artifacts carry 48D normalization stats.
- The controller accepts the regenerated metadata without the legacy-dimension failure.

### 11. Remove hardcoded stale action/default-pose assumptions from export tooling

Issue:
- Export-side metadata still hardcodes default joint positions and action scaling assumptions that can drift from the sim/deployment source of truth.

Changes:
- Source default pose and joint ranges from the canonical stance/config path used by training and deployment.
- Ensure export metadata reflects the actual training-time action scale and joint limits.

Acceptance:
- Export metadata and deployment conversion logic match the active training configuration.
- No stale 50D-era defaults remain in the export path.

## Phase 4: Verification

### 12. Static and unit-level checks

Run after each phase:
- `python3 -m py_compile` on edited Python files.
- `source ~/Desktop/env_isaaclab/bin/activate && python -m pytest deployment/tests/test_inference.py -q`
- Add targeted tests for frame alignment, reward gating, EMA reset, and export-dimension handling where practical.

### 13. Flat-task smoke validation

Run after Phase 1:
- Enable the Isaac Lab venv.
- Start a short flat-task smoke run through `python scripts/harold.py train --duration fast --hypothesis "audit phase1 smoke" --tags "audit,phase1"` and confirm video exists.
- Run one small multi-env logging smoke with `HAROLD_POLICY_LOG_DIR` set to verify JSONL capture.
- Use `python scripts/harold.py status` and `validate` to confirm metrics/termination telemetry are coherent.

Acceptance:
- Run starts cleanly, logs video, and no policy-log or reset-reporting exceptions occur.

### 14. Rough-task realism validation

Run after Phase 2:
- Add a rough-task smoke run through `python scripts/harold.py train --task rough --duration fast --hypothesis "audit phase2 smoke" --tags "audit,phase2,rough"`.
- Capture a terrain-level summary and the effective randomization values.
- Confirm the run uses the intended terrain span and only claims randomization that is truly active.

Acceptance:
- Terrain distribution and randomization behavior match the repaired design.

### 15. Export/deployment validation

Run after Phase 3:
- Export a fresh 48D policy artifact.
- Run the existing inference-side tests and a local controller initialization check against the regenerated metadata.
- Re-run any ONNX-vs-sim validation scripts after updating them to 48D expectations.

Acceptance:
- Export, controller load, and validation tooling all agree on 48D.

## Recommended Order Of Implementation

1. Phase 1 items 1, 2, 4, 5, and 6.
2. Phase 1 item 3, so the clipping story is clean before new training results are interpreted.
3. Phase 2 items 7 and 8.
4. Phase 3 items 10 and 11.
5. Phase 4 verification after each completed phase, not only at the end.

## Definition Of Done

- Flat-task reward/telemetry frame mismatch is removed.
- Fallen states no longer earn forward reward.
- Reset state and optional policy logging are stable.
- Rough-task terrain sampling and domain-randomization claims reflect reality.
- Exported policy artifacts, metadata, and controller all use 48D consistently.
- Comments/docs in the audited paths are synchronized with the code.
- Smoke runs and lightweight tests pass after each phase.
