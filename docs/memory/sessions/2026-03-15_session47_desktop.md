# 2026-03-15 Session 47 (desktop)

## Summary
- Addressed three follow-up review regressions from the audit-remediation patch set.
- Restored explicit Isaac Lab interpreter selection for background training launch.
- Fixed repo-relative checkpoint resolution in the ONNX quick validator.
- Fixed autoresearch backfill timestamps to use current manifest fields.

## Fixes
- `scripts/harold.py`
  - `build_train_command()` now launches training with `~/Desktop/env_isaaclab/bin/python` instead of the caller's `sys.executable`.
  - `start_watchdog()` now uses the same explicit Isaac Lab interpreter for consistency with the training process.
- `deployment/validation/validate_onnx_quick.py`
  - `resolve_checkpoint_path()` now resolves relative `checkpoint_path` metadata against the repo root, so the documented `cd deployment/validation && python validate_onnx_quick.py` workflow works.
- `scripts/autoresearch.py`
  - `backfill_results()` now uses `manifest["started_at"]` and falls back to `created` only for older manifests.

## Verification
- `python3 -m py_compile scripts/harold.py deployment/validation/validate_onnx_quick.py scripts/autoresearch.py`
- Verified `build_train_command(...)[0] == /home/matteo/Desktop/env_isaaclab/bin/python`
- Verified `resolve_checkpoint_path()` from `deployment/validation/` returns the correct repo-root checkpoint path
- Verified timestamp fallback logic prefers `started_at` and still accepts legacy `created`

## Notes
- No experiments or hardware tests were run in this follow-up session.
- The remaining post-audit work is unchanged: default-8192 smoke and legacy 50D debug-script cleanup are still pending.
