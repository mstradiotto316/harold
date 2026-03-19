# Harold Coordinate Frame Investigation

## Status: RESOLVED — No Bug Found

Investigation on 2026-03-18/19 determined that the original identity quaternion setup was correct. A 180° Z rotation was temporarily applied (EXP-429) based on incorrect analysis, then reverted after diagnostic testing proved the original setup was sound.

## Timeline

1. **Pre-EXP-429**: Robot trained with identity quaternion `(1,0,0,0)`. Performance appeared poor — robot seemed to walk backward in some video views.
2. **EXP-429**: 180° Z rotation applied based on URDF joint naming analysis (FL/FR at -X in URDF → assumed body +X pointed backward). Performance appeared to improve.
3. **2026-03-18 diagnostic**: Velocity push test (+2.0 m/s world +X) showed `vx_b = +vx_w` with identity quaternion — body +X *does* align with world +X. The URDF analysis was wrong because the USD file (converted from URDF) has FL/FR at +X, not -X.
4. **2026-03-19 reward diagnostic**: Logged `forward_motion` and `track_lin_vel_xy` rewards alongside velocity during the push test. Rewards correctly increase with positive body-frame velocity. No misalignment found.
5. **Resolution**: Reverted all quaternion rotations to identity. The performance difference attributed to the rotation was likely coincidental (other reward/config changes were made simultaneously).

## What Was Investigated

### USD Asset Inspection
Used `pxr.Usd` API from Isaac Sim packages to inspect `part_files/V4/harold_8.usd`:
- FL/FR joints are at +X in the USD (opposite from URDF convention)
- All `orient` values are identity — no hidden rotations baked in
- The URDF-to-USD conversion flipped the joint positions but this is consistent — body +X still aligns with visual forward

### Diagnostic Test Results
With identity quaternion, pushing env 0 at +2.0 m/s in world +X:
```
step=  1 | vx_w=+2.002 vx_b=+1.994 | cmd_vx=0.194 | fwd_reward=+9.966 track_reward=+0.000
step=  3 | vx_w=+1.533 vx_b=+1.530 | cmd_vx=0.194 | fwd_reward=+7.649 track_reward=+0.000
step= 24 | vx_w=+0.274 vx_b=+0.272 | cmd_vx=0.297 | fwd_reward=+1.360 track_reward=+3.590
```
- `vx_b ≈ vx_w` (identity quaternion, no yaw rotation)
- `forward_motion` reward is positive when vx_b is positive (correct)
- `track_lin_vel_xy` peaks when velocity matches command (correct)

## Key Facts

- **Isaac Lab uses (w, x, y, z) quaternion format.** Identity = `(1.0, 0.0, 0.0, 0.0)`.
- **Body-frame velocity** (`root_lin_vel_b`) is computed via `quat_apply_inverse(root_quat_w, root_lin_vel_w)`.
- **Commands are body-frame.** `cmd_vx` range: [0.0, 0.3] m/s. Both commands and rewards use body-frame velocity, making the reward system rotation-invariant.
- **With identity quaternion**: body +X = world +X = robot visual forward. Positive `vx_b` = moving forward = positive reward. Everything aligns.

## Files Changed During This Investigation

| File | Change |
|---|---|
| `harold_flat/harold.py` | Kept at identity quaternion `(1,0,0,0)`, cleaned up diagnostic comment |
| `harold_rough/harold.py` | Reverted from 180° Z rotation back to identity quaternion |
| `harold_flat/train_env.py` | Removed diagnostic velocity push and reward logging code |
| `deployment/inference/observation_builder.py` | Updated comment — no sign flip needed with identity quat |
| `common/multi_camera_video.py` | Fixed axis overlay gizmos to show robot-relative directions |

## Camera Axis Overlays

During this investigation, the camera axis overlay gizmos in `multi_camera_video.py` were corrected to show **robot-relative** directions (not camera-relative). The overlays now show where the robot's own +X (forward), +Y (left), and +Z (up) project onto each camera view.
