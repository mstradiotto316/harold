# Harold Coordinate Frame Bug

## Status: PATCHED (not fully resolved)

The robot's body-frame coordinate system is misaligned with its visual mesh orientation. This document explains the bug, the current patch, and what needs to happen for a proper fix.

## The Problem in Plain English

When Harold stands in the simulator facing to the right (+X direction), his internal coordinate system thinks "forward" is to the LEFT (-X direction). So when the reward function says "good job moving forward," it's actually rewarding the robot for walking backward.

## Technical Details

### What We Know

1. **The URDF joint naming**: FL/FR ("front-left"/"front-right") shoulder joints are at URDF -X. BL/BR ("back-left"/"back-right") are at URDF +X. However, the robot's visual front (the end that looks like the "head") is at URDF +X — the BL/BR end.

2. **The USD asset** (`part_files/V4/harold_8.usd`): There appears to be a coordinate frame transformation baked into the USD file (binary USDC format, not human-readable) that causes body-frame +X to map to world -X motion, even with an identity quaternion. This hidden transform is the root cause.

3. **Isaac Lab convention**: Uses (w, x, y, z) quaternion format. Identity = `(1.0, 0.0, 0.0, 0.0)`. Body-frame velocity (`root_lin_vel_b`) is computed via `quat_apply_inverse(root_quat_w, root_lin_vel_w)`.

### Observed Behavior

| Quaternion | Robot Faces | Reward Direction | Forward Walking? |
|---|---|---|---|
| Identity `(1,0,0,0)` | +X (correct) | -X (wrong!) | No — trained backward |
| 180° Z `(0,0,0,1)` | -X (wrong) | -X (now correct!) | Yes — but faces wrong way |

The reward function uses `vx_b` (body-frame X velocity) to reward forward motion. Due to the USD coordinate flip, positive `vx_b` produces -X world motion. With identity quaternion, the robot visually faces +X but is rewarded for moving -X — walking backward.

### The Hidden USD Transform

The USD file likely contains an internal rotation or axis remapping that was introduced during the SolidWorks → URDF → USD conversion pipeline. This transform is invisible in the simulation config (it's baked into the binary USD) but affects the physics frame orientation.

**This needs to be confirmed** by inspecting the USD file directly with `usdcat` or USD Composer.

## Current Patch

**Applied in EXP-429 (2026-03-18)**: 180° Z rotation on the spawn quaternion.

```python
# In harold.py InitialStateCfg:
rot=(0.0, 0.0, 0.0, 1.0),  # 180° Z rotation (Isaac Lab w,x,y,z format)
```

This makes the robot face -X (visually wrong) but aligns the reward direction with the robot's visual forward. The robot now walks forward relative to itself, which is what matters for learning locomotion.

**Trade-offs of this patch:**
- Training works correctly (robot walks forward)
- Robot faces -X instead of +X in world frame (cosmetic issue)
- Camera names are misleading ("front" camera shows the back)
- World-frame telemetry (`vx_w_mean`, `x_displacement`) has inverted sign for forward motion

## Proper Fix (Future Work)

### Option A: Fix the USD Asset (Recommended)
Edit `part_files/V4/harold_8.usd` directly to align the physics frame with the visual mesh. The body-frame +X should point in the same direction as the robot's visual front. This fixes the root cause and requires no code patches.

**Steps:**
1. Open `harold_8.usd` in USD Composer or equivalent tool
2. Identify the root transform that causes the axis flip
3. Correct it so body-frame +X = visual forward = world +X at identity quaternion
4. Revert the quaternion patch back to identity `(1,0,0,0)`
5. Verify: with identity quat, positive `vx_b` should produce +X world motion

### Option B: Negate Body-Frame Velocity in Rewards
Keep identity quaternion (robot faces +X) and negate `vx_b` in `train_env.py`:
```python
vx_b = -root_lin_vel_b[:, 0]  # Compensate for USD body frame orientation
```
This is a cleaner patch than the quaternion rotation but still doesn't fix the root cause.

### Option C: Current Patch (Keep 180° Z Rotation)
What we have now. Works for training, but the robot faces the wrong direction in world frame.

## Areas for Further Investigation

1. **USD internal transforms**: Use `usdcat` or Python USD API (`pxr.Usd`) to inspect the root transform in `harold_8.usd`. Look for rotation on the root prim or any xformOp that could cause the axis flip.

2. **URDF-to-USD conversion pipeline**: How was the USD created from the URDF? Was `urdf_to_usd` or a manual process used? The conversion step may have introduced the axis flip.

3. **URDF joint naming**: Verify whether FL/FR truly correspond to the robot's visual front or if the naming was arbitrary in the SolidWorks export. The visual front appears to be at the BL/BR end (+X in URDF).

4. **Hardware IMU alignment**: The MPU6050 IMU on the physical robot has its own axis orientation. When deploying trained policies, `deployment/inference/observation_builder.py` reads IMU body-frame velocity directly. The IMU's +X axis direction on the physical robot must match the simulation's body-frame +X convention. A TODO comment has been added to `observation_builder.py` flagging this.

5. **Rough terrain task**: The same patch has been applied to `harold_rough/harold.py`. Any USD fix must be validated on both flat and rough tasks.

## Files Involved

| File | Role |
|---|---|
| `part_files/V4/harold_8.usd` | USD asset with suspected baked-in transform (ROOT CAUSE) |
| `harold_isaac_lab/.../harold_flat/harold.py` | Spawn quaternion (PATCHED) |
| `harold_isaac_lab/.../harold_rough/harold.py` | Spawn quaternion (PATCHED) |
| `harold_isaac_lab/.../harold_flat/train_env.py` | Reward computation using `vx_b` |
| `deployment/inference/observation_builder.py` | Hardware IMU → observation mapping (TODO added) |
| `docs/memory/OBSERVATIONS.md` | Quaternion convention documentation |

## History

- **Pre-EXP-429**: All experiments trained with identity quaternion. Robot was rewarded for walking backward. Walk_score baseline of 29.8 was achieved by a backward-walking robot.
- **EXP-429**: Applied 180° Z rotation. Robot walked forward for the first time with correct reward alignment. Walk_score ~55 at 10-min peak.
- **2026-03-18**: Bug documented. Patch applied to both flat and rough terrain tasks.
