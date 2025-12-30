#!/usr/bin/env python3
"""Test direct policy mode (no CPG).

The training running_mean for prev_target thighs is near 0, suggesting
the policy was trained WITHOUT CPG mode. This tests if direct mode works better.
"""
import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed")
    sys.exit(1)

from inference.action_converter import ActionConverter, ActionConfig


def test_direct_mode():
    """Test direct policy mode without CPG."""
    print("=" * 70)
    print("DIRECT MODE TEST (NO CPG)")
    print("=" * 70)

    # Load configs
    config_dir = Path(__file__).parent / "config"
    action_config = ActionConfig.from_yaml(config_dir / "cpg.yaml", config_dir / "hardware.yaml")
    metadata_path = Path(__file__).parent / "policy" / "policy_metadata.json"
    policy_path = Path(__file__).parent / "policy" / "harold_policy.onnx"

    with open(metadata_path) as f:
        metadata = json.load(f)

    running_mean = np.array(metadata["running_mean"], dtype=np.float32)
    running_var = np.array(metadata["running_variance"], dtype=np.float32)

    # Override lin_vel stats
    running_mean[0:3] = 0.0
    running_var[0:3] = 0.25

    policy = ort.InferenceSession(str(policy_path), providers=['CPUExecutionProvider'])
    action_conv = ActionConverter(action_config)

    # Create observation matching training mean (normalized = 0)
    obs = running_mean.copy()
    obs[0:3] = 0.0  # lin_vel overridden anyway

    print(f"\nUsing training mean observation (normalized ≈ 0)")

    # Normalize
    obs_norm = (obs - running_mean) / np.sqrt(running_var + 1e-8)
    obs_norm = np.clip(obs_norm, -5.0, 5.0)

    # Run policy
    outputs = policy.run(['mean'], {'obs': obs_norm.reshape(1, -1).astype(np.float32)})
    action = outputs[0][0]

    print(f"\nPolicy output:")
    print(f"  shoulders: {action[0:4]}")
    print(f"  thighs: {action[4:8]}")
    print(f"  calves: {action[8:12]}")
    print(f"  Range: [{action.min():.3f}, {action.max():.3f}]")

    # Simulate what training mode would do (direct, no CPG)
    # From simulation: target = default_joint_pos + action_scale * joint_range * actions
    # action_scale is typically 0.5 or 1.0, let's check metadata
    action_scale = metadata.get("action_scale", 1.0)
    joint_range = np.array([
        action_config.joint_range["shoulder"]] * 4 +
        [action_config.joint_range["thigh"]] * 4 +
        [action_config.joint_range["calf"]] * 4,
        dtype=np.float32
    )

    print(f"\nTraining action processing:")
    print(f"  action_scale: {action_scale}")
    print(f"  joint_range: {joint_range}")

    # Direct mode: target = default + action_scale * joint_range * action
    hw_default = action_config.hw_default_pose
    direct_target = hw_default + action_scale * joint_range * action

    print(f"\nDirect mode targets (hw convention):")
    print(f"  shoulders: {direct_target[0:4]}")
    print(f"  thighs: {direct_target[4:8]}")
    print(f"  calves: {direct_target[8:12]}")

    # Check if targets are within safe limits
    print(f"\nSafety check:")
    print(f"  Thigh range: [{direct_target[4:8].min():.2f}, {direct_target[4:8].max():.2f}] rad")
    print(f"  Calf range: [{direct_target[8:12].min():.2f}, {direct_target[8:12].max():.2f}] rad")

    # Compare with ActionConverter direct mode
    print(f"\n" + "-" * 70)
    print("Testing ActionConverter in direct mode (use_cpg=False)")
    print("-" * 70)

    action_conv.reset()
    cpg_dummy = np.zeros(12, dtype=np.float32)  # CPG not used in direct mode

    rl_targets, hw_targets = action_conv.compute(cpg_dummy, action, use_cpg=False)

    print(f"\nActionConverter results:")
    print(f"  RL targets (thighs): {rl_targets[4:8]}")
    print(f"  HW targets (thighs): {hw_targets[4:8]}")

    # What prev_target_delta would be in direct mode
    prev_delta = rl_targets - hw_default
    print(f"\n  prev_target_delta (thighs): {prev_delta[4:8]}")
    print(f"  Training mean (thighs): {running_mean[40:44]}")

    # Check alignment
    delta_diff = prev_delta - running_mean[36:48]
    print(f"\n  Difference from training mean: {np.abs(delta_diff).max():.3f}")

    if np.abs(delta_diff).max() < 0.5:
        print("  -> prev_target_delta aligns well with training!")
    else:
        print("  -> Large mismatch with training expectations")


if __name__ == "__main__":
    test_direct_mode()
