#!/usr/bin/env python3
"""Test whether sign conversion is needed in observation builder.

This compares policy behavior with and without sign conversion.
"""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed")
    sys.exit(1)


def test_sign_conversion():
    """Test policy with different sign conversion options."""
    print("=" * 70)
    print("SIGN CONVERSION TEST")
    print("=" * 70)

    # Load configs
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

    # Constants from config
    config_path = Path(__file__).parent / "config" / "cpg.yaml"
    obs_config = ObservationConfig.from_yaml(config_path)
    hw_default_pose = obs_config.hw_default_pose
    joint_sign = obs_config.joint_sign

    # Test with different HW position scenarios
    test_positions = [
        ("At hw_default", hw_default_pose.copy()),
        ("Thighs +0.1", hw_default_pose + np.array([0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0])),
        ("Thighs -0.1", hw_default_pose + np.array([0, 0, 0, 0, -0.1, -0.1, -0.1, -0.1, 0, 0, 0, 0])),
        ("Calves +0.1", hw_default_pose + np.array([0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1])),
    ]

    for test_name, hw_positions in test_positions:
        print(f"\n{'='*70}")
        print(f"TEST: {test_name}")
        print(f"{'='*70}")
        print(f"HW positions: {hw_positions}")

        # Compute relative positions both ways
        hw_relative = hw_positions - hw_default_pose
        rl_relative_with_sign = hw_relative * joint_sign
        rl_relative_no_sign = hw_relative

        print(f"\nHW relative: {hw_relative}")
        print(f"RL relative (WITH sign): {rl_relative_with_sign}")
        print(f"RL relative (NO sign): {rl_relative_no_sign}")

        # Build observation for both cases
        for sign_name, joint_rel in [
            ("WITH sign conversion", rl_relative_with_sign),
            ("NO sign conversion", rl_relative_no_sign),
        ]:
            obs = np.zeros(50, dtype=np.float32)
            obs[0:3] = 0.0  # lin_vel
            obs[3:6] = 0.0  # ang_vel
            obs[6:9] = np.array([0.0, 0.0, -1.0])  # gravity (sim convention)
            obs[9:21] = joint_rel
            obs[21:33] = 0.0  # joint_vel
            obs[33:36] = np.array([0.3, 0.0, 0.0])  # commands
            obs[36:48] = running_mean[36:48]  # prev_targets (training mean)
            obs[48:50] = np.array([0.0, 1.0])  # phase

            # Normalize
            obs_norm = (obs - running_mean) / np.sqrt(running_var + 1e-8)
            obs_norm = np.clip(obs_norm, -5.0, 5.0)

            # Run policy
            outputs = policy.run(['mean'], {'obs': obs_norm.reshape(1, -1).astype(np.float32)})
            action = outputs[0][0]

            print(f"\n  {sign_name}:")
            print(f"    Normalized joint_pos_rel thighs: {obs_norm[13:17]}")
            print(f"    Normalized joint_pos_rel calves: {obs_norm[17:21]}")
            print(f"    Policy output range: [{action.min():.3f}, {action.max():.3f}]")
            print(f"    Policy thigh outputs: {action[4:8]}")

    # Special test: what if we match training mean exactly?
    print(f"\n{'='*70}")
    print("SPECIAL TEST: Observation matching training mean exactly")
    print(f"{'='*70}")

    obs = running_mean.copy()  # Start from training mean
    obs[0:3] = 0.0  # But set lin_vel to 0 (overridden anyway)

    obs_norm = (obs - running_mean) / np.sqrt(running_var + 1e-8)
    obs_norm = np.clip(obs_norm, -5.0, 5.0)

    print(f"\nNormalized observation should be all zeros (or very close):")
    print(f"  lin_vel: {obs_norm[0:3]}")
    print(f"  joint_pos: {obs_norm[9:21]}")
    print(f"  prev_targets: {obs_norm[36:48]}")

    outputs = policy.run(['mean'], {'obs': obs_norm.reshape(1, -1).astype(np.float32)})
    action = outputs[0][0]

    print(f"\nPolicy output with 'zero' observation:")
    print(f"  shoulders: {action[0:4]}")
    print(f"  thighs: {action[4:8]}")
    print(f"  calves: {action[8:12]}")
    print(f"  Range: [{action.min():.3f}, {action.max():.3f}]")

    if np.abs(action).max() < 3.0:
        print("\n  GOOD: Policy outputs reasonable values when observation matches training mean")
    else:
        print("\n  WARNING: Policy outputs are still large even with training mean observation")


if __name__ == "__main__":
    test_sign_conversion()
from inference.observation_builder import ObservationConfig
