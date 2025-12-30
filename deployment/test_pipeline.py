#!/usr/bin/env python3
"""Test the full inference pipeline with mock or real hardware.

This tests:
1. Observation building with correct conventions
2. Normalization with corrected stats
3. Policy inference
4. Action conversion
"""
import argparse
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

from inference.cpg_generator import CPGGenerator, CPGConfig
from inference.observation_builder import ObservationConfig, normalize_observation
from inference.action_converter import ActionConverter, ActionConfig


def test_with_mock_data():
    """Test pipeline with mock hardware data."""
    print("=" * 70)
    print("PIPELINE TEST WITH MOCK DATA")
    print("=" * 70)

    # Load configs
    config_dir = Path(__file__).parent / "config"
    cpg_config = CPGConfig.from_yaml(config_dir / "cpg.yaml")
    action_config = ActionConfig.from_yaml(config_dir / "cpg.yaml", config_dir / "hardware.yaml")

    # Load policy
    policy_path = Path(__file__).parent / "policy" / "harold_policy.onnx"
    metadata_path = Path(__file__).parent / "policy" / "policy_metadata.json"

    with open(metadata_path) as f:
        metadata = json.load(f)

    running_mean = np.array(metadata["running_mean"], dtype=np.float32)
    running_var = np.array(metadata["running_variance"], dtype=np.float32)

    # Apply lin_vel override
    print("\nApplying lin_vel statistics override:")
    print(f"  Original lin_vel mean: {running_mean[0:3]}")
    print(f"  Original lin_vel std: {np.sqrt(running_var[0:3])}")
    running_mean[0:3] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    running_var[0:3] = np.array([0.25, 0.25, 0.25], dtype=np.float32)
    print(f"  New lin_vel mean: {running_mean[0:3]}")
    print(f"  New lin_vel std: {np.sqrt(running_var[0:3])}")

    policy = ort.InferenceSession(str(policy_path), providers=['CPUExecutionProvider'])
    print(f"\nPolicy loaded: {policy_path}")

    # Initialize CPG and action converter
    cpg = CPGGenerator(cpg_config)
    action_conv = ActionConverter(action_config)
    obs_config = ObservationConfig.from_yaml(config_dir / "cpg.yaml")

    print(f"\nConfigurations:")
    print(f"  HW default pose: {action_config.hw_default_pose}")
    print(f"  RL default pose: {action_config.rl_default_pose}")
    print(f"  Joint sign: {action_config.joint_sign}")

    # Create mock hardware observation
    # Simulating robot at rest, level, at hw_default_pose
    print("\n" + "-" * 70)
    print("TEST 1: Robot at rest (hw_default_pose)")
    print("-" * 70)

    # Mock IMU data
    imu_lin_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    imu_ang_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    imu_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)  # Sim convention: -1 for level

    # Mock servo positions at hw_default
    hw_positions = obs_config.hw_default_pose.copy()

    # Build observation manually
    obs = np.zeros(50, dtype=np.float32)

    # [0:3] lin_vel
    obs[0:3] = imu_lin_vel
    # [3:6] ang_vel
    obs[3:6] = imu_ang_vel
    # [6:9] gravity
    obs[6:9] = imu_gravity
    # [9:21] joint_pos_relative (HW -> RL conversion)
    hw_relative = hw_positions - obs_config.hw_default_pose
    rl_relative = hw_relative * obs_config.joint_sign
    obs[9:21] = rl_relative
    # [21:33] joint_vel (zeros at rest)
    obs[21:33] = 0.0
    # [33:36] commands
    obs[33:36] = np.array([0.3, 0.0, 0.0], dtype=np.float32)
    # [36:48] prev_target_delta (zeros at start)
    obs[36:48] = 0.0
    # [48:50] phase (t=0)
    obs[48] = 0.0  # sin(0)
    obs[49] = 1.0  # cos(0)

    print(f"\nRaw observation (key components):")
    print(f"  lin_vel: {obs[0:3]}")
    print(f"  ang_vel: {obs[3:6]}")
    print(f"  gravity: {obs[6:9]}")
    print(f"  joint_pos_rel[0:4] (shoulders): {obs[9:13]}")
    print(f"  joint_pos_rel[4:8] (thighs): {obs[13:17]}")
    print(f"  joint_pos_rel[8:12] (calves): {obs[17:21]}")
    print(f"  commands: {obs[33:36]}")
    print(f"  prev_targets[0:4]: {obs[36:40]}")
    print(f"  phase: {obs[48:50]}")

    # Normalize
    obs_norm = normalize_observation(obs, running_mean, running_var)

    print(f"\nNormalized observation (key components):")
    print(f"  lin_vel: {obs_norm[0:3]}")
    print(f"  ang_vel: {obs_norm[3:6]}")
    print(f"  gravity: {obs_norm[6:9]}")
    print(f"  joint_pos_rel[0:4] (shoulders): {obs_norm[9:13]}")
    print(f"  joint_pos_rel[4:8] (thighs): {obs_norm[13:17]}")
    print(f"  joint_pos_rel[8:12] (calves): {obs_norm[17:21]}")
    print(f"  commands: {obs_norm[33:36]}")
    print(f"  prev_targets[0:4]: {obs_norm[36:40]}")
    print(f"  phase: {obs_norm[48:50]}")

    # Check for extreme values
    extreme_mask = np.abs(obs_norm) > 3.0
    if np.any(extreme_mask):
        extreme_indices = np.where(extreme_mask)[0]
        print(f"\n  WARNING: Extreme normalized values at indices: {extreme_indices}")
        for idx in extreme_indices:
            print(f"    [{idx}]: {obs_norm[idx]:.2f}")
    else:
        print(f"\n  All normalized values within ±3.0 range")

    # Run policy inference
    outputs = policy.run(['mean'], {'obs': obs_norm.reshape(1, -1).astype(np.float32)})
    action = outputs[0][0]

    print(f"\nPolicy output (raw):")
    print(f"  shoulders: {action[0:4]}")
    print(f"  thighs: {action[4:8]}")
    print(f"  calves: {action[8:12]}")
    print(f"  Range: [{action.min():.3f}, {action.max():.3f}]")

    # Compute CPG targets at t=0
    cpg_targets = cpg.compute(0.0)

    print(f"\nCPG targets (t=0):")
    print(f"  shoulders: {cpg_targets[0:4]}")
    print(f"  thighs: {cpg_targets[4:8]}")
    print(f"  calves: {cpg_targets[8:12]}")

    # Compute final targets
    rl_targets, hw_targets = action_conv.compute(cpg_targets, action)

    print(f"\nFinal RL targets:")
    print(f"  shoulders: {rl_targets[0:4]}")
    print(f"  thighs: {rl_targets[4:8]}")
    print(f"  calves: {rl_targets[8:12]}")

    print(f"\nFinal HW targets (for servos):")
    print(f"  shoulders: {hw_targets[0:4]}")
    print(f"  thighs: {hw_targets[4:8]}")
    print(f"  calves: {hw_targets[8:12]}")

    # Verify prev_target_delta
    hw_default = action_conv.get_hw_default_pose()
    prev_target_delta = rl_targets - hw_default
    print(f"\nPrev target delta (for next observation):")
    print(f"  shoulders: {prev_target_delta[0:4]}")
    print(f"  thighs: {prev_target_delta[4:8]}")
    print(f"  calves: {prev_target_delta[8:12]}")

    print("\n" + "-" * 70)
    print("TEST 2: Simulate 5 control loop iterations (with training mean init)")
    print("-" * 70)

    action_conv.reset()
    # Initialize prev_targets to training mean values (prevents extreme normalized values)
    prev_targets = running_mean[36:48].copy()
    print(f"Initializing prev_targets to training mean: {prev_targets[:4]} ...")

    for step in range(5):
        t = step * 0.05  # 20 Hz

        # Update phase
        phase = 2 * np.pi * cpg.cfg.frequency_hz * t
        obs[48] = np.sin(phase)
        obs[49] = np.cos(phase)

        # Update prev_targets
        obs[36:48] = prev_targets

        # Normalize and run policy
        obs_norm = normalize_observation(obs, running_mean, running_var)
        outputs = policy.run(['mean'], {'obs': obs_norm.reshape(1, -1).astype(np.float32)})
        action = outputs[0][0]

        # Compute targets
        cpg_targets = cpg.compute(t)
        rl_targets, hw_targets = action_conv.compute(cpg_targets, action)

        # Update prev_targets for next iteration
        prev_targets = rl_targets - hw_default

        print(f"\nStep {step} (t={t:.3f}s):")
        print(f"  Phase: sin={obs[48]:.3f}, cos={obs[49]:.3f}")
        print(f"  Policy output range: [{action.min():.3f}, {action.max():.3f}]")
        print(f"  HW targets (thighs): {hw_targets[4:8]}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

    # Check if policy outputs are reasonable
    if np.abs(action).max() > 5.0:
        print("\nWARNING: Policy outputs are extreme (> 5.0)")
        print("This suggests observation mismatch with training")
    else:
        print("\nPolicy outputs look reasonable (<= 5.0)")


if __name__ == "__main__":
    test_with_mock_data()
