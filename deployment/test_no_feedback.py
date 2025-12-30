#!/usr/bin/env python3
"""Test policy with prev_targets disabled (no feedback).

This tests if the feedback loop is causing the divergence.
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

from inference.cpg_generator import CPGGenerator, CPGConfig
from inference.observation_builder import ObservationConfig
from inference.action_converter import ActionConverter, ActionConfig


def test_no_feedback():
    """Test with prev_targets fixed to training mean (no feedback)."""
    print("=" * 70)
    print("NO FEEDBACK TEST (prev_targets fixed to training mean)")
    print("=" * 70)

    # Load configs
    config_dir = Path(__file__).parent / "config"
    cpg_config = CPGConfig.from_yaml(config_dir / "cpg.yaml")
    action_config = ActionConfig.from_yaml(config_dir / "cpg.yaml", config_dir / "hardware.yaml")
    obs_config = ObservationConfig.from_yaml(config_dir / "cpg.yaml")
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
    cpg = CPGGenerator(cpg_config)
    action_conv = ActionConverter(action_config)
    hw_default = action_config.hw_default_pose

    print("\nSimulating 10 iterations with NO prev_target feedback...")
    print("(prev_targets fixed to training mean)")

    # Fixed prev_targets to training mean
    prev_targets_fixed = running_mean[36:48].copy()

    action_conv.reset()

    for step in range(10):
        t = step * 0.05  # 20 Hz

        # Simulate observation
        obs = np.zeros(50, dtype=np.float32)

        # Compute CPG targets and use them to estimate joint positions
        cpg_targets = cpg.compute(t)

        # Convert CPG targets to HW convention (what servo would achieve)
        _, hw_targets = action_conv.compute(cpg_targets, np.zeros(12), use_cpg=True)

        # Simulate servo positions close to targets (with some lag)
        if step == 0:
            simulated_hw_pos = hw_default.copy()
        else:
            simulated_hw_pos = 0.8 * simulated_hw_pos + 0.2 * hw_targets

        # Build observation
        obs[0:3] = 0.0  # lin_vel
        obs[3:6] = 0.0  # ang_vel
        obs[6:9] = np.array([0.0, 0.0, -1.0])  # gravity

        # Joint positions (with sign conversion)
        hw_relative = simulated_hw_pos - obs_config.hw_default_pose
        rl_relative = hw_relative * obs_config.joint_sign
        obs[9:21] = rl_relative

        obs[21:33] = 0.0  # joint_vel
        obs[33:36] = np.array([0.3, 0.0, 0.0])  # commands
        obs[36:48] = prev_targets_fixed  # FIXED to training mean
        phase = 2 * np.pi * cpg.cfg.frequency_hz * t
        obs[48] = np.sin(phase)
        obs[49] = np.cos(phase)

        # Normalize
        obs_norm = (obs - running_mean) / np.sqrt(running_var + 1e-8)
        obs_norm = np.clip(obs_norm, -5.0, 5.0)

        # Run policy
        outputs = policy.run(['mean'], {'obs': obs_norm.reshape(1, -1).astype(np.float32)})
        action = outputs[0][0]

        # Compute targets (but don't update prev_targets)
        rl_targets, hw_targets = action_conv.compute(cpg_targets, action)

        print(f"\nStep {step} (t={t:.3f}s):")
        print(f"  Normalized joint_pos_rel[calves]: {obs_norm[17:21]}")
        print(f"  Policy output range: [{action.min():.2f}, {action.max():.2f}]")
        print(f"  HW targets (thighs): {hw_targets[4:8]}")

    print("\n" + "-" * 70)
    print("Now testing WITH feedback (for comparison)")
    print("-" * 70)

    action_conv.reset()
    prev_targets = prev_targets_fixed.copy()  # Start from training mean

    for step in range(10):
        t = step * 0.05

        obs = np.zeros(50, dtype=np.float32)
        cpg_targets = cpg.compute(t)
        _, hw_targets_pure = action_conv.compute(cpg_targets, np.zeros(12), use_cpg=True)

        if step == 0:
            simulated_hw_pos = hw_default.copy()
        else:
            simulated_hw_pos = 0.8 * simulated_hw_pos + 0.2 * hw_targets_pure

        obs[0:3] = 0.0
        obs[3:6] = 0.0
        obs[6:9] = np.array([0.0, 0.0, -1.0])

        hw_relative = simulated_hw_pos - obs_config.hw_default_pose
        rl_relative = hw_relative * obs_config.joint_sign
        obs[9:21] = rl_relative

        obs[21:33] = 0.0
        obs[33:36] = np.array([0.3, 0.0, 0.0])
        obs[36:48] = prev_targets  # WITH feedback
        phase = 2 * np.pi * cpg.cfg.frequency_hz * t
        obs[48] = np.sin(phase)
        obs[49] = np.cos(phase)

        obs_norm = (obs - running_mean) / np.sqrt(running_var + 1e-8)
        obs_norm = np.clip(obs_norm, -5.0, 5.0)

        outputs = policy.run(['mean'], {'obs': obs_norm.reshape(1, -1).astype(np.float32)})
        action = outputs[0][0]

        rl_targets, hw_targets = action_conv.compute(cpg_targets, action)

        # Update prev_targets for next iteration (with blending)
        actual_delta = rl_targets - hw_default
        training_mean = running_mean[36:48]
        blend_factor = 0.05
        prev_targets = blend_factor * actual_delta + (1 - blend_factor) * training_mean

        print(f"\nStep {step} (t={t:.3f}s):")
        print(f"  Normalized prev_targets[calves]: {obs_norm[44:48]}")
        print(f"  Policy output range: [{action.min():.2f}, {action.max():.2f}]")

    if action.max() > 20:
        print("\n\nCONCLUSION: Feedback causes divergence!")
    else:
        print("\n\nCONCLUSION: Both approaches similar - feedback not main issue")


if __name__ == "__main__":
    test_no_feedback()
