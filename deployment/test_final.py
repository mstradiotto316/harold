#!/usr/bin/env python3
"""Final integration test with all fixes applied.

Tests:
1. Lin_vel override
2. Prev_target blending
3. Correct coordinate conversions
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
from inference.observation_builder import ObservationConfig, normalize_observation
from inference.action_converter import ActionConverter, ActionConfig


def main():
    print("=" * 70)
    print("FINAL INTEGRATION TEST")
    print("All fixes: lin_vel override, prev_target blend, correct conventions")
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

    # FIX 1: Override lin_vel statistics
    print("\nFIX 1: Overriding lin_vel statistics")
    print(f"  Original mean: {running_mean[0:3]}")
    running_mean[0:3] = 0.0
    running_var[0:3] = 0.25
    print(f"  New mean: {running_mean[0:3]}")

    policy = ort.InferenceSession(str(policy_path), providers=['CPUExecutionProvider'])
    cpg = CPGGenerator(cpg_config)
    action_conv = ActionConverter(action_config)
    hw_default = action_config.hw_default_pose

    print("\nSimulating 20 control loop iterations...")
    print("FIX 2: joint_pos blending (30% actual during warmup, transition to 100%)")
    print("FIX 3: prev_target blending (10% actual, 90% training mean)")

    # Initialize prev_targets to training mean
    prev_targets = running_mean[36:48].copy()
    prev_targets_training_mean = running_mean[36:48].copy()
    prev_target_blend = 0.1  # 10% actual, 90% training mean

    # Warmup and transition parameters (in seconds)
    warmup_duration = 3 * (1.0 / cpg_config.frequency_hz)  # 3 CPG cycles
    transition_duration = warmup_duration + 2 * (1.0 / cpg_config.frequency_hz)

    action_conv.reset()

    # Track statistics
    action_ranges = []

    for step in range(20):
        t = step * 0.05  # 20 Hz

        # Compute CPG targets
        cpg_targets = cpg.compute(t)
        _, hw_targets_pure = action_conv.compute(cpg_targets, np.zeros(12), use_cpg=True)

        # Simulate servo positions (tracking CPG with lag)
        if step == 0:
            simulated_hw_pos = hw_default.copy()
        else:
            simulated_hw_pos = 0.8 * simulated_hw_pos + 0.2 * hw_targets_pure

        # Calculate joint_pos blend factor (FIX 2)
        if t < warmup_duration:
            joint_pos_blend = 0.3  # 30% actual during warmup
        elif t < transition_duration:
            progress = (t - warmup_duration) / (transition_duration - warmup_duration)
            joint_pos_blend = 0.3 + 0.7 * progress
        else:
            joint_pos_blend = 1.0

        # Build observation
        obs = np.zeros(50, dtype=np.float32)

        # [0:3] lin_vel (zeros at rest/walking)
        obs[0:3] = 0.0

        # [3:6] ang_vel
        obs[3:6] = 0.0

        # [6:9] gravity (sim convention: -1 for level)
        obs[6:9] = np.array([0.0, 0.0, -1.0])

        # [9:21] joint_pos_relative (with sign conversion and blending)
        hw_relative = simulated_hw_pos - obs_config.hw_default_pose
        rl_relative = hw_relative * obs_config.joint_sign
        # Apply joint_pos blending
        rl_relative_blended = joint_pos_blend * rl_relative + (1 - joint_pos_blend) * running_mean[9:21]
        obs[9:21] = rl_relative_blended

        # [21:33] joint_vel
        obs[21:33] = 0.0

        # [33:36] commands
        obs[33:36] = np.array([0.3, 0.0, 0.0])

        # [36:48] prev_targets (blended with training mean)
        obs[36:48] = prev_targets

        # [48:50] phase
        phase = 2 * np.pi * cpg.cfg.frequency_hz * t
        obs[48] = np.sin(phase)
        obs[49] = np.cos(phase)

        # Normalize
        obs_norm = normalize_observation(obs, running_mean, running_var)

        # Run policy
        outputs = policy.run(['mean'], {'obs': obs_norm.reshape(1, -1).astype(np.float32)})
        action = outputs[0][0]

        # Compute targets
        rl_targets, hw_targets = action_conv.compute(cpg_targets, action)

        # FIX 3: Update prev_targets with blending (10% actual, 90% training mean)
        actual_delta = rl_targets - hw_default
        prev_targets = prev_target_blend * actual_delta + (1 - prev_target_blend) * prev_targets_training_mean

        action_ranges.append((action.min(), action.max()))

        if step % 5 == 0:
            print(f"\nStep {step} (t={t:.2f}s, blend={joint_pos_blend:.1f}):")
            print(f"  CPG thighs: {cpg_targets[4:8]}")
            print(f"  Policy range: [{action.min():.1f}, {action.max():.1f}]")
            print(f"  HW targets thighs: {hw_targets[4:8]}")

    # Summary
    action_mins = [r[0] for r in action_ranges]
    action_maxs = [r[1] for r in action_ranges]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nPolicy output statistics over 20 iterations:")
    print(f"  Min: {min(action_mins):.1f}")
    print(f"  Max: {max(action_maxs):.1f}")
    print(f"  Average range: [{np.mean(action_mins):.1f}, {np.mean(action_maxs):.1f}]")

    max_correction = max(abs(min(action_mins)), abs(max(action_maxs))) * 0.05 * 0.9
    print(f"\n  Max correction magnitude: {max_correction:.3f} rad ({np.degrees(max_correction):.1f}°)")

    if max(abs(min(action_mins)), abs(max(action_maxs))) < 25:
        print("\n  STATUS: STABLE - Policy outputs bounded")
        print("  The controller should be ready for hardware testing")
    else:
        print("\n  STATUS: UNSTABLE - Policy outputs still growing")
        print("  Further investigation needed")


if __name__ == "__main__":
    main()
