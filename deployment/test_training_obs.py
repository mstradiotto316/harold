#!/usr/bin/env python3
"""Test with observation set to training mean.

If policy works well with training mean observation, the issue is
our observation being far from training distribution.
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
from inference.observation_builder import normalize_observation
from inference.action_converter import ActionConverter, ActionConfig


def main():
    print("=" * 70)
    print("TRAINING MEAN OBSERVATION TEST")
    print("Using training mean for observation (normalized = 0)")
    print("=" * 70)

    config_dir = Path(__file__).parent / "config"
    cpg_config = CPGConfig.from_yaml(config_dir / "cpg.yaml")
    action_config = ActionConfig.from_yaml(config_dir / "cpg.yaml", config_dir / "hardware.yaml")
    metadata_path = Path(__file__).parent / "policy" / "policy_metadata.json"
    policy_path = Path(__file__).parent / "policy" / "harold_policy.onnx"

    with open(metadata_path) as f:
        metadata = json.load(f)

    running_mean = np.array(metadata["running_mean"], dtype=np.float32)
    running_var = np.array(metadata["running_variance"], dtype=np.float32)

    # Override lin_vel
    running_mean[0:3] = 0.0
    running_var[0:3] = 0.25

    policy = ort.InferenceSession(str(policy_path), providers=['CPUExecutionProvider'])
    cpg = CPGGenerator(cpg_config)
    action_conv = ActionConverter(action_config)

    print("\nSimulating with observation = training mean (normalized → 0)")

    for step in range(10):
        t = step * 0.05

        # Use training mean as observation (except phase)
        obs = running_mean.copy()
        obs[0:3] = 0.0  # lin_vel

        # Update phase only
        phase = 2 * np.pi * cpg.cfg.frequency_hz * t
        obs[48] = np.sin(phase)
        obs[49] = np.cos(phase)

        # Normalize (should be close to 0)
        obs_norm = normalize_observation(obs, running_mean, running_var)

        # Run policy
        outputs = policy.run(['mean'], {'obs': obs_norm.reshape(1, -1).astype(np.float32)})
        action = outputs[0][0]

        cpg_targets = cpg.compute(t)
        rl_targets, hw_targets = action_conv.compute(cpg_targets, action)

        print(f"\nStep {step} (t={t:.2f}s, phase={obs[48]:.2f}/{obs[49]:.2f}):")
        print(f"  Policy range: [{action.min():.2f}, {action.max():.2f}]")
        print(f"  Corrections (thigh): {action[4:8] * 0.05 * 0.9}")
        print(f"  HW targets thighs: {hw_targets[4:8]}")

    print("\n" + "=" * 70)
    print("If policy outputs are reasonable (~±3), observation mismatch is the issue")
    print("If policy outputs are still large, the policy itself may have issues")
    print("=" * 70)


if __name__ == "__main__":
    main()
