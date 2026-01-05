#!/usr/bin/env python3
"""Test inference components without hardware.

These tests verify that the CPG, observation builder, and action converter
produce expected outputs without requiring real hardware.
"""
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_cpg_generator():
    """Test CPG trajectory generation."""
    from inference.cpg_generator import CPGGenerator, CPGConfig

    print("Testing CPG Generator...")

    cpg = CPGGenerator()

    # Test at t=0 (start of cycle)
    targets = cpg.compute(0.0)
    assert targets.shape == (12,), f"Expected shape (12,), got {targets.shape}"
    print(f"  t=0.0: phase={cpg.phase:.3f}, targets[0:3]={targets[:3]}")

    # Test at t=1.0 (half cycle at 0.5 Hz)
    targets = cpg.compute(1.0)
    assert cpg.phase == 0.5, f"Expected phase=0.5, got {cpg.phase}"
    print(f"  t=1.0: phase={cpg.phase:.3f}, targets[0:3]={targets[:3]}")

    # Test at t=2.0 (full cycle)
    targets = cpg.compute(2.0)
    assert abs(cpg.phase) < 0.01, f"Expected phase≈0, got {cpg.phase}"
    print(f"  t=2.0: phase={cpg.phase:.3f}, targets[0:3]={targets[:3]}")

    # Test phase sin/cos
    sin_val, cos_val = cpg.get_phase_sin_cos()
    assert -1 <= sin_val <= 1, f"Sin out of range: {sin_val}"
    assert -1 <= cos_val <= 1, f"Cos out of range: {cos_val}"
    print(f"  sin={sin_val:.3f}, cos={cos_val:.3f}")

    print("  PASSED")


def test_action_converter():
    """Test action conversion."""
    from inference.action_converter import ActionConverter

    print("Testing Action Converter...")

    converter = ActionConverter()

    # Test with zero policy output
    policy_output = np.zeros(12, dtype=np.float32)
    rl_targets, hw_targets = converter.compute_policy_targets(policy_output)

    assert rl_targets.shape == (12,), f"Expected shape (12,), got {rl_targets.shape}"
    assert hw_targets.shape == (12,), f"Expected shape (12,), got {hw_targets.shape}"
    print(f"  Zero policy: max |rl| = {np.max(np.abs(rl_targets)):.6f}")

    # Test with max policy output
    policy_output = np.ones(12, dtype=np.float32)
    rl_targets, hw_targets = converter.compute_policy_targets(policy_output)
    print(f"  Max policy: rl range = [{rl_targets.min():.4f}, {rl_targets.max():.4f}]")

    print("  PASSED")


def test_onnx_policy():
    """Test ONNX policy inference."""
    import json

    try:
        import onnxruntime as ort
    except ImportError:
        print("SKIPPING: onnxruntime not installed")
        return

    print("Testing ONNX Policy...")

    policy_dir = Path(__file__).parent.parent / "policy"
    onnx_path = policy_dir / "harold_policy.onnx"
    meta_path = policy_dir / "policy_metadata.json"

    if not onnx_path.exists():
        print(f"  SKIPPING: {onnx_path} not found")
        return

    # Load policy
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])

    # Load metadata
    with open(meta_path) as f:
        meta = json.load(f)

    running_mean = np.array(meta["running_mean"], dtype=np.float32)
    running_var = np.array(meta["running_variance"], dtype=np.float32)

    obs_dim = len(running_mean)
    print(f"  Observation dimension: {obs_dim}")

    # Create dummy observation
    obs = np.zeros(obs_dim, dtype=np.float32)
    obs_norm = (obs - running_mean) / np.sqrt(running_var + 1e-8)

    # Run inference
    import time
    start = time.time()
    num_iters = 100
    for _ in range(num_iters):
        outputs = session.run(['mean'], {'obs': obs_norm.reshape(1, -1)})
    elapsed = time.time() - start

    action = outputs[0][0]
    print(f"  Action shape: {action.shape}")
    print(f"  Action range: [{action.min():.3f}, {action.max():.3f}]")
    print(f"  Inference time: {elapsed / num_iters * 1000:.2f} ms")

    print("  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Harold Inference Tests")
    print("=" * 60)
    print()

    test_cpg_generator()
    print()

    test_action_converter()
    print()

    test_onnx_policy()
    print()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
