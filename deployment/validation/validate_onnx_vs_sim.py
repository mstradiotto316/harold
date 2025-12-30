#!/usr/bin/env python3
"""Validate ONNX policy outputs against recorded simulation data.

This script loads observation/action pairs recorded from Isaac Lab simulation
and compares the ONNX policy outputs against the recorded simulation actions.

Usage:
    python validate_onnx_vs_sim.py [--data sim_episode.json]

Expected JSON format:
{
    "metadata": {
        "checkpoint": "path/to/checkpoint.pt",
        "num_steps": 200,
        "running_mean": [...],  # 50D
        "running_var": [...]    # 50D
    },
    "timesteps": [
        {
            "step": 0,
            "obs_raw": [...],        # 50D raw observation
            "obs_normalized": [...], # 50D normalized observation
            "action": [...]          # 12D policy action
        },
        ...
    ]
}
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Validate ONNX vs simulation")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "sim_episode.json",
        help="Path to recorded simulation data",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path(__file__).parent.parent / "policy" / "harold_policy.onnx",
        help="Path to ONNX policy",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Tolerance for action comparison",
    )
    args = parser.parse_args()

    # Check files exist
    if not args.data.exists():
        print(f"ERROR: Simulation data not found: {args.data}")
        print("\nTo generate this file:")
        print("1. On desktop, run: python harold_isaac_lab/scripts/skrl/record_episode.py")
        print("2. Transfer: scp deployment/validation/sim_episode.json pi@harold.local:~/Desktop/harold/deployment/validation/")
        sys.exit(1)

    if not args.policy.exists():
        print(f"ERROR: Policy not found: {args.policy}")
        sys.exit(1)

    # Load simulation data
    print(f"Loading simulation data: {args.data}")
    with open(args.data) as f:
        data = json.load(f)

    metadata = data["metadata"]
    timesteps = data["timesteps"]
    print(f"  Checkpoint: {metadata.get('checkpoint', 'unknown')}")
    print(f"  Timesteps: {len(timesteps)}")

    # Load ONNX policy
    print(f"\nLoading ONNX policy: {args.policy}")
    policy = ort.InferenceSession(str(args.policy), providers=['CPUExecutionProvider'])

    # Compare outputs
    print(f"\nComparing {len(timesteps)} timesteps (tolerance={args.tolerance})...")
    print("=" * 70)

    errors = []
    max_error = 0.0
    max_error_step = 0

    for ts in timesteps:
        step = ts["step"]
        obs_norm = np.array(ts["obs_normalized"], dtype=np.float32)
        sim_action = np.array(ts["action"], dtype=np.float32)

        # Run ONNX inference
        outputs = policy.run(['mean'], {'obs': obs_norm.reshape(1, -1)})
        onnx_action = outputs[0][0]

        # Compare
        diff = np.abs(onnx_action - sim_action)
        step_max_error = diff.max()

        if step_max_error > max_error:
            max_error = step_max_error
            max_error_step = step

        if step_max_error > args.tolerance:
            errors.append({
                "step": step,
                "max_diff": step_max_error,
                "diff": diff.tolist(),
                "sim_action": sim_action.tolist(),
                "onnx_action": onnx_action.tolist(),
            })

        # Progress
        if step % 50 == 0:
            status = "PASS" if step_max_error <= args.tolerance else "FAIL"
            print(f"  Step {step:4d}: max_diff={step_max_error:.6f} [{status}]")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if len(errors) == 0:
        print(f"\nRESULT: PASS - All {len(timesteps)} timesteps within tolerance")
        print(f"  Max error: {max_error:.6f} at step {max_error_step}")
        print("\nThe ONNX policy matches simulation outputs!")
    else:
        print(f"\nRESULT: FAIL - {len(errors)}/{len(timesteps)} timesteps exceed tolerance")
        print(f"  Max error: {max_error:.6f} at step {max_error_step}")

        print("\nFirst 5 failures:")
        for err in errors[:5]:
            print(f"\n  Step {err['step']}:")
            print(f"    Max diff: {err['max_diff']:.6f}")
            print(f"    Sim action:  {np.array(err['sim_action'])[:4]}...")
            print(f"    ONNX action: {np.array(err['onnx_action'])[:4]}...")

        # Analyze systematic differences
        print("\n\nAnalyzing systematic differences...")
        all_diffs = []
        for ts in timesteps:
            obs_norm = np.array(ts["obs_normalized"], dtype=np.float32)
            sim_action = np.array(ts["action"], dtype=np.float32)
            outputs = policy.run(['mean'], {'obs': obs_norm.reshape(1, -1)})
            onnx_action = outputs[0][0]
            all_diffs.append(onnx_action - sim_action)

        all_diffs = np.array(all_diffs)
        mean_diff = all_diffs.mean(axis=0)
        std_diff = all_diffs.std(axis=0)

        print("\nPer-joint mean difference (ONNX - SIM):")
        joint_names = ['sh_FL', 'sh_FR', 'sh_BL', 'sh_BR',
                       'th_FL', 'th_FR', 'th_BL', 'th_BR',
                       'ca_FL', 'ca_FR', 'ca_BL', 'ca_BR']
        for i, name in enumerate(joint_names):
            print(f"  {name}: mean={mean_diff[i]:+.4f}, std={std_diff[i]:.4f}")

        if np.abs(mean_diff).max() > 0.1:
            print("\nWARNING: Large systematic bias detected!")
            print("This suggests coordinate convention mismatch.")


if __name__ == "__main__":
    main()
