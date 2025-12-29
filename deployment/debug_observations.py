#!/usr/bin/env python3
"""Debug observation values during deployment.

Logs raw observations, normalized observations, and policy outputs
to diagnose the observation mismatch issue.

Usage:
    python debug_observations.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed")
    sys.exit(1)

from drivers.esp32_serial import ESP32Interface
from drivers.imu_reader_rpi5 import IMUReaderRPi5
from inference.cpg_generator import CPGGenerator
from inference.observation_builder import ObservationBuilder, ObservationConfig, normalize_observation


def main():
    print("=" * 70)
    print("Harold Observation Debug")
    print("=" * 70)

    # Load policy metadata
    metadata_path = Path(__file__).parent / "policy" / "policy_metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    running_mean = np.array(metadata["running_mean"], dtype=np.float32)
    running_var = np.array(metadata["running_variance"], dtype=np.float32)

    print("\n--- Running Statistics from Training ---")
    print(f"running_mean shape: {running_mean.shape}")
    print(f"running_var shape: {running_var.shape}")
    print()

    # Print key running stats
    obs_labels = [
        ("root_lin_vel_b[0] (vx)", 0),
        ("root_lin_vel_b[1] (vy)", 1),
        ("root_lin_vel_b[2] (vz)", 2),
        ("root_ang_vel_b[0]", 3),
        ("root_ang_vel_b[1]", 4),
        ("root_ang_vel_b[2]", 5),
        ("projected_gravity_b[0]", 6),
        ("projected_gravity_b[1]", 7),
        ("projected_gravity_b[2]", 8),
        ("joint_pos_rel[0]", 9),
        ("joint_vel[0]", 21),
        ("commands[0] (vx cmd)", 33),
        ("gait_phase sin", 48),
        ("gait_phase cos", 49),
    ]

    print("Index | Label                    | Mean          | Var           | Std")
    print("-" * 75)
    for label, idx in obs_labels:
        mean = running_mean[idx]
        var = running_var[idx]
        std = np.sqrt(var)
        print(f"{idx:5} | {label:24} | {mean:+13.4f} | {var:13.4f} | {std:10.4f}")

    # Load ONNX policy
    policy_path = Path(__file__).parent / "policy" / "harold_policy.onnx"
    print(f"\nLoading policy: {policy_path}")
    policy = ort.InferenceSession(str(policy_path), providers=['CPUExecutionProvider'])

    # Initialize hardware
    print("\n--- Initializing Hardware ---")

    esp32 = ESP32Interface()
    if not esp32.connect():
        print("ERROR: ESP32 connection failed")
        return 1

    imu = IMUReaderRPi5()
    if not imu.connect():
        print("ERROR: IMU connection failed")
        esp32.disconnect()
        return 1

    print("Calibrating IMU (keep robot still)...")
    imu.calibrate(duration=2.0)

    cpg = CPGGenerator()
    obs_builder = ObservationBuilder(imu, esp32)

    print("\n--- Collecting Observations ---")
    print("Reading 10 samples over 2 seconds...")

    samples = []
    start_time = time.time()

    for i in range(10):
        t = time.time() - start_time
        phase_sin = np.sin(2 * np.pi * cpg.cfg.frequency_hz * t)
        phase_cos = np.cos(2 * np.pi * cpg.cfg.frequency_hz * t)

        # Build raw observation
        obs_raw = obs_builder.build(t, phase_sin, phase_cos)

        # Normalize observation
        obs_norm = normalize_observation(obs_raw, running_mean, running_var)

        # Run policy
        outputs = policy.run(['mean'], {'obs': obs_norm.reshape(1, -1)})
        action = outputs[0][0]

        samples.append({
            'time': t,
            'obs_raw': obs_raw.copy(),
            'obs_norm': obs_norm.copy(),
            'action': action.copy(),
        })

        time.sleep(0.2)

    esp32.disconnect()
    imu.disconnect()

    # Analyze samples
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Average raw observations
    obs_raw_mean = np.mean([s['obs_raw'] for s in samples], axis=0)
    obs_norm_mean = np.mean([s['obs_norm'] for s in samples], axis=0)
    action_mean = np.mean([s['action'] for s in samples], axis=0)

    print("\n--- Raw Observation Statistics (averaged over 10 samples) ---")
    print("Index | Label                    | Hardware Value | Training Mean | Difference")
    print("-" * 85)
    for label, idx in obs_labels:
        hw_val = obs_raw_mean[idx]
        train_mean = running_mean[idx]
        diff = hw_val - train_mean
        print(f"{idx:5} | {label:24} | {hw_val:+14.4f} | {train_mean:+13.4f} | {diff:+10.4f}")

    print("\n--- Normalized Observation Statistics ---")
    print("Index | Label                    | Normalized Value | Expected ~N(0,1)?")
    print("-" * 70)
    for label, idx in obs_labels:
        norm_val = obs_norm_mean[idx]
        status = "OK" if abs(norm_val) < 5.0 else "EXTREME!"
        print(f"{idx:5} | {label:24} | {norm_val:+17.4f} | {status}")

    # Check for extreme values
    print("\n--- Extreme Normalized Values (|val| > 5) ---")
    extreme_indices = np.where(np.abs(obs_norm_mean) > 5.0)[0]
    if len(extreme_indices) == 0:
        print("None found.")
    else:
        for idx in extreme_indices:
            print(f"  Index {idx}: norm_val = {obs_norm_mean[idx]:+.2f}")

    print("\n--- Action Statistics ---")
    print(f"Action mean: {action_mean}")
    print(f"Action min:  {action_mean.min():.4f}")
    print(f"Action max:  {action_mean.max():.4f}")

    # Check each sample's action extremes
    print("\n--- Per-Sample Action Extremes ---")
    for i, s in enumerate(samples):
        action = s['action']
        print(f"Sample {i}: min={action.min():+8.2f}, max={action.max():+8.2f}")

    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    # Key diagnostics
    lin_vel_hw = obs_raw_mean[0:3]
    lin_vel_train = running_mean[0:3]
    lin_vel_var_train = running_var[0:3]

    print("\nLinear Velocity Comparison:")
    print(f"  Hardware reads:  {lin_vel_hw}")
    print(f"  Training mean:   {lin_vel_train}")
    print(f"  Training std:    {np.sqrt(lin_vel_var_train)}")

    if np.abs(lin_vel_train[2]) > 10:
        print("\n  WARNING: Training mean for vz is unrealistic!")
        print("  This suggests a simulation issue or units mismatch.")

    any_extreme = np.any(np.abs(obs_norm_mean) > 5.0)
    if any_extreme:
        print("\n  PROBLEM: Normalized observations have extreme values!")
        print("  This causes the policy to output garbage.")
    else:
        print("\n  Normalized observations look reasonable.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
