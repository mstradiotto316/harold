#!/usr/bin/env python3
"""Diagnose observation normalization to find problematic components.

This compares hardware observations to training statistics to find
which observation components are causing extreme normalized values.

Observation layout (48D, manager-based):
    [0:3]   lin_vel (zeros)
    [3:6]   ang_vel
    [6:9]   projected_gravity
    [9:12]  velocity_commands
    [12:24] joint_pos_relative (12D)
    [24:36] joint_vel (12D)
    [36:48] last_action (12D, raw policy output)
"""
import json
import numpy as np
from pathlib import Path

# Load policy metadata
metadata_path = Path(__file__).parent / "policy" / "policy_metadata.json"
with open(metadata_path) as f:
    metadata = json.load(f)

running_mean = np.array(metadata["running_mean"], dtype=np.float32)
running_var = np.array(metadata["running_variance"], dtype=np.float32)
running_std = np.sqrt(running_var + 1e-8)

# Observation component names and indices (manager-based layout)
OBS_COMPONENTS = [
    ("lin_vel_x", 0),
    ("lin_vel_y", 1),
    ("lin_vel_z", 2),
    ("ang_vel_x", 3),
    ("ang_vel_y", 4),
    ("ang_vel_z", 5),
    ("grav_x", 6),
    ("grav_y", 7),
    ("grav_z", 8),
    ("cmd_vx", 9),
    ("cmd_vy", 10),
    ("cmd_yaw_rate", 11),
    ("joint_pos_rel[0] (FL_sh)", 12),
    ("joint_pos_rel[1] (FR_sh)", 13),
    ("joint_pos_rel[2] (BL_sh)", 14),
    ("joint_pos_rel[3] (BR_sh)", 15),
    ("joint_pos_rel[4] (FL_th)", 16),
    ("joint_pos_rel[5] (FR_th)", 17),
    ("joint_pos_rel[6] (BL_th)", 18),
    ("joint_pos_rel[7] (BR_th)", 19),
    ("joint_pos_rel[8] (FL_ca)", 20),
    ("joint_pos_rel[9] (FR_ca)", 21),
    ("joint_pos_rel[10] (BL_ca)", 22),
    ("joint_pos_rel[11] (BR_ca)", 23),
    ("joint_vel[0]", 24),
    ("joint_vel[1]", 25),
    ("joint_vel[2]", 26),
    ("joint_vel[3]", 27),
    ("joint_vel[4]", 28),
    ("joint_vel[5]", 29),
    ("joint_vel[6]", 30),
    ("joint_vel[7]", 31),
    ("joint_vel[8]", 32),
    ("joint_vel[9]", 33),
    ("joint_vel[10]", 34),
    ("joint_vel[11]", 35),
    ("last_action[0]", 36),
    ("last_action[1]", 37),
    ("last_action[2]", 38),
    ("last_action[3]", 39),
    ("last_action[4]", 40),
    ("last_action[5]", 41),
    ("last_action[6]", 42),
    ("last_action[7]", 43),
    ("last_action[8]", 44),
    ("last_action[9]", 45),
    ("last_action[10]", 46),
    ("last_action[11]", 47),
]

print("=" * 80)
print("TRAINING STATISTICS ANALYSIS")
print("=" * 80)
print(f"\nRunning count: {metadata['running_count']:,} timesteps")
print(f"\nCheckpoint: {metadata.get('checkpoint_path', 'unknown')}")

print("\n" + "-" * 80)
print("COMPONENT         |   MEAN   |   STD    | EXPECTED HARDWARE |  NORM @ HW VALUE")
print("-" * 80)

# Define expected hardware values at rest (default stance, stationary)
EXPECTED_HW_VALUES = {
    # Velocity-blind: always zeros
    "lin_vel_x": 0.0, "lin_vel_y": 0.0, "lin_vel_z": 0.0,
    # IMU at rest
    "ang_vel_x": 0.0, "ang_vel_y": 0.0, "ang_vel_z": 0.0,
    # Gravity (sim convention: -1 for level)
    "grav_x": 0.0, "grav_y": 0.0, "grav_z": -1.0,
    # Default velocity commands
    "cmd_vx": 0.3, "cmd_vy": 0.0, "cmd_yaw_rate": 0.0,
}
# Joint positions at default stance → relative = 0
for i in range(12):
    suffix = ["FL_sh", "FR_sh", "BL_sh", "BR_sh",
              "FL_th", "FR_th", "BL_th", "BR_th",
              "FL_ca", "FR_ca", "BL_ca", "BR_ca"][i]
    EXPECTED_HW_VALUES[f"joint_pos_rel[{i}] ({suffix})"] = 0.0
# Joint velocities at rest = 0
for i in range(12):
    EXPECTED_HW_VALUES[f"joint_vel[{i}]"] = 0.0
# Last action at startup = 0
for i in range(12):
    EXPECTED_HW_VALUES[f"last_action[{i}]"] = 0.0

# Analyze each component
problems = []
for name, idx in OBS_COMPONENTS:
    if idx >= len(running_mean):
        continue
    mean = running_mean[idx]
    std = running_std[idx]
    expected = EXPECTED_HW_VALUES.get(name, 0.0)
    normalized = (expected - mean) / std

    flag = ""
    if abs(normalized) > 3.0:
        flag = " *** EXTREME"
        problems.append((name, idx, mean, std, expected, normalized))
    elif abs(normalized) > 2.0:
        flag = " * HIGH"

    print(f"{name:30s} | {mean:+8.3f} | {std:8.3f} | {expected:+17.3f} | {normalized:+8.2f}{flag}")

print("\n" + "=" * 80)
print("SUMMARY: PROBLEMATIC COMPONENTS")
print("=" * 80)

if problems:
    print(f"\n{len(problems)} components have extreme normalized values (|norm| > 3.0):")
    for name, idx, mean, std, expected, normalized in problems:
        print(f"\n  [{idx:2d}] {name}")
        print(f"       Training mean: {mean:+.3f}, std: {std:.3f}")
        print(f"       Hardware value: {expected:+.3f}")
        print(f"       Normalized: {normalized:+.2f} (clips to {np.clip(normalized, -5, 5):+.2f})")
else:
    print("\nAll components look reasonable!")

print("\n" + "=" * 80)
print("VELOCITY-BLIND CHECK")
print("=" * 80)
print(f"\nobs[0:3] (lin_vel) training mean: [{running_mean[0]:.3f}, {running_mean[1]:.3f}, {running_mean[2]:.3f}]")
print(f"obs[0:3] hardware value: [0, 0, 0] (velocity-blind)")
print("If training mean is near zero, normalization is safe.")
print("If training mean is far from zero, the ONNX model handles normalization internally.")
