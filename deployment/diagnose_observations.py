#!/usr/bin/env python3
"""Diagnose observation normalization to find problematic components.

This compares hardware observations to training statistics to find
which observation components are causing extreme normalized values.
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

# Observation component names and indices
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
    ("joint_pos_rel[0] (FL_sh)", 9),
    ("joint_pos_rel[1] (FR_sh)", 10),
    ("joint_pos_rel[2] (BL_sh)", 11),
    ("joint_pos_rel[3] (BR_sh)", 12),
    ("joint_pos_rel[4] (FL_th)", 13),
    ("joint_pos_rel[5] (FR_th)", 14),
    ("joint_pos_rel[6] (BL_th)", 15),
    ("joint_pos_rel[7] (BR_th)", 16),
    ("joint_pos_rel[8] (FL_ca)", 17),
    ("joint_pos_rel[9] (FR_ca)", 18),
    ("joint_pos_rel[10] (BL_ca)", 19),
    ("joint_pos_rel[11] (BR_ca)", 20),
    ("joint_vel[0]", 21),
    ("joint_vel[1]", 22),
    ("joint_vel[2]", 23),
    ("joint_vel[3]", 24),
    ("joint_vel[4]", 25),
    ("joint_vel[5]", 26),
    ("joint_vel[6]", 27),
    ("joint_vel[7]", 28),
    ("joint_vel[8]", 29),
    ("joint_vel[9]", 30),
    ("joint_vel[10]", 31),
    ("joint_vel[11]", 32),
    ("cmd_vx", 33),
    ("cmd_vy", 34),
    ("cmd_yaw_rate", 35),
    ("prev_target[0]", 36),
    ("prev_target[1]", 37),
    ("prev_target[2]", 38),
    ("prev_target[3]", 39),
    ("prev_target[4]", 40),
    ("prev_target[5]", 41),
    ("prev_target[6]", 42),
    ("prev_target[7]", 43),
    ("prev_target[8]", 44),
    ("prev_target[9]", 45),
    ("prev_target[10]", 46),
    ("prev_target[11]", 47),
    ("phase_sin", 48),
    ("phase_cos", 49),
]

print("=" * 80)
print("TRAINING STATISTICS ANALYSIS")
print("=" * 80)
print(f"\nRunning count: {metadata['running_count']:,} timesteps")
print(f"\nCheckpoint: {metadata['checkpoint_path']}")

print("\n" + "-" * 80)
print("COMPONENT         |   MEAN   |   STD    | EXPECTED HARDWARE |  NORM @ HW VALUE")
print("-" * 80)

# Define expected hardware values at rest
EXPECTED_HW_VALUES = {
    # IMU at rest on level surface
    "lin_vel_x": 0.0,
    "lin_vel_y": 0.0,
    "lin_vel_z": 0.0,  # No vertical movement
    "ang_vel_x": 0.0,
    "ang_vel_y": 0.0,
    "ang_vel_z": 0.0,
    # Gravity points down (sim convention: -1 for level)
    "grav_x": 0.0,
    "grav_y": 0.0,
    "grav_z": -1.0,
    # Joint positions relative to RL default (at ready stance).
    # With the canonical stance applied, rel pos should be ~0 for all joints.
    "joint_pos_rel[0] (FL_sh)": 0.0,
    "joint_pos_rel[1] (FR_sh)": 0.0,
    "joint_pos_rel[2] (BL_sh)": 0.0,
    "joint_pos_rel[3] (BR_sh)": 0.0,
    "joint_pos_rel[4] (FL_th)": 0.0,
    "joint_pos_rel[5] (FR_th)": 0.0,
    "joint_pos_rel[6] (BL_th)": 0.0,
    "joint_pos_rel[7] (BR_th)": 0.0,
    "joint_pos_rel[8] (FL_ca)": 0.0,
    "joint_pos_rel[9] (FR_ca)": 0.0,
    "joint_pos_rel[10] (BL_ca)": 0.0,
    "joint_pos_rel[11] (BR_ca)": 0.0,
    # Joint velocities at rest
    "joint_vel[0]": 0.0,
    "joint_vel[1]": 0.0,
    "joint_vel[2]": 0.0,
    "joint_vel[3]": 0.0,
    "joint_vel[4]": 0.0,
    "joint_vel[5]": 0.0,
    "joint_vel[6]": 0.0,
    "joint_vel[7]": 0.0,
    "joint_vel[8]": 0.0,
    "joint_vel[9]": 0.0,
    "joint_vel[10]": 0.0,
    "joint_vel[11]": 0.0,
    # Commands
    "cmd_vx": 0.3,  # Default command
    "cmd_vy": 0.0,
    "cmd_yaw_rate": 0.0,
    # Previous targets (zeros at start)
    "prev_target[0]": 0.0,
    "prev_target[1]": 0.0,
    "prev_target[2]": 0.0,
    "prev_target[3]": 0.0,
    "prev_target[4]": 0.0,
    "prev_target[5]": 0.0,
    "prev_target[6]": 0.0,
    "prev_target[7]": 0.0,
    "prev_target[8]": 0.0,
    "prev_target[9]": 0.0,
    "prev_target[10]": 0.0,
    "prev_target[11]": 0.0,
    # Phase at t=0
    "phase_sin": 0.0,
    "phase_cos": 1.0,
}

# Analyze each component
problems = []
for name, idx in OBS_COMPONENTS:
    mean = running_mean[idx]
    std = running_std[idx]
    expected = EXPECTED_HW_VALUES.get(name, 0.0)
    normalized = (expected - mean) / std

    # Flag if normalized value is extreme (> 3.0 or < -3.0)
    flag = ""
    if abs(normalized) > 3.0:
        flag = " *** EXTREME"
        problems.append((name, idx, mean, std, expected, normalized))
    elif abs(normalized) > 2.0:
        flag = " * HIGH"

    print(f"{name:25s} | {mean:+8.3f} | {std:8.3f} | {expected:+17.3f} | {normalized:+8.2f}{flag}")

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

# Check linear velocity specifically
print("\n" + "=" * 80)
print("LINEAR VELOCITY DEEP ANALYSIS")
print("=" * 80)

print("\nTraining statistics suggest:")
print(f"  Mean X velocity: {running_mean[0]:.2f} m/s")
print(f"  Mean Y velocity: {running_mean[1]:.2f} m/s")
print(f"  Mean Z velocity: {running_mean[2]:.2f} m/s")

print("\nThese are PHYSICALLY IMPOSSIBLE for a walking robot:")
print(f"  - X mean of {running_mean[0]:.1f} m/s = {abs(running_mean[0]*3.6):.0f} km/h backwards")
print(f"  - Z mean of {running_mean[2]:.1f} m/s = falling at {abs(running_mean[2]*3.6):.0f} km/h")

print("\nPOSSIBLE CAUSES:")
print("  1. Corrupted running statistics during export")
print("  2. Bug in simulation during training (robots falling through terrain)")
print("  3. Domain randomization that adds extreme values")

print("\nRECOMMENDED FIX:")
print("  Override lin_vel normalization with reasonable values:")
print("  - Use mean=[0, 0, 0] and std=[0.5, 0.5, 0.5] for lin_vel")
print("  - OR zero out lin_vel in observation (set to 0)")
