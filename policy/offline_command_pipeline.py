#!/usr/bin/env python3
"""Replay logged observations through exported policy and emit servo-ready commands."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort

OBS_DIM = 48
ACTION_DIM = 12
UNITS_PER_DEG = 4096.0 / 360.0
SERVO_MID = 2047
# Direction multipliers copied from firmware (index 0 unused, IDs 1-12)
DIR = [
    0,
    +1, +1, +1,  # Front-left shoulder/thigh/calf
    -1, -1, -1,  # Front-right
    +1, +1, +1,  # Rear-left
    -1, -1, -1,  # Rear-right
]

JOINT_CATEGORIES = (
    "shoulder", "shoulder", "shoulder", "shoulder",
    "thigh", "thigh", "thigh", "thigh",
    "calf", "calf", "calf", "calf",
)


def load_metadata(meta_path: Path):
    meta = json.loads(meta_path.read_text())
    joint_order = meta["joint_order"]
    default = np.array([meta["default_joint_pos"][name] for name in joint_order], dtype=np.float32)
    joint_range = meta["joint_range"]
    joint_limits = {
        "shoulder": (-meta["joint_angle_limits"]["shoulder"], meta["joint_angle_limits"]["shoulder"]),
        "thigh": (-meta["joint_angle_limits"]["thigh"], meta["joint_angle_limits"]["thigh"]),
        "calf": (-meta["joint_angle_limits"]["calf"], meta["joint_angle_limits"]["calf"]),
    }
    joint_sign = np.array(meta.get("joint_sign", [1.0] * ACTION_DIM), dtype=np.float32)
    return joint_order, default, joint_range, joint_limits, joint_sign


def read_observations(path: Path, num_samples: int | None) -> np.ndarray:
    lines = path.read_text().strip().splitlines()
    if num_samples is not None:
        lines = lines[:num_samples]
    obs = []
    for idx, line in enumerate(lines):
        values = [float(x) for x in line.strip().strip("[]").split(",")]
        if len(values) < OBS_DIM:
            raise ValueError(f"Observation line {idx} has {len(values)} values (< {OBS_DIM})")
        obs.append(values[:OBS_DIM])
    return np.asarray(obs, dtype=np.float32)


def clip_actions(actions: np.ndarray) -> np.ndarray:
    return np.clip(actions, -1.0, 1.0)


def scale_to_joint_targets(actions: np.ndarray, default: np.ndarray, joint_range: dict, joint_limits: dict, joint_sign: np.ndarray) -> np.ndarray:
    targets = np.zeros_like(actions)
    for i in range(ACTION_DIM):
        category = JOINT_CATEGORIES[i]
        span = joint_range[category]
        raw = default[i] + span * actions[:, i]
        lo, hi = joint_limits[category]
        targets[:, i] = np.clip(raw, lo, hi)
    return targets * joint_sign
    return targets


def radians_to_servo_units(rad: np.ndarray) -> np.ndarray:
    deg = np.degrees(rad)
    servo = np.zeros_like(deg, dtype=np.float32)
    for i in range(ACTION_DIM):
        servo[:, i] = SERVO_MID + DIR[i + 1] * deg[:, i] * UNITS_PER_DEG
    return deg, servo


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline policy command pipeline")
    parser.add_argument("--observations", type=Path, default=Path("simulation_logs/observations.log"))
    parser.add_argument("--policy", type=Path, default=Path("deployment_artifacts/terrain_64_2/harold_policy.onnx"))
    parser.add_argument("--metadata", type=Path, default=Path("deployment_artifacts/terrain_64_2/policy_metadata.json"))
    parser.add_argument("--samples", type=int, default=500, help="Number of observations to replay (None for all)")
    parser.add_argument("--output", type=Path, default=Path("deployment_artifacts/terrain_64_2/offline_commands.csv"))
    args = parser.parse_args()

    joint_order, default_pose, joint_range, joint_limits, joint_sign = load_metadata(args.metadata)
    obs = read_observations(args.observations, None if args.samples <= 0 else args.samples)

    sess = ort.InferenceSession(str(args.policy))
    mean, value, log_std = sess.run(None, {"obs": obs})
    actions = clip_actions(mean)

    targets = scale_to_joint_targets(actions, default_pose, joint_range, joint_limits, joint_sign)
    deg, servo = radians_to_servo_units(targets)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["step"]
        header += [f"{name}_rad" for name in joint_order]
        header += [f"{name}_deg" for name in joint_order]
        header += [f"{name}_servo" for name in joint_order]
        writer.writerow(header)
        for idx in range(targets.shape[0]):
            row = [idx]
            row.extend(f"{targets[idx, j]:.6f}" for j in range(ACTION_DIM))
            row.extend(f"{deg[idx, j]:.3f}" for j in range(ACTION_DIM))
            row.extend(f"{servo[idx, j]:.1f}" for j in range(ACTION_DIM))
            writer.writerow(row)

    max_clip = np.max(np.abs(actions) >= 0.999)
    rad_limits = []
    for i in range(ACTION_DIM):
        category = JOINT_CATEGORIES[i]
        lo, hi = joint_limits[category]
        within = np.logical_or(np.isclose(targets[:, i], lo, atol=1e-4), np.isclose(targets[:, i], hi, atol=1e-4))
        rad_limits.append(int(np.count_nonzero(within)))
    print(f"Processed {targets.shape[0]} samples. Any action clipped to ±1.0: {bool(max_clip)}")
    print("Joint saturation counts (rad target at clamp):")
    for name, count in zip(joint_order, rad_limits):
        print(f"  {name}: {count}")
    print(f"CSV written to {args.output}")


if __name__ == "__main__":
    main()
