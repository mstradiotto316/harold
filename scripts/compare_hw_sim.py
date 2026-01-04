#!/usr/bin/env python3
"""Compare hardware telemetry logs against simulation gait logs."""
import argparse
import csv
import math
from pathlib import Path

import yaml


JOINT_NAMES = [
    "fl_sh", "fr_sh", "bl_sh", "br_sh",
    "fl_th", "fr_th", "bl_th", "br_th",
    "fl_ca", "fr_ca", "bl_ca", "br_ca",
]


def _expand_joint_sign(data: dict) -> list[float]:
    js = data.get("joint_sign", {}) if isinstance(data, dict) else {}
    shoulders = js.get("shoulders", 1.0)
    if isinstance(shoulders, (list, tuple)) and len(shoulders) == 4:
        shoulder_vals = list(shoulders)
    else:
        shoulder_vals = [
            js.get("shoulder_fl", shoulders),
            js.get("shoulder_fr", shoulders),
            js.get("shoulder_bl", shoulders),
            js.get("shoulder_br", shoulders),
        ]
    thigh_val = js.get("thighs", -1.0)
    calf_val = js.get("calves", -1.0)
    return [float(v) for v in shoulder_vals] + [float(thigh_val)] * 4 + [float(calf_val)] * 4


def _load_hw_default(stance_path: Path) -> list[float]:
    with stance_path.open("r", encoding="utf-8") as f:
        stance = yaml.safe_load(f) or {}
    hw = stance.get("hw_default_pose", {})
    return [
        float(hw.get("shoulders", 0.0)),
        float(hw.get("shoulders", 0.0)),
        float(hw.get("shoulders", 0.0)),
        float(hw.get("shoulders", 0.0)),
        float(hw.get("thighs", 0.65)),
        float(hw.get("thighs", 0.65)),
        float(hw.get("thighs", 0.65)),
        float(hw.get("thighs", 0.65)),
        float(hw.get("calves", -1.13)),
        float(hw.get("calves", -1.13)),
        float(hw.get("calves", -1.13)),
        float(hw.get("calves", -1.13)),
    ]


def _load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        for k, v in list(r.items()):
            if k in ("mode",):
                continue
            try:
                r[k] = float(v)
            except Exception:
                pass
    return rows


def _to_rl(hw_pos: list[float], hw_default: list[float], joint_sign: list[float]) -> list[float]:
    return [d + (p - d) * s for p, d, s in zip(hw_pos, hw_default, joint_sign)]


def _convert_hw_rows(rows: list[dict], hw_default: list[float], joint_sign: list[float]) -> None:
    for r in rows:
        hw_pos = [r[f"pos_{name}"] for name in JOINT_NAMES]
        hw_cmd = [r[f"cmd_pos_{name}"] for name in JOINT_NAMES]
        rl_pos = _to_rl(hw_pos, hw_default, joint_sign)
        rl_cmd = _to_rl(hw_cmd, hw_default, joint_sign)
        for name, val in zip(JOINT_NAMES, rl_pos):
            r[f"pos_{name}"] = val
        for name, val in zip(JOINT_NAMES, rl_cmd):
            r[f"cmd_pos_{name}"] = val


def _bin_means(rows: list[dict], prefix: str, bins: int) -> dict[str, list[float]]:
    sums = {name: [0.0] * bins for name in JOINT_NAMES}
    counts = [0] * bins
    for r in rows:
        phase = r.get("cpg_phase")
        if phase is None:
            continue
        idx = int(math.floor(phase * bins))
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1
        for name in JOINT_NAMES:
            key = f"{prefix}_{name}"
            val = r.get(key)
            if val is None:
                continue
            sums[name][idx] += val
    means = {}
    for name, vals in sums.items():
        means[name] = [(s / counts[i]) if counts[i] else float("nan") for i, s in enumerate(vals)]
    return means


def _rms_phase_diff(hw_cmd: dict[str, list[float]], sim_cmd: dict[str, list[float]]) -> dict[str, float]:
    rms = {}
    for name in JOINT_NAMES:
        diffs = []
        for a, b in zip(hw_cmd[name], sim_cmd[name]):
            if math.isnan(a) or math.isnan(b):
                continue
            diffs.append((a - b) ** 2)
        rms[name] = math.sqrt(sum(diffs) / len(diffs)) if diffs else float("nan")
    return rms


def _mean_abs_error(rows: list[dict], a_prefix: str, b_prefix: str) -> dict[str, float]:
    out = {}
    for name in JOINT_NAMES:
        errs = []
        key_a = f"{a_prefix}_{name}"
        key_b = f"{b_prefix}_{name}"
        for r in rows:
            a = r.get(key_a)
            b = r.get(key_b)
            if a is None or b is None:
                continue
            errs.append(abs(a - b))
        out[name] = sum(errs) / len(errs) if errs else float("nan")
    return out


def _imu_stats(rows: list[dict]) -> dict[str, tuple[float, float]]:
    keys = [
        "imu_lin_vel_x", "imu_lin_vel_y", "imu_lin_vel_z",
        "imu_gyro_x", "imu_gyro_y", "imu_gyro_z",
        "imu_proj_g_x", "imu_proj_g_y", "imu_proj_g_z",
    ]
    stats = {}
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r.get(k), float)]
        if not vals:
            stats[k] = (float("nan"), float("nan"))
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        stats[k] = (mean, math.sqrt(var))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare hardware vs simulation gait logs.")
    parser.add_argument("--hw", type=Path, required=True, help="Hardware session CSV")
    parser.add_argument("--sim", type=Path, required=True, help="Simulation CSV")
    parser.add_argument("--stance", type=Path, default=Path("deployment/config/stance.yaml"), help="Stance YAML")
    parser.add_argument("--cpg-yaml", type=Path, default=Path("deployment/config/cpg.yaml"), help="CPG YAML")
    parser.add_argument("--phase-bins", type=int, default=20, help="Bins for phase-averaged comparison")
    args = parser.parse_args()

    with args.cpg_yaml.open("r", encoding="utf-8") as f:
        cpg = yaml.safe_load(f) or {}

    hw_default = _load_hw_default(args.stance)
    joint_sign = _expand_joint_sign(cpg)

    hw_rows = _load_rows(args.hw)
    sim_rows = _load_rows(args.sim)
    _convert_hw_rows(hw_rows, hw_default, joint_sign)

    hw_cmd = _bin_means(hw_rows, "cmd_pos", args.phase_bins)
    sim_cmd = _bin_means(sim_rows, "cmd_pos", args.phase_bins)
    rms = _rms_phase_diff(hw_cmd, sim_cmd)

    hw_track = _mean_abs_error(hw_rows, "cmd_pos", "pos")
    sim_track = _mean_abs_error(sim_rows, "cmd_pos", "pos")

    hw_imu = _imu_stats(hw_rows)
    sim_imu = _imu_stats(sim_rows)

    print(f"HW rows: {len(hw_rows)} | SIM rows: {len(sim_rows)}")
    print("\nRMS cmd_pos diff across phase bins (rad, HW converted to RL):")
    for name in JOINT_NAMES:
        print(f"  {name}: {rms[name]:.4f}")
    print("\nMean |cmd - pos| tracking error (rad):")
    for name in JOINT_NAMES:
        print(f"  {name}: hw={hw_track[name]:.4f} sim={sim_track[name]:.4f}")

    print("\nIMU mean +/- std:")
    for k in [
        "imu_lin_vel_x", "imu_lin_vel_y", "imu_lin_vel_z",
        "imu_gyro_x", "imu_gyro_y", "imu_gyro_z",
        "imu_proj_g_x", "imu_proj_g_y", "imu_proj_g_z",
    ]:
        h = hw_imu[k]
        s = sim_imu[k]
        print(f"  {k}: hw={h[0]:+.4f}+/-{h[1]:.4f}, sim={s[0]:+.4f}+/-{s[1]:.4f}")


if __name__ == "__main__":
    main()
