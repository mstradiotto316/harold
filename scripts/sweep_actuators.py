#!/usr/bin/env python3
"""Sweep actuator parameters and compare sim tracking to hardware."""
import argparse
import csv
import math
import os
import subprocess
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


def _aggregate_group(values: dict[str, float], indices: range) -> float:
    selected = [values[JOINT_NAMES[i]] for i in indices]
    return sum(selected) / len(selected)


def _compute_tracking_metrics(hw_rows: list[dict], sim_rows: list[dict]) -> dict[str, float]:
    hw_track = _mean_abs_error(hw_rows, "cmd_pos", "pos")
    sim_track = _mean_abs_error(sim_rows, "cmd_pos", "pos")

    hw_sh = _aggregate_group(hw_track, range(0, 4))
    hw_th = _aggregate_group(hw_track, range(4, 8))
    hw_ca = _aggregate_group(hw_track, range(8, 12))

    sim_sh = _aggregate_group(sim_track, range(0, 4))
    sim_th = _aggregate_group(sim_track, range(4, 8))
    sim_ca = _aggregate_group(sim_track, range(8, 12))

    cost_sh = abs(sim_sh - hw_sh)
    cost_th = abs(sim_th - hw_th)
    cost_ca = abs(sim_ca - hw_ca)
    cost_all = (cost_sh + cost_th + cost_ca) / 3.0

    return {
        "hw_sh": hw_sh,
        "hw_th": hw_th,
        "hw_ca": hw_ca,
        "sim_sh": sim_sh,
        "sim_th": sim_th,
        "sim_ca": sim_ca,
        "cost_sh": cost_sh,
        "cost_th": cost_th,
        "cost_ca": cost_ca,
        "cost_all": cost_all,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep actuator params for sim-to-real tracking.")
    parser.add_argument("--hw-log", type=Path, required=True, help="Hardware session CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("deployment/validation/sim_logs"))
    parser.add_argument("--mode", choices=("scripted", "cpg"), default="scripted")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--log-rate-hz", type=float, default=5.0)
    parser.add_argument("--skip-seconds", type=float, default=3.0)
    parser.add_argument("--cmd-vx", type=float, default=0.3)
    parser.add_argument("--cmd-vy", type=float, default=0.0)
    parser.add_argument("--cmd-yaw", type=float, default=0.0)
    parser.add_argument("--cpg-yaml", type=Path, default=Path("deployment/config/cpg.yaml"))
    parser.add_argument("--stance", type=Path, default=Path("deployment/config/stance.yaml"))
    parser.add_argument("--enable-domain-rand", action="store_true")
    args = parser.parse_args()

    param_sets = [
        {"stiffness": 400, "damping": 150, "effort": 2.8},
        {"stiffness": 600, "damping": 120, "effort": 2.8},
        {"stiffness": 800, "damping": 100, "effort": 2.8},
        {"stiffness": 1200, "damping": 75, "effort": 2.8},
        {"stiffness": 1200, "damping": 50, "effort": 2.8},
        {"stiffness": 1200, "damping": 50, "effort": 4.0},
        {"stiffness": 800, "damping": 50, "effort": 4.0},
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.cpg_yaml.open("r", encoding="utf-8") as f:
        cpg = yaml.safe_load(f) or {}
    hw_default = _load_hw_default(args.stance)
    joint_sign = _expand_joint_sign(cpg)

    hw_rows = _load_rows(args.hw_log)
    _convert_hw_rows(hw_rows, hw_default, joint_sign)

    skip_rows = int(round(args.skip_seconds * args.log_rate_hz))
    if skip_rows > 0:
        hw_rows = hw_rows[skip_rows:]

    results = []

    for params in param_sets:
        stiffness = params["stiffness"]
        damping = params["damping"]
        effort = params["effort"]

        tag = f"S{stiffness}_D{damping}_E{str(effort).replace('.', 'p')}"
        out_path = args.output_dir / f"sim_{args.mode}_sweep_{tag}.csv"

        env = os.environ.copy()
        env["HAROLD_ACTUATOR_STIFFNESS"] = str(stiffness)
        env["HAROLD_ACTUATOR_DAMPING"] = str(damping)
        env["HAROLD_ACTUATOR_EFFORT_LIMIT"] = str(effort)

        cmd = [
            "python",
            "harold_isaac_lab/scripts/log_gait_playback.py",
            "--mode",
            args.mode,
            "--duration",
            str(args.duration),
            "--log-rate-hz",
            str(args.log_rate_hz),
            "--cmd-vx",
            str(args.cmd_vx),
            "--cmd-vy",
            str(args.cmd_vy),
            "--cmd-yaw",
            str(args.cmd_yaw),
            "--cpg-yaml",
            str(args.cpg_yaml),
            "--output",
            str(out_path),
        ]
        if args.enable_domain_rand:
            cmd.append("--enable-domain-rand")

        print(f"\n[RUN] stiffness={stiffness} damping={damping} effort={effort}")
        subprocess.run(cmd, check=True, env=env)

        sim_rows = _load_rows(out_path)
        if skip_rows > 0:
            sim_rows = sim_rows[skip_rows:]

        metrics = _compute_tracking_metrics(hw_rows, sim_rows)
        metrics.update({
            "stiffness": stiffness,
            "damping": damping,
            "effort": effort,
            "sim_log": str(out_path),
        })
        results.append(metrics)

        print(
            f"[METRIC] cost_all={metrics['cost_all']:.4f} "
            f"(sim_sh={metrics['sim_sh']:.4f}, sim_th={metrics['sim_th']:.4f}, sim_ca={metrics['sim_ca']:.4f})"
        )

    results.sort(key=lambda r: r["cost_all"])

    result_path = args.output_dir / "actuator_sweep_results.csv"
    fieldnames = [
        "stiffness", "damping", "effort", "cost_all", "cost_sh", "cost_th", "cost_ca",
        "sim_sh", "sim_th", "sim_ca", "hw_sh", "hw_th", "hw_ca", "sim_log",
    ]
    with result_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in fieldnames})

    best = results[0]
    print("\n[RESULT] Best match:")
    print(
        f"  stiffness={best['stiffness']} damping={best['damping']} effort={best['effort']} "
        f"cost_all={best['cost_all']:.4f}"
    )
    print(f"  results saved: {result_path}")


if __name__ == "__main__":
    main()
