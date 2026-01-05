#!/usr/bin/env python3
"""Compare sim CPG cmd_pos against hardware CPG generator outputs."""
import argparse
import csv
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from deployment.inference.cpg_generator import CPGConfig, CPGGenerator


JOINT_NAMES = [
    "fl_sh", "fr_sh", "bl_sh", "br_sh",
    "fl_th", "fr_th", "bl_th", "br_th",
    "fl_ca", "fr_ca", "bl_ca", "br_ca",
]


def _load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        for k, v in list(r.items()):
            try:
                r[k] = float(v)
            except Exception:
                pass
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare sim CPG cmd_pos vs hardware generator.")
    parser.add_argument("--sim", type=Path, required=True, help="Simulation log CSV")
    parser.add_argument("--cpg-yaml", type=Path, default=Path("deployment/config/cpg.yaml"))
    parser.add_argument("--skip-seconds", type=float, default=0.0)
    args = parser.parse_args()

    if not args.sim.exists():
        raise FileNotFoundError(f"Sim log not found: {args.sim}")
    if not args.cpg_yaml.exists():
        raise FileNotFoundError(f"CPG config not found: {args.cpg_yaml}")

    cfg = CPGConfig.from_yaml(args.cpg_yaml)
    cpg = CPGGenerator(cfg)

    rows = _load_rows(args.sim)
    if args.skip_seconds > 0.0:
        rows = [r for r in rows if isinstance(r.get("sim_time"), float) and r["sim_time"] >= args.skip_seconds]

    if not rows:
        raise RuntimeError("No rows to compare after filtering.")

    sums = {name: 0.0 for name in JOINT_NAMES}
    counts = {name: 0 for name in JOINT_NAMES}

    for r in rows:
        phase = r.get("cpg_phase")
        sim_time = r.get("sim_time")
        if isinstance(phase, float) and cfg.frequency_hz > 0.0:
            ref_time = phase / cfg.frequency_hz
        elif isinstance(sim_time, float):
            ref_time = sim_time
        else:
            continue
        ref = cpg.compute(ref_time)
        for idx, name in enumerate(JOINT_NAMES):
            key = f"cmd_pos_{name}"
            val = r.get(key)
            if not isinstance(val, float):
                continue
            diff = val - float(ref[idx])
            sums[name] += diff * diff
            counts[name] += 1

    print(f"Rows compared: {len(rows)}")
    print("RMS cmd_pos diff vs hardware generator (rad):")
    for name in JOINT_NAMES:
        if counts[name] == 0:
            rms = float("nan")
        else:
            rms = math.sqrt(sums[name] / counts[name])
        print(f"  {name}: {rms:.6f}")


if __name__ == "__main__":
    main()
