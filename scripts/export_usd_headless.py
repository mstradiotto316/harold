#!/usr/bin/env python3
"""Headless USD export via Isaac Sim (Kit).

Exports readable USDA copies and a joint metadata CSV for inspection.

Run with Isaac Sim's python:
  ~/isaacsim/python.sh scripts/export_usd_headless.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(description="Headless USD export (Isaac Sim)")
    parser.add_argument(
        "--usd",
        type=Path,
        default=repo_root / "part_files" / "V4" / "harold_8.usd",
        help="Path to the USD file to export",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "docs" / "usd_exports",
        help="Directory for USDA/CSV outputs",
    )
    parser.add_argument(
        "--wait-frames",
        type=int,
        default=10,
        help="Number of Kit update frames to process after opening stage",
    )
    return parser.parse_args()


def _get_attr_str(prim, name: str) -> str:
    attr = prim.GetAttribute(name)
    if not attr:
        return ""
    value = attr.Get()
    return "" if value is None else str(value)


def _export_joint_table(stage, csv_path: Path) -> None:
    fields = [
        "path",
        "type",
        "axis",
        "localPos0",
        "localRot0",
        "localPos1",
        "localRot1",
        "body0",
        "body1",
        "lowerLimit",
        "upperLimit",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for prim in stage.Traverse():
            prim_type = prim.GetTypeName()
            if "Joint" not in prim_type:
                continue
            writer.writerow(
                {
                    "path": str(prim.GetPath()),
                    "type": prim_type,
                    "axis": _get_attr_str(prim, "physics:axis"),
                    "localPos0": _get_attr_str(prim, "physics:localPos0"),
                    "localRot0": _get_attr_str(prim, "physics:localRot0"),
                    "localPos1": _get_attr_str(prim, "physics:localPos1"),
                    "localRot1": _get_attr_str(prim, "physics:localRot1"),
                    "body0": _get_attr_str(prim, "physics:body0"),
                    "body1": _get_attr_str(prim, "physics:body1"),
                    "lowerLimit": _get_attr_str(prim, "physics:lowerLimit"),
                    "upperLimit": _get_attr_str(prim, "physics:upperLimit"),
                }
            )


def main() -> int:
    args = _parse_args()
    usd_path = args.usd.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not usd_path.exists():
        print(f"ERROR: USD not found: {usd_path}")
        return 2

    try:
        from isaacsim.simulation_app import SimulationApp
    except Exception as exc:
        print("ERROR: Isaac Sim python not available.")
        print("Run with: ~/isaacsim/python.sh scripts/export_usd_headless.py")
        print(f"Details: {exc}")
        return 1

    simulation_app = SimulationApp(
        {
            "headless": True,
            "sync_loads": True,
            "create_new_stage": False,
            "renderer": "RaytracedLighting",
        }
    )
    try:
        import omni.usd
        import omni.kit.app
        from isaacsim.core.utils.stage import is_stage_loading
    except Exception as exc:
        print(f"ERROR: Failed to import Kit modules: {exc}")
        simulation_app.close()
        return 1

    try:
        ctx = omni.usd.get_context()
        if not ctx.open_stage(str(usd_path)):
            print(f"ERROR: Failed to open stage: {usd_path}")
            return 3

        app = omni.kit.app.get_app()
        for _ in range(max(0, args.wait_frames)):
            app.update()

        while is_stage_loading():
            app.update()

        stage = ctx.get_stage()
        if stage is None:
            print("ERROR: Stage not available after open.")
            return 4

        root_path = out_dir / f"{usd_path.stem}_root.usda"
        flat_path = out_dir / f"{usd_path.stem}_flat.usda"
        joints_path = out_dir / f"{usd_path.stem}_joints.csv"

        stage.GetRootLayer().Export(str(root_path))
        stage.Flatten().Export(str(flat_path))
        _export_joint_table(stage, joints_path)

        print(f"Wrote root USDA: {root_path}")
        print(f"Wrote flat USDA: {flat_path}")
        print(f"Wrote joints CSV: {joints_path}")
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
