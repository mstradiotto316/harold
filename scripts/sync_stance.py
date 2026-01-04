#!/usr/bin/env python3
"""Sync firmware DEFAULT_POSE from deployment/config/stance.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
STANCE_PATH = ROOT / "deployment" / "config" / "stance.yaml"
FIRMWARE_PATH = (
    ROOT
    / "firmware"
    / "StreamingControl"
    / "HaroldStreamingControl"
    / "HaroldStreamingControl.ino"
)


def _expand_group(value: Any, fallback: float) -> list[float]:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return [float(v) for v in value]
    if value is None:
        return [float(fallback)] * 4
    return [float(value)] * 4


def _expand_pose(pose: dict[str, Any]) -> list[float]:
    shoulders_value = pose.get("shoulders")
    if isinstance(shoulders_value, (list, tuple)) and len(shoulders_value) == 4:
        shoulders = [float(v) for v in shoulders_value]
    else:
        shoulders = [
            pose.get("shoulder_fl", shoulders_value),
            pose.get("shoulder_fr", shoulders_value),
            pose.get("shoulder_bl", shoulders_value),
            pose.get("shoulder_br", shoulders_value),
        ]
        shoulders = [0.0 if v is None else float(v) for v in shoulders]

    thighs = _expand_group(pose.get("thighs"), 0.0)
    calves = _expand_group(pose.get("calves"), 0.0)
    return shoulders + thighs + calves


def _format_pose(values: list[float]) -> str:
    chunks = [values[i : i + 4] for i in range(0, len(values), 4)]
    lines = []
    for chunk in chunks:
        line = ", ".join(f"{v:.2f}f" for v in chunk)
        lines.append(f"  {line},")
    # Replace trailing comma on last line
    if lines:
        lines[-1] = lines[-1].rstrip(",")
    return "\n".join(lines)


def _load_stance() -> dict[str, Any]:
    with open(STANCE_PATH, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("stance.yaml did not parse to a dict")
    return data


def update_firmware_default_pose() -> None:
    data = _load_stance()
    pose = data.get("hw_default_pose")
    if not isinstance(pose, dict):
        raise ValueError("stance.yaml missing hw_default_pose")

    values = _expand_pose(pose)
    if len(values) != 12:
        raise ValueError("hw_default_pose did not expand to 12 values")

    content = FIRMWARE_PATH.read_text(encoding="utf-8")
    marker = "const float DEFAULT_POSE[12] = {"
    start = content.find(marker)
    if start == -1:
        raise ValueError("DEFAULT_POSE array not found in firmware file")

    end = content.find("};", start)
    if end == -1:
        raise ValueError("DEFAULT_POSE terminator not found in firmware file")

    before = content[: start + len(marker)]
    after = content[end:]
    replacement = f"\n{_format_pose(values)}\n"
    updated = before + replacement + after
    FIRMWARE_PATH.write_text(updated, encoding="utf-8")
    print(f"Updated DEFAULT_POSE in {FIRMWARE_PATH}")


def main() -> None:
    if not STANCE_PATH.exists():
        raise SystemExit(f"Missing stance config: {STANCE_PATH}")
    if not FIRMWARE_PATH.exists():
        raise SystemExit(f"Missing firmware file: {FIRMWARE_PATH}")
    update_firmware_default_pose()


if __name__ == "__main__":
    main()
