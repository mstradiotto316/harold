"""Stance helpers for Isaac Lab tasks (RL convention)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError:  # Isaac Lab env should include PyYAML; fall back to defaults.
    yaml = None


DEFAULT_RL_POSE = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.65,
    0.65,
    0.65,
    0.65,
    -1.13,
    -1.13,
    -1.13,
    -1.13,
]

JOINT_NAMES = [
    "fl_shoulder_joint",
    "fr_shoulder_joint",
    "bl_shoulder_joint",
    "br_shoulder_joint",
    "fl_thigh_joint",
    "fr_thigh_joint",
    "bl_thigh_joint",
    "br_thigh_joint",
    "fl_calf_joint",
    "fr_calf_joint",
    "bl_calf_joint",
    "br_calf_joint",
]


def _resolve_stance_path() -> Optional[Path]:
    env_path = os.getenv("HAROLD_STANCE_PATH", "").strip()
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.exists():
            return candidate

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "deployment" / "config" / "stance.yaml"
        if candidate.exists():
            return candidate

    return None


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _expand_group(value: Any, defaults: list[float]) -> list[float]:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return [float(v) for v in value]
    if value is None:
        return [float(defaults[0])] * 4
    return [float(value)] * 4


def _expand_pose(pose: Any, defaults: list[float]) -> list[float]:
    if not isinstance(pose, dict):
        return list(defaults)

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
        shoulders = [
            float(val) if val is not None else float(defaults[i])
            for i, val in enumerate(shoulders)
        ]

    thighs = _expand_group(pose.get("thighs"), defaults[4:8])
    calves = _expand_group(pose.get("calves"), defaults[8:12])
    return shoulders + thighs + calves


def load_rl_default_pose() -> list[float]:
    path = _resolve_stance_path()
    if path is None:
        return list(DEFAULT_RL_POSE)

    if yaml is None:
        return list(DEFAULT_RL_POSE)
    data = _load_yaml(path)
    pose = data.get("rl_default_pose", {})
    return _expand_pose(pose, DEFAULT_RL_POSE)


def load_ready_pose_dict() -> dict[str, float]:
    pose = load_rl_default_pose()
    return {name: float(value) for name, value in zip(JOINT_NAMES, pose)}
