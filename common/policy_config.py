"""Shared policy configuration for training, export, and deployment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # Deployment and export normally have PyYAML available.
    yaml = None


ACTION_DIM = 12
DEFAULT_ACTION_SCALE = 0.5

JOINT_ORDER = [
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

JOINT_CATEGORIES = (
    "shoulder",
    "shoulder",
    "shoulder",
    "shoulder",
    "thigh",
    "thigh",
    "thigh",
    "thigh",
    "calf",
    "calf",
    "calf",
    "calf",
)

JOINT_SIGN = [
    1.0,
    1.0,
    1.0,
    1.0,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
]

JOINT_RANGE_BY_CATEGORY = {
    "shoulder": 0.30,
    "thigh": 0.90,
    "calf": 0.90,
}
JOINT_RANGE = tuple(JOINT_RANGE_BY_CATEGORY[name] for name in JOINT_CATEGORIES)

FLAT_JOINT_LIMITS_BY_CATEGORY = {
    "shoulder": (-0.4363, 0.4363),
    "thigh": (-0.0873, 0.9599),
    "calf": (-1.3963, 0.0873),
}
FLAT_JOINT_ANGLE_MIN = tuple(FLAT_JOINT_LIMITS_BY_CATEGORY[name][0] for name in JOINT_CATEGORIES)
FLAT_JOINT_ANGLE_MAX = tuple(FLAT_JOINT_LIMITS_BY_CATEGORY[name][1] for name in JOINT_CATEGORIES)

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


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "AGENTS.md").exists():
            return parent
    return Path(__file__).resolve().parents[1]


def _resolve_stance_path() -> Path | None:
    env_path = os.getenv("HAROLD_STANCE_PATH", "").strip()
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.exists():
            return candidate

    candidate = _repo_root() / "deployment" / "config" / "stance.yaml"
    if candidate.exists():
        return candidate
    return None


def _resolve_hardware_path() -> Path | None:
    candidate = _repo_root() / "deployment" / "config" / "hardware.yaml"
    if candidate.exists():
        return candidate
    return None


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _expand_group(value: Any, defaults: list[float]) -> list[float]:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return [float(item) for item in value]
    if value is None:
        return [float(defaults[0])] * 4
    return [float(value)] * 4


def expand_joint_sign(value: Any, defaults: list[float] | None = None) -> list[float]:
    """Expand a joint-sign config into the canonical 12D joint order."""
    joint_defaults = list(defaults or JOINT_SIGN)

    if isinstance(value, (list, tuple)) and len(value) == ACTION_DIM:
        return [float(item) for item in value]

    if not isinstance(value, dict):
        return list(joint_defaults)

    shoulders_value = value.get("shoulders")
    if isinstance(shoulders_value, (list, tuple)) and len(shoulders_value) == 4:
        shoulders = [float(item) for item in shoulders_value]
    else:
        shoulders = [
            value.get("shoulder_fl", shoulders_value),
            value.get("shoulder_fr", shoulders_value),
            value.get("shoulder_bl", shoulders_value),
            value.get("shoulder_br", shoulders_value),
        ]
        shoulders = [
            float(item) if item is not None else float(joint_defaults[index])
            for index, item in enumerate(shoulders)
        ]

    thighs = _expand_group(value.get("thighs"), joint_defaults[4:8])
    calves = _expand_group(value.get("calves"), joint_defaults[8:12])
    return shoulders + thighs + calves


def load_hardware_joint_sign(path: Path | None = None) -> list[float]:
    """Load the deployment joint-sign convention from hardware.yaml."""
    hardware_path = path or _resolve_hardware_path()
    if hardware_path is None:
        return list(JOINT_SIGN)

    hardware_data = _load_yaml(hardware_path)
    servos = hardware_data.get("servos", {}) if isinstance(hardware_data, dict) else {}
    return expand_joint_sign(servos.get("joint_sign", {}), defaults=list(JOINT_SIGN))


def resolve_deployment_joint_sign(
    *,
    metadata: dict[str, Any] | None = None,
    hardware_path: Path | None = None,
) -> list[float]:
    """Resolve the deployment sign convention and reject mismatched metadata."""
    hardware_sign = load_hardware_joint_sign(hardware_path)
    metadata = metadata or {}

    metadata_sign = metadata.get("joint_sign")
    if metadata_sign is None:
        return hardware_sign

    if not isinstance(metadata_sign, (list, tuple)) or len(metadata_sign) != ACTION_DIM:
        raise ValueError(f"Expected metadata joint_sign to have {ACTION_DIM} entries")

    normalized_metadata = [float(item) for item in metadata_sign]
    if normalized_metadata != hardware_sign:
        raise ValueError(
            "Exported metadata joint_sign does not match deployment hardware.yaml. "
            "Refuse to start with inconsistent sign conventions."
        )

    return normalized_metadata


def _expand_pose(pose: Any, defaults: list[float]) -> list[float]:
    if not isinstance(pose, dict):
        return list(defaults)

    shoulders_value = pose.get("shoulders")
    if isinstance(shoulders_value, (list, tuple)) and len(shoulders_value) == 4:
        shoulders = [float(item) for item in shoulders_value]
    else:
        shoulders = [
            pose.get("shoulder_fl", shoulders_value),
            pose.get("shoulder_fr", shoulders_value),
            pose.get("shoulder_bl", shoulders_value),
            pose.get("shoulder_br", shoulders_value),
        ]
        shoulders = [
            float(value) if value is not None else float(defaults[index])
            for index, value in enumerate(shoulders)
        ]

    thighs = _expand_group(pose.get("thighs"), defaults[4:8])
    calves = _expand_group(pose.get("calves"), defaults[8:12])
    return shoulders + thighs + calves


def load_rl_default_pose() -> list[float]:
    stance_path = _resolve_stance_path()
    if stance_path is None:
        return list(DEFAULT_RL_POSE)
    stance_data = _load_yaml(stance_path)
    return _expand_pose(stance_data.get("rl_default_pose", {}), DEFAULT_RL_POSE)


def load_rl_default_pose_dict() -> dict[str, float]:
    return {name: float(value) for name, value in zip(JOINT_ORDER, load_rl_default_pose())}
