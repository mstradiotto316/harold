"""Shared stance loader for hardware defaults.

Reads deployment/config/stance.yaml (or HAROLD_STANCE_PATH) and exposes
ready-stance defaults in both hardware and RL conventions.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml


DEFAULT_HW_POSE = np.array(
    [
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
    ],
    dtype=np.float32,
)

DEFAULT_RL_POSE = np.array(
    [
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
    ],
    dtype=np.float32,
)


def _resolve_stance_path(cpg_path: Optional[Path] = None) -> Optional[Path]:
    env_path = os.getenv("HAROLD_STANCE_PATH", "").strip()
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.exists():
            return candidate

    if cpg_path is not None:
        candidate = Path(cpg_path).parent / "stance.yaml"
        if candidate.exists():
            return candidate

    candidate = Path(__file__).resolve().parents[1] / "config" / "stance.yaml"
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


def _expand_group(value: Any, defaults: np.ndarray) -> list[float]:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return [float(v) for v in value]
    if value is None:
        return [float(defaults[0])] * 4
    return [float(value)] * 4


def _expand_pose(pose: Any, defaults: np.ndarray) -> np.ndarray:
    if not isinstance(pose, dict):
        return defaults.copy()

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

    return np.array(shoulders + thighs + calves, dtype=np.float32)


def load_stance_data(cpg_path: Optional[Path] = None) -> dict[str, Any]:
    path = _resolve_stance_path(cpg_path)
    if path is None:
        return {}
    return _load_yaml(path)


def load_hw_default_pose(cpg_path: Optional[Path] = None) -> np.ndarray:
    data = load_stance_data(cpg_path)
    return _expand_pose(data.get("hw_default_pose", {}), DEFAULT_HW_POSE)


def load_rl_default_pose(cpg_path: Optional[Path] = None) -> np.ndarray:
    data = load_stance_data(cpg_path)
    return _expand_pose(data.get("rl_default_pose", {}), DEFAULT_RL_POSE)
