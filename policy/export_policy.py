#!/usr/bin/env python3
"""Export a trained Harold policy to TorchScript and ONNX."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import yaml
except ImportError:  # Export normally runs with PyYAML available.
    yaml = None

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.policy_config import (
    ACTION_DIM,
    DEFAULT_ACTION_SCALE,
    JOINT_CATEGORIES,
    JOINT_ORDER,
    JOINT_RANGE_BY_CATEGORY,
    JOINT_SIGN,
    TASK_ACTION_SCALE_DEFAULTS,
    TASK_JOINT_LIMITS_BY_CATEGORY,
    load_rl_default_pose_dict,
)


class SharedPolicyValue(torch.nn.Module):
    """Minimal replica of the skrl shared policy/value network."""

    def __init__(self, obs_dim: int, hidden_dims: tuple[int, int, int], action_dim: int):
        super().__init__()
        h1, h2, h3 = hidden_dims
        self.net_container = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, h1),
            torch.nn.ELU(),
            torch.nn.Linear(h1, h2),
            torch.nn.ELU(),
            torch.nn.Linear(h2, h3),
            torch.nn.ELU(),
        )
        self.policy_layer = torch.nn.Linear(h3, action_dim)
        self.value_layer = torch.nn.Linear(h3, 1)
        self.log_std_parameter = torch.nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor):
        features = self.net_container(obs)
        mean = self.policy_layer(features)
        value = self.value_layer(features)
        log_std = self.log_std_parameter.expand_as(mean)
        return mean, value, log_std


class NormalizedPolicy(torch.nn.Module):
    """Wrap the policy with the same running-stat normalization used in training."""

    def __init__(self, base: SharedPolicyValue, running_mean: torch.Tensor, running_var: torch.Tensor):
        super().__init__()
        self.base = base
        self.register_buffer("running_mean", running_mean.clone())
        self.register_buffer("running_var", running_var.clone())
        self.eps = 1.0e-8

    def forward(self, obs: torch.Tensor):
        normalized_obs = (obs - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        mean, value, log_std = self.base(normalized_obs)
        return mean, value, log_std


_CATEGORY_ORDER = ("shoulder", "thigh", "calf")


@dataclass(frozen=True)
class ExportTrainingConfig:
    """Training-time action and joint metadata needed by deployment/export."""

    action_scale: float
    joint_range: dict[str, float]
    joint_angle_min: dict[str, float]
    joint_angle_max: dict[str, float]


def infer_policy_dims(policy_state: dict[str, torch.Tensor]) -> tuple[int, tuple[int, int, int], int]:
    """Infer network dimensions directly from checkpoint weights."""
    obs_dim = int(policy_state["net_container.0.weight"].shape[1])
    hidden_dims = (
        int(policy_state["net_container.0.weight"].shape[0]),
        int(policy_state["net_container.2.weight"].shape[0]),
        int(policy_state["net_container.4.weight"].shape[0]),
    )
    action_dim = int(policy_state["policy_layer.weight"].shape[0])
    return obs_dim, hidden_dims, action_dim


def _candidate_run_dirs(checkpoint_path: Path) -> list[Path]:
    candidates: list[Path] = []
    for candidate in (checkpoint_path.parent, checkpoint_path.parent.parent):
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _read_env_snapshot(path: Path) -> dict[str, Any] | None:
    if yaml is None:
        return None

    try:
        data = yaml.full_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _find_run_manifest(checkpoint_path: Path) -> dict[str, Any] | None:
    for run_dir in _candidate_run_dirs(checkpoint_path):
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            return _read_json(manifest_path)
    return None


def _find_env_snapshot(checkpoint_path: Path) -> dict[str, Any] | None:
    for run_dir in _candidate_run_dirs(checkpoint_path):
        env_path = run_dir / "params" / "env.yaml"
        if env_path.exists():
            return _read_env_snapshot(env_path)
    return None


def _manifest_action_scale(manifest: dict[str, Any] | None) -> float | None:
    if not manifest:
        return None

    meta = manifest.get("metadata", {})
    if isinstance(meta, dict) and "action_scale" in meta:
        try:
            return float(meta["action_scale"])
        except (TypeError, ValueError):
            pass

    training_config = manifest.get("training_config", {})
    if isinstance(training_config, dict) and "action_scale" in training_config:
        try:
            return float(training_config["action_scale"])
        except (TypeError, ValueError):
            pass

    if "action_scale" in manifest:
        try:
            return float(manifest["action_scale"])
        except (TypeError, ValueError):
            pass

    return None


def _manifest_task_key(manifest: dict[str, Any] | None) -> str | None:
    if not manifest:
        return None

    training_config = manifest.get("training_config", {})
    if isinstance(training_config, dict):
        task = training_config.get("task")
        if isinstance(task, str) and task in TASK_ACTION_SCALE_DEFAULTS:
            return task

    task = manifest.get("task")
    if isinstance(task, str) and task in TASK_ACTION_SCALE_DEFAULTS:
        return task

    return None


def _collapse_category_values(raw: Any, field_name: str) -> dict[str, float] | None:
    if raw is None:
        return None

    if isinstance(raw, dict):
        collapsed = {}
        for category in _CATEGORY_ORDER:
            if category not in raw:
                return None
            collapsed[category] = float(raw[category])
        return collapsed

    if not isinstance(raw, (list, tuple)):
        return None

    if len(raw) == len(_CATEGORY_ORDER):
        return {category: float(value) for category, value in zip(_CATEGORY_ORDER, raw)}

    if len(raw) != len(JOINT_ORDER):
        return None

    collapsed: dict[str, float] = {}
    for category, value in zip(JOINT_CATEGORIES, raw):
        numeric = float(value)
        if category in collapsed and abs(collapsed[category] - numeric) > 1.0e-6:
            raise ValueError(
                f"{field_name} varies per joint within '{category}'. "
                "Category-level export metadata cannot represent that safely."
            )
        collapsed[category] = numeric
    return collapsed


def _default_export_training_config(task_key: str | None) -> ExportTrainingConfig:
    normalized_task = task_key if task_key in TASK_ACTION_SCALE_DEFAULTS else "flat"
    joint_limits = TASK_JOINT_LIMITS_BY_CATEGORY[normalized_task]
    return ExportTrainingConfig(
        action_scale=float(TASK_ACTION_SCALE_DEFAULTS.get(normalized_task, DEFAULT_ACTION_SCALE)),
        joint_range={key: float(value) for key, value in JOINT_RANGE_BY_CATEGORY.items()},
        joint_angle_min={key: float(bounds[0]) for key, bounds in joint_limits.items()},
        joint_angle_max={key: float(bounds[1]) for key, bounds in joint_limits.items()},
    )


def resolve_export_training_config(checkpoint_path: Path) -> ExportTrainingConfig:
    """Resolve action scaling and joint limits from the run before using repo defaults."""

    manifest = _find_run_manifest(checkpoint_path)
    env_snapshot = _find_env_snapshot(checkpoint_path)
    config = _default_export_training_config(_manifest_task_key(manifest))

    action_scale = _manifest_action_scale(manifest)
    if action_scale is None and env_snapshot and "action_scale" in env_snapshot:
        try:
            action_scale = float(env_snapshot["action_scale"])
        except (TypeError, ValueError):
            action_scale = None

    joint_range = config.joint_range
    joint_angle_min = config.joint_angle_min
    joint_angle_max = config.joint_angle_max
    if env_snapshot:
        env_joint_range = _collapse_category_values(env_snapshot.get("joint_range"), "joint_range")
        if env_joint_range is not None:
            joint_range = env_joint_range

        env_joint_angle_min = _collapse_category_values(env_snapshot.get("joint_angle_min"), "joint_angle_min")
        if env_joint_angle_min is not None:
            joint_angle_min = env_joint_angle_min

        env_joint_angle_max = _collapse_category_values(env_snapshot.get("joint_angle_max"), "joint_angle_max")
        if env_joint_angle_max is not None:
            joint_angle_max = env_joint_angle_max

    return ExportTrainingConfig(
        action_scale=config.action_scale if action_scale is None else action_scale,
        joint_range=joint_range,
        joint_angle_min=joint_angle_min,
        joint_angle_max=joint_angle_max,
    )


def load_reference_policy(checkpoint_path: Path) -> tuple[NormalizedPolicy, torch.Tensor, torch.Tensor, int]:
    """Load the checkpoint and rebuild the normalized policy."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy_state = checkpoint["policy"]
    running_mean = checkpoint["state_preprocessor"]["running_mean"].float()
    running_var = checkpoint["state_preprocessor"]["running_variance"].float()

    obs_dim, hidden_dims, action_dim = infer_policy_dims(policy_state)
    if running_mean.numel() != obs_dim:
        raise ValueError(
            f"Checkpoint normalization stats are {running_mean.numel()}D but the policy expects {obs_dim}D observations."
        )

    base = SharedPolicyValue(obs_dim=obs_dim, hidden_dims=hidden_dims, action_dim=action_dim)
    base.load_state_dict(policy_state)
    base.eval()
    wrapper = NormalizedPolicy(base, running_mean, running_var).eval()
    return wrapper, running_mean, running_var, action_dim


def _compute_effective_action_scale(training_cfg: ExportTrainingConfig) -> list[float]:
    """Compute the per-joint effective action scale used during training.

    For manager-based envs (e.g. harold_mgr with scale=0.2), this is uniform:
        effective_scale[i] = action_scale (0.2 for all joints)

    For direct envs (e.g. flat with scale=0.5), this includes per-joint ranges:
        effective_scale[i] = action_scale * joint_range[category_i]

    The deployment pipeline should use this directly:
        target = default_pose + action * effective_scale
    """
    scale = training_cfg.action_scale
    joint_range = training_cfg.joint_range

    # Check if joint_range is the default JOINT_RANGE_BY_CATEGORY.
    # Manager-based envs use uniform scale (scale parameter IS the effective scale),
    # while direct envs multiply scale * joint_range per category.
    #
    # Heuristic: if action_scale <= 0.25, it's likely a manager-based env where
    # scale=0.2 is the full effective scale. If action_scale >= 0.5, it's a direct env
    # where scale is multiplied by joint_range.
    #
    # To be safe, we always compute scale * joint_range when joint_range differs
    # from the default, and use uniform scale when joint_range matches defaults.
    is_default_range = all(
        abs(joint_range.get(cat, JOINT_RANGE_BY_CATEGORY[cat]) - JOINT_RANGE_BY_CATEGORY[cat]) < 1e-6
        for cat in JOINT_RANGE_BY_CATEGORY
    )

    if is_default_range and scale < 0.3:
        # Manager-based env: scale IS the effective per-joint scale (uniform)
        return [scale] * ACTION_DIM
    else:
        # Direct env: effective_scale = action_scale * joint_range_per_category
        return [
            scale * joint_range.get(JOINT_CATEGORIES[i], JOINT_RANGE_BY_CATEGORY[JOINT_CATEGORIES[i]])
            for i in range(ACTION_DIM)
        ]


def build_policy_metadata(
    checkpoint_path: Path,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    log_std_parameter: torch.Tensor,
) -> dict[str, object]:
    """Build deployment metadata from the active training configuration."""
    training_cfg = resolve_export_training_config(checkpoint_path)
    effective_scale = _compute_effective_action_scale(training_cfg)
    return {
        "schema_version": 3,
        "observation_dim": int(running_mean.numel()),
        "action_dim": len(JOINT_ORDER),
        "action_scale": training_cfg.action_scale,
        "effective_action_scale": effective_scale,
        "joint_order": JOINT_ORDER,
        "default_joint_pos": load_rl_default_pose_dict(),
        "joint_range": training_cfg.joint_range,
        "joint_angle_min": training_cfg.joint_angle_min,
        "joint_angle_max": training_cfg.joint_angle_max,
        "joint_angle_limits": {
            key: float(max(abs(training_cfg.joint_angle_min[key]), abs(training_cfg.joint_angle_max[key])))
            for key in _CATEGORY_ORDER
        },
        "joint_sign": JOINT_SIGN,
        "running_mean": running_mean.tolist(),
        "running_variance": running_var.tolist(),
        "log_std_parameter": log_std_parameter.tolist(),
        "checkpoint_path": str(checkpoint_path),
    }


def export_policy(checkpoint_path: Path, output_dir: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    wrapper, running_mean, running_var, _ = load_reference_policy(checkpoint_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    example = torch.zeros(1, running_mean.numel(), dtype=torch.float32)

    traced = torch.jit.trace(wrapper, example)
    traced.save(str(output_dir / "harold_policy.ts"))

    torch.onnx.export(
        wrapper,
        example,
        str(output_dir / "harold_policy.onnx"),
        input_names=["obs"],
        output_names=["mean", "value", "log_std"],
        opset_version=17,
        dynamic_axes={
            "obs": {0: "batch"},
            "mean": {0: "batch"},
            "value": {0: "batch"},
            "log_std": {0: "batch"},
        },
    )

    policy_meta = build_policy_metadata(
        checkpoint_path=checkpoint_path,
        running_mean=running_mean,
        running_var=running_var,
        log_std_parameter=checkpoint["policy"]["log_std_parameter"],
    )
    policy_meta["running_count"] = int(checkpoint["state_preprocessor"]["current_count"].item())
    with open(output_dir / "policy_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(policy_meta, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Harold PPO policy")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the checkpoint")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("deployment/policy"),
        help="Output directory for TorchScript, ONNX, and metadata",
    )
    args = parser.parse_args()

    export_policy(args.checkpoint, args.output)
    print(f"Policy exported to {args.output}")


if __name__ == "__main__":
    main()
