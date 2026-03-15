#!/usr/bin/env python3
"""Export a trained Harold policy to TorchScript and ONNX."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.policy_config import (
    DEFAULT_ACTION_SCALE,
    FLAT_JOINT_LIMITS_BY_CATEGORY,
    JOINT_ORDER,
    JOINT_RANGE_BY_CATEGORY,
    JOINT_SIGN,
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


def _read_training_action_scale(checkpoint_path: Path) -> float:
    """Read action_scale from the training run's manifest, falling back to the default.

    Checks three locations within the manifest (in priority order):
      1. metadata.action_scale  (legacy / direct field)
      2. training_config.action_scale  (written by harold.py since 2026-03-15)
      3. top-level action_scale
    """
    # Checkpoint is typically at <run_dir>/checkpoints/<file> or <run_dir>/<file>
    for parent in [checkpoint_path.parent, checkpoint_path.parent.parent]:
        manifest = parent / "manifest.json"
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                # Check metadata (legacy)
                meta = data.get("metadata", {})
                if "action_scale" in meta:
                    return float(meta["action_scale"])
                # Check training_config (current)
                tc = data.get("training_config", {})
                if "action_scale" in tc:
                    return float(tc["action_scale"])
                # Check top-level
                if "action_scale" in data:
                    return float(data["action_scale"])
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
    return DEFAULT_ACTION_SCALE


def build_policy_metadata(
    checkpoint_path: Path,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    log_std_parameter: torch.Tensor,
) -> dict[str, object]:
    """Build deployment metadata from the active training configuration."""
    return {
        "schema_version": 2,
        "observation_dim": int(running_mean.numel()),
        "action_dim": len(JOINT_ORDER),
        "action_scale": _read_training_action_scale(checkpoint_path),
        "joint_order": JOINT_ORDER,
        "default_joint_pos": load_rl_default_pose_dict(),
        "joint_range": JOINT_RANGE_BY_CATEGORY,
        "joint_angle_min": {key: float(bounds[0]) for key, bounds in FLAT_JOINT_LIMITS_BY_CATEGORY.items()},
        "joint_angle_max": {key: float(bounds[1]) for key, bounds in FLAT_JOINT_LIMITS_BY_CATEGORY.items()},
        "joint_angle_limits": {
            key: float(max(abs(bounds[0]), abs(bounds[1])))
            for key, bounds in FLAT_JOINT_LIMITS_BY_CATEGORY.items()
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
