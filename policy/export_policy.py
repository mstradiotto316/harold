#!/usr/bin/env python3
"""Export terrain_64_2 PPO policy to TorchScript and ONNX."""
import argparse
import json
import math
import os
from pathlib import Path

import torch

# --- Constants from training config ---
OBS_DIM = 48
ACTION_DIM = 12
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
JOINT_SIGN = [
    1.0, 1.0, 1.0, 1.0,
   -1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,
]
DEFAULT_JOINT_POS = {
    "fl_shoulder_joint": 0.0,
    "fr_shoulder_joint": 0.0,
    "bl_shoulder_joint": 0.0,
    "br_shoulder_joint": 0.0,
    "fl_thigh_joint": 0.3,
    "fr_thigh_joint": 0.3,
    "bl_thigh_joint": 0.3,
    "br_thigh_joint": 0.3,
    "fl_calf_joint": -0.75,
    "fr_calf_joint": -0.75,
    "bl_calf_joint": -0.75,
    "br_calf_joint": -0.75,
}
JOINT_RANGE = {
    "shoulder": 0.30,
    "thigh": 0.90,
    "calf": 0.90,
}
JOINT_LIMITS = {
    "shoulder": math.radians(30.0),
    "thigh": math.radians(90.0),
    "calf": math.radians(90.0),
}


class SharedPolicyValue(torch.nn.Module):
    """Minimal replica of skrl shared policy/value network."""

    def __init__(self):
        super().__init__()
        self.net_container = torch.nn.Sequential(
            torch.nn.Linear(OBS_DIM, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
        )
        self.policy_layer = torch.nn.Linear(128, ACTION_DIM)
        self.value_layer = torch.nn.Linear(128, 1)
        self.log_std_parameter = torch.nn.Parameter(torch.zeros(ACTION_DIM))

    def forward(self, obs: torch.Tensor):
        x = self.net_container(obs)
        mean = self.policy_layer(x)
        value = self.value_layer(x)
        log_std = self.log_std_parameter.expand_as(mean)
        return mean, value, log_std


class NormalizedPolicy(torch.nn.Module):
    """Wrap policy with running stats normalization."""

    def __init__(self, base: SharedPolicyValue, running_mean: torch.Tensor, running_var: torch.Tensor):
        super().__init__()
        self.base = base
        self.register_buffer("running_mean", running_mean.clone())
        self.register_buffer("running_var", running_var.clone())
        self.eps = 1.0e-8

    def forward(self, obs: torch.Tensor):
        norm_obs = (obs - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        mean, value, log_std = self.base(norm_obs)
        return mean, value, log_std


def export_policy(checkpoint_path: Path, output_dir: Path) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    base = SharedPolicyValue()
    base.load_state_dict(ckpt["policy"])
    base.eval()

    running_mean = ckpt["state_preprocessor"]["running_mean"].float()
    running_var = ckpt["state_preprocessor"]["running_variance"].float()

    wrapper = NormalizedPolicy(base, running_mean, running_var).eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    example = torch.zeros(1, OBS_DIM)

    # TorchScript export
    traced = torch.jit.trace(wrapper, example)
    traced.save(str(output_dir / "harold_policy.ts"))

    # ONNX export (mean, value, log_std outputs)
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

    # Write accompanying metadata
    policy_meta = {
        "action_scale": 1.0,
        "joint_order": JOINT_ORDER,
        "default_joint_pos": DEFAULT_JOINT_POS,
        "joint_range": JOINT_RANGE,
        "joint_angle_limits": {
            k: float(v) for k, v in JOINT_LIMITS.items()
        },
        "joint_sign": JOINT_SIGN,
        "running_mean": running_mean.tolist(),
        "running_variance": running_var.tolist(),
        "log_std_parameter": ckpt["policy"]["log_std_parameter"].tolist(),
        "running_count": int(ckpt["state_preprocessor"]["current_count"].item()),
        "checkpoint_path": str(checkpoint_path),
    }
    with open(output_dir / "policy_metadata.json", "w", encoding="utf-8") as f:
        json.dump(policy_meta, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Harold PPO policy")
    parser.add_argument("--checkpoint", type=Path, default=Path("logs/skrl/harold_direct/terrain_64_2/checkpoints/best_agent.pt"))
    parser.add_argument("--output", type=Path, default=Path("deployment_artifacts/terrain_64_2"))
    args = parser.parse_args()

    export_policy(args.checkpoint, args.output)
    print(f"Policy exported to {args.output}")


if __name__ == "__main__":
    main()
