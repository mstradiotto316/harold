import importlib.util
from pathlib import Path
import sys

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_export_policy_module():
    script_path = REPO_ROOT / "policy" / "export_policy.py"
    spec = importlib.util.spec_from_file_location("export_policy_module", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_env_snapshot(path: Path, *, action_scale: float, joint_range, joint_angle_min, joint_angle_max) -> None:
    payload = {
        "action_scale": action_scale,
        "joint_range": tuple(joint_range),
        "joint_angle_min": tuple(joint_angle_min),
        "joint_angle_max": tuple(joint_angle_max),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(payload), encoding="utf-8")


def test_build_policy_metadata_uses_manifestless_env_snapshot_for_nonflat_export(tmp_path):
    export_policy = _load_export_policy_module()

    checkpoint_path = tmp_path / "terrain_58" / "checkpoints" / "best_agent.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"")

    _write_env_snapshot(
        tmp_path / "terrain_58" / "params" / "env.yaml",
        action_scale=1.0,
        joint_range=[0.30] * 4 + [0.90] * 8,
        joint_angle_min=[-0.5236] * 4 + [-1.5708] * 8,
        joint_angle_max=[0.5236] * 4 + [1.5708] * 8,
    )

    metadata = export_policy.build_policy_metadata(
        checkpoint_path=checkpoint_path,
        running_mean=torch.zeros(48),
        running_var=torch.ones(48),
        log_std_parameter=torch.zeros(12),
    )

    assert metadata["action_scale"] == 1.0
    assert metadata["joint_angle_min"]["shoulder"] == -0.5236
    assert metadata["joint_angle_max"]["calf"] == 1.5708


def test_resolve_export_training_config_falls_back_to_manifest_task_defaults(tmp_path):
    export_policy = _load_export_policy_module()

    checkpoint_path = tmp_path / "2026-03-15_02-03-30_ppo_torch" / "checkpoints" / "best_agent.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"")
    (checkpoint_path.parent.parent / "manifest.json").write_text(
        (
            "{\n"
            '  "training_config": {\n'
            '    "task": "rough"\n'
            "  }\n"
            "}\n"
        ),
        encoding="utf-8",
    )

    config = export_policy.resolve_export_training_config(checkpoint_path)

    assert config.action_scale == 1.0
    assert config.joint_angle_min["thigh"] == -1.5708
    assert config.joint_angle_max["shoulder"] == 0.5236
