import argparse
import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_harold_cli_module():
    script_path = REPO_ROOT / "scripts" / "harold.py"
    spec = importlib.util.spec_from_file_location("harold_cli", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_args(module):
    return argparse.Namespace(
        task=None,
        iterations=None,
        duration=module.DEFAULT_DURATION,
        num_envs=None,
        hypothesis="",
        tags="",
        mode="rl",
        checkpoint=None,
        gait_scale=None,
        no_watchdog=False,
    )


def test_compare_displays_manager_metrics_with_missing_values(tmp_path, capsys):
    harold = _load_harold_cli_module()
    runs = []
    for alias, metrics in (("EXP-001", {"gait": 12.75}), ("EXP-002", {})):
        path = tmp_path / alias
        path.mkdir()
        harold.save_manifest(path, {"alias": alias, "summary": {"final": metrics}})
        runs.append(str(path))

    assert harold.cmd_compare(argparse.Namespace(experiments=runs)) == 0
    rows = capsys.readouterr().out.splitlines()
    assert any(row.split() == ["EXP-001", "EXP-002"] for row in rows)
    assert any(row.split() == ["gait", "12.750", "?"] for row in rows)


@pytest.mark.parametrize("extra_args", [[], ["--video", "--video-interval", "100", "--video-length", "12"]])
def test_train_cli_always_builds_video_command(monkeypatch, extra_args):
    harold = _load_harold_cli_module()
    command = []
    build = harold.build_train_command

    def capture(*args, **kwargs):
        command.extend(build(*args, **kwargs))
        return command

    monkeypatch.setattr(sys, "argv", ["harold", "train", *extra_args])
    monkeypatch.setattr(harold, "build_train_command", capture)
    # Stop after command construction, before launching a simulator.
    monkeypatch.setattr(harold, "is_training_running", lambda: harold.TrainingStatus(True, 123, 1.0))
    assert harold.main() == 1
    assert "--video" in command
    assert command[command.index("--video_interval") + 1] == str(100 if extra_args else harold.TRAINING_DEFAULTS["video_interval"])
    assert command[command.index("--video_length") + 1] == str(12 if extra_args else harold.TRAINING_DEFAULTS["video_length"])


def test_cmd_train_rejects_active_run(monkeypatch, capsys):
    """Re-running `harold train` must not kill a tracked active experiment."""
    harold = _load_harold_cli_module()
    stop_calls = []

    monkeypatch.setattr(harold, "build_train_command", lambda *args, **kwargs: ["train"])
    monkeypatch.setattr(
        harold,
        "is_training_running",
        lambda: harold.TrainingStatus(running=True, pid=4321, elapsed_seconds=10.0),
    )
    monkeypatch.setattr(harold, "find_training_processes", lambda: [])
    monkeypatch.setattr(harold, "cmd_stop", lambda args: stop_calls.append(args))

    result = harold.cmd_train(_make_args(harold))
    output = capsys.readouterr().out

    assert result == 1
    assert "already running" in output
    assert stop_calls == []


def test_cmd_train_rejects_orphans_without_auto_stop(monkeypatch, capsys):
    """Orphan cleanup should remain explicit instead of happening on train launch."""
    harold = _load_harold_cli_module()
    stop_calls = []

    monkeypatch.setattr(harold, "build_train_command", lambda *args, **kwargs: ["train"])
    monkeypatch.setattr(
        harold,
        "is_training_running",
        lambda: harold.TrainingStatus(running=False, pid=None, elapsed_seconds=None),
    )
    monkeypatch.setattr(harold, "find_training_processes", lambda: [{"pid": 9876, "tracked": False}])
    monkeypatch.setattr(harold, "cmd_stop", lambda args: stop_calls.append(args))

    result = harold.cmd_train(_make_args(harold))
    output = capsys.readouterr().out

    assert result == 1
    assert "harold stop" in output
    assert stop_calls == []


def test_get_metrics_reads_termination_counters_with_info_prefix(tmp_path):
    """Status diagnostics should resolve the actual TensorBoard scalar names."""
    tensorboard = pytest.importorskip("torch.utils.tensorboard")
    harold = _load_harold_cli_module()

    writer = tensorboard.SummaryWriter(log_dir=str(tmp_path))
    writer.add_scalar("Info / Episode_Termination/orientation", 3.0, 1)
    writer.add_scalar("Info / Episode_Termination/body_contact", 5.0, 1)
    writer.add_scalar("Info / Episode_Metric/termination_body_contact", 99.0, 1)
    writer.add_scalar("Info / Episode_Metric/termination_height", 2.0, 1)
    writer.add_scalar("Info / Episode_Termination/time_out", 7.0, 1)
    writer.add_scalar("Info / Episode_Metric/vx_w_mean", 0.12, 1)
    writer.flush()
    writer.close()

    metrics = harold.get_metrics(tmp_path)

    assert metrics["term_orientation"] == 3.0
    assert metrics["term_body_contact"] == 5.0
    assert metrics["term_height"] == 2.0
    assert "term_elbow_pose" not in metrics
    assert metrics["term_timeout"] == 7.0
