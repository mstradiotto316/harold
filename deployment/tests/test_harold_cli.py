import argparse
import importlib.util
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
    writer.add_scalar("Info / Episode_Termination/time_out", 7.0, 1)
    writer.add_scalar("Info / Episode_Metric/vx_w_mean", 0.12, 1)
    writer.flush()
    writer.close()

    metrics = harold.get_metrics(tmp_path)

    assert metrics["term_orientation"] == 3.0
    assert metrics["term_body_contact"] == 5.0
    assert metrics["term_timeout"] == 7.0
