import argparse
import importlib.util
from pathlib import Path


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
