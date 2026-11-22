#!/usr/bin/env python3
"""Test inference components without hardware.

These tests verify that the CPG, observation builder, and action converter
produce expected outputs without requiring real hardware.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_train_env_module():
    train_env_path = (
        REPO_ROOT
        / "harold_isaac_lab/source/harold_isaac_lab/harold_isaac_lab/tasks/direct/harold_flat/train_env.py"
    )
    spec = importlib.util.spec_from_file_location("harold_flat_train_env", train_env_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cpg_generator():
    """Test CPG trajectory generation."""
    from inference.cpg_generator import CPGGenerator, CPGConfig

    print("Testing CPG Generator...")

    cpg = CPGGenerator()

    # Test at t=0 (start of cycle)
    targets = cpg.compute(0.0)
    assert targets.shape == (12,), f"Expected shape (12,), got {targets.shape}"
    print(f"  t=0.0: phase={cpg.phase:.3f}, targets[0:3]={targets[:3]}")

    # Test at t=1.0 (half cycle at 0.5 Hz)
    targets = cpg.compute(1.0)
    assert cpg.phase == 0.5, f"Expected phase=0.5, got {cpg.phase}"
    print(f"  t=1.0: phase={cpg.phase:.3f}, targets[0:3]={targets[:3]}")

    # Test at t=2.0 (full cycle)
    targets = cpg.compute(2.0)
    assert abs(cpg.phase) < 0.01, f"Expected phase≈0, got {cpg.phase}"
    print(f"  t=2.0: phase={cpg.phase:.3f}, targets[0:3]={targets[:3]}")

    # Test phase sin/cos
    sin_val, cos_val = cpg.get_phase_sin_cos()
    assert -1 <= sin_val <= 1, f"Sin out of range: {sin_val}"
    assert -1 <= cos_val <= 1, f"Cos out of range: {cos_val}"
    print(f"  sin={sin_val:.3f}, cos={cos_val:.3f}")

    print("  PASSED")


def test_action_converter():
    """Test action conversion."""
    from inference.action_converter import ActionConverter

    print("Testing Action Converter...")

    converter = ActionConverter()

    # Test with zero policy output
    policy_output = np.zeros(12, dtype=np.float32)
    rl_targets, hw_targets = converter.compute_policy_targets(policy_output)

    assert rl_targets.shape == (12,), f"Expected shape (12,), got {rl_targets.shape}"
    assert hw_targets.shape == (12,), f"Expected shape (12,), got {hw_targets.shape}"
    print(f"  Zero policy: max |rl| = {np.max(np.abs(rl_targets)):.6f}")

    # Test with max policy output
    policy_output = np.ones(12, dtype=np.float32)
    rl_targets, hw_targets = converter.compute_policy_targets(policy_output)
    print(f"  Max policy: rl range = [{rl_targets.min():.4f}, {rl_targets.max():.4f}]")

    print("  PASSED")


def test_action_converter_respects_metadata_action_scale():
    """Deployment action conversion should honor export metadata action_scale."""
    from inference.action_converter import ActionConfig, ActionConverter

    config_dir = REPO_ROOT / "deployment" / "config"
    cfg = ActionConfig.from_yaml(
        config_dir / "cpg.yaml",
        config_dir / "hardware.yaml",
        metadata={
            "action_scale": 0.25,
            "joint_range": {"shoulder": 0.30, "thigh": 0.90, "calf": 0.90},
        },
    )
    converter = ActionConverter(cfg)
    converter.reset()

    rl_targets, _ = converter.compute_policy_targets(np.ones(12, dtype=np.float32))
    expected_delta = 0.25 * 0.90

    assert np.isclose(rl_targets[4] - cfg.rl_default_pose[4], expected_delta)


def test_deployment_joint_sign_uses_hardware_convention(tmp_path):
    """Action and observation conversion should share the hardware-backed sign convention."""
    from common.policy_config import JOINT_SIGN
    from inference.action_converter import ActionConfig
    from inference.observation_builder import ObservationConfig

    cpg_path = tmp_path / "cpg.yaml"
    cpg_path.write_text(
        "\n".join(
            [
                "joint_sign:",
                "  shoulders: [1.0, -1.0, 1.0, -1.0]",
                "  thighs: -1.0",
                "  calves: -1.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    hw_path = tmp_path / "hardware.yaml"
    hw_path.write_text(
        "\n".join(
            [
                "servos:",
                "  joint_sign:",
                "    shoulders: 1.0",
                "    thighs: -1.0",
                "    calves: -1.0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    action_cfg = ActionConfig.from_yaml(cpg_path, hw_path)
    obs_cfg = ObservationConfig.from_yaml(cpg_path, hw_path)
    expected = np.array(JOINT_SIGN, dtype=np.float32)

    assert np.array_equal(action_cfg.joint_sign, expected)
    assert np.array_equal(obs_cfg.joint_sign, expected)


def test_deployment_joint_sign_rejects_mismatched_metadata(tmp_path):
    """Deployment should fail closed if export metadata disagrees with hardware.yaml."""
    from inference.action_converter import ActionConfig
    from inference.observation_builder import ObservationConfig

    cpg_path = tmp_path / "cpg.yaml"
    cpg_path.write_text("joint_range: {}\n", encoding="utf-8")
    hw_path = tmp_path / "hardware.yaml"
    hw_path.write_text(
        "\n".join(
            [
                "servos:",
                "  joint_sign:",
                "    shoulders: 1.0",
                "    thighs: -1.0",
                "    calves: -1.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    bad_metadata = {
        "joint_sign": [1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    }

    with pytest.raises(ValueError, match="joint_sign"):
        ActionConfig.from_yaml(cpg_path, hw_path, metadata=bad_metadata)

    with pytest.raises(ValueError, match="joint_sign"):
        ObservationConfig.from_yaml(cpg_path, hw_path, metadata=bad_metadata)


def test_flat_command_errors_use_body_frame():
    """Flat-task command telemetry should compare body-frame commands to body-frame velocity."""
    train_env = _load_train_env_module()

    root_lin_vel_b = torch.tensor(
        [
            [0.20, -0.10, 0.00],
            [0.05, 0.30, 0.00],
        ],
        dtype=torch.float32,
    )
    commands = torch.tensor(
        [
            [0.10, 0.00, 0.00],
            [0.00, 0.20, 0.00],
        ],
        dtype=torch.float32,
    )

    cmd_vx_error, cmd_vy_error = train_env.compute_body_frame_command_errors(root_lin_vel_b, commands)

    assert torch.allclose(cmd_vx_error, torch.tensor([0.10, 0.05]))
    assert torch.allclose(cmd_vy_error, torch.tensor([0.10, 0.10]))


def test_forward_motion_reward_rejects_fallen_pose():
    """Forward reward should not leak to obviously fallen states."""
    train_env = _load_train_env_module()

    healthy_reward, healthy_mask = train_env.compute_forward_motion_reward(
        vx_b=torch.tensor([0.25]),
        upright=torch.tensor([0.92]),
        current_height=torch.tensor([0.21]),
        target_height=0.22,
        undesired_contacts=torch.tensor([0.0]),
        weight=3.0,
    )
    fallen_reward, fallen_mask = train_env.compute_forward_motion_reward(
        vx_b=torch.tensor([0.25]),
        upright=torch.tensor([0.35]),
        current_height=torch.tensor([0.08]),
        target_height=0.22,
        undesired_contacts=torch.tensor([1.0]),
        weight=3.0,
    )

    assert bool(healthy_mask.item())
    assert healthy_reward.item() > 0.0
    assert not bool(fallen_mask.item())
    assert fallen_reward.item() <= 0.0


def test_reset_policy_state_buffers_clears_ema_state():
    """Episode reset must clear smoothed actions and delayed action history."""
    from common.env_state import reset_policy_state_buffers

    env_ids = torch.tensor([0, 2], dtype=torch.long)
    actions = torch.ones(3, 12)
    previous_actions = torch.full((3, 12), 2.0)
    prev_target_delta = torch.full((3, 12), 3.0)
    actions_smooth = torch.full((3, 12), 4.0)
    action_delay_buffer = torch.full((3, 2, 12), 5.0)

    reset_policy_state_buffers(
        env_ids=env_ids,
        actions=actions,
        previous_actions=previous_actions,
        prev_target_delta=prev_target_delta,
        actions_smooth=actions_smooth,
        action_delay_buffer=action_delay_buffer,
    )

    assert torch.count_nonzero(actions[env_ids]) == 0
    assert torch.count_nonzero(previous_actions[env_ids]) == 0
    assert torch.count_nonzero(prev_target_delta[env_ids]) == 0
    assert torch.count_nonzero(actions_smooth[env_ids]) == 0
    assert torch.count_nonzero(action_delay_buffer[env_ids]) == 0
    assert torch.all(actions[1] == 1.0)


def test_onnx_policy():
    """Test ONNX policy inference."""
    import json

    try:
        import onnxruntime as ort
    except ImportError:
        print("SKIPPING: onnxruntime not installed")
        return

    print("Testing ONNX Policy...")

    policy_dir = Path(__file__).parent.parent / "policy"
    onnx_path = policy_dir / "harold_policy.onnx"
    meta_path = policy_dir / "policy_metadata.json"

    if not onnx_path.exists():
        print(f"  SKIPPING: {onnx_path} not found")
        return

    # Load policy
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])

    # Load metadata
    with open(meta_path) as f:
        meta = json.load(f)

    obs_dim = int(meta.get("observation_dim", len(meta["running_mean"])))
    print(f"  Observation dimension: {obs_dim}")
    assert obs_dim == len(meta["running_mean"])

    # Create dummy observation
    obs = np.zeros(obs_dim, dtype=np.float32)

    # Run inference
    import time
    start = time.time()
    num_iters = 100
    for _ in range(num_iters):
        outputs = session.run(['mean'], {'obs': obs.reshape(1, -1)})
    elapsed = time.time() - start

    action = outputs[0][0]
    print(f"  Action shape: {action.shape}")
    print(f"  Action range: [{action.min():.3f}, {action.max():.3f}]")
    print(f"  Inference time: {elapsed / num_iters * 1000:.2f} ms")

    print("  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Harold Inference Tests")
    print("=" * 60)
    print()

    test_cpg_generator()
    print()

    test_action_converter()
    print()

    test_onnx_policy()
    print()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


def test_manager_action_filter_starts_at_zero_after_every_reset():
    from inference.action_converter import ActionConfig, ActionConverter

    config_dir = REPO_ROOT / "deployment/config"
    cfg = ActionConfig.from_yaml(config_dir / "cpg.yaml", config_dir / "hardware.yaml",
                                 metadata={"action_filter_beta": 0.2, "effective_action_scale": [0.4] * 12})
    converter = ActionConverter(cfg)
    action = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
    for _ in range(2):
        first, wire = converter.compute_policy_targets(action)
        np.testing.assert_allclose(first, cfg.rl_default_pose + 0.08 * action, atol=1e-6)
        np.testing.assert_allclose(wire, cfg.hw_default_pose + 0.08 * action * cfg.joint_sign, atol=1e-6)
        second, _ = converter.compute_policy_targets(action)
        np.testing.assert_allclose(second, cfg.rl_default_pose + 0.144 * action, atol=1e-6)
        converter.reset()


@pytest.mark.parametrize("source", ["zero", "leg_odometry"])
def test_policy_differentiates_each_encoder_packet_once(source):
    from unittest.mock import Mock

    from drivers.esp32_serial import Telemetry
    from inference.observation_builder import ObservationBuilder, ObservationConfig

    cfg = ObservationConfig(base_lin_vel_source=source, vel_filter_alpha=1.0)
    esp, imu = Mock(), Mock()
    imu.read.return_value = Mock(valid=True, gyro=np.array([0.1, 0.2, 0.3]),
                                 projected_gravity=np.array([0., 0., 1.]))
    builder = ObservationBuilder(imu, esp, cfg)
    builder._leg_odom = Mock()
    builder._leg_odom.update.return_value = np.array([0.1, 0.2, 0.3])
    sample = Telemetry(valid=True, timestamp_ms=2**32 - 40, positions=cfg.hw_default_pose.copy())
    sample.temperatures[:] = 26
    esp.read_telemetry.return_value = sample
    obs = builder.build(1.0)
    np.testing.assert_array_equal(obs[24:36], 0.0)
    sample.timestamp_ms = 60  # A 100 ms packet interval across the millis rollover
    sample.positions = sample.positions + 0.1
    obs = builder.build(1.2)
    np.testing.assert_allclose(obs[24:36], cfg.joint_sign, atol=1e-5)
    repeated = builder.build(1.25)
    np.testing.assert_array_equal(repeated, obs)
    if source == "leg_odometry":
        assert builder._leg_odom.update.call_count == 2
        np.testing.assert_array_equal(builder._leg_odom.update.call_args.kwargs["angular_velocity"], obs[3:6])
        np.testing.assert_allclose(obs[:3], builder._leg_odom.update.return_value)
    else:
        builder._leg_odom.update.assert_not_called()
        np.testing.assert_array_equal(obs[:3], 0.0)


def test_hardware_csv_records_sensor_data_used_at_inference(tmp_path, monkeypatch):
    import csv
    from unittest.mock import Mock

    from drivers.esp32_serial import Telemetry
    from telemetry.session_logger import ControllerState, SessionLogger

    monkeypatch.setattr("telemetry.session_logger.signal.signal", Mock())
    logger = SessionLogger(tmp_path, fsync=False)
    imu = Mock(valid=True, accel=[0.05, 0.0, 1.0], gyro=[0.0, 0.1, 0.0],
               projected_gravity=[0.0, 0.0, 1.0], lin_vel=[0.0, 0.0, 0.2])
    logger.log(Telemetry(timestamp_ms=123, valid=True),
               ControllerState(imu=imu, base_velocity=[0.23, 0.0, -0.1]))
    imu.valid = False
    logger.log(Telemetry(), ControllerState(imu=imu))
    logger.close()
    with logger.session_path.open() as handle:
        good, invalid = list(csv.DictReader(handle))
    assert good["timestamp_ms"] == "123"
    for column, value in (("imu_accel_g_x", 0.05), ("imu_gyro_y", 0.1),
                          ("imu_proj_g_z", 1), ("imu_lin_vel_z", 0.2),
                          ("policy_lin_vel_b_m_s_x", 0.23)):
        assert float(good[column]) == value
        assert np.isnan(float(invalid[column]))


@pytest.mark.parametrize(("fault", "message", "expected_sends"), [
    ("frozen", "stopped updating", 5),
    ("imu", "Invalid encoder or IMU", 0),
    ("negative_load", "Critical servo load", 0),
    ("heat", "Servo temperature", 0),
    ("no_packet", "stopped updating", 0),
    ("substituted_position", "Missing servo feedback at joint indices", 0),
])
def test_policy_loop_stops_on_sensor_fault(fault, message, expected_sends, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from drivers.esp32_serial import ESP32Config, Telemetry
    from inference import harold_controller as module
    from inference.action_converter import ActionConverter
    from inference.observation_builder import ObservationBuilder, ObservationConfig

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(module, "time", SimpleNamespace(
        monotonic=lambda: clock.now,
        sleep=lambda seconds: setattr(clock, "now", clock.now + seconds),
    ))
    monkeypatch.setattr(module.signal, "signal", Mock())
    ctrl = module.HaroldController(None, None, None)
    ctrl.manager_action_filter = True
    ctrl.running_mean = np.zeros(48, dtype=np.float32)
    ctrl.esp32 = Mock(cfg=ESP32Config())
    ctrl.action_conv = ActionConverter()
    ctrl.imu = Mock()
    ctrl.imu.read.return_value = Mock(valid=fault != "imu", gyro=np.zeros(3),
                                      projected_gravity=np.array([0., 0., 1.]))
    ctrl.obs_builder = ObservationBuilder(ctrl.imu, ctrl.esp32, ObservationConfig(base_lin_vel_source="zero"))
    telem = Telemetry(valid=fault != "no_packet", timestamp_ms=123, positions=ctrl.obs_builder.cfg.hw_default_pose)
    telem.loads[4] = -900 if fault == "negative_load" else -899
    telem.temperatures[:] = 28
    telem.temperatures[5] = 50 if fault == "heat" else 49
    if fault == "substituted_position":
        telem.temperatures[11] = 0
    ctrl.esp32.read_telemetry.return_value = telem
    ctrl.policy = Mock()
    ctrl.policy.run.return_value = [np.zeros((1, 12), dtype=np.float32)]
    ctrl.logger = Mock()
    ctrl._capture_diagnostics = Mock()

    assert not ctrl.run(max_seconds=1.0, commands=np.array([0.1, 0.0, -0.2]))
    assert message in str(ctrl._capture_diagnostics.call_args.args[1])
    assert ctrl.esp32.send_targets.call_count == expected_sends
    ctrl.esp32.stop_streaming.assert_called_once()
    if fault == "substituted_position":
        ctrl.policy.run.assert_not_called()
    if expected_sends:
        recorded_state = ctrl.logger.log.call_args.args[1]
        np.testing.assert_allclose([recorded_state.command_vx, recorded_state.command_vy,
                                    recorded_state.command_yaw], [0.1, 0.0, -0.2])
        assert recorded_state.imu is ctrl.imu.read.return_value
