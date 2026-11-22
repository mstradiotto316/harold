"""Check absolute manager limits against the runtime's forward transformation."""
from pathlib import Path
import sys

import numpy as np

from deployment.validation.firmware_limits import manager_joint_target_limits

DEPLOYMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOYMENT))
from inference.action_converter import ActionConfig, ActionConverter


def test_clipping_manager_targets_matches_streaming_control(monkeypatch):
    monkeypatch.delenv("HAROLD_STANCE_PATH", raising=False)
    config = DEPLOYMENT / "config"
    converter = ActionConverter(ActionConfig.from_yaml(config / "cpg.yaml", config / "hardware.yaml"))
    lower, upper = manager_joint_target_limits()
    # StreamingControl signs and physical-degree stops in canonical joint order.
    signs = np.repeat([1, -1, -1], 4)
    hard_min = np.deg2rad(np.repeat([-20, -55, -5], 4))
    hard_max = np.deg2rad(np.repeat([20, 5, 80], 4))
    # Increasing any manager target increases its serial angle. StreamingControl
    # inverts thighs and calves once, so their lower manager endpoints hit the upper stops.
    physical_at_lower = np.where(signs > 0, hard_min, hard_max)
    physical_at_upper = np.where(signs > 0, hard_max, hard_min)
    np.testing.assert_allclose(signs * converter.rl_to_hardware(lower), physical_at_lower, atol=2e-7)
    np.testing.assert_allclose(signs * converter.rl_to_hardware(upper), physical_at_upper, atol=2e-7)

    targets = np.stack([lower - 0.5, converter.get_rl_default_pose(), upper + 0.5])
    wire = converter.rl_to_hardware(targets)
    expected = np.clip(signs * wire, hard_min, hard_max)
    clipped = converter.rl_to_hardware(np.clip(targets, lower, upper))
    np.testing.assert_allclose(signs * clipped, expected, atol=2e-7)
