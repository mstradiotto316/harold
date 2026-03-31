"""Action Converter for Harold Robot.

Converts policy output to servo commands:
    1. Scale policy outputs around the default pose
    2. Apply sign corrections for RL -> hardware mapping
    3. Apply safety limits (if enabled)
"""
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.policy_config import (
    DEFAULT_ACTION_SCALE,
    JOINT_RANGE_BY_CATEGORY,
    JOINT_SIGN,
    resolve_deployment_joint_sign,
)
from inference.stance import load_hw_default_pose, load_rl_default_pose


@dataclass
class ActionConfig:
    """Action converter configuration."""
    # Joint action ranges (for scaling policy output)
    joint_range: dict = None

    # Hardware default pose (ready stance in hardware convention)
    hw_default_pose: np.ndarray = None

    # RL default pose (ready stance in simulation convention)
    rl_default_pose: np.ndarray = None

    # Joint sign correction (RL <-> hardware)
    joint_sign: np.ndarray = None

    # Training-time action scale.
    action_scale: float = DEFAULT_ACTION_SCALE

    # Safe joint limits (degrees, in hardware convention)
    safe_limits_deg: dict = None

    def __post_init__(self):
        if self.joint_range is None:
            self.joint_range = dict(JOINT_RANGE_BY_CATEGORY)

        if self.hw_default_pose is None:
            # Hardware convention - ready stance (from config/stance.yaml)
            self.hw_default_pose = load_hw_default_pose()

        if self.rl_default_pose is None:
            # RL convention - default pose (from config/stance.yaml)
            self.rl_default_pose = load_rl_default_pose()

        if self.joint_sign is None:
            # Sign conversion: hw = hw_default + (rl - rl_default) * joint_sign
            self.joint_sign = np.array(JOINT_SIGN, dtype=np.float32)

        if self.safe_limits_deg is None:
            self.safe_limits_deg = {
                "shoulder": (-25, 25),
                "thigh": (-55, 5),
                "calf": (-5, 80),
            }

    @classmethod
    def from_yaml(cls, cpg_path: Path, hw_path: Path, metadata: dict | None = None) -> "ActionConfig":
        """Load config from YAML files."""
        with open(cpg_path) as f:
            cpg_data = yaml.safe_load(f)
        with open(hw_path) as f:
            hw_data = yaml.safe_load(f)
        metadata = metadata or {}

        # Get joint ranges from CPG config
        jr = metadata.get("joint_range", cpg_data.get("joint_range", {}))
        joint_range = {
            "shoulder": jr.get("shoulder", JOINT_RANGE_BY_CATEGORY["shoulder"]),
            "thigh": jr.get("thigh", JOINT_RANGE_BY_CATEGORY["thigh"]),
            "calf": jr.get("calf", JOINT_RANGE_BY_CATEGORY["calf"]),
        }

        # Get safe limits from hardware config
        servos = hw_data.get("servos", {})
        limits = servos.get("safe_limits_deg", {})
        safe_limits_deg = {
            "shoulder": tuple(limits.get("shoulder", [-25, 25])),
            "thigh": tuple(limits.get("thigh", [-55, 5])),
            "calf": tuple(limits.get("calf", [-5, 80])),
        }

        # Get hardware default pose (ready stance, from config/stance.yaml)
        hw_default_pose = load_hw_default_pose(cpg_path)

        # Get RL default pose (ready stance, from config/stance.yaml)
        rl_default_pose = load_rl_default_pose(cpg_path)

        # Joint sign comes from the hardware-facing deployment convention.
        joint_sign = np.array(
            resolve_deployment_joint_sign(metadata=metadata, hardware_path=hw_path),
            dtype=np.float32,
        )

        return cls(
            joint_range=joint_range,
            hw_default_pose=hw_default_pose,
            rl_default_pose=rl_default_pose,
            joint_sign=joint_sign,
            safe_limits_deg=safe_limits_deg,
            action_scale=float(metadata.get("action_scale", DEFAULT_ACTION_SCALE)),
        )


# Joint category for each of the 12 joints
JOINT_CATEGORIES = (
    "shoulder", "shoulder", "shoulder", "shoulder",
    "thigh", "thigh", "thigh", "thigh",
    "calf", "calf", "calf", "calf",
)


class ActionConverter:
    """Converts policy output to servo commands.

    Usage:
        converter = ActionConverter()
        rl_targets, hw_targets = converter.compute_policy_targets(policy_output)
    """

    def __init__(self, config: ActionConfig | None = None):
        self.cfg = config or ActionConfig()

        # Convert limits to radians for efficiency
        self._limits_rad = {}
        for cat, (lo, hi) in self.cfg.safe_limits_deg.items():
            self._limits_rad[cat] = (math.radians(lo), math.radians(hi))

        # Pre-compute joint ranges array
        self._joint_ranges = np.array([
            self.cfg.joint_range[JOINT_CATEGORIES[i]] for i in range(12)
        ], dtype=np.float32)

        # Action smoothing (EMA filter) — must match training (flat_env_cfg.py EMAJointPositionActionCfg)
        self._smooth_action: Optional[np.ndarray] = None
        self._action_beta = 0.2  # Filter coefficient (aligned with training ema_beta=0.2)

    def compute_policy_targets(
        self,
        policy_output: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute joint targets from policy output (direct mode).

        Args:
            policy_output: 12D raw policy output (NOT clipped - matches training)

        Returns:
            Tuple of:
            - rl_targets: 12D joint targets in RL convention (for observation prev_target_delta)
            - hw_targets: 12D joint targets in HARDWARE convention (for ESP32)
        """
        # NOTE: Training does NOT clip policy outputs before scaling.
        # We match training by scaling around the default pose.
        action = policy_output.astype(np.float32)

        # Apply action smoothing (EMA filter)
        if self._smooth_action is None:
            self._smooth_action = action.copy()
        else:
            self._smooth_action = (
                (1 - self._action_beta) * self._smooth_action +
                self._action_beta * action
            )

        # targets = rl_default + action * action_scale * joint_range
        scaled = self._smooth_action * self.cfg.action_scale * self._joint_ranges
        rl_targets = self.cfg.rl_default_pose + scaled

        # Convert from RL convention to hardware convention
        # Formula: hw = hw_default + (rl - rl_default) * joint_sign
        hw_targets = self.rl_to_hardware(rl_targets)

        return rl_targets.astype(np.float32), hw_targets.astype(np.float32)

    def convert_rl_targets(self, rl_targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert RL-convention targets to hardware convention.

        Args:
            rl_targets: 12D joint targets in RL convention

        Returns:
            Tuple of RL targets (float32) and hardware targets (float32).
        """
        hw_targets = self.rl_to_hardware(rl_targets)
        return rl_targets.astype(np.float32), hw_targets.astype(np.float32)

    def get_rl_default_pose(self) -> np.ndarray:
        """Get the RL default pose for prev_target_delta computation."""
        return self.cfg.rl_default_pose.copy()

    def get_hw_default_pose(self) -> np.ndarray:
        """Get the hardware default pose (ready stance).

        IMPORTANT: Simulation uses this for prev_target_delta computation:
            prev_target_delta = processed_actions - default_joint_pos
        """
        return self.cfg.hw_default_pose.copy()

    def rl_to_hardware(self, rl_targets: np.ndarray) -> np.ndarray:
        """Convert joint targets from RL convention to hardware convention.

        The transformation handles both sign inversion and offset differences:
            hw = hw_default + (rl - rl_default) * joint_sign

        Args:
            rl_targets: 12D joint targets in RL convention

        Returns:
            12D joint targets in hardware convention
        """
        rl_relative = rl_targets - self.cfg.rl_default_pose
        hw_relative = rl_relative * self.cfg.joint_sign
        hw_targets = self.cfg.hw_default_pose + hw_relative
        return hw_targets

    def _apply_limits(self, targets: np.ndarray) -> np.ndarray:
        """Apply safe joint limits.

        Args:
            targets: 12D joint targets (radians)

        Returns:
            Clamped joint targets
        """
        clamped = np.zeros_like(targets)
        for i in range(12):
            cat = JOINT_CATEGORIES[i]
            lo, hi = self._limits_rad[cat]
            clamped[i] = np.clip(targets[i], lo, hi)
        return clamped

    def reset(self) -> None:
        """Reset action smoothing state."""
        self._smooth_action = None

    def get_safe_stance(self) -> np.ndarray:
        """Get safe default stance (ready pose) in hardware convention.

        Returns:
            12D joint positions for safe stance (hardware convention)
        """
        return self.cfg.hw_default_pose.copy()


if __name__ == "__main__":
    # Test action converter
    print("Action Converter Test")
    print("=" * 50)

    converter = ActionConverter()

    # Test with zero policy output
    policy_output = np.zeros(12, dtype=np.float32)
    rl_targets, hw_targets = converter.compute_policy_targets(policy_output)
    print(f"Policy output: {policy_output}")
    print(f"RL targets: {rl_targets}")
    print(f"HW targets: {hw_targets}")
    print()

    # Test with non-zero policy output
    policy_output = np.ones(12, dtype=np.float32) * 0.5

    rl_targets, hw_targets = converter.compute_policy_targets(policy_output)
    print(f"Policy output (0.5): {policy_output}")
    print(f"RL targets: {rl_targets}")
    print(f"HW targets: {hw_targets}")
