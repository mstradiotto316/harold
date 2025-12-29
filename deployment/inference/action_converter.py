"""Action Converter for Harold Robot.

Converts policy output to servo commands:
    1. Combine CPG base trajectory with policy residual corrections
    2. Apply safety limits
    3. Apply sign corrections for RL -> hardware mapping

Action pipeline:
    cpg_targets = cpg.compute(time)
    corrections = policy_output * residual_scale * joint_range
    final_targets = cpg_targets + corrections
    final_targets = clip(final_targets, safe_limits)
"""
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


@dataclass
class ActionConfig:
    """Action converter configuration."""
    # Residual scale (limits policy authority)
    residual_scale: float = 0.05

    # Joint action ranges (for scaling policy output)
    joint_range: dict = None

    # Default joint positions
    default_pose: np.ndarray = None

    # Joint sign correction (RL -> hardware)
    joint_sign: np.ndarray = None

    # Safe joint limits (degrees)
    safe_limits_deg: dict = None

    def __post_init__(self):
        if self.joint_range is None:
            self.joint_range = {
                "shoulder": 0.30,
                "thigh": 0.90,
                "calf": 0.90,
            }

        if self.default_pose is None:
            self.default_pose = np.array([
                0.0, 0.0, 0.0, 0.0,       # Shoulders
                0.3, 0.3, 0.3, 0.3,       # Thighs
                -0.75, -0.75, -0.75, -0.75  # Calves
            ], dtype=np.float32)

        if self.joint_sign is None:
            # Thighs and calves are inverted relative to simulation
            self.joint_sign = np.array([
                1.0, 1.0, 1.0, 1.0,       # Shoulders (same)
                -1.0, -1.0, -1.0, -1.0,   # Thighs (inverted)
                -1.0, -1.0, -1.0, -1.0,   # Calves (inverted)
            ], dtype=np.float32)

        if self.safe_limits_deg is None:
            self.safe_limits_deg = {
                "shoulder": (-25, 25),
                "thigh": (-55, 5),
                "calf": (-5, 80),
            }

    @classmethod
    def from_yaml(cls, cpg_path: Path, hw_path: Path) -> "ActionConfig":
        """Load config from YAML files."""
        with open(cpg_path) as f:
            cpg_data = yaml.safe_load(f)
        with open(hw_path) as f:
            hw_data = yaml.safe_load(f)

        # Get joint ranges from CPG config
        jr = cpg_data.get("joint_range", {})
        joint_range = {
            "shoulder": jr.get("shoulder", 0.30),
            "thigh": jr.get("thigh", 0.90),
            "calf": jr.get("calf", 0.90),
        }

        # Get safe limits from hardware config
        servos = hw_data.get("servos", {})
        limits = servos.get("safe_limits_deg", {})
        safe_limits_deg = {
            "shoulder": tuple(limits.get("shoulder", [-25, 25])),
            "thigh": tuple(limits.get("thigh", [-55, 5])),
            "calf": tuple(limits.get("calf", [-5, 80])),
        }

        # Get default pose
        pose = cpg_data.get("default_pose", {})
        default_pose = np.array([
            pose.get("shoulders", 0.0),
            pose.get("shoulders", 0.0),
            pose.get("shoulders", 0.0),
            pose.get("shoulders", 0.0),
            pose.get("thighs", 0.3),
            pose.get("thighs", 0.3),
            pose.get("thighs", 0.3),
            pose.get("thighs", 0.3),
            pose.get("calves", -0.75),
            pose.get("calves", -0.75),
            pose.get("calves", -0.75),
            pose.get("calves", -0.75),
        ], dtype=np.float32)

        # Get joint sign from hardware config
        js = servos.get("joint_sign", {})
        joint_sign = np.array([
            js.get("shoulders", 1.0),
            js.get("shoulders", 1.0),
            js.get("shoulders", 1.0),
            js.get("shoulders", 1.0),
            js.get("thighs", -1.0),
            js.get("thighs", -1.0),
            js.get("thighs", -1.0),
            js.get("thighs", -1.0),
            js.get("calves", -1.0),
            js.get("calves", -1.0),
            js.get("calves", -1.0),
            js.get("calves", -1.0),
        ], dtype=np.float32)

        return cls(
            residual_scale=cpg_data.get("residual_scale", 0.05),
            joint_range=joint_range,
            default_pose=default_pose,
            joint_sign=joint_sign,
            safe_limits_deg=safe_limits_deg,
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
        targets = converter.compute(cpg_targets, policy_output)
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

        # Action smoothing (EMA filter)
        self._smooth_action: Optional[np.ndarray] = None
        self._action_beta = 0.18  # Filter coefficient

    def compute(
        self,
        cpg_targets: np.ndarray,
        policy_output: np.ndarray,
        use_cpg: bool = True,
    ) -> np.ndarray:
        """Compute final joint targets from CPG and policy.

        Args:
            cpg_targets: 12D CPG base trajectory (radians)
            policy_output: 12D raw policy output (NOT clipped - matches training)
            use_cpg: If True, use CPG + residual; if False, use policy directly

        Returns:
            12D joint targets (radians, RL convention)
        """
        # NOTE: Training does NOT clip policy outputs before scaling.
        # The residual_scale * joint_range scales them down, then joint
        # limits are applied. We must match this behavior exactly.
        action = policy_output.astype(np.float32)

        # Apply action smoothing (EMA filter)
        if self._smooth_action is None:
            self._smooth_action = action.copy()
        else:
            self._smooth_action = (
                (1 - self._action_beta) * self._smooth_action +
                self._action_beta * action
            )

        if use_cpg:
            # CPG + residual mode
            # corrections = action * residual_scale * joint_range
            corrections = (
                self._smooth_action *
                self.cfg.residual_scale *
                self._joint_ranges
            )
            targets = cpg_targets + corrections
        else:
            # Direct mode (no CPG)
            # targets = default + action * action_scale * joint_range
            action_scale = 0.5  # Standard action scale
            scaled = self._smooth_action * action_scale * self._joint_ranges
            targets = self.cfg.default_pose + scaled

        # Apply safety limits
        targets = self._apply_limits(targets)

        return targets.astype(np.float32)

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

    def apply_joint_sign(self, targets: np.ndarray) -> np.ndarray:
        """Apply sign correction for RL -> hardware mapping.

        This is applied BEFORE sending to ESP32.

        Args:
            targets: 12D joint targets (RL convention)

        Returns:
            12D joint targets (hardware convention)
        """
        return targets * self.cfg.joint_sign

    def reset(self) -> None:
        """Reset action smoothing state."""
        self._smooth_action = None

    def get_safe_stance(self) -> np.ndarray:
        """Get safe default stance (athletic pose).

        Returns:
            12D joint positions for safe stance
        """
        return self.cfg.default_pose.copy()


if __name__ == "__main__":
    # Test action converter
    print("Action Converter Test")
    print("=" * 50)

    converter = ActionConverter()

    # Test with zero policy output (should get CPG targets)
    cpg_targets = np.array([
        0.05, -0.05, 0.05, -0.05,   # Shoulders
        0.6, 0.6, 0.6, 0.6,         # Thighs
        -1.0, -1.0, -1.0, -1.0,     # Calves
    ], dtype=np.float32)

    policy_output = np.zeros(12, dtype=np.float32)

    targets = converter.compute(cpg_targets, policy_output)
    print(f"CPG targets: {cpg_targets}")
    print(f"Policy output: {policy_output}")
    print(f"Final targets: {targets}")
    print()

    # Test with non-zero policy output
    policy_output = np.ones(12, dtype=np.float32) * 0.5

    targets = converter.compute(cpg_targets, policy_output)
    print(f"Policy output (0.5): {policy_output}")
    print(f"Final targets: {targets}")
    print(f"Residual added: {targets - cpg_targets}")
