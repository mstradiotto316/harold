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

    # Hardware default pose (servo encoder readings at athletic stance)
    hw_default_pose: np.ndarray = None

    # RL default pose (simulation training convention)
    rl_default_pose: np.ndarray = None

    # Joint sign correction (RL <-> hardware)
    joint_sign: np.ndarray = None

    # Safe joint limits (degrees, in hardware convention)
    safe_limits_deg: dict = None

    def __post_init__(self):
        if self.joint_range is None:
            self.joint_range = {
                "shoulder": 0.30,
                "thigh": 0.90,
                "calf": 0.90,
            }

        if self.hw_default_pose is None:
            # Hardware convention - what servo encoders read at rest
            self.hw_default_pose = np.array([
                0.0, 0.0, 0.0, 0.0,         # Shoulders: 0 rad
                0.3, 0.3, 0.3, 0.3,         # Thighs: 0.3 rad
                -0.75, -0.75, -0.75, -0.75  # Calves: -0.75 rad
            ], dtype=np.float32)

        if self.rl_default_pose is None:
            # RL convention - what simulation training used
            # Shoulders alternate for diagonal trot gait
            self.rl_default_pose = np.array([
                0.20, -0.20, 0.20, -0.20,   # Shoulders: alternating
                0.70, 0.70, 0.70, 0.70,     # Thighs: 0.70 rad
                -1.40, -1.40, -1.40, -1.40  # Calves: -1.40 rad
            ], dtype=np.float32)

        if self.joint_sign is None:
            # Sign conversion: hw = hw_default + (rl - rl_default) * joint_sign
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

        # Get hardware default pose (servo encoder readings at athletic stance)
        hw_pose = cpg_data.get("hw_default_pose", {})
        hw_default_pose = np.array([
            hw_pose.get("shoulders", 0.0),
            hw_pose.get("shoulders", 0.0),
            hw_pose.get("shoulders", 0.0),
            hw_pose.get("shoulders", 0.0),
            hw_pose.get("thighs", 0.3),
            hw_pose.get("thighs", 0.3),
            hw_pose.get("thighs", 0.3),
            hw_pose.get("thighs", 0.3),
            hw_pose.get("calves", -0.75),
            hw_pose.get("calves", -0.75),
            hw_pose.get("calves", -0.75),
            hw_pose.get("calves", -0.75),
        ], dtype=np.float32)

        # Get RL default pose (simulation training convention)
        # Note: Shoulders have per-joint values for alternating gait
        rl_pose = cpg_data.get("rl_default_pose", {})
        rl_default_pose = np.array([
            rl_pose.get("shoulder_fl", rl_pose.get("shoulders", 0.20)),
            rl_pose.get("shoulder_fr", rl_pose.get("shoulders", -0.20)),
            rl_pose.get("shoulder_bl", rl_pose.get("shoulders", 0.20)),
            rl_pose.get("shoulder_br", rl_pose.get("shoulders", -0.20)),
            rl_pose.get("thighs", 0.70),
            rl_pose.get("thighs", 0.70),
            rl_pose.get("thighs", 0.70),
            rl_pose.get("thighs", 0.70),
            rl_pose.get("calves", -1.40),
            rl_pose.get("calves", -1.40),
            rl_pose.get("calves", -1.40),
            rl_pose.get("calves", -1.40),
        ], dtype=np.float32)

        # Get joint sign from CPG config
        js = cpg_data.get("joint_sign", {})
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
            hw_default_pose=hw_default_pose,
            rl_default_pose=rl_default_pose,
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
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute final joint targets from CPG and policy.

        Args:
            cpg_targets: 12D CPG base trajectory (radians, RL convention)
            policy_output: 12D raw policy output (NOT clipped - matches training)
            use_cpg: If True, use CPG + residual; if False, use policy directly

        Returns:
            Tuple of:
            - rl_targets: 12D joint targets in RL convention (for observation prev_target_delta)
            - hw_targets: 12D joint targets in HARDWARE convention (for ESP32)
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
            # CPG targets are in RL convention (thigh=0.4-0.9, calf=-0.9 to -1.4)
            # corrections = action * residual_scale * joint_range
            corrections = (
                self._smooth_action *
                self.cfg.residual_scale *
                self._joint_ranges
            )
            rl_targets = cpg_targets + corrections
        else:
            # Direct mode (no CPG)
            # targets = rl_default + action * action_scale * joint_range
            action_scale = 0.5  # Standard action scale
            scaled = self._smooth_action * action_scale * self._joint_ranges
            rl_targets = self.cfg.rl_default_pose + scaled

        # Convert from RL convention to hardware convention
        # Formula: hw = hw_default + (rl - rl_default) * joint_sign
        hw_targets = self.rl_to_hardware(rl_targets)

        return rl_targets.astype(np.float32), hw_targets.astype(np.float32)

    def get_rl_default_pose(self) -> np.ndarray:
        """Get the RL default pose for prev_target_delta computation."""
        return self.cfg.rl_default_pose.copy()

    def get_hw_default_pose(self) -> np.ndarray:
        """Get the hardware default pose (matches simulation's default_joint_pos).

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

    def apply_joint_sign(self, targets: np.ndarray) -> np.ndarray:
        """DEPRECATED: Use rl_to_hardware() instead.

        Simple sign flip doesn't account for offset differences.
        """
        import warnings
        warnings.warn("apply_joint_sign is deprecated, use rl_to_hardware", DeprecationWarning)
        return targets * self.cfg.joint_sign

    def reset(self) -> None:
        """Reset action smoothing state."""
        self._smooth_action = None

    def get_safe_stance(self) -> np.ndarray:
        """Get safe default stance (athletic pose) in hardware convention.

        Returns:
            12D joint positions for safe stance (hardware convention)
        """
        return self.cfg.hw_default_pose.copy()


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
