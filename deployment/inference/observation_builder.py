"""Observation Builder for Harold Robot.

Constructs the 50D observation vector from hardware sensors:
    - IMU: linear velocity, angular velocity, projected gravity
    - Servo feedback: joint positions, joint velocities
    - Control state: commands, previous targets, gait phase

Observation layout (50D):
    [0:3]   root_lin_vel_b      - Body linear velocity (m/s)
    [3:6]   root_ang_vel_b      - Body angular velocity (rad/s)
    [6:9]   projected_gravity_b - Gravity in body frame (normalized)
    [9:21]  joint_pos_relative  - Joint angles - default pose (rad)
    [21:33] joint_vel           - Joint velocities (rad/s)
    [33:36] commands            - Velocity commands [vx, vy, yaw_rate]
    [36:48] prev_target_delta   - Previous policy output
    [48:50] gait_phase          - [sin(phase), cos(phase)]
"""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from drivers.imu_reader_rpi5 import IMUReaderRPi5, IMUData
from drivers.esp32_serial import ESP32Interface, Telemetry


def _expand_joint_sign(js: dict) -> np.ndarray:
    """Expand joint_sign config into a 12D array (shoulders, thighs, calves)."""
    if not isinstance(js, dict):
        js = {}

    shoulders = js.get("shoulders", 1.0)
    if isinstance(shoulders, (list, tuple)) and len(shoulders) == 4:
        shoulder_vals = list(shoulders)
    else:
        shoulder_vals = [
            js.get("shoulder_fl", shoulders),
            js.get("shoulder_fr", shoulders),
            js.get("shoulder_bl", shoulders),
            js.get("shoulder_br", shoulders),
        ]

    thigh_val = js.get("thighs", -1.0)
    calf_val = js.get("calves", -1.0)

    return np.array(
        shoulder_vals + [thigh_val] * 4 + [calf_val] * 4,
        dtype=np.float32,
    )


@dataclass
class ObservationConfig:
    """Observation builder configuration."""
    # Hardware default pose (servo encoder readings at athletic stance)
    hw_default_pose: np.ndarray = None

    # Joint sign for HW -> RL convention conversion
    # rl_relative = hw_relative * joint_sign
    joint_sign: np.ndarray = None

    # Velocity commands [vx, vy, yaw_rate]
    default_commands: np.ndarray = None

    # Joint velocity estimation
    vel_filter_alpha: float = 0.5  # Low-pass filter coefficient

    def __post_init__(self):
        if self.hw_default_pose is None:
            # [shoulders(4), thighs(4), calves(4)]
            # Hardware convention - what servo encoders read at rest
            self.hw_default_pose = np.array([
                0.0, 0.0, 0.0, 0.0,         # Shoulders: 0 rad
                0.40, 0.40, 0.40, 0.40,     # Thighs: tuned stance
                -0.74, -0.74, -0.74, -0.74  # Calves: tuned stance
            ], dtype=np.float32)

        if self.joint_sign is None:
            # Sign conversion: rl_relative = hw_relative * joint_sign
            self.joint_sign = np.array([
                1.0, 1.0, 1.0, 1.0,         # Shoulders: same
                -1.0, -1.0, -1.0, -1.0,     # Thighs: inverted
                -1.0, -1.0, -1.0, -1.0      # Calves: inverted
            ], dtype=np.float32)

        if self.default_commands is None:
            # NOTE: Training used commands around 0.3 m/s (see running_mean[33])
            # Using 0.1 creates extreme normalized values
            self.default_commands = np.array([0.3, 0.0, 0.0], dtype=np.float32)

    @classmethod
    def from_yaml(cls, cpg_path: Path) -> "ObservationConfig":
        """Load config from CPG YAML file."""
        with open(cpg_path) as f:
            data = yaml.safe_load(f)

        # Hardware default pose (servo encoder readings at athletic stance)
        hw_pose = data.get("hw_default_pose", {})
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

        # Joint sign for HW -> RL conversion (supports per-shoulder overrides)
        joint_sign = _expand_joint_sign(data.get("joint_sign", {}))

        return cls(hw_default_pose=hw_default_pose, joint_sign=joint_sign)


class ObservationBuilder:
    """Builds 50D observation vector from hardware sensors.

    Usage:
        obs_builder = ObservationBuilder(imu, esp32)
        obs = obs_builder.build(time, cpg_phase, prev_targets)
    """

    OBS_DIM = 50

    def __init__(
        self,
        imu: IMUReaderRPi5,
        esp32: ESP32Interface,
        config: ObservationConfig | None = None,
    ):
        self.imu = imu
        self.esp32 = esp32
        self.cfg = config or ObservationConfig()

        # State for velocity estimation
        self._prev_positions: Optional[np.ndarray] = None
        self._prev_time: Optional[float] = None
        self._joint_vel = np.zeros(12, dtype=np.float32)

        # Previous targets (for observation)
        self._prev_targets = np.zeros(12, dtype=np.float32)

    def build(
        self,
        time_sec: float,
        cpg_phase_sin: float,
        cpg_phase_cos: float,
        commands: Optional[np.ndarray] = None,
        training_mean: Optional[np.ndarray] = None,
        joint_pos_blend: float = 1.0,
    ) -> np.ndarray:
        """Build 50D observation vector.

        Args:
            time_sec: Current time in seconds (for velocity estimation)
            cpg_phase_sin: sin(2*pi*phase) from CPG generator
            cpg_phase_cos: cos(2*pi*phase) from CPG generator
            commands: Optional [vx, vy, yaw_rate] commands
            training_mean: Optional 50D training mean for blending
            joint_pos_blend: Blend factor for joint positions (0=training mean, 1=actual)

        Returns:
            50D observation vector (numpy array)
        """
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        # Read IMU data
        imu_data = self.imu.read()

        # [0:3] Body linear velocity (m/s)
        obs[0:3] = imu_data.lin_vel if imu_data.valid else np.zeros(3)

        # [3:6] Body angular velocity (rad/s)
        obs[3:6] = imu_data.gyro if imu_data.valid else np.zeros(3)

        # [6:9] Projected gravity (normalized)
        # NOTE: Hardware IMU uses Z-up convention (+1 for level)
        # Simulation uses Z-down convention (-1 for level)
        # We must flip the sign to match training
        projected_gravity = imu_data.projected_gravity if imu_data.valid else np.array([0, 0, 1])
        obs[6:9] = -projected_gravity  # Flip sign for sim convention

        # Read servo telemetry
        telem = self.esp32.read_telemetry()
        positions = telem.positions if telem.valid else np.zeros(12)

        # Convert hardware positions to RL-convention relative positions:
        # 1. Compute relative position in hardware convention
        # 2. Apply sign conversion: rl_relative = hw_relative * joint_sign
        # This is needed because thighs/calves have opposite sign in RL vs hardware
        hw_relative = positions - self.cfg.hw_default_pose
        rl_relative = hw_relative * self.cfg.joint_sign

        # [9:21] Joint positions relative to default pose (RL convention)
        # Optionally blend with training mean for smoother startup
        if training_mean is not None and joint_pos_blend < 1.0:
            obs[9:21] = joint_pos_blend * rl_relative + (1 - joint_pos_blend) * training_mean[9:21]
        else:
            obs[9:21] = rl_relative

        # [21:33] Joint velocities (estimated via differentiation)
        # Also needs sign conversion since velocities are derived from HW positions
        hw_joint_vel = self._estimate_joint_velocities(positions, time_sec)
        rl_joint_vel = hw_joint_vel * self.cfg.joint_sign
        obs[21:33] = rl_joint_vel

        # [33:36] Velocity commands
        if commands is not None:
            obs[33:36] = commands
        else:
            obs[33:36] = self.cfg.default_commands

        # [36:48] Previous target deltas
        obs[36:48] = self._prev_targets

        # [48:50] Gait phase (sin, cos)
        obs[48] = cpg_phase_sin
        obs[49] = cpg_phase_cos

        return obs

    def update_prev_target_delta(
        self,
        rl_targets: np.ndarray,
        default_pose: np.ndarray,
        training_mean: np.ndarray | None = None,
        blend_factor: float = 0.3
    ) -> None:
        """Update previous target delta for next observation.

        IMPORTANT: Simulation stores prev_target_delta = processed_actions - default_pose
        This is the final targets (in RL convention) minus the default pose.

        To prevent feedback divergence, we blend actual values with training mean.

        Args:
            rl_targets: 12D final joint targets in RL convention (CPG + residual)
            default_pose: 12D default pose (hw_default_pose, matches simulation)
            training_mean: 12D training mean for prev_targets (optional, for blending)
            blend_factor: How much to trust actual values vs training mean (0=all training, 1=all actual)
        """
        delta = (rl_targets - default_pose).astype(np.float32)

        if training_mean is not None:
            # Blend actual delta with training mean to prevent divergence
            # This keeps normalized values closer to 0
            self._prev_targets = blend_factor * delta + (1 - blend_factor) * training_mean
        else:
            # No blending - use actual values (may cause divergence)
            self._prev_targets = delta

    def _estimate_joint_velocities(
        self,
        positions: np.ndarray,
        time_sec: float,
    ) -> np.ndarray:
        """Estimate joint velocities via differentiation with low-pass filter.

        Args:
            positions: Current joint positions (12D)
            time_sec: Current time in seconds

        Returns:
            Estimated joint velocities (12D, rad/s)
        """
        if self._prev_positions is None or self._prev_time is None:
            self._prev_positions = positions.copy()
            self._prev_time = time_sec
            return self._joint_vel

        dt = time_sec - self._prev_time
        if dt < 1e-6:
            return self._joint_vel

        # Compute raw velocity
        raw_vel = (positions - self._prev_positions) / dt

        # Low-pass filter
        alpha = self.cfg.vel_filter_alpha
        self._joint_vel = alpha * raw_vel + (1 - alpha) * self._joint_vel

        # Update state
        self._prev_positions = positions.copy()
        self._prev_time = time_sec

        return self._joint_vel

    def reset(self, prev_targets_init: np.ndarray | None = None) -> None:
        """Reset observation builder state.

        Args:
            prev_targets_init: Optional initial values for prev_targets.
                              If None, uses zeros. For better stability,
                              initialize to training mean values.
        """
        self._prev_positions = None
        self._prev_time = None
        self._joint_vel = np.zeros(12, dtype=np.float32)
        if prev_targets_init is not None:
            self._prev_targets = prev_targets_init.astype(np.float32)
        else:
            self._prev_targets = np.zeros(12, dtype=np.float32)


def normalize_observation(
    obs: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    eps: float = 1e-8,
    clip_obs: float = 5.0,
) -> np.ndarray:
    """Apply running stats normalization to observation.

    This is the same normalization used during training, with clipping
    to prevent extreme values from causing policy instability.

    Args:
        obs: Raw 50D observation
        running_mean: 50D running mean from training
        running_var: 50D running variance from training
        eps: Small value to avoid division by zero
        clip_obs: Clip normalized observations to [-clip_obs, clip_obs]
                  (matches rl_games clip_observations: 5.0)

    Returns:
        Normalized and clipped observation
    """
    normalized = (obs - running_mean) / np.sqrt(running_var + eps)
    return np.clip(normalized, -clip_obs, clip_obs)


if __name__ == "__main__":
    # Test observation builder with mock data
    print("Observation Builder Test")
    print("=" * 50)

    # Create mock IMU and ESP32
    class MockIMU:
        def read(self):
            from drivers.imu_reader_rpi5 import IMUData
            return IMUData(
                accel=np.array([0.0, 0.0, 1.0]),
                gyro=np.array([0.0, 0.0, 0.0]),
                projected_gravity=np.array([0.0, 0.0, -1.0]),
                orientation=np.array([0.0, 0.0]),
                lin_vel=np.array([0.0, 0.0, 0.0]),
                valid=True,
            )

    class MockESP32:
        def read_telemetry(self):
            from drivers.esp32_serial import Telemetry
            return Telemetry(
                timestamp_ms=0,
                positions=np.array([
                    0.0, 0.0, 0.0, 0.0,
                    0.3, 0.3, 0.3, 0.3,
                    -0.75, -0.75, -0.75, -0.75
                ]),
                valid=True,
            )

    imu = MockIMU()
    esp32 = MockESP32()
    builder = ObservationBuilder(imu, esp32)

    obs = builder.build(0.0, 0.0, 1.0)  # phase=0 -> sin=0, cos=1
    print(f"Observation shape: {obs.shape}")
    print(f"Observation: {obs}")
