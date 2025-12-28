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

from ..drivers.imu_reader_rpi5 import IMUReaderRPi5, IMUData
from ..drivers.esp32_serial import ESP32Interface, Telemetry


@dataclass
class ObservationConfig:
    """Observation builder configuration."""
    # Default joint positions (athletic stance)
    default_pose: np.ndarray = None

    # Velocity commands [vx, vy, yaw_rate]
    default_commands: np.ndarray = None

    # Joint velocity estimation
    vel_filter_alpha: float = 0.5  # Low-pass filter coefficient

    def __post_init__(self):
        if self.default_pose is None:
            # [shoulders(4), thighs(4), calves(4)]
            self.default_pose = np.array([
                0.0, 0.0, 0.0, 0.0,       # Shoulders
                0.3, 0.3, 0.3, 0.3,       # Thighs
                -0.75, -0.75, -0.75, -0.75  # Calves
            ], dtype=np.float32)

        if self.default_commands is None:
            self.default_commands = np.array([0.1, 0.0, 0.0], dtype=np.float32)

    @classmethod
    def from_yaml(cls, cpg_path: Path) -> "ObservationConfig":
        """Load config from CPG YAML file."""
        with open(cpg_path) as f:
            data = yaml.safe_load(f)

        pose = data.get("default_pose", {})
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

        return cls(default_pose=default_pose)


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
    ) -> np.ndarray:
        """Build 50D observation vector.

        Args:
            time_sec: Current time in seconds (for velocity estimation)
            cpg_phase_sin: sin(2*pi*phase) from CPG generator
            cpg_phase_cos: cos(2*pi*phase) from CPG generator
            commands: Optional [vx, vy, yaw_rate] commands

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
        obs[6:9] = imu_data.projected_gravity if imu_data.valid else np.array([0, 0, -1])

        # Read servo telemetry
        telem = self.esp32.read_telemetry()
        positions = telem.positions if telem.valid else np.zeros(12)

        # [9:21] Joint positions relative to default pose
        obs[9:21] = positions - self.cfg.default_pose

        # [21:33] Joint velocities (estimated via differentiation)
        joint_vel = self._estimate_joint_velocities(positions, time_sec)
        obs[21:33] = joint_vel

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

    def update_prev_targets(self, targets: np.ndarray) -> None:
        """Update previous targets for next observation.

        Args:
            targets: 12D joint targets from action converter
        """
        # Store the delta from default pose, not absolute targets
        self._prev_targets = (targets - self.cfg.default_pose).astype(np.float32)

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

    def reset(self) -> None:
        """Reset observation builder state."""
        self._prev_positions = None
        self._prev_time = None
        self._joint_vel = np.zeros(12, dtype=np.float32)
        self._prev_targets = np.zeros(12, dtype=np.float32)


def normalize_observation(
    obs: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Apply running stats normalization to observation.

    This is the same normalization used during training.

    Args:
        obs: Raw 50D observation
        running_mean: 50D running mean from training
        running_var: 50D running variance from training
        eps: Small value to avoid division by zero

    Returns:
        Normalized observation
    """
    return (obs - running_mean) / np.sqrt(running_var + eps)


if __name__ == "__main__":
    # Test observation builder with mock data
    print("Observation Builder Test")
    print("=" * 50)

    # Create mock IMU and ESP32
    class MockIMU:
        def read(self):
            from ..drivers.imu_reader_rpi5 import IMUData
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
            from ..drivers.esp32_serial import Telemetry
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
