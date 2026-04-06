"""Observation Builder for Harold Robot.

Constructs the 48D observation vector from hardware sensors.
Layout matches the manager-based training env (HaroldObservationsCfg in flat_env_cfg.py).

Observation layout (48D):
    [0:3]   root_lin_vel_b      - Body linear velocity (ZEROED — velocity-blind policy)
    [3:6]   root_ang_vel_b      - Body angular velocity (rad/s)
    [6:9]   projected_gravity_b - Gravity in body frame (normalized)
    [9:12]  velocity_commands   - [vx, vy, yaw_rate] (m/s, rad/s)
    [12:24] joint_pos_relative  - Joint angles - default pose (rad)
    [24:36] joint_vel           - Joint velocities (rad/s)
    [36:48] last_action         - Previous raw policy output (before EMA/scaling)
"""
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.policy_config import JOINT_SIGN, DEFAULT_RL_POSE, resolve_deployment_joint_sign
from drivers.imu_reader_rpi5 import IMUReaderRPi5, IMUData
from drivers.esp32_serial import ESP32Interface, Telemetry
from inference.leg_odometry import LegOdometry
from inference.stance import load_hw_default_pose


@dataclass
class ObservationConfig:
    """Observation builder configuration."""
    # Hardware default pose (ready stance in hardware convention)
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
            # Hardware convention - ready stance (from config/stance.yaml)
            self.hw_default_pose = load_hw_default_pose()

        if self.joint_sign is None:
            # Sign conversion: rl_relative = hw_relative * joint_sign
            self.joint_sign = np.array(JOINT_SIGN, dtype=np.float32)

        if self.default_commands is None:
            # NOTE: Training used commands around 0.3 m/s (see running_mean[9])
            # Using 0.1 creates extreme normalized values
            self.default_commands = np.array([0.3, 0.0, 0.0], dtype=np.float32)

    @classmethod
    def from_yaml(
        cls,
        cpg_path: Path,
        hw_path: Path | None = None,
        metadata: dict | None = None,
    ) -> "ObservationConfig":
        """Load config from deployment config files."""
        # Hardware default pose (ready stance, from config/stance.yaml)
        hw_default_pose = load_hw_default_pose(cpg_path)

        # Joint sign must match the same hardware-facing convention used by action conversion.
        joint_sign = np.array(
            resolve_deployment_joint_sign(metadata=metadata, hardware_path=hw_path),
            dtype=np.float32,
        )

        return cls(hw_default_pose=hw_default_pose, joint_sign=joint_sign)


class ObservationBuilder:
    """Builds 48D observation vector from hardware sensors.

    Usage:
        obs_builder = ObservationBuilder(imu, esp32)
        obs = obs_builder.build(time)
    """

    OBS_DIM = 48

    def __init__(
        self,
        imu: IMUReaderRPi5,
        esp32: ESP32Interface,
        config: ObservationConfig | None = None,
    ):
        self.imu = imu
        self.esp32 = esp32
        self.cfg = config or ObservationConfig()

        # Leg odometry for velocity estimation (replaces velocity-blind zeros)
        self._leg_odom = LegOdometry()
        self._rl_default_pose = np.array(DEFAULT_RL_POSE, dtype=np.float32)

        # State for velocity estimation
        self._prev_positions: Optional[np.ndarray] = None
        self._prev_time: Optional[float] = None
        self._joint_vel = np.zeros(12, dtype=np.float32)

        # Previous raw policy output (for observation [36:48])
        self._prev_raw_action = np.zeros(12, dtype=np.float32)

        self.last_imu_data: Optional[IMUData] = None
        self.last_telemetry: Optional[Telemetry] = None

    def build(
        self,
        time_sec: float,
        commands: Optional[np.ndarray] = None,
        training_mean: Optional[np.ndarray] = None,
        joint_pos_blend: float = 1.0,
    ) -> np.ndarray:
        """Build 48D observation vector.

        Args:
            time_sec: Current time in seconds (for velocity estimation)
            commands: Optional [vx, vy, yaw_rate] commands
            training_mean: Optional 48D training mean for blending
            joint_pos_blend: Blend factor for joint positions (0=training mean, 1=actual)

        Returns:
            48D observation vector (numpy array)
        """
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        # Read IMU data
        imu_data = self.imu.read()
        self.last_imu_data = imu_data

        # [0:3] Body linear velocity via leg odometry.
        # Computed from joint encoder FK + Jacobian during stance phases.
        # Replaces the previous velocity-blind approach (zeros).
        # Populated below after joint positions/velocities are computed.

        # [3:6] Body angular velocity (rad/s)
        obs[3:6] = imu_data.gyro if imu_data.valid else np.zeros(3)

        # [6:9] Projected gravity (normalized)
        # Hardware IMU: Z-up (+1 level), Sim: Z-down (-1 level) → flip sign
        projected_gravity = imu_data.projected_gravity if imu_data.valid else np.array([0, 0, 1])
        obs[6:9] = -projected_gravity

        # [9:12] Velocity commands
        if commands is not None:
            obs[9:12] = commands
        else:
            obs[9:12] = self.cfg.default_commands

        # Read servo telemetry
        telem = self.esp32.read_telemetry()
        self.last_telemetry = telem
        positions = telem.positions if telem.valid else np.zeros(12)

        # Convert hardware positions to RL-convention relative positions:
        # hw_relative → rl_relative via joint_sign (thighs/calves are inverted)
        hw_relative = positions - self.cfg.hw_default_pose
        rl_relative = hw_relative * self.cfg.joint_sign

        # [12:24] Joint positions relative to default pose (RL convention)
        if training_mean is not None and joint_pos_blend < 1.0:
            obs[12:24] = joint_pos_blend * rl_relative + (1 - joint_pos_blend) * training_mean[12:24]
        else:
            obs[12:24] = rl_relative

        # [24:36] Joint velocities (estimated via differentiation, sign-corrected)
        hw_joint_vel = self._estimate_joint_velocities(positions, time_sec)
        rl_joint_vel = hw_joint_vel * self.cfg.joint_sign
        obs[24:36] = rl_joint_vel

        # [0:3] Body linear velocity via leg odometry (uses joint pos + vel computed above)
        # Convert rl_relative back to absolute RL angles for FK
        rl_absolute = self._rl_default_pose + rl_relative
        servo_loads = telem.loads if (telem.valid and telem.loads is not None) else None
        obs[0:3] = self._leg_odom.update(rl_absolute, rl_joint_vel, servo_loads)

        # [36:48] Previous raw policy output (before EMA/scaling)
        obs[36:48] = self._prev_raw_action

        return obs

    def update_prev_action(self, raw_action: np.ndarray) -> None:
        """Store the raw policy output for the next observation's [36:48] slot.

        Training's mdp.last_action returns env.action_manager.action — the raw
        network output BEFORE EMA smoothing, scaling, or offset. We must store
        the same quantity here: the 12D ONNX 'mean' output, unprocessed.

        Args:
            raw_action: 12D raw policy network output (ONNX 'mean')
        """
        self._prev_raw_action = np.asarray(raw_action, dtype=np.float32)

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

    def reset(self, prev_action_init: np.ndarray | None = None) -> None:
        """Reset observation builder state.

        Args:
            prev_action_init: Optional initial values for prev_raw_action [36:48].
                             If None, uses zeros. For stability, initialize to
                             training mean of last_action (running_mean[36:48]).
        """
        self._prev_positions = None
        self._prev_time = None
        self._joint_vel = np.zeros(12, dtype=np.float32)
        self._leg_odom.reset()
        if prev_action_init is not None:
            self._prev_raw_action = prev_action_init.astype(np.float32)
        else:
            self._prev_raw_action = np.zeros(12, dtype=np.float32)


def normalize_observation(
    obs: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    eps: float = 1e-8,
    clip_obs: float = 5.0,
) -> np.ndarray:
    """Apply running-stat normalization for debugging and offline analysis.

    The deployed ONNX policy normalizes internally and does not call this helper.
    The optional clip is only for diagnostics when inspecting normalized vectors.

    Args:
        obs: Raw 48D observation
        running_mean: 48D running mean from training
        running_var: 48D running variance from training
        eps: Small value to avoid division by zero
        clip_obs: Optional diagnostic clip for normalized observations

        Returns:
            Normalized observation, optionally clipped for diagnostics
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
