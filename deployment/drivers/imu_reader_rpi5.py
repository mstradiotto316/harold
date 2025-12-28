"""MPU6050 IMU Reader for Raspberry Pi 5.

Uses smbus2 for I2C communication (RPi 5 compatible).
Provides raw sensor data and filtered orientation for robot control.

Sensor outputs:
    - Accelerometer: 3-axis (g)
    - Gyroscope: 3-axis (rad/s)
    - Projected gravity: 3-axis (normalized)
    - Orientation: roll, pitch (radians)
"""
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

try:
    import smbus2 as smbus
except ImportError:
    import smbus  # Fallback for older systems


@dataclass
class IMUConfig:
    """IMU configuration."""
    bus: int = 1
    address: int = 0x68
    sample_rate_hz: int = 100

    @classmethod
    def from_yaml(cls, path: Path) -> "IMUConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        imu_cfg = data.get("imu", {})
        return cls(
            bus=imu_cfg.get("bus", 1),
            address=imu_cfg.get("address", 0x68),
            sample_rate_hz=imu_cfg.get("sample_rate_hz", 100),
        )


@dataclass
class IMUData:
    """Parsed IMU data."""
    # Raw sensor values
    accel: np.ndarray  # [ax, ay, az] in g
    gyro: np.ndarray   # [gx, gy, gz] in rad/s

    # Derived values
    projected_gravity: np.ndarray  # Normalized gravity in body frame
    orientation: np.ndarray        # [roll, pitch] in radians

    # Estimated linear velocity (integrated accelerometer)
    lin_vel: np.ndarray  # [vx, vy, vz] in m/s (body frame)

    valid: bool = True


class IMUReaderRPi5:
    """MPU6050 IMU interface for Raspberry Pi 5.

    Usage:
        imu = IMUReaderRPi5()
        if imu.connect():
            imu.calibrate(duration=3)
            data = imu.read()
            print(f"Roll: {data.orientation[0]:.2f} rad")
            imu.disconnect()
    """

    # MPU6050 registers
    REG_PWR_MGMT_1 = 0x6B
    REG_ACCEL_XOUT_H = 0x3B
    REG_GYRO_XOUT_H = 0x43

    # Sensitivity scale factors
    ACCEL_SCALE = 16384.0  # LSB/g (±2g range)
    GYRO_SCALE = 131.0     # LSB/(°/s) (±250°/s range)
    DEG_TO_RAD = math.pi / 180.0

    def __init__(self, config: IMUConfig | None = None):
        self.cfg = config or IMUConfig()
        self.bus: Optional[smbus.SMBus] = None

        # Calibration offsets
        self.accel_offset = np.zeros(3, dtype=np.float32)
        self.gyro_offset = np.zeros(3, dtype=np.float32)

        # Complementary filter state
        self.alpha = 0.98  # Gyro weight (higher = more gyro trust)
        self.roll = 0.0
        self.pitch = 0.0
        self._last_time = 0.0

        # Velocity estimation state
        self.lin_vel = np.zeros(3, dtype=np.float32)
        self._vel_decay = 0.95  # Velocity decay (damping)

    @property
    def connected(self) -> bool:
        """Check if I2C bus is open."""
        return self.bus is not None

    def connect(self) -> bool:
        """Connect to MPU6050 over I2C.

        Returns:
            True if connection successful
        """
        try:
            self.bus = smbus.SMBus(self.cfg.bus)
            # Wake up MPU6050 (clear sleep bit)
            self.bus.write_byte_data(self.cfg.address, self.REG_PWR_MGMT_1, 0)
            time.sleep(0.1)
            print(f"IMU connected on I2C bus {self.cfg.bus}")
            self._last_time = time.time()
            return True
        except Exception as e:
            print(f"ERROR: Could not connect to IMU: {e}")
            return False

    def disconnect(self) -> None:
        """Close I2C connection."""
        if self.bus:
            self.bus.close()
            self.bus = None

    def calibrate(self, duration: float = 3.0) -> None:
        """Calibrate accelerometer and gyroscope offsets.

        Robot must be stationary and level during calibration.

        Args:
            duration: Calibration duration in seconds
        """
        if not self.connected:
            print("ERROR: IMU not connected")
            return

        print(f"Calibrating IMU for {duration}s (keep robot still)...")

        accel_sum = np.zeros(3)
        gyro_sum = np.zeros(3)
        samples = 0

        start = time.time()
        while time.time() - start < duration:
            raw = self._read_raw()
            if raw is not None:
                accel_sum += raw[:3]
                gyro_sum += raw[3:]
                samples += 1
            time.sleep(0.01)

        if samples > 0:
            self.accel_offset = accel_sum / samples
            # Keep gravity in Z, zero out X/Y offsets for level surface
            self.accel_offset[2] -= 1.0  # Expect 1g in Z
            self.gyro_offset = gyro_sum / samples

            print(f"Calibration complete ({samples} samples)")
            print(f"  Accel offset: {self.accel_offset}")
            print(f"  Gyro offset: {self.gyro_offset}")
        else:
            print("WARNING: No samples collected during calibration")

        # Reset filter state
        self.roll = 0.0
        self.pitch = 0.0
        self.lin_vel = np.zeros(3, dtype=np.float32)
        self._last_time = time.time()

    def read(self) -> IMUData:
        """Read and process IMU data.

        Returns:
            IMUData with all sensor values and derived quantities
        """
        if not self.connected:
            return IMUData(
                accel=np.zeros(3, dtype=np.float32),
                gyro=np.zeros(3, dtype=np.float32),
                projected_gravity=np.array([0, 0, -1], dtype=np.float32),
                orientation=np.zeros(2, dtype=np.float32),
                lin_vel=np.zeros(3, dtype=np.float32),
                valid=False,
            )

        raw = self._read_raw()
        if raw is None:
            return IMUData(
                accel=np.zeros(3, dtype=np.float32),
                gyro=np.zeros(3, dtype=np.float32),
                projected_gravity=np.array([0, 0, -1], dtype=np.float32),
                orientation=np.zeros(2, dtype=np.float32),
                lin_vel=np.zeros(3, dtype=np.float32),
                valid=False,
            )

        # Apply calibration offsets
        accel = (raw[:3] - self.accel_offset).astype(np.float32)
        gyro_dps = (raw[3:] - self.gyro_offset).astype(np.float32)
        gyro = (gyro_dps * self.DEG_TO_RAD).astype(np.float32)  # Convert to rad/s

        # Compute dt for filters
        now = time.time()
        dt = now - self._last_time
        self._last_time = now

        # Limit dt to avoid instability after pauses
        dt = min(dt, 0.1)

        # Compute orientation using complementary filter
        acc_roll, acc_pitch = self._accel_to_orientation(accel)
        self.roll = self.alpha * (self.roll + gyro[0] * dt) + (1 - self.alpha) * acc_roll
        self.pitch = self.alpha * (self.pitch + gyro[1] * dt) + (1 - self.alpha) * acc_pitch

        # Compute projected gravity (normalized accelerometer)
        accel_norm = np.linalg.norm(accel)
        if accel_norm > 0.1:
            projected_gravity = (accel / accel_norm).astype(np.float32)
        else:
            projected_gravity = np.array([0, 0, -1], dtype=np.float32)

        # Estimate linear velocity (simple integration with decay)
        # Remove gravity component from acceleration
        gravity_world = np.array([0, 0, 1])  # Assume flat terrain
        accel_linear = accel - gravity_world  # Linear acceleration in body frame
        accel_ms2 = accel_linear * 9.81  # Convert g to m/s²

        # Integrate with decay to prevent drift
        self.lin_vel = self._vel_decay * self.lin_vel + accel_ms2 * dt

        return IMUData(
            accel=accel,
            gyro=gyro,
            projected_gravity=projected_gravity,
            orientation=np.array([self.roll, self.pitch], dtype=np.float32),
            lin_vel=self.lin_vel.copy(),
            valid=True,
        )

    def _read_raw(self) -> Optional[np.ndarray]:
        """Read raw sensor values.

        Returns:
            Array of [ax, ay, az, gx, gy, gz] or None on error
        """
        try:
            # Read 14 bytes: accel (6) + temp (2) + gyro (6)
            data = self.bus.read_i2c_block_data(self.cfg.address, self.REG_ACCEL_XOUT_H, 14)

            # Parse accelerometer
            ax = self._to_signed_16((data[0] << 8) | data[1]) / self.ACCEL_SCALE
            ay = self._to_signed_16((data[2] << 8) | data[3]) / self.ACCEL_SCALE
            az = self._to_signed_16((data[4] << 8) | data[5]) / self.ACCEL_SCALE

            # Skip temperature (bytes 6-7)

            # Parse gyroscope
            gx = self._to_signed_16((data[8] << 8) | data[9]) / self.GYRO_SCALE
            gy = self._to_signed_16((data[10] << 8) | data[11]) / self.GYRO_SCALE
            gz = self._to_signed_16((data[12] << 8) | data[13]) / self.GYRO_SCALE

            return np.array([ax, ay, az, gx, gy, gz], dtype=np.float64)

        except Exception as e:
            print(f"WARNING: IMU read error: {e}")
            return None

    @staticmethod
    def _to_signed_16(val: int) -> int:
        """Convert unsigned 16-bit to signed."""
        return val - 0x10000 if val >= 0x8000 else val

    @staticmethod
    def _accel_to_orientation(accel: np.ndarray) -> tuple[float, float]:
        """Convert accelerometer reading to roll/pitch angles.

        Args:
            accel: [ax, ay, az] in g

        Returns:
            (roll, pitch) in radians
        """
        ax, ay, az = accel

        # Avoid division by zero
        denom_roll = math.sqrt(ay * ay + az * az)
        denom_pitch = math.sqrt(ax * ax + az * az)

        if denom_roll < 1e-6:
            roll = 0.0
        else:
            roll = math.atan2(-ax, denom_roll)

        if denom_pitch < 1e-6:
            pitch = 0.0
        else:
            pitch = math.atan2(ay, denom_pitch)

        return roll, pitch


if __name__ == "__main__":
    # Test IMU reader
    imu = IMUReaderRPi5()

    if imu.connect():
        imu.calibrate(duration=3)

        print("\nReading IMU (press Ctrl+C to stop)...")
        try:
            while True:
                data = imu.read()
                if data.valid:
                    print(
                        f"Roll: {data.orientation[0]:+.2f} rad, "
                        f"Pitch: {data.orientation[1]:+.2f} rad, "
                        f"Gyro: [{data.gyro[0]:+.2f}, {data.gyro[1]:+.2f}, {data.gyro[2]:+.2f}] rad/s"
                    )
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            imu.disconnect()
    else:
        print("IMU connection failed!")
