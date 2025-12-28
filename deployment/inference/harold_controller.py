#!/usr/bin/env python3
"""Harold Robot Controller - Main 20 Hz Control Loop.

Runs CPG + RL policy inference for quadruped walking gait.

Usage:
    python harold_controller.py [--config-dir CONFIG_DIR]

Components:
    - CPG Generator: Base walking trajectory
    - ONNX Policy: Residual corrections for balance
    - IMU Reader: Body orientation and velocity
    - ESP32 Interface: Servo control and telemetry
"""
import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed. Run: pip install onnxruntime")
    sys.exit(1)

from drivers.esp32_serial import ESP32Interface, ESP32Config
from drivers.imu_reader_rpi5 import IMUReaderRPi5, IMUConfig
from inference.cpg_generator import CPGGenerator, CPGConfig
from inference.observation_builder import ObservationBuilder, ObservationConfig, normalize_observation
from inference.action_converter import ActionConverter, ActionConfig


class HaroldController:
    """Main controller for Harold quadruped robot.

    Runs a 20 Hz control loop with:
        1. CPG base trajectory generation
        2. Observation building from IMU + servo feedback
        3. ONNX policy inference for residual corrections
        4. Action conversion and servo command transmission
    """

    CONTROL_RATE_HZ = 20
    CONTROL_PERIOD = 1.0 / CONTROL_RATE_HZ  # 50ms

    def __init__(
        self,
        policy_path: Path,
        metadata_path: Path,
        config_dir: Path,
    ):
        """Initialize controller.

        Args:
            policy_path: Path to harold_policy.onnx
            metadata_path: Path to policy_metadata.json
            config_dir: Directory containing hardware.yaml and cpg.yaml
        """
        self.policy_path = policy_path
        self.metadata_path = metadata_path
        self.config_dir = config_dir

        # Components (initialized in connect())
        self.imu: Optional[IMUReaderRPi5] = None
        self.esp32: Optional[ESP32Interface] = None
        self.cpg: Optional[CPGGenerator] = None
        self.obs_builder: Optional[ObservationBuilder] = None
        self.action_conv: Optional[ActionConverter] = None
        self.policy: Optional[ort.InferenceSession] = None

        # Normalization stats
        self.running_mean: Optional[np.ndarray] = None
        self.running_var: Optional[np.ndarray] = None

        # Control loop state
        self._running = False
        self._start_time = 0.0
        self._loop_count = 0
        self._overrun_count = 0

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\nShutdown signal received...")
        self._running = False

    def connect(self) -> bool:
        """Initialize and connect all components.

        Returns:
            True if all components connected successfully
        """
        print("=" * 60)
        print("Harold Controller Initialization")
        print("=" * 60)

        # Load configs
        hw_config_path = self.config_dir / "hardware.yaml"
        cpg_config_path = self.config_dir / "cpg.yaml"

        if not hw_config_path.exists():
            print(f"ERROR: Hardware config not found: {hw_config_path}")
            return False
        if not cpg_config_path.exists():
            print(f"ERROR: CPG config not found: {cpg_config_path}")
            return False

        # Load policy metadata
        if not self.metadata_path.exists():
            print(f"ERROR: Policy metadata not found: {self.metadata_path}")
            return False

        with open(self.metadata_path) as f:
            metadata = json.load(f)

        self.running_mean = np.array(metadata["running_mean"], dtype=np.float32)
        self.running_var = np.array(metadata["running_variance"], dtype=np.float32)
        print(f"Loaded normalization stats (dim={len(self.running_mean)})")

        # Load ONNX policy
        if not self.policy_path.exists():
            print(f"ERROR: Policy not found: {self.policy_path}")
            return False

        print(f"Loading policy: {self.policy_path}")
        self.policy = ort.InferenceSession(
            str(self.policy_path),
            providers=['CPUExecutionProvider'],
        )
        print("  Policy loaded successfully")

        # Initialize CPG
        cpg_config = CPGConfig.from_yaml(cpg_config_path)
        self.cpg = CPGGenerator(cpg_config)
        print(f"CPG initialized: {cpg_config.frequency_hz} Hz, residual_scale={cpg_config.residual_scale}")

        # Initialize ESP32
        esp32_config = ESP32Config.from_yaml(hw_config_path)
        self.esp32 = ESP32Interface(esp32_config)
        print(f"Connecting to ESP32 on {esp32_config.port}...")
        if not self.esp32.connect():
            print("ERROR: ESP32 connection failed")
            return False
        print("  ESP32 connected")

        # Initialize IMU
        imu_config = IMUConfig.from_yaml(hw_config_path)
        self.imu = IMUReaderRPi5(imu_config)
        print(f"Connecting to IMU on I2C bus {imu_config.bus}...")
        if not self.imu.connect():
            print("ERROR: IMU connection failed")
            return False
        print("  IMU connected")

        # Initialize observation builder
        obs_config = ObservationConfig.from_yaml(cpg_config_path)
        self.obs_builder = ObservationBuilder(self.imu, self.esp32, obs_config)
        print("Observation builder initialized")

        # Initialize action converter
        action_config = ActionConfig.from_yaml(cpg_config_path, hw_config_path)
        self.action_conv = ActionConverter(action_config)
        print("Action converter initialized")

        print("=" * 60)
        print("All components initialized successfully")
        print("=" * 60)
        return True

    def disconnect(self) -> None:
        """Disconnect all components."""
        if self.esp32:
            self.esp32.stop_streaming()
            self.esp32.disconnect()
        if self.imu:
            self.imu.disconnect()
        print("Disconnected")

    def calibrate(self, duration: float = 3.0) -> None:
        """Calibrate IMU.

        Args:
            duration: Calibration duration in seconds
        """
        if self.imu:
            self.imu.calibrate(duration)

    def run(self, use_cpg: bool = True) -> None:
        """Run the main control loop.

        Args:
            use_cpg: If True, use CPG + residual mode; if False, direct policy mode
        """
        print("\n" + "=" * 60)
        print("Starting Control Loop")
        print(f"  Mode: {'CPG + Residual' if use_cpg else 'Direct Policy'}")
        print(f"  Rate: {self.CONTROL_RATE_HZ} Hz")
        print("  Press Ctrl+C to stop")
        print("=" * 60 + "\n")

        # Start streaming
        if not self.esp32.start_streaming():
            print("ERROR: Failed to start streaming")
            return

        self._running = True
        self._start_time = time.time()
        self._loop_count = 0
        self._overrun_count = 0

        # Reset components
        self.obs_builder.reset()
        self.action_conv.reset()

        try:
            while self._running:
                loop_start = time.time()
                t = loop_start - self._start_time

                # 1. Compute CPG base trajectory
                cpg_targets = self.cpg.compute(t)
                phase_sin, phase_cos = self.cpg.get_phase_sin_cos()

                # 2. Build observation (50D)
                obs = self.obs_builder.build(t, phase_sin, phase_cos)

                # 3. Normalize observation
                obs_norm = normalize_observation(obs, self.running_mean, self.running_var)

                # 4. Run policy inference
                outputs = self.policy.run(
                    ['mean'],
                    {'obs': obs_norm.reshape(1, -1).astype(np.float32)}
                )
                action = outputs[0][0]  # [12]

                # 5. Compute final targets (CPG + residual)
                targets = self.action_conv.compute(cpg_targets, action, use_cpg=use_cpg)

                # 6. Update observation builder state
                self.obs_builder.update_prev_targets(targets)

                # 7. Apply joint sign and send to ESP32
                # Note: ESP32 firmware already handles sign internally
                self.esp32.send_targets(targets)

                # 8. Logging (every 20 loops = 1 second)
                self._loop_count += 1
                if self._loop_count % 20 == 0:
                    elapsed = time.time() - self._start_time
                    rate = self._loop_count / elapsed
                    telem = self.esp32.read_telemetry()
                    if telem.valid:
                        print(
                            f"t={t:.1f}s | rate={rate:.1f}Hz | "
                            f"phase={self.cpg.phase:.2f} | "
                            f"pos[0:3]={telem.positions[:3]}"
                        )

                # 9. Sleep to maintain control rate
                elapsed = time.time() - loop_start
                if elapsed < self.CONTROL_PERIOD:
                    time.sleep(self.CONTROL_PERIOD - elapsed)
                else:
                    self._overrun_count += 1
                    if self._overrun_count % 10 == 1:
                        print(f"WARNING: Loop overrun ({elapsed * 1000:.1f}ms > {self.CONTROL_PERIOD * 1000:.0f}ms)")

        except Exception as e:
            print(f"ERROR: Control loop exception: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Stop streaming and return to safe stance
            print("\nStopping...")
            self.esp32.stop_streaming()

            # Print summary
            total_time = time.time() - self._start_time
            print(f"\nControl loop summary:")
            print(f"  Total time: {total_time:.1f}s")
            print(f"  Total loops: {self._loop_count}")
            print(f"  Average rate: {self._loop_count / total_time:.1f} Hz")
            print(f"  Overruns: {self._overrun_count}")


def main():
    parser = argparse.ArgumentParser(description="Harold Robot Controller")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).parent.parent / "config",
        help="Directory containing hardware.yaml and cpg.yaml",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path(__file__).parent.parent / "policy" / "harold_policy.onnx",
        help="Path to ONNX policy file",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(__file__).parent.parent / "policy" / "policy_metadata.json",
        help="Path to policy metadata JSON",
    )
    parser.add_argument(
        "--no-cpg",
        action="store_true",
        help="Run in direct policy mode (no CPG base trajectory)",
    )
    parser.add_argument(
        "--calibrate",
        type=float,
        default=3.0,
        help="IMU calibration duration in seconds (0 to skip)",
    )
    args = parser.parse_args()

    # Create controller
    controller = HaroldController(
        policy_path=args.policy,
        metadata_path=args.metadata,
        config_dir=args.config_dir,
    )

    # Connect
    if not controller.connect():
        print("Connection failed!")
        sys.exit(1)

    try:
        # Calibrate IMU
        if args.calibrate > 0:
            controller.calibrate(duration=args.calibrate)

        # Run control loop
        controller.run(use_cpg=not args.no_cpg)

    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()
