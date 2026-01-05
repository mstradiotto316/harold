#!/usr/bin/env python3
"""Harold Robot Controller - Main 20 Hz Control Loop.

Runs either open-loop CPG playback or direct RL policy inference.

Usage:
    python harold_controller.py [--config-dir CONFIG_DIR] [--cpg-only] [--max-seconds N]

Components:
    - CPG Generator: Base walking trajectory (CPG-only mode)
    - ONNX Policy: Direct joint targets (policy-only mode)
    - IMU Reader: Body orientation and velocity
    - ESP32 Interface: Servo control and telemetry
"""
import argparse
import json
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
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
from inference.observation_builder import ObservationBuilder, ObservationConfig
from inference.action_converter import ActionConverter, ActionConfig
from telemetry.session_logger import SessionLogger, ControllerState


class HaroldController:
    """Main controller for Harold quadruped robot.

    Runs a 20 Hz control loop with:
        1. CPG base trajectory generation (CPG-only mode)
        2. Observation building from IMU + servo feedback (policy-only mode)
        3. ONNX policy inference (policy-only mode)
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

        # Session logging (5 Hz = every 4th cycle of 20 Hz loop)
        self.logger: Optional[SessionLogger] = None
        self._log_counter = 0
        self._LOG_DIVISOR = 4  # 20 Hz / 4 = 5 Hz

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\nShutdown signal received...")
        self._running = False

    def _run_diag_command(self, cmd: list[str] | str, timeout_s: float = 2.0) -> str:
        """Run a diagnostic command and return combined output."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
                shell=isinstance(cmd, str),
            )
            stdout = (result.stdout or "").strip()
            stderr = (result.stderr or "").strip()
            output = stdout if stdout else "(no stdout)"
            if stderr:
                output = f"{output}\n(stderr) {stderr}"
            return f"{output}\n(exit={result.returncode})"
        except FileNotFoundError:
            return "(command not found)"
        except Exception as exc:
            return f"(diagnostic error: {exc})"

    def _capture_diagnostics(self, reason: str, exc: Exception | None = None) -> None:
        """Capture system diagnostics to a local log file."""
        try:
            log_dir = self.config_dir.parent / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "controller_diagnostics.log"

            lines = []
            lines.append("=" * 60)
            lines.append(f"{datetime.now().isoformat(timespec='seconds')} | {reason}")

            if exc is not None:
                lines.append(f"exception: {exc!r}")
                lines.append("traceback:")
                lines.append(traceback.format_exc().strip())

            if self.logger is not None:
                lines.append(f"session_path: {self.logger.session_path}")
                lines.append(f"session_rows: {self.logger.row_count}")

            if self.esp32 is not None:
                lines.append(f"esp32_connected: {self.esp32.connected}")
                lines.append(f"esp32_streaming: {self.esp32.streaming}")
                telem = self.esp32.last_telemetry
                lines.append(f"last_telem_valid: {telem.valid}")
                if telem.valid:
                    lines.append(f"last_voltage_v: {telem.voltage_V:.2f}")
                    lines.append(f"last_pos_0_3: {np.array2string(telem.positions[:3], precision=4)}")

            lines.append("-- vcgencmd get_throttled --")
            lines.append(self._run_diag_command(["vcgencmd", "get_throttled"]))
            lines.append("-- dmesg tail --")
            lines.append(self._run_diag_command("/bin/sh -c 'dmesg -T | tail -n 80'"))
            lines.append("-- journalctl kernel tail --")
            lines.append(self._run_diag_command(["journalctl", "-k", "-n", "200", "--no-pager"]))
            lines.append("-- lsusb --")
            lines.append(self._run_diag_command(["lsusb"]))
            lines.append("-- ttyUSB devices --")
            lines.append(self._run_diag_command("/bin/sh -c 'ls -l /dev/ttyUSB* 2>/dev/null'"))

            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write("\n".join(lines) + "\n")
        except Exception as exc:
            print(f"WARNING: Diagnostics capture failed: {exc}")

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
        obs_dim = len(self.running_mean)
        print(f"Loaded normalization stats (dim={obs_dim})")
        if obs_dim != ObservationBuilder.OBS_DIM:
            print(
                "ERROR: Policy observation dimension mismatch. "
                f"Expected {ObservationBuilder.OBS_DIM}D, got {obs_dim}D. "
                "Legacy 50D policies are no longer supported; export a 48D policy."
            )
            return False

        # NOTE: The ONNX model (harold_policy.onnx) already includes normalization!
        # The export script wraps the policy with NormalizedPolicy which applies
        # running stats normalization internally. We keep running_mean/var for:
        # 1. Initializing prev_targets to training mean values
        # 2. Joint position blending during warmup
        # We do NOT use them for normalizing observations before ONNX inference.
        # See: policy/export_policy.py - NormalizedPolicy class

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
        print(f"CPG initialized: {cpg_config.frequency_hz} Hz")

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

        # Initialize session logger
        self.logger = SessionLogger()
        print(f"Session logging to: {self.logger.session_path}")

        print("=" * 60)
        print("All components initialized successfully")
        print("=" * 60)
        return True

    def disconnect(self) -> None:
        """Disconnect all components."""
        if self.logger:
            self.logger.close()
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

    def run(
        self,
        mode: str = "policy",
        max_seconds: float | None = None,
    ) -> None:
        """Run the main control loop.

        Args:
            mode: "cpg" for open-loop CPG playback, "policy" for direct policy mode.
            max_seconds: Optional auto-stop after this many seconds
        """
        if mode not in ("cpg", "policy"):
            raise ValueError(f"Unknown mode '{mode}' (expected 'cpg' or 'policy').")

        print("\n" + "=" * 60)
        print("Starting Control Loop")
        print(f"  Mode: {'CPG only' if mode == 'cpg' else 'Direct Policy'}")
        print(f"  Rate: {self.CONTROL_RATE_HZ} Hz")
        if max_seconds is not None:
            print(f"  Max runtime: {max_seconds:.0f}s")
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
        if mode == "policy":
            prev_targets_init = self.running_mean[36:48].copy()
            self.obs_builder.reset(prev_targets_init=prev_targets_init)
        self.action_conv.reset()

        try:
            while self._running:
                loop_start = time.time()
                t = loop_start - self._start_time

                if mode == "cpg":
                    # Open-loop CPG playback
                    cpg_targets = self.cpg.compute(t)
                    rl_targets, hw_targets = self.action_conv.convert_rl_targets(cpg_targets)
                    command_vx, command_vy, command_yaw = self.obs_builder.cfg.default_commands
                    cpg_phase = self.cpg.phase
                else:
                    # Direct policy mode (no CPG)
                    obs = self.obs_builder.build(
                        t,
                        training_mean=self.running_mean,
                        joint_pos_blend=1.0,
                    )
                    outputs = self.policy.run(
                        ['mean'],
                        {'obs': obs.reshape(1, -1).astype(np.float32)}
                    )
                    action = outputs[0][0]  # [12]
                    rl_targets, hw_targets = self.action_conv.compute_policy_targets(action)

                    # Update prev_target_delta for next obs
                    hw_default_pose = self.action_conv.get_hw_default_pose()
                    prev_targets_training_mean = self.running_mean[36:48]
                    self.obs_builder.update_prev_target_delta(
                        rl_targets, hw_default_pose,
                        training_mean=prev_targets_training_mean,
                        blend_factor=0.1,
                    )
                    command_vx, command_vy, command_yaw = obs[33], obs[34], obs[35]
                    cpg_phase = 0.0

                # Send HW targets to ESP32
                self.esp32.send_targets(hw_targets)

                # Log telemetry at 5 Hz (every 4th cycle)
                self._log_counter += 1
                if self._log_counter >= self._LOG_DIVISOR:
                    self._log_counter = 0
                    state = ControllerState(
                        mode="CPG_ONLY" if mode == "cpg" else "POLICY",
                        cpg_phase=cpg_phase,
                        command_vx=command_vx,
                        command_vy=command_vy,
                        command_yaw=command_yaw,
                        cmd_positions=hw_targets,
                    )
                    self.logger.log(self.esp32.last_telemetry, state)

                # Logging (every 20 loops = 1 second)
                self._loop_count += 1
                if self._loop_count % 20 == 0:
                    elapsed = time.time() - self._start_time
                    rate = self._loop_count / elapsed
                    telem = self.esp32.read_telemetry()
                    mode_str = "CPG_ONLY" if mode == "cpg" else "POLICY"
                    if telem.valid:
                        print(
                            f"t={t:.1f}s [{mode_str}] | rate={rate:.1f}Hz | "
                            f"phase={cpg_phase:.2f} | "
                            f"pos[0:3]={telem.positions[:3]}"
                        )

                # Sleep to maintain control rate
                elapsed = time.time() - loop_start
                if elapsed < self.CONTROL_PERIOD:
                    time.sleep(self.CONTROL_PERIOD - elapsed)
                else:
                    self._overrun_count += 1
                    if self._overrun_count % 10 == 1:
                        print(f"WARNING: Loop overrun ({elapsed * 1000:.1f}ms > {self.CONTROL_PERIOD * 1000:.0f}ms)")

                if max_seconds is not None and t >= max_seconds:
                    self._running = False

        except Exception as e:
            print(f"ERROR: Control loop exception: {e}")
            traceback.print_exc()
            self._capture_diagnostics("control_loop_exception", e)

        finally:
            # Stop streaming and return to safe stance
            print("\nStopping...")
            try:
                self.esp32.stop_streaming()
            except Exception as e:
                print(f"WARNING: stop_streaming failed: {e}")
                self._capture_diagnostics("stop_streaming_exception", e)

            # Close session logger
            if self.logger:
                self.logger.close()
                print(f"Session saved: {self.logger.session_path} ({self.logger.row_count} rows)")

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
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--no-cpg",
        action="store_true",
        help="Run in direct policy mode (no CPG base trajectory)",
    )
    mode_group.add_argument(
        "--cpg-only",
        action="store_true",
        help="Run open-loop CPG only (no policy)",
    )
    parser.add_argument(
        "--calibrate",
        type=float,
        default=3.0,
        help="IMU calibration duration in seconds (0 to skip)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Auto-stop after this many seconds (optional)",
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
        mode = "cpg" if args.cpg_only else "policy"
        controller.run(
            mode=mode,
            max_seconds=args.max_seconds,
        )

    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()
