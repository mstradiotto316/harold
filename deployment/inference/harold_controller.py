#!/usr/bin/env python3
"""Harold Robot Controller - Main 20 Hz Control Loop.

Runs CPG + RL policy inference for quadruped walking gait.

Usage:
    python harold_controller.py [--config-dir CONFIG_DIR] [--cpg-only] [--max-seconds N]

Components:
    - CPG Generator: Base walking trajectory
    - ONNX Policy: Residual corrections for balance
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
        print(f"Loaded normalization stats (dim={len(self.running_mean)})")

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
        use_cpg: bool = True,
        warmup_cycles: int = 3,
        enable_policy: bool = True,
        max_seconds: float | None = None,
    ) -> None:
        """Run the main control loop.

        Args:
            use_cpg: If True, use CPG + residual mode; if False, direct policy mode
            warmup_cycles: Number of CPG cycles before enabling policy (avoids
                          cold-start observation mismatch)
            enable_policy: If False, run open-loop CPG only (zero policy residuals)
            max_seconds: Optional auto-stop after this many seconds
        """
        print("\n" + "=" * 60)
        print("Starting Control Loop")
        if use_cpg and enable_policy:
            mode_label = "CPG + Residual"
        elif use_cpg and not enable_policy:
            mode_label = "CPG only"
        else:
            mode_label = "Direct Policy"
        print(f"  Mode: {mode_label}")
        print(f"  Rate: {self.CONTROL_RATE_HZ} Hz")
        if enable_policy:
            print(f"  Warmup: {warmup_cycles} CPG cycles before policy")
        else:
            print("  Warmup: disabled (policy off)")
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
        # Initialize prev_targets to training mean values for stability
        # This prevents extreme normalized values at startup
        prev_targets_init = self.running_mean[36:48].copy()  # prev_target portion of obs
        self.obs_builder.reset(prev_targets_init=prev_targets_init)
        self.action_conv.reset()

        # Calculate warmup duration based on CPG frequency
        if enable_policy:
            warmup_duration = warmup_cycles / self.cpg.cfg.frequency_hz
            # Add extra transition period for blending (2 additional CPG cycles)
            transition_duration = warmup_duration + (2.0 / self.cpg.cfg.frequency_hz)
        else:
            warmup_duration = 0.0
            transition_duration = 0.0
        policy_enabled = False

        try:
            while self._running:
                loop_start = time.time()
                t = loop_start - self._start_time

                # 1. Compute CPG base trajectory
                cpg_targets = self.cpg.compute(t)
                phase_sin, phase_cos = self.cpg.get_phase_sin_cos()

                # 2. Calculate joint_pos blend factor
                # During warmup: blend=0.0 (use training mean)
                # During transition: gradually increase to 1.0
                # After transition: blend=1.0 (use actual values)
                if not enable_policy:
                    joint_pos_blend = 1.0
                elif t < warmup_duration:
                    joint_pos_blend = 0.3  # 30% actual during warmup
                elif t < transition_duration:
                    # Linear interpolation from 0.3 to 1.0 during transition
                    progress = (t - warmup_duration) / (transition_duration - warmup_duration)
                    joint_pos_blend = 0.3 + 0.7 * progress
                else:
                    joint_pos_blend = 1.0  # Full actual values

                # 3. Build observation (50D) with optional blending
                obs = self.obs_builder.build(
                    t, phase_sin, phase_cos,
                    training_mean=self.running_mean,
                    joint_pos_blend=joint_pos_blend
                )

                # NOTE: Do NOT normalize here! The ONNX model (harold_policy.onnx)
                # already includes normalization internally via NormalizedPolicy wrapper.
                # See policy/export_policy.py for details.

                # Check if warmup period is complete
                if enable_policy and not policy_enabled and t >= warmup_duration:
                    policy_enabled = True
                    print(f"[t={t:.1f}s] Warmup complete, enabling policy")

                if enable_policy and policy_enabled:
                    # 4. Run policy inference with RAW observations
                    # ONNX model normalizes internally - pass raw obs!
                    outputs = self.policy.run(
                        ['mean'],
                        {'obs': obs.reshape(1, -1).astype(np.float32)}
                    )
                    action = outputs[0][0]  # [12]
                else:
                    # During warmup: use zero action (pure CPG)
                    action = np.zeros(12, dtype=np.float32)

                # 5. Compute final targets (CPG + residual)
                # Returns both RL targets (for observation) and HW targets (for ESP32)
                rl_targets, hw_targets = self.action_conv.compute(cpg_targets, action, use_cpg=use_cpg)

                # 6. Update observation builder state with prev_target_delta
                # IMPORTANT: Simulation stores prev_target_delta = processed_actions - default_joint_pos
                # where default_joint_pos is the USD model default (same as hw_default_pose)
                # Blend with training mean to prevent feedback divergence
                hw_default_pose = self.action_conv.get_hw_default_pose()
                prev_targets_training_mean = self.running_mean[36:48]  # Indices 36-48 are prev_targets
                self.obs_builder.update_prev_target_delta(
                    rl_targets, hw_default_pose,
                    training_mean=prev_targets_training_mean,
                    blend_factor=0.1  # 10% actual, 90% training mean (prevents divergence)
                )

                # 7. Send HW targets to ESP32
                self.esp32.send_targets(hw_targets)

                # 7.5 Log telemetry at 5 Hz (every 4th cycle)
                self._log_counter += 1
                if self._log_counter >= self._LOG_DIVISOR:
                    self._log_counter = 0
                    if enable_policy:
                        mode = "POLICY" if policy_enabled else "WARMUP"
                    else:
                        mode = "CPG_ONLY" if use_cpg else "ZERO"
                    state = ControllerState(
                        mode=mode,
                        cpg_phase=self.cpg.phase,
                        command_vx=obs[33],
                        command_vy=obs[34],
                        command_yaw=obs[35],
                        cmd_positions=hw_targets,
                    )
                    imu_data = self.obs_builder.last_imu_data if self.obs_builder else None
                    self.logger.log(self.esp32.last_telemetry, state, imu_data)

                # 8. Logging (every 20 loops = 1 second)
                self._loop_count += 1
                if self._loop_count % 20 == 0:
                    elapsed = time.time() - self._start_time
                    rate = self._loop_count / elapsed
                    telem = self.esp32.read_telemetry()
                    if enable_policy:
                        mode_str = "POLICY" if policy_enabled else "WARMUP"
                    else:
                        mode_str = "CPG_ONLY" if use_cpg else "ZERO"
                    if telem.valid:
                        print(
                            f"t={t:.1f}s [{mode_str}] | rate={rate:.1f}Hz | "
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
        help="Run open-loop CPG only (no policy residuals)",
    )
    parser.add_argument(
        "--calibrate",
        type=float,
        default=3.0,
        help="IMU calibration duration in seconds (0 to skip)",
    )
    parser.add_argument(
        "--warmup-cycles",
        type=int,
        default=0,
        help="CPG cycles before enabling policy (default: 0, immediate)",
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
        controller.run(
            use_cpg=not args.no_cpg,
            warmup_cycles=args.warmup_cycles,
            enable_policy=not args.cpg_only,
            max_seconds=args.max_seconds,
        )

    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()
