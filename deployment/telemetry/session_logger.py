"""Session logger for hardware telemetry and system metrics.

Logs servo telemetry and RPi metrics at 5 Hz to CSV files.
Implements Ousterhout's deep module pattern: simple interface, complex internals.

Usage:
    logger = SessionLogger()
    logger.log(telemetry, controller_state)  # Called at 5 Hz
    logger.close()  # Graceful shutdown
"""
import atexit
import csv
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .system_metrics import SystemMetricsCollector

if TYPE_CHECKING:
    from drivers.esp32_serial import Telemetry


# Joint names in RL order
JOINT_NAMES = [
    "fl_sh", "fr_sh", "bl_sh", "br_sh",  # Shoulders
    "fl_th", "fr_th", "bl_th", "br_th",  # Thighs
    "fl_ca", "fr_ca", "bl_ca", "br_ca",  # Calves
]


@dataclass
class ControllerState:
    """Optional controller state for extended logging.

    Attributes:
        mode: Current controller mode ('CPG', 'POLICY', 'WARMUP')
        cpg_phase: Current CPG phase [0, 1)
        command_vx: Commanded forward velocity (m/s)
        command_vy: Commanded lateral velocity (m/s)
        command_yaw: Commanded yaw rate (rad/s)
        cmd_positions: 12D commanded joint targets (hardware convention, radians)
    """
    mode: str = "UNKNOWN"
    cpg_phase: float = 0.0
    command_vx: float = 0.0
    command_vy: float = 0.0
    command_yaw: float = 0.0
    cmd_positions: list[float] | None = None


class SessionLogger:
    """Logs hardware telemetry to CSV files at 5 Hz.

    Deep module design:
        - Simple public interface: log(), close()
        - Hides buffering, format details, system metric collection
        - Graceful degradation on errors (no exceptions to control loop)

    Usage:
        from logging.session_logger import SessionLogger
        from drivers.esp32_serial import Telemetry

        logger = SessionLogger()
        # In control loop (every 4th cycle):
        logger.log(telemetry, controller_state)
        # On shutdown:
        logger.close()

    Attributes:
        session_path: Path to current session CSV file
        row_count: Number of rows written
    """

    BUFFER_SIZE = 1  # Crash-safe default: flush every row at 5 Hz

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        buffer_size: Optional[int] = None,
        fsync: bool = True,
    ):
        """Initialize session logger.

        Args:
            output_dir: Directory for session files.
                       Default: deployment/sessions/
            buffer_size: Rows to buffer before flushing.
                        Default: 1 (flush every row at 5 Hz)
            fsync: If True, fsync after flush for crash-safe logging.

        Creates output directory if it doesn't exist.
        Creates new session file with timestamp: session_YYYY-MM-DD_HH-MM-SS.csv
        """
        # Default output directory
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "sessions"
        self._output_dir = Path(output_dir)

        # Buffer configuration
        self._buffer_size = buffer_size or self.BUFFER_SIZE
        self._fsync = fsync

        # State
        self._buffer: list[list] = []
        self._row_count = 0
        self._closed = False
        self._file = None
        self._writer = None

        # System metrics collector
        self._sys_metrics = SystemMetricsCollector(cache_duration_s=0.5)

        # Create output directory
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"WARNING: Could not create sessions directory: {e}", file=sys.stderr)

        # Create session file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._session_path = self._output_dir / f"session_{timestamp}.csv"

        try:
            self._file = open(self._session_path, "w", newline="", buffering=1)
            self._writer = csv.writer(self._file)
            self._write_header()
            self._file.flush()
            if self._fsync:
                os.fsync(self._file.fileno())
        except OSError as e:
            print(f"WARNING: Could not create session file: {e}", file=sys.stderr)
            self._file = None
            self._writer = None

        # Register cleanup handlers
        atexit.register(self.close)
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _write_header(self) -> None:
        """Write CSV header row."""
        if self._writer is None:
            return

        header = ["timestamp", "timestamp_ms"]

        # Position columns (measured)
        for name in JOINT_NAMES:
            header.append(f"pos_{name}")

        # Commanded position columns (hardware convention)
        for name in JOINT_NAMES:
            header.append(f"cmd_pos_{name}")

        # Load columns
        for name in JOINT_NAMES:
            header.append(f"load_{name}")

        # Current columns
        for name in JOINT_NAMES:
            header.append(f"curr_{name}")

        # Temperature columns
        for name in JOINT_NAMES:
            header.append(f"temp_{name}")

        # Voltage
        header.append("bus_voltage_v")

        # RPi metrics
        header.extend(["rpi_cpu_temp_c", "rpi_cpu_percent", "rpi_memory_percent", "rpi_disk_percent"])

        # Controller state
        header.extend(["mode", "cpg_phase", "cmd_vx", "cmd_vy", "cmd_yaw"])

        self._writer.writerow(header)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals - flush and call original handler."""
        self.close()

        # Call original handler
        if signum == signal.SIGINT and self._original_sigint:
            if callable(self._original_sigint):
                self._original_sigint(signum, frame)
        elif signum == signal.SIGTERM and self._original_sigterm:
            if callable(self._original_sigterm):
                self._original_sigterm(signum, frame)

    def log(
        self,
        telemetry: "Telemetry",
        state: Optional[ControllerState] = None,
    ) -> None:
        """Log one timestep of telemetry and system metrics.

        This method is designed to be fast and non-blocking:
            - Buffers rows in memory
            - Collects system metrics asynchronously
            - Never raises exceptions (logs warnings instead)

        Args:
            telemetry: Telemetry dataclass from ESP32Interface
            state: Optional controller state (mode, phase, commands)

        Note:
            If telemetry.valid is False, logs a row with NaN values
            for servo data but still captures system metrics.
        """
        if self._closed:
            return

        try:
            # Build row
            row = self._build_row(telemetry, state)
            self._buffer.append(row)
            self._row_count += 1

            # Flush if buffer full
            if len(self._buffer) >= self._buffer_size:
                self._flush()

        except Exception as e:
            print(f"WARNING: Logging failed: {e}", file=sys.stderr)

    def _build_row(
        self,
        telemetry: "Telemetry",
        state: Optional[ControllerState],
    ) -> list:
        """Build a CSV row from telemetry and state."""
        import math

        row = []

        # Timestamp
        row.append(datetime.now().isoformat(timespec="milliseconds"))
        row.append(telemetry.timestamp_ms if telemetry.valid else 0)

        # Servo data (NaN if invalid)
        if telemetry.valid:
            row.extend(telemetry.positions)
        else:
            row.extend([math.nan] * 12)

        # Commanded positions (hardware convention)
        if state is not None and state.cmd_positions is not None:
            row.extend([float(val) for val in state.cmd_positions])
        else:
            row.extend([math.nan] * 12)

        if telemetry.valid:
            row.extend(telemetry.loads)
            row.extend(telemetry.currents)
            row.extend(telemetry.temperatures)
            row.append(telemetry.voltage_V)
        else:
            # 12*3 servo values + 1 voltage = 37 NaN values
            row.extend([math.nan] * 36)
            row.append(math.nan)

        # RPi metrics
        sys_metrics = self._sys_metrics.get()
        row.append(sys_metrics.cpu_temp_c)
        row.append(sys_metrics.cpu_percent)
        row.append(sys_metrics.memory_percent)
        row.append(sys_metrics.disk_percent)

        # Controller state
        if state is not None:
            row.append(state.mode)
            row.append(f"{state.cpg_phase:.4f}")
            row.append(f"{state.command_vx:.4f}")
            row.append(f"{state.command_vy:.4f}")
            row.append(f"{state.command_yaw:.4f}")
        else:
            row.extend(["UNKNOWN", "0.0", "0.0", "0.0", "0.0"])

        return row

    def _flush(self) -> None:
        """Flush buffer to disk."""
        if self._writer is None or not self._buffer:
            return

        try:
            for row in self._buffer:
                self._writer.writerow(row)
            self._file.flush()
            if self._fsync:
                os.fsync(self._file.fileno())
            self._buffer.clear()
        except OSError as e:
            print(f"WARNING: Could not flush session log: {e}", file=sys.stderr)

    def close(self) -> None:
        """Flush buffer and close session file.

        Safe to call multiple times. Registers with signal handlers
        for SIGINT/SIGTERM to ensure graceful shutdown.
        """
        if self._closed:
            return

        self._closed = True

        # Flush remaining buffer
        self._flush()

        # Close file
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None
            self._writer = None

        # Unregister atexit (already closing)
        try:
            atexit.unregister(self.close)
        except Exception:
            pass

    @property
    def session_path(self) -> Path:
        """Path to current session CSV file."""
        return self._session_path

    @property
    def row_count(self) -> int:
        """Number of rows written (including buffered)."""
        return self._row_count


if __name__ == "__main__":
    # Test session logger with mock telemetry
    import numpy as np

    @dataclass
    class MockTelemetry:
        timestamp_ms: int = 0
        positions: np.ndarray = None
        loads: np.ndarray = None
        currents: np.ndarray = None
        temperatures: np.ndarray = None
        voltage_dV: int = 120
        valid: bool = True

        @property
        def voltage_V(self) -> float:
            return self.voltage_dV / 10.0

        def __post_init__(self):
            if self.positions is None:
                self.positions = np.zeros(12, dtype=np.float32)
            if self.loads is None:
                self.loads = np.zeros(12, dtype=np.int32)
            if self.currents is None:
                self.currents = np.zeros(12, dtype=np.int32)
            if self.temperatures is None:
                self.temperatures = np.full(12, 30, dtype=np.int32)

    print("Testing session logger...")
    logger = SessionLogger()
    print(f"Session file: {logger.session_path}")

    # Log some mock data
    for i in range(10):
        telem = MockTelemetry(
            timestamp_ms=i * 200,
            positions=np.random.uniform(-0.5, 0.5, 12).astype(np.float32),
            loads=np.random.randint(0, 100, 12).astype(np.int32),
            currents=np.random.randint(100, 500, 12).astype(np.int32),
            temperatures=np.random.randint(28, 35, 12).astype(np.int32),
        )
        state = ControllerState(
            mode="POLICY" if i > 3 else "WARMUP",
            cpg_phase=(i * 0.1) % 1.0,
            command_vx=0.3,
        )
        logger.log(telem, state)
        print(f"  Logged row {i + 1}")
        time.sleep(0.1)

    logger.close()
    print(f"Logged {logger.row_count} rows to {logger.session_path}")
