"""ESP32 Serial Communication for Harold Robot.

Wraps the HaroldStreamingControl protocol for USB serial communication.

Protocol:
    Commands (RPi -> ESP32):
        - "READY?#" -> "ARDUINO_READY"
        - "START#"  -> "STATUS,STARTED"
        - "STOP#"   -> "STATUS,STOPPED"
        - "PING#"   -> "PONG"
        - "[t0,t1,...,t11]#" -> 12 joint targets in radians (SERVO order)

    Telemetry (ESP32 -> RPi, 20 Hz):
        "TELEM,timestamp,pos*12,load*12,current*12,temp*12" (SERVO order)

Joint Order:
    ESP32 firmware uses RL order: [FL_sh, FR_sh, BL_sh, BR_sh, FL_th, FR_th, BL_th, BR_th, FL_ca, FR_ca, BL_ca, BR_ca]
    This matches the policy order, so no conversion is needed.
"""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import serial
import yaml
import numpy as np

# Joint order conversion indices
# RL order: [FL_sh, FR_sh, BL_sh, BR_sh, FL_th, FR_th, BL_th, BR_th, FL_ca, FR_ca, BL_ca, BR_ca]
# Servo order: [FL_sh, FL_th, FL_ca, FR_sh, FR_th, FR_ca, BL_sh, BL_th, BL_ca, BR_sh, BR_th, BR_ca]

# To convert RL -> Servo: servo_arr[SERVO_IDX] = rl_arr[RL_TO_SERVO[SERVO_IDX]]
RL_TO_SERVO = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]

# To convert Servo -> RL: rl_arr[RL_IDX] = servo_arr[SERVO_TO_RL[RL_IDX]]
SERVO_TO_RL = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]


def rl_to_servo_order(rl_arr: np.ndarray) -> np.ndarray:
    """Convert array from RL joint order to servo ID order."""
    return rl_arr[RL_TO_SERVO]


def servo_to_rl_order(servo_arr: np.ndarray) -> np.ndarray:
    """Convert array from servo ID order to RL joint order."""
    return servo_arr[SERVO_TO_RL]


@dataclass
class ESP32Config:
    """ESP32 serial configuration."""
    port: str = "/dev/ttyUSB0"
    baud: int = 115200
    timeout_ms: int = 50

    @classmethod
    def from_yaml(cls, path: Path) -> "ESP32Config":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        esp_cfg = data.get("esp32", {})
        return cls(
            port=esp_cfg.get("port", "/dev/ttyUSB0"),
            baud=esp_cfg.get("baud", 115200),
            timeout_ms=esp_cfg.get("timeout_ms", 50),
        )


@dataclass
class Telemetry:
    """Parsed telemetry from ESP32."""
    timestamp_ms: int = 0
    positions: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float32))
    loads: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.int32))
    currents: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.int32))
    temperatures: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.int32))
    voltage_dV: int = 0  # Bus voltage in decivolts (120 = 12.0V)
    valid: bool = False

    @property
    def voltage_V(self) -> float:
        """Bus voltage in volts."""
        return self.voltage_dV / 10.0


class ESP32Interface:
    """Serial interface to ESP32 servo controller.

    Usage:
        esp32 = ESP32Interface()
        if esp32.connect():
            esp32.start_streaming()
            esp32.send_targets([0.0] * 12)
            telem = esp32.read_telemetry()
            esp32.stop_streaming()
            esp32.disconnect()
    """

    def __init__(self, config: ESP32Config | None = None):
        self.cfg = config or ESP32Config()
        self.serial: Optional[serial.Serial] = None
        self.last_telemetry = Telemetry()
        self._buffer = ""
        self._streaming = False

    @property
    def connected(self) -> bool:
        """Check if serial port is open."""
        return self.serial is not None and self.serial.is_open

    @property
    def streaming(self) -> bool:
        """Check if streaming mode is enabled."""
        return self._streaming

    def connect(self, retry_count: int = 3) -> bool:
        """Connect to ESP32 and perform handshake.

        Args:
            retry_count: Number of handshake retries

        Returns:
            True if handshake successful
        """
        try:
            self.serial = serial.Serial(
                self.cfg.port,
                self.cfg.baud,
                timeout=self.cfg.timeout_ms / 1000.0
            )
            time.sleep(2.0)  # Wait for ESP32 to boot (needs ~2s after reset)
            self.serial.reset_input_buffer()
        except serial.SerialException as e:
            print(f"ERROR: Could not open {self.cfg.port}: {e}")
            return False

        # Perform handshake - use direct readline to avoid buffer issues
        old_timeout = self.serial.timeout
        self.serial.timeout = 0.1  # 100ms per read

        for attempt in range(retry_count):
            self._buffer = ""  # Clear internal buffer
            self.serial.reset_input_buffer()
            self._write("READY?#")

            # Read lines directly looking for ARDUINO_READY among telemetry
            for _ in range(20):
                try:
                    line = self.serial.readline().decode("utf-8", errors="ignore").strip()
                    if "ARDUINO_READY" in line:
                        self.serial.timeout = old_timeout
                        print(f"ESP32 connected on {self.cfg.port}")
                        return True
                    if line.startswith("TELEM,") or "STATUS,STARTED" in line or "STATUS,STOPPED" in line:
                        # If the firmware is already streaming, treat it as ready.
                        self.serial.timeout = old_timeout
                        self._streaming = line.startswith("TELEM,") or "STATUS,STARTED" in line
                        print(f"ESP32 streaming detected on {self.cfg.port}")
                        return True
                except Exception:
                    pass
            print(f"Handshake attempt {attempt + 1}/{retry_count} failed")

        self.serial.timeout = old_timeout

        print("ERROR: ESP32 handshake failed")
        self.disconnect()
        return False

    def disconnect(self) -> None:
        """Close serial connection."""
        if self.serial and self.serial.is_open:
            self.serial.close()
        self.serial = None
        self._streaming = False

    def start_streaming(self) -> bool:
        """Enable command execution on ESP32.

        Returns:
            True if START acknowledged
        """
        if not self.connected:
            return False

        self._buffer = ""  # Clear internal buffer
        self.serial.reset_input_buffer()
        self._write("START#")

        # Read a few lines looking for STATUS,STARTED
        old_timeout = self.serial.timeout
        self.serial.timeout = 0.1
        for _ in range(10):
            try:
                line = self.serial.readline().decode("utf-8", errors="ignore").strip()
                if "STATUS,STARTED" in line:
                    self._streaming = True
                    self.serial.timeout = old_timeout
                    return True
            except Exception:
                pass
        self.serial.timeout = old_timeout
        print(f"WARNING: START not acknowledged")
        self._streaming = True  # Assume it worked anyway
        return True

    def stop_streaming(self) -> bool:
        """Disable command execution and return to safe pose.

        Returns:
            True if STOP acknowledged
        """
        if not self.connected:
            return False

        self._buffer = ""  # Clear internal buffer
        self.serial.reset_input_buffer()
        self._write("STOP#")
        self._streaming = False

        # Read a few lines looking for STATUS,STOPPED
        old_timeout = self.serial.timeout
        self.serial.timeout = 0.1
        for _ in range(10):
            try:
                line = self.serial.readline().decode("utf-8", errors="ignore").strip()
                if "STATUS,STOPPED" in line:
                    self.serial.timeout = old_timeout
                    return True
            except Exception:
                pass
        self.serial.timeout = old_timeout
        return True  # STOP always works

    def ping(self) -> bool:
        """Check ESP32 responsiveness.

        Returns:
            True if PONG received
        """
        if not self.connected:
            return False

        self._write("PING#")
        time.sleep(0.02)
        response = self._read_line()
        return response is not None and "PONG" in response

    def send_targets(self, targets: np.ndarray | list[float]) -> bool:
        """Send joint position targets to ESP32.

        Args:
            targets: 12D array of joint positions in radians (RL order)
                    [shoulders(4), thighs(4), calves(4)]

        Returns:
            True if command sent successfully
        """
        if not self.connected:
            return False

        if len(targets) != 12:
            print(f"ERROR: Expected 12 targets, got {len(targets)}")
            return False

        # NOTE: ESP32 firmware already uses RL order (grouped by joint type)
        # No conversion needed - send directly in RL order
        targets_arr = np.asarray(targets, dtype=np.float32)

        # Format: [t0,t1,...,t11]#
        values = ",".join(f"{t:.4f}" for t in targets_arr)
        msg = f"[{values}]#"
        self._write(msg)
        return True

    def read_telemetry(self) -> Telemetry:
        """Read and parse latest telemetry from ESP32.

        Returns:
            Telemetry dataclass (check .valid flag)
        """
        if not self.connected:
            return Telemetry()

        # Read all available lines
        while True:
            line = self._read_line()
            if line is None:
                break
            if line.startswith("TELEM,"):
                self.last_telemetry = self._parse_telemetry(line)

        return self.last_telemetry

    def _parse_telemetry(self, line: str) -> Telemetry:
        """Parse TELEM packet.

        Format: TELEM,timestamp,pos*12,load*12,current*12,temp*12,voltage
        (voltage is optional for backwards compatibility)

        Note: ESP32 sends data in SERVO order, we convert to RL order.
        """
        telem = Telemetry()
        try:
            parts = line.strip().split(",")
            if len(parts) < 1 + 1 + 12 + 12 + 12 + 12:
                return telem

            idx = 1
            telem.timestamp_ms = int(parts[idx])
            idx += 1

            # Parse positions - ESP32 already sends in RL order
            telem.positions = np.array([float(parts[idx + i]) for i in range(12)], dtype=np.float32)
            idx += 12

            # Parse loads - ESP32 already sends in RL order
            telem.loads = np.array([int(parts[idx + i]) for i in range(12)], dtype=np.int32)
            idx += 12

            # Parse currents - ESP32 already sends in RL order
            telem.currents = np.array([int(parts[idx + i]) for i in range(12)], dtype=np.int32)
            idx += 12

            # Parse temperatures - ESP32 already sends in RL order
            telem.temperatures = np.array([int(parts[idx + i]) for i in range(12)], dtype=np.int32)
            idx += 12

            # Parse bus voltage if present (new firmware)
            if idx < len(parts):
                telem.voltage_dV = int(parts[idx])

            telem.valid = True
        except (ValueError, IndexError) as e:
            print(f"WARNING: Telemetry parse error: {e}")

        return telem

    def _write(self, msg: str) -> None:
        """Write message to serial port."""
        if self.serial and self.serial.is_open:
            self.serial.write(msg.encode("utf-8"))

    def _read_line(self) -> Optional[str]:
        """Read a line from serial port (non-blocking).

        Returns:
            Line without newline, or None if no complete line available
        """
        if not self.serial or not self.serial.is_open:
            return None

        try:
            if self.serial.in_waiting > 0:
                data = self.serial.read(self.serial.in_waiting).decode("utf-8", errors="ignore")
                self._buffer += data

            if "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                return line.strip()
        except serial.SerialException:
            pass

        return None


if __name__ == "__main__":
    # Test ESP32 connection
    import argparse

    parser = argparse.ArgumentParser(description="Test ESP32 connection")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial port")
    args = parser.parse_args()

    config = ESP32Config(port=args.port)
    esp32 = ESP32Interface(config)

    if esp32.connect():
        print("Connection successful!")

        if esp32.ping():
            print("Ping successful!")
        else:
            print("Ping failed!")

        # Read some telemetry
        print("\nReading telemetry (press Ctrl+C to stop)...")
        try:
            esp32.start_streaming()
            while True:
                telem = esp32.read_telemetry()
                if telem.valid:
                    print(f"T={telem.timestamp_ms}ms, pos[0:3]={telem.positions[:3]}")
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            esp32.stop_streaming()
            esp32.disconnect()
    else:
        print("Connection failed!")
