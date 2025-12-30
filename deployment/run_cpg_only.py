#!/usr/bin/env python3
"""Harold CPG-only deployment.

Runs a pure Central Pattern Generator gait without policy corrections.
This provides stable walking without the complexity of policy inference.

Usage:
    python run_cpg_only.py [--duration SECONDS] [--calibrate SECONDS]
"""
import argparse
import signal
import sys
import time

sys.path.insert(0, str(__file__).rsplit('/', 1)[0])

from drivers.esp32_serial import ESP32Interface
from drivers.imu_reader_rpi5 import IMUReaderRPi5
from inference.cpg_generator import CPGGenerator


def main():
    parser = argparse.ArgumentParser(description="Harold CPG-only deployment")
    parser.add_argument("--duration", type=float, default=60, help="Walking duration (seconds)")
    parser.add_argument("--calibrate", type=float, default=3.0, help="IMU calibration time (seconds)")
    args = parser.parse_args()

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        print("\nShutdown signal received...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 60)
    print("Harold CPG-Only Deployment")
    print("=" * 60)

    # Initialize components
    esp32 = ESP32Interface()
    if not esp32.connect():
        print("ERROR: Failed to connect to ESP32")
        return 1

    imu = IMUReaderRPi5()
    if not imu.connect():
        print("ERROR: Failed to connect to IMU")
        esp32.disconnect()
        return 1

    print(f"Calibrating IMU for {args.calibrate}s (keep robot still)...")
    imu.calibrate(duration=args.calibrate)

    cpg = CPGGenerator()
    print(f"CPG initialized: {cpg.cfg.frequency_hz} Hz gait")

    print("=" * 60)
    print(f"Starting walking for {args.duration}s")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    # Start streaming
    esp32.start_streaming()

    # Run CPG loop
    start_time = time.time()
    loop_count = 0

    while running and (time.time() - start_time) < args.duration:
        t = time.time() - start_time

        # Compute and send CPG targets
        targets = cpg.compute(t)
        esp32.send_targets(targets)

        loop_count += 1

        # Log every 0.5 seconds
        if loop_count % 10 == 0:
            elapsed = time.time() - start_time
            rate = loop_count / elapsed
            telem = esp32.read_telemetry()
            if telem.valid:
                import numpy as np
                abs_loads = np.abs(telem.loads) / 10.0  # Convert to %
                max_load = np.max(abs_loads)
                avg_load = np.mean(abs_loads)
                voltage = telem.voltage_V
                v_str = f"{voltage:.1f}V" if voltage > 0 else "N/A"
                print(f"t={elapsed:.1f}s | {v_str} | load: max={max_load:.0f}% avg={avg_load:.0f}% | phase={cpg.phase:.2f}")
            else:
                print(f"t={elapsed:.1f}s | rate={rate:.1f}Hz | phase={cpg.phase:.2f}")

        # Maintain 20 Hz
        time.sleep(0.05)

    # Stop
    print("\nStopping...")
    esp32.stop_streaming()

    total_time = time.time() - start_time
    print(f"\nSummary:")
    print(f"  Duration: {total_time:.1f}s")
    print(f"  Loops: {loop_count}")
    print(f"  Average rate: {loop_count/total_time:.1f} Hz")

    esp32.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
