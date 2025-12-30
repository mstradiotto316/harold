#!/usr/bin/env python3
"""Real-time voltage and load monitoring for Harold robot.

This script monitors the servo bus voltage and load to diagnose power supply issues.

Usage:
    python scripts/voltage_monitor.py [--port /dev/ttyUSB0] [--duration 30]
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deployment.drivers.esp32_serial import ESP32Interface, ESP32Config
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Monitor servo bus voltage and load")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial port")
    parser.add_argument("--duration", type=int, default=30, help="Duration in seconds")
    args = parser.parse_args()

    config = ESP32Config(port=args.port)
    esp32 = ESP32Interface(config)

    print(f"Connecting to ESP32 on {args.port}...")
    if not esp32.connect():
        print("ERROR: Failed to connect to ESP32")
        return 1

    print("Connected! Starting monitoring...")
    print("="*80)
    print("NOTE: Voltage reading requires updated firmware with voltage telemetry.")
    print("      If voltage shows 0.0V, flash the updated HaroldStreamingControl.ino")
    print("="*80)
    print()

    esp32.start_streaming()

    # Statistics tracking
    voltage_samples = []
    load_samples = []
    start_time = time.time()

    joint_names = [
        'FL_sh', 'FR_sh', 'BL_sh', 'BR_sh',
        'FL_th', 'FR_th', 'BL_th', 'BR_th',
        'FL_ca', 'FR_ca', 'BL_ca', 'BR_ca'
    ]

    try:
        print(f"{'Time':>6} {'Voltage':>8} {'Max Load':>10} {'Loaded Joint':>14} {'Avg Load':>10}")
        print("-"*60)

        sample_count = 0
        while time.time() - start_time < args.duration:
            telem = esp32.read_telemetry()
            if telem.valid:
                sample_count += 1

                # Get voltage (will be 0 if old firmware)
                voltage = telem.voltage_V

                # Get max absolute load (raw values are -1000 to +1000, divide by 10 for %)
                abs_loads = np.abs(telem.loads) / 10.0  # Convert to percentage
                max_load_idx = np.argmax(abs_loads)
                max_load = abs_loads[max_load_idx]
                avg_load = np.mean(abs_loads)

                # Track statistics
                if voltage > 0:
                    voltage_samples.append(voltage)
                load_samples.append(max_load)

                # Print every 5 samples (~250ms)
                if sample_count % 5 == 0:
                    elapsed = time.time() - start_time
                    v_str = f"{voltage:.1f}V" if voltage > 0 else "N/A"
                    print(f"{elapsed:>5.1f}s {v_str:>8} {max_load:>8.1f}% {joint_names[max_load_idx]:>14} {avg_load:>9.1f}%")

                    # Warning messages
                    if voltage > 0 and voltage < 10.0:
                        print(f"        *** WARNING: LOW VOLTAGE ({voltage:.1f}V) - Check power supply! ***")
                    if max_load > 80:
                        print(f"        *** WARNING: HIGH LOAD ({max_load:.1f}%) on {joint_names[max_load_idx]} ***")

            time.sleep(0.05)  # 20 Hz

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        esp32.stop_streaming()
        esp32.disconnect()

    # Print summary
    print()
    print("="*60)
    print("SUMMARY")
    print("="*60)

    if voltage_samples:
        print(f"\nVoltage Statistics:")
        print(f"  Min: {min(voltage_samples):.1f}V")
        print(f"  Max: {max(voltage_samples):.1f}V")
        print(f"  Avg: {np.mean(voltage_samples):.1f}V")

        min_v = min(voltage_samples)
        if min_v < 10.0:
            print(f"\n  *** PROBLEM DETECTED: Voltage dropped to {min_v:.1f}V ***")
            print(f"  At 10V, servo torque is only ~70% of rated capacity.")
            print(f"  At 8V, servo torque is only ~45% of rated capacity.")
            print(f"  RECOMMENDATION: Upgrade to power supply with higher current capacity.")
    else:
        print("\nVoltage: Not available (requires updated firmware)")
        print("  Flash HaroldStreamingControl.ino to enable voltage monitoring.")

    if load_samples:
        print(f"\nLoad Statistics:")
        print(f"  Min: {min(load_samples):.1f}%")
        print(f"  Max: {max(load_samples):.1f}%")
        print(f"  Avg: {np.mean(load_samples):.1f}%")

        max_l = max(load_samples)
        if max_l > 80:
            high_load_pct = sum(1 for l in load_samples if l > 80) / len(load_samples) * 100
            print(f"\n  *** NOTE: Load exceeded 80% ({high_load_pct:.0f}% of samples) ***")
            print(f"  High sustained load may indicate mechanical resistance or heavy motion.")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
