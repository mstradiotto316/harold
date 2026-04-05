#!/usr/bin/env python3
"""M2: Hold default stance for N seconds and log telemetry.

Connects to ESP32, sends hw_default_pose at 20 Hz, logs telemetry,
then stops. Used to verify servo tracking at rest.

Usage:
    python -m tests.test_stance_hold [--duration 10] [--port /dev/ttyUSB0]

Pass criteria:
    - Position tracking error: |cmd - actual| < 0.05 rad for all joints
    - Load < 30% when suspended
    - Current < 400 mA per servo
"""
import argparse
import csv
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from drivers.esp32_serial import ESP32Interface, ESP32Config
from inference.stance import load_hw_default_pose


JOINT_NAMES = [
    "FL_sh", "FR_sh", "BL_sh", "BR_sh",
    "FL_th", "FR_th", "BL_th", "BR_th",
    "FL_ca", "FR_ca", "BL_ca", "BR_ca",
]

CONTROL_RATE_HZ = 20
CONTROL_PERIOD = 1.0 / CONTROL_RATE_HZ


def main():
    parser = argparse.ArgumentParser(description="Hold default stance and log telemetry")
    parser.add_argument("--duration", type=float, default=10.0, help="Hold duration (seconds)")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0", help="ESP32 serial port")
    parser.add_argument("--output", type=str, default=None, help="CSV output path")
    args = parser.parse_args()

    # Default pose
    hw_default = load_hw_default_pose()
    print(f"Default pose (hw): {np.array2string(hw_default, precision=3)}")

    # Connect
    config = ESP32Config(port=args.port)
    esp32 = ESP32Interface(config)

    running = True
    def sighandler(signum, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, sighandler)

    print(f"Connecting to ESP32 on {args.port}...")
    if not esp32.connect():
        print("ERROR: ESP32 connection failed")
        sys.exit(1)

    # Prepare CSV output
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = Path(args.output) if args.output else Path(__file__).parent.parent / "sessions" / f"stance_hold_{ts}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect data
    rows = []
    errors = []

    print(f"\nStarting streaming, holding stance for {args.duration}s...")
    if not esp32.start_streaming():
        print("ERROR: Failed to start streaming")
        esp32.disconnect()
        sys.exit(1)

    start_time = time.time()
    loop_count = 0

    try:
        while running and (time.time() - start_time) < args.duration:
            loop_start = time.time()
            t = loop_start - start_time

            # Send default pose
            esp32.send_targets(hw_default)

            # Read telemetry
            telem = esp32.read_telemetry()
            if telem.valid:
                tracking_error = np.abs(telem.positions - hw_default)
                errors.append(tracking_error)

                row = {
                    "time_s": round(t, 3),
                    "voltage_V": telem.voltage_V,
                }
                for i, name in enumerate(JOINT_NAMES):
                    row[f"cmd_{name}"] = round(float(hw_default[i]), 4)
                    row[f"pos_{name}"] = round(float(telem.positions[i]), 4)
                    row[f"err_{name}"] = round(float(tracking_error[i]), 4)
                    row[f"load_{name}"] = int(telem.loads[i])
                    row[f"curr_{name}"] = int(telem.currents[i])
                    row[f"temp_{name}"] = int(telem.temperatures[i])
                rows.append(row)

            loop_count += 1

            # Print status every second
            if loop_count % CONTROL_RATE_HZ == 0 and telem.valid:
                max_err = tracking_error.max()
                max_load = np.abs(telem.loads).max()
                max_curr = telem.currents.max()
                print(f"  t={t:5.1f}s | max_err={max_err:.4f} rad | max_load={max_load} | "
                      f"max_curr={max_curr} mA | voltage={telem.voltage_V:.1f}V")

            # Rate control
            elapsed = time.time() - loop_start
            if elapsed < CONTROL_PERIOD:
                time.sleep(CONTROL_PERIOD - elapsed)

    finally:
        print("\nStopping...")
        esp32.stop_streaming()
        esp32.disconnect()

    # Write CSV
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {len(rows)} rows to {csv_path}")

    # Analysis
    if errors:
        all_errors = np.array(errors)
        print("\n" + "=" * 70)
        print("STANCE HOLD RESULTS")
        print("=" * 70)

        print(f"\nDuration: {args.duration}s, Samples: {len(errors)}")

        print("\nTracking Error (|cmd - actual|):")
        print(f"  {'Joint':10s}  {'Mean':>8s}  {'Max':>8s}  {'Pass?':>6s}")
        print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*6}")
        all_pass = True
        for i, name in enumerate(JOINT_NAMES):
            mean_err = all_errors[:, i].mean()
            max_err = all_errors[:, i].max()
            passed = max_err < 0.05
            all_pass = all_pass and passed
            print(f"  {name:10s}  {mean_err:8.4f}  {max_err:8.4f}  {'PASS' if passed else 'FAIL':>6s}")

        # Load analysis
        all_loads = np.array([r for r in rows])
        max_load_any = max(abs(r[f"load_{name}"]) for r in rows for name in JOINT_NAMES)
        max_curr_any = max(r[f"curr_{name}"] for r in rows for name in JOINT_NAMES)

        print(f"\nMax load across all joints: {max_load_any}")
        print(f"Max current across all joints: {max_curr_any} mA")
        print(f"Voltage range: {min(r['voltage_V'] for r in rows):.1f}V - {max(r['voltage_V'] for r in rows):.1f}V")

        load_pass = max_load_any < 300  # 30%
        curr_pass = max_curr_any < 400
        print(f"\nLoad < 30%: {'PASS' if load_pass else 'FAIL'}")
        print(f"Current < 400 mA: {'PASS' if curr_pass else 'FAIL'}")
        print(f"Tracking < 0.05 rad: {'PASS' if all_pass else 'FAIL'}")

        overall = all_pass and load_pass and curr_pass
        print(f"\n{'='*70}")
        print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
        print(f"{'='*70}")
    else:
        print("\nWARNING: No valid telemetry received")


if __name__ == "__main__":
    main()
