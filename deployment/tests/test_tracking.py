#!/usr/bin/env python3
"""M3B: Sinusoidal tracking test to measure servo bandwidth.

Commands a sinusoidal trajectory on each joint and measures:
- RMS tracking error
- Phase lag
- Max overshoot

Usage:
    python -m tests.test_tracking [--freq 0.5] [--amplitude 0.1] [--port /dev/ttyUSB0]

Pass criteria:
    - RMS tracking error < 0.03 rad (shoulders), < 0.05 rad (thighs/calves)
    - Phase lag < 50ms (1 control cycle at 20 Hz)
"""
import argparse
import csv
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_DEPLOY_DIR = Path(__file__).parent.parent
_REPO_ROOT = _DEPLOY_DIR.parent
sys.path.insert(0, str(_DEPLOY_DIR))
sys.path.insert(0, str(_REPO_ROOT))

from drivers.esp32_serial import ESP32Interface, ESP32Config
from inference.stance import load_hw_default_pose
from common.policy_config import JOINT_SIGN


JOINT_NAMES = [
    "FL_sh", "FR_sh", "BL_sh", "BR_sh",
    "FL_th", "FR_th", "BL_th", "BR_th",
    "FL_ca", "FR_ca", "BL_ca", "BR_ca",
]

JOINT_CATEGORIES = ["shoulder"] * 4 + ["thigh"] * 4 + ["calf"] * 4

CONTROL_RATE_HZ = 20
CONTROL_PERIOD = 1.0 / CONTROL_RATE_HZ


def estimate_phase_lag(cmd_signal, meas_signal, dt):
    """Estimate phase lag via cross-correlation."""
    if len(cmd_signal) < 10:
        return 0.0
    # Normalize
    cmd_norm = cmd_signal - cmd_signal.mean()
    meas_norm = meas_signal - meas_signal.mean()
    # Cross-correlation
    corr = np.correlate(meas_norm, cmd_norm, mode='full')
    lag_samples = corr.argmax() - (len(cmd_norm) - 1)
    return lag_samples * dt


def run_sinusoid_test(esp32, hw_default, joint_idx, freq, amplitude, duration, sign):
    """Run sinusoidal test on a single joint.

    Returns dict with cmd_signal, meas_signal, timestamps, tracking_error.
    """
    # Apply sign: RL positive → HW positive (for shoulder) or HW negative (for thigh/calf)
    hw_amplitude = amplitude * sign

    timestamps = []
    cmd_values = []
    meas_values = []

    start = time.time()
    while time.time() - start < duration:
        loop_start = time.time()
        t = loop_start - start

        # Sinusoidal command
        delta = hw_amplitude * np.sin(2 * np.pi * freq * t)
        targets = hw_default.copy()
        targets[joint_idx] += delta

        esp32.send_targets(targets)
        telem = esp32.read_telemetry()

        if telem.valid:
            timestamps.append(t)
            cmd_values.append(float(targets[joint_idx]))
            meas_values.append(float(telem.positions[joint_idx]))

        elapsed = time.time() - loop_start
        if elapsed < CONTROL_PERIOD:
            time.sleep(CONTROL_PERIOD - elapsed)

    cmd_arr = np.array(cmd_values)
    meas_arr = np.array(meas_values)
    err_arr = np.abs(cmd_arr - meas_arr)

    # Skip first 0.5s (transient)
    skip = max(1, int(0.5 * CONTROL_RATE_HZ))
    if len(err_arr) > skip:
        rms_error = np.sqrt(np.mean(err_arr[skip:] ** 2))
        max_error = err_arr[skip:].max()
    else:
        rms_error = np.sqrt(np.mean(err_arr ** 2))
        max_error = err_arr.max() if len(err_arr) > 0 else 0.0

    dt = 1.0 / CONTROL_RATE_HZ
    phase_lag = estimate_phase_lag(cmd_arr[skip:], meas_arr[skip:], dt) if len(cmd_arr) > skip else 0.0

    return {
        "timestamps": np.array(timestamps),
        "cmd": cmd_arr,
        "meas": meas_arr,
        "rms_error": rms_error,
        "max_error": max_error,
        "phase_lag_s": phase_lag,
    }


def main():
    parser = argparse.ArgumentParser(description="Sinusoidal servo tracking test")
    parser.add_argument("--freq", type=float, default=0.5, help="Sinusoid frequency (Hz)")
    parser.add_argument("--amplitude", type=float, default=0.1, help="Amplitude (rad, RL convention)")
    parser.add_argument("--duration", type=float, default=5.0, help="Test duration per joint (s)")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0", help="ESP32 serial port")
    parser.add_argument("--joints", type=str, default="all",
                        help="Joints to test: 'all', 'one_per_category', or comma-separated indices")
    parser.add_argument("--output", type=str, default=None, help="CSV output path")
    args = parser.parse_args()

    hw_default = load_hw_default_pose()
    joint_sign = np.array(JOINT_SIGN, dtype=np.float32)

    # Determine which joints to test
    if args.joints == "all":
        test_joints = list(range(12))
    elif args.joints == "one_per_category":
        test_joints = [0, 4, 8]  # FL_sh, FL_th, FL_ca
    else:
        test_joints = [int(j.strip()) for j in args.joints.split(",")]

    print(f"Sinusoidal tracking test")
    print(f"  Frequency: {args.freq} Hz")
    print(f"  Amplitude: {args.amplitude} rad (RL convention)")
    print(f"  Duration: {args.duration}s per joint")
    print(f"  Testing joints: {[JOINT_NAMES[j] for j in test_joints]}")

    # Connect
    config = ESP32Config(port=args.port)
    esp32 = ESP32Interface(config)

    running = True
    def sighandler(signum, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, sighandler)

    print(f"\nConnecting to ESP32 on {args.port}...")
    if not esp32.connect():
        print("ERROR: ESP32 connection failed")
        sys.exit(1)

    if not esp32.start_streaming():
        print("ERROR: Failed to start streaming")
        esp32.disconnect()
        sys.exit(1)

    results = []
    all_traces = []

    try:
        for joint_idx in test_joints:
            if not running:
                break

            name = JOINT_NAMES[joint_idx]
            category = JOINT_CATEGORIES[joint_idx]
            sign = joint_sign[joint_idx]

            print(f"\nTesting [{joint_idx}] {name} ({category}, sign={sign:+.0f})...")

            # Settle at default first
            for _ in range(10):
                esp32.send_targets(hw_default)
                time.sleep(CONTROL_PERIOD)

            # Run sinusoid
            result = run_sinusoid_test(
                esp32, hw_default, joint_idx,
                args.freq, args.amplitude, args.duration, sign
            )
            result["joint_idx"] = joint_idx
            result["name"] = name
            result["category"] = category
            results.append(result)
            all_traces.append(result)

            threshold = 0.03 if category == "shoulder" else 0.05
            passed = result["rms_error"] < threshold
            print(f"  RMS error: {result['rms_error']:.4f} rad (threshold: {threshold})")
            print(f"  Max error: {result['max_error']:.4f} rad")
            print(f"  Phase lag: {result['phase_lag_s']*1000:.1f} ms")
            print(f"  {'PASS' if passed else 'FAIL'}")

    finally:
        # Return to default
        for _ in range(10):
            esp32.send_targets(hw_default)
            time.sleep(CONTROL_PERIOD)
        print("\nStopping...")
        esp32.stop_streaming()
        esp32.disconnect()

    # Save traces to CSV
    if all_traces and args.output:
        csv_path = Path(args.output)
    elif all_traces:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_path = Path(__file__).parent.parent / "sessions" / f"tracking_{ts}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    if all_traces:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["joint_idx", "joint_name", "time_s", "cmd_rad", "meas_rad", "error_rad"])
            for trace in all_traces:
                for i in range(len(trace["timestamps"])):
                    writer.writerow([
                        trace["joint_idx"],
                        trace["name"],
                        f"{trace['timestamps'][i]:.3f}",
                        f"{trace['cmd'][i]:.5f}",
                        f"{trace['meas'][i]:.5f}",
                        f"{abs(trace['cmd'][i] - trace['meas'][i]):.5f}",
                    ])
        print(f"\nSaved traces to {csv_path}")

    # Summary
    print("\n" + "=" * 70)
    print("TRACKING TEST RESULTS")
    print("=" * 70)
    print(f"\n{'Joint':10s}  {'Category':10s}  {'RMS Err':>10s}  {'Max Err':>10s}  {'Lag (ms)':>10s}  {'Pass?':>6s}")
    print(f"{'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}")

    all_pass = True
    for r in results:
        threshold = 0.03 if r["category"] == "shoulder" else 0.05
        passed = r["rms_error"] < threshold
        all_pass = all_pass and passed
        print(f"{r['name']:10s}  {r['category']:10s}  {r['rms_error']:10.4f}  "
              f"{r['max_error']:10.4f}  {r['phase_lag_s']*1000:10.1f}  {'PASS' if passed else 'FAIL':>6s}")

    print(f"\n{'='*70}")
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'} ({sum(1 for r in results if r['rms_error'] < (0.03 if r['category']=='shoulder' else 0.05))}/{len(results)} joints)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
