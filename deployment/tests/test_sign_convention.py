#!/usr/bin/env python3
"""M3A: Verify joint sign convention by applying small deltas.

For each joint category (shoulder, thigh, calf), sends a small positive
delta in RL convention through the action converter and checks that the
servo moves in the expected direction.

Usage:
    python -m tests.test_sign_convention [--delta 0.1] [--port /dev/ttyUSB0]

Expected behavior:
    - FL shoulder +delta → servo position increases (joint_sign = +1)
    - FL thigh +delta    → servo position DECREASES (joint_sign = -1)
    - FL calf +delta     → servo position DECREASES (joint_sign = -1)
"""
import argparse
import signal
import sys
import time
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

CONTROL_RATE_HZ = 20
CONTROL_PERIOD = 1.0 / CONTROL_RATE_HZ


def hold_and_measure(esp32, targets, duration=2.0):
    """Send targets for duration and return average measured positions."""
    positions_list = []
    start = time.time()
    while time.time() - start < duration:
        loop_start = time.time()
        esp32.send_targets(targets)
        telem = esp32.read_telemetry()
        if telem.valid:
            positions_list.append(telem.positions.copy())
        elapsed = time.time() - loop_start
        if elapsed < CONTROL_PERIOD:
            time.sleep(CONTROL_PERIOD - elapsed)
    if positions_list:
        # Use last half for steady-state
        half = len(positions_list) // 2
        return np.mean(positions_list[half:], axis=0)
    return None


def main():
    parser = argparse.ArgumentParser(description="Verify joint sign convention")
    parser.add_argument("--delta", type=float, default=0.1, help="Delta in RL convention (rad)")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0", help="ESP32 serial port")
    parser.add_argument("--hold-time", type=float, default=2.0, help="Hold duration per test (s)")
    args = parser.parse_args()

    hw_default = load_hw_default_pose()
    joint_sign = np.array(JOINT_SIGN, dtype=np.float32)

    print(f"Default pose (hw): {np.array2string(hw_default, precision=3)}")
    print(f"Joint signs: {joint_sign}")
    print(f"RL delta: +{args.delta} rad")

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

    try:
        # 1. Measure baseline at default pose
        print(f"\nHolding default pose for {args.hold_time}s...")
        baseline = hold_and_measure(esp32, hw_default, args.hold_time)
        if baseline is None:
            print("ERROR: No telemetry at baseline")
            return
        print(f"  Baseline positions: {np.array2string(baseline, precision=4)}")

        # 2. Test each joint individually
        for joint_idx in range(12):
            if not running:
                break

            name = JOINT_NAMES[joint_idx]
            sign = joint_sign[joint_idx]

            # Apply positive RL delta → convert to HW delta
            # rl_delta = +delta
            # hw_delta = rl_delta * joint_sign (because hw = hw_default + rl_relative * joint_sign)
            hw_delta = args.delta * sign
            expected_direction = "increase" if hw_delta > 0 else "decrease"

            targets = hw_default.copy()
            targets[joint_idx] += hw_delta

            print(f"\n[{joint_idx:2d}] {name}: RL +{args.delta:.2f} → HW {hw_delta:+.3f} → expect {expected_direction}")
            measured = hold_and_measure(esp32, targets, args.hold_time)

            if measured is not None:
                actual_delta = measured[joint_idx] - baseline[joint_idx]
                actual_direction = "increase" if actual_delta > 0 else "decrease"
                match = actual_direction == expected_direction
                results.append((name, joint_idx, sign, hw_delta, actual_delta, match))

                print(f"     Actual delta: {actual_delta:+.4f} rad ({actual_direction})")
                print(f"     {'PASS' if match else '*** FAIL ***'}")
            else:
                print(f"     ERROR: No telemetry")
                results.append((name, joint_idx, sign, hw_delta, 0.0, False))

            # Return to default between tests
            hold_and_measure(esp32, hw_default, 0.5)

    finally:
        print("\nStopping...")
        esp32.stop_streaming()
        esp32.disconnect()

    # Summary
    print("\n" + "=" * 70)
    print("SIGN CONVENTION RESULTS")
    print("=" * 70)
    print(f"\n{'Joint':10s}  {'Sign':>5s}  {'HW Delta':>10s}  {'Actual':>10s}  {'Result':>8s}")
    print(f"{'-'*10}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*8}")

    all_pass = True
    for name, idx, sign, hw_d, actual_d, match in results:
        all_pass = all_pass and match
        print(f"{name:10s}  {sign:+5.0f}  {hw_d:+10.3f}  {actual_d:+10.4f}  {'PASS' if match else 'FAIL':>8s}")

    print(f"\n{'='*70}")
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'} ({sum(m for *_, m in results)}/{len(results)} joints correct)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
