#!/usr/bin/env python3
"""M3A: Verify joint sign convention by applying small deltas.

For each joint category (shoulder, thigh, calf), sends a small positive
delta using the configured sign and checks the direction reported by the
firmware. An operator observes the supported leg to verify anatomical motion.

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
from inference.action_converter import ActionConfig, ActionConverter
from deployment.validation.firmware_limits import manager_joint_target_limits


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
    last_stamp = esp32.last_telemetry.timestamp_ms
    start = time.monotonic()
    last_received = start
    while time.monotonic() - start < duration:
        loop_start = time.monotonic()
        telem = esp32.read_telemetry()
        if telem.valid and telem.timestamp_ms != last_stamp:
            last_received = loop_start
            telem.check_servo_limits(esp32.cfg)
            if telem.missing_servo_indices.size:
                raise RuntimeError(f"Missing servo feedback at {telem.missing_servo_indices.tolist()}")
            if loop_start - start >= duration / 2:
                positions_list.append(telem.positions.copy())
            last_stamp = telem.timestamp_ms
        if loop_start - last_received >= esp32.cfg.watchdog_ms / 1000.0:
            raise RuntimeError("ESP32 encoder telemetry stopped updating")
        if telem.valid and not esp32.send_targets(targets):
            raise RuntimeError("Sending servo targets failed")
        elapsed = time.monotonic() - loop_start
        if elapsed < CONTROL_PERIOD:
            time.sleep(CONTROL_PERIOD - elapsed)
    if positions_list:
        return np.mean(positions_list, axis=0)
    raise RuntimeError("No fresh encoder samples during the second half of the hold")


def main():
    parser = argparse.ArgumentParser(description="Verify joint sign convention")
    parser.add_argument("--delta", type=float, default=0.1, help="Delta in RL convention (rad)")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0", help="ESP32 serial port")
    parser.add_argument("--hold-time", type=float, default=2.0, help="Hold duration per test (s)")
    parser.add_argument("--joint", choices=JOINT_NAMES, help="Test only this joint (default: all 12)")
    args = parser.parse_args()
    max_delta = 0.20 if args.joint else 0.10
    if not 0.0 < args.delta <= max_delta or not 0.1 <= args.hold_time <= 2.0:
        parser.error(f"--delta must be in (0, {max_delta}] rad and --hold-time in [0.1, 2] seconds")

    config_dir = _DEPLOY_DIR / "config"
    hw_path = config_dir / "hardware.yaml"
    converter = ActionConverter(ActionConfig.from_yaml(config_dir / "cpg.yaml", hw_path))
    hw_default = converter.get_hw_default_pose()
    joint_sign = converter.cfg.joint_sign
    selected = [JOINT_NAMES.index(args.joint)] if args.joint else range(len(JOINT_NAMES))

    lower, upper = manager_joint_target_limits()
    ready = converter.get_rl_default_pose()
    if np.any(ready < lower) or np.any(ready > upper) or np.any(ready[selected] + args.delta > upper[selected]):
        parser.error("Pulse or ready stance exceeds the configured firmware target envelope")

    print(f"Default pose (hw): {np.array2string(hw_default, precision=3)}")
    print(f"Joint signs: {joint_sign}")
    print(f"RL delta: +{args.delta} rad")

    # Connect
    config = ESP32Config.from_yaml(hw_path)
    config.port = args.port
    esp32 = ESP32Interface(config)

    def stop_on_signal(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop_on_signal)

    print(f"\nConnecting to ESP32 on {args.port}...")
    if not esp32.connect():
        print("ERROR: ESP32 connection failed")
        sys.exit(1)

    results = []

    try:
        if not esp32.start_streaming():
            raise RuntimeError("ESP32 did not acknowledge streaming start")
        # 1. Measure baseline at default pose
        print(f"\nHolding default pose for {args.hold_time}s...")
        baseline = hold_and_measure(esp32, hw_default, args.hold_time)
        print(f"  Baseline positions: {np.array2string(baseline, precision=4)}")

        # 2. Test each joint individually
        for joint_idx in selected:
            name = JOINT_NAMES[joint_idx]
            sign = joint_sign[joint_idx]

            rl_target = converter.get_rl_default_pose()
            rl_target[joint_idx] += args.delta
            _, targets = converter.convert_rl_targets(rl_target)
            hw_delta = targets[joint_idx] - hw_default[joint_idx]
            expected_direction = "increase" if hw_delta > 0 else "decrease"

            print(f"\n[{joint_idx:2d}] {name}: RL +{args.delta:.2f} → HW {hw_delta:+.3f} → expect {expected_direction}")
            measured = hold_and_measure(esp32, targets, args.hold_time)

            actual_delta = measured[joint_idx] - baseline[joint_idx]
            match = np.sign(actual_delta) == np.sign(hw_delta)
            results.append((name, joint_idx, sign, hw_delta, actual_delta, match))

            print(f"     Actual encoder delta: {actual_delta:+.4f} rad")
            print(f"     {'PASS' if match else '*** FAIL ***'}")
            if not match:
                break

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

    all_pass = len(results) == len(selected) and all(result[-1] for result in results)
    for name, idx, sign, hw_d, actual_d, match in results:
        print(f"{name:10s}  {sign:+5.0f}  {hw_d:+10.3f}  {actual_d:+10.4f}  {'PASS' if match else 'FAIL':>8s}")

    print(f"\n{'='*70}")
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'} ({sum(m for *_, m in results)}/{len(selected)} encoder directions correct)")
    print(f"{'='*70}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
