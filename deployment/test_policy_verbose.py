#!/usr/bin/env python3
"""Verbose policy test with observation/action logging.

Run for a few seconds and log observations and actions to diagnose issues.
"""
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import onnxruntime as ort

from drivers.esp32_serial import ESP32Interface
from drivers.imu_reader_rpi5 import IMUReaderRPi5
from inference.cpg_generator import CPGGenerator
from inference.observation_builder import ObservationBuilder, normalize_observation
from inference.action_converter import ActionConverter


def main():
    print("=" * 70)
    print("Harold Verbose Policy Test")
    print("=" * 70)

    # Load metadata
    with open("policy/policy_metadata.json") as f:
        meta = json.load(f)
    running_mean = np.array(meta["running_mean"], dtype=np.float32)
    running_var = np.array(meta["running_variance"], dtype=np.float32)

    # Load policy
    policy = ort.InferenceSession("policy/harold_policy.onnx", providers=['CPUExecutionProvider'])

    # Initialize hardware
    esp32 = ESP32Interface()
    if not esp32.connect():
        print("ERROR: ESP32 connection failed")
        return 1

    imu = IMUReaderRPi5()
    if not imu.connect():
        print("ERROR: IMU connection failed")
        esp32.disconnect()
        return 1

    print("Calibrating IMU...")
    imu.calibrate(duration=2.0)

    cpg = CPGGenerator()
    obs_builder = ObservationBuilder(imu, esp32)
    action_conv = ActionConverter()

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    print("\nStarting verbose test for 10 seconds...")
    print("-" * 70)

    esp32.start_streaming()
    start_time = time.time()
    loop_count = 0

    while running and (time.time() - start_time) < 10.0:
        t = time.time() - start_time

        # CPG targets
        cpg_targets = cpg.compute(t)
        phase_sin, phase_cos = cpg.get_phase_sin_cos()

        # Build observation
        obs_raw = obs_builder.build(t, phase_sin, phase_cos)

        # Normalize
        obs_norm = normalize_observation(obs_raw, running_mean, running_var)

        # Policy inference
        outputs = policy.run(['mean'], {'obs': obs_norm.reshape(1, -1)})
        action_raw = outputs[0][0]

        # Compute targets
        rl_targets, hw_targets = action_conv.compute(cpg_targets, action_raw, use_cpg=True)

        # Send to robot
        esp32.send_targets(hw_targets)
        prev_targets_training_mean = running_mean[36:48]
        obs_builder.update_prev_target_delta(
            rl_targets,
            action_conv.get_hw_default_pose(),
            training_mean=prev_targets_training_mean,
            blend_factor=0.1,
        )

        loop_count += 1

        # Log every 0.5 seconds
        if loop_count % 10 == 0:
            # Key observations
            gravity_z = obs_norm[8]  # Gravity Z (should be ~-0.14)
            joint_pos_0 = obs_norm[9]  # First joint pos

            # Action stats
            action_min = action_raw.min()
            action_max = action_raw.max()

            # Correction stats
            corrections = action_raw * 0.05 * np.array([0.3]*4 + [0.9]*8)
            correction_max = np.abs(corrections).max()

            print(f"t={t:5.1f}s | "
                  f"grav_z={gravity_z:+6.2f} | "
                  f"jpos0={joint_pos_0:+6.2f} | "
                  f"act=[{action_min:+6.1f},{action_max:+6.1f}] | "
                  f"corr_max={correction_max:.3f}rad")

        time.sleep(0.05)

    esp32.stop_streaming()
    esp32.disconnect()

    print("-" * 70)
    print(f"Test complete: {loop_count} loops in {time.time()-start_time:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
