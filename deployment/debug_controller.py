#!/usr/bin/env python3
"""Debug version of harold_controller - prints observations to diagnose drift."""
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed")
    sys.exit(1)

from drivers.esp32_serial import ESP32Interface, ESP32Config
from drivers.imu_reader_rpi5 import IMUReaderRPi5, IMUConfig
from inference.cpg_generator import CPGGenerator, CPGConfig
from inference.observation_builder import ObservationBuilder, ObservationConfig, normalize_observation
from inference.action_converter import ActionConverter, ActionConfig


def main():
    config_dir = Path(__file__).parent / "config"
    policy_path = Path(__file__).parent / "policy" / "harold_policy.onnx"
    metadata_path = Path(__file__).parent / "policy" / "policy_metadata.json"

    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)
    running_mean = np.array(metadata["running_mean"], dtype=np.float32)
    running_var = np.array(metadata["running_variance"], dtype=np.float32)

    # Load policy
    policy = ort.InferenceSession(str(policy_path), providers=['CPUExecutionProvider'])

    # Initialize components
    cpg_config = CPGConfig.from_yaml(config_dir / "cpg.yaml")
    cpg = CPGGenerator(cpg_config)

    esp32_config = ESP32Config.from_yaml(config_dir / "hardware.yaml")
    esp32 = ESP32Interface(esp32_config)
    if not esp32.connect():
        print("ESP32 connection failed!")
        return

    imu_config = IMUConfig.from_yaml(config_dir / "hardware.yaml")
    imu = IMUReaderRPi5(imu_config)
    if not imu.connect():
        print("IMU connection failed!")
        return

    obs_config = ObservationConfig.from_yaml(config_dir / "cpg.yaml")
    obs_builder = ObservationBuilder(imu, esp32, obs_config)

    action_config = ActionConfig.from_yaml(config_dir / "cpg.yaml", config_dir / "hardware.yaml")
    action_conv = ActionConverter(action_config)

    # Calibrate
    imu.calibrate(duration=3)

    print("\n" + "=" * 80)
    print("DEBUG MODE - Printing observations every second")
    print("=" * 80 + "\n")

    esp32.start_streaming()
    obs_builder.reset()
    action_conv.reset()

    start_time = time.time()
    warmup_duration = 3 / cpg.cfg.frequency_hz
    policy_enabled = False
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, signal_handler)

    loop_count = 0
    try:
        while running:
            loop_start = time.time()
            t = loop_start - start_time

            # CPG
            cpg_targets = cpg.compute(t)
            phase_sin, phase_cos = cpg.get_phase_sin_cos()

            # Build observation
            obs = obs_builder.build(t, phase_sin, phase_cos)
            obs_norm = normalize_observation(obs, running_mean, running_var)

            if not policy_enabled and t >= warmup_duration:
                policy_enabled = True
                print(f"\n[t={t:.1f}s] *** POLICY ENABLED ***\n")

            if policy_enabled:
                outputs = policy.run(['mean'], {'obs': obs_norm.reshape(1, -1).astype(np.float32)})
                action = outputs[0][0]
            else:
                action = np.zeros(12, dtype=np.float32)

            rl_targets, hw_targets = action_conv.compute(cpg_targets, action, use_cpg=True)
            prev_targets_training_mean = running_mean[36:48]
            obs_builder.update_prev_target_delta(
                rl_targets,
                action_conv.get_hw_default_pose(),
                training_mean=prev_targets_training_mean,
                blend_factor=0.1,
            )
            esp32.send_targets(hw_targets)

            loop_count += 1

            # Debug output every second
            if loop_count % 20 == 0:
                print(f"\n[t={t:.1f}s] {'POLICY' if policy_enabled else 'WARMUP'}")
                print(f"  Raw obs[0:3] lin_vel:     {obs[0:3]}")
                print(f"  Raw obs[3:6] ang_vel:     {obs[3:6]}")
                print(f"  Raw obs[6:9] gravity:     {obs[6:9]}")
                print(f"  Norm obs[0:3]:            {obs_norm[0:3]}")
                print(f"  Norm obs[33:36] commands: {obs_norm[33:36]}")
                print(f"  Policy action[0:4]:       {action[0:4]}")
                print(f"  Policy action range:      [{action.min():.3f}, {action.max():.3f}]")

                # Check for clipping
                clipped = np.sum(np.abs(obs_norm) >= 4.9)
                if clipped > 0:
                    print(f"  *** WARNING: {clipped} observations near clip limit! ***")
                    clip_indices = np.where(np.abs(obs_norm) >= 4.9)[0]
                    print(f"      Clipped indices: {clip_indices}")

            # Sleep
            elapsed = time.time() - loop_start
            if elapsed < 0.05:
                time.sleep(0.05 - elapsed)

    finally:
        esp32.stop_streaming()
        esp32.disconnect()
        imu.disconnect()
        print("\nDone")


if __name__ == "__main__":
    main()
