#!/usr/bin/env python3
"""Analyze joint position values to debug observation issues."""
import json
import time
import numpy as np
from pathlib import Path

from drivers.esp32_serial import ESP32Interface
from inference.observation_builder import ObservationConfig

# Load expected values
with open('policy/policy_metadata.json') as f:
    metadata = json.load(f)
mean = np.array(metadata['running_mean'])
var = np.array(metadata['running_variance'])
std = np.sqrt(var)

# Get default pose
config = ObservationConfig.from_yaml(Path('config/cpg.yaml'))
default_pose = config.default_pose
joint_sign = config.joint_sign

print('=== Joint Position Analysis ===')
print(f'Default pose (hw convention): {default_pose}')
print()

# Connect to ESP32
esp32 = ESP32Interface()
if esp32.connect():
    esp32.start_streaming()
    time.sleep(0.5)

    telem = esp32.read_telemetry()
    if telem.valid:
        hw_pos = telem.positions
        # No sign conversion - both positions and default_pose are in hardware convention
        rel_pos = hw_pos - default_pose

        print('Hardware positions:', hw_pos)
        print('Default pose:      ', default_pose)
        print('Relative (hw-def): ', rel_pos)
        print()

        # Check normalization for each joint
        print('Normalized joint positions:')
        for i in range(12):
            idx = 9 + i
            norm_val = (rel_pos[i] - mean[idx]) / std[idx]
            clip_warn = ' *** CLIPS!' if abs(norm_val) > 5 else ''
            print(f'  [{i:2d}] hw={hw_pos[i]:+.3f} rel={rel_pos[i]:+.3f} mean={mean[idx]:+.3f} norm={norm_val:+6.2f}{clip_warn}')

    esp32.stop_streaming()
    esp32.disconnect()
