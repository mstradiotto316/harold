#!/usr/bin/env python3
"""Debug script to trace the full observation -> action pipeline."""
import json
import numpy as np
from pathlib import Path

from drivers.esp32_serial import ESP32Interface, ESP32Config
from drivers.imu_reader_rpi5 import IMUReaderRPi5, IMUConfig
from inference.cpg_generator import CPGGenerator, CPGConfig
from inference.observation_builder import ObservationBuilder, ObservationConfig, normalize_observation
from inference.action_converter import ActionConverter, ActionConfig

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed")
    exit(1)

# Load configs
cpg_path = Path('config/cpg.yaml')
hw_path = Path('config/hardware.yaml')
policy_path = Path('policy/harold_policy.onnx')
metadata_path = Path('policy/policy_metadata.json')

# Load metadata
with open(metadata_path) as f:
    metadata = json.load(f)
running_mean = np.array(metadata['running_mean'], dtype=np.float32)
running_var = np.array(metadata['running_variance'], dtype=np.float32)
std = np.sqrt(running_var + 1e-8)

# Initialize components
cpg_cfg = CPGConfig.from_yaml(cpg_path)
obs_cfg = ObservationConfig.from_yaml(cpg_path)
action_cfg = ActionConfig.from_yaml(cpg_path, hw_path)

cpg = CPGGenerator(cpg_cfg)
action_conv = ActionConverter(action_cfg)

# Load policy
policy = ort.InferenceSession(str(policy_path), providers=['CPUExecutionProvider'])

# Connect to hardware
esp32_cfg = ESP32Config.from_yaml(hw_path)
esp32 = ESP32Interface(esp32_cfg)
imu_cfg = IMUConfig.from_yaml(hw_path)
imu = IMUReaderRPi5(imu_cfg)

print("Connecting to ESP32...")
if not esp32.connect():
    print("ESP32 connection failed!")
    exit(1)

print("Connecting to IMU...")
if not imu.connect():
    print("IMU connection failed!")
    esp32.disconnect()
    exit(1)

# Initialize observation builder
obs_builder = ObservationBuilder(imu, esp32, obs_cfg)

print("\n" + "=" * 70)
print("Pipeline Debug at t=0")
print("=" * 70)

# Step 1: Read hardware state
telem = esp32.read_telemetry()
if telem.valid:
    hw_pos = telem.positions
    print("\n1. HARDWARE POSITIONS (raw from ESP32):")
    print(f"   Shoulders: {hw_pos[0:4]}")
    print(f"   Thighs:    {hw_pos[4:8]}")
    print(f"   Calves:    {hw_pos[8:12]}")

# Step 2: Compute observation
t = 0.0
cpg_targets = cpg.compute(t)
phase_sin, phase_cos = cpg.get_phase_sin_cos()
obs = obs_builder.build(t, phase_sin, phase_cos)

print("\n2. OBSERVATION (50D):")
print(f"   [0:3] lin_vel:     {obs[0:3]}")
print(f"   [3:6] ang_vel:     {obs[3:6]}")
print(f"   [6:9] gravity:     {obs[6:9]}")
print(f"   [9:21] joint_pos_rel (RL convention):")
print(f"      Shoulders: {obs[9:13]}")
print(f"      Thighs:    {obs[13:17]}")
print(f"      Calves:    {obs[17:21]}")
print(f"   [21:33] joint_vel: {obs[21:33]}")
print(f"   [33:36] commands:  {obs[33:36]}")
print(f"   [36:48] prev_act:  {obs[36:48]}")
print(f"   [48:50] phase:     {obs[48:50]}")

# Step 3: Normalize observation
obs_norm = normalize_observation(obs, running_mean, running_var)

print("\n3. NORMALIZED OBSERVATION:")
print(f"   [9:21] joint_pos_rel normalized:")
for i in range(12):
    idx = 9 + i
    clip_warn = " *** CLIPS!" if abs(obs_norm[idx]) > 4.5 else ""
    print(f"      [{i:2d}] raw={obs[idx]:+.3f} mean={running_mean[idx]:+.3f} std={std[idx]:.3f} -> norm={obs_norm[idx]:+6.2f}{clip_warn}")

# Step 4: Run policy
outputs = policy.run(['mean'], {'obs': obs_norm.reshape(1, -1).astype(np.float32)})
action = outputs[0][0]

print("\n4. POLICY OUTPUT (12D):")
print(f"   Shoulders: {action[0:4]}")
print(f"   Thighs:    {action[4:8]}")
print(f"   Calves:    {action[8:12]}")
print(f"   Range: [{action.min():.2f}, {action.max():.2f}]")

# Step 5: Compute CPG targets
print("\n5. CPG TARGETS (RL convention):")
print(f"   Shoulders: {cpg_targets[0:4]}")
print(f"   Thighs:    {cpg_targets[4:8]}")
print(f"   Calves:    {cpg_targets[8:12]}")

# Step 6: Compute final targets
hw_targets = action_conv.compute(cpg_targets, action)

print("\n6. FINAL TARGETS (HW convention, ready for ESP32):")
print(f"   Shoulders: {hw_targets[0:4]}")
print(f"   Thighs:    {hw_targets[4:8]}")
print(f"   Calves:    {hw_targets[8:12]}")

# Verify conversion math
print("\n7. CONVERSION VERIFICATION:")
print(f"   HW default:  {action_cfg.hw_default_pose[[0,4,8]]}")
print(f"   RL default:  {action_cfg.rl_default_pose[[0,4,8]]}")
print(f"   Joint sign:  {action_cfg.joint_sign[[0,4,8]]}")

# Check if any values are outside expected range
shoulder_range = (-0.5, 0.5)
thigh_range = (-0.3, 0.8)  # HW convention
calf_range = (-1.5, 0.0)   # HW convention

print("\n8. SANITY CHECK:")
for i in range(4):
    if not shoulder_range[0] <= hw_targets[i] <= shoulder_range[1]:
        print(f"   WARNING: Shoulder[{i}] = {hw_targets[i]:.3f} outside {shoulder_range}")
for i in range(4):
    if not thigh_range[0] <= hw_targets[4+i] <= thigh_range[1]:
        print(f"   WARNING: Thigh[{i}] = {hw_targets[4+i]:.3f} outside {thigh_range}")
for i in range(4):
    if not calf_range[0] <= hw_targets[8+i] <= calf_range[1]:
        print(f"   WARNING: Calf[{i}] = {hw_targets[8+i]:.3f} outside {calf_range}")

# Cleanup
esp32.disconnect()
imu.disconnect()
print("\nDone.")
