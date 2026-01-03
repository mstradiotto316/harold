# Harold Deployment - Raspberry Pi 5

Real-time inference for Harold quadruped robot using CPG + residual RL policy.

## Architecture

```
RPi 5 (inference + CPG) <--USB Serial 115200--> ESP32 (servo control) <--1Mbps--> 12 Servos
         |
         | I2C
         v
     MPU6050 (IMU)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run controller (robot must be connected)
python inference/harold_controller.py
```

## Components

| Component | Description |
|-----------|-------------|
| `policy/harold_policy.onnx` | Exported neural network (50D -> 12D) |
| `policy/policy_metadata.json` | Normalization stats + joint config |
| `inference/harold_controller.py` | Main 20 Hz control loop |
| `inference/cpg_generator.py` | CPG trajectory generator |
| `inference/observation_builder.py` | IMU + servo -> 50D observation |
| `inference/action_converter.py` | Policy output -> servo commands |
| `drivers/imu_reader_rpi5.py` | MPU6050 I2C driver |
| `drivers/esp32_serial.py` | ESP32 serial wrapper |

## Control Loop (20 Hz)

1. Compute CPG base trajectory from time
2. Build 50D observation from IMU + servo feedback
3. Normalize observation using running stats
4. Run ONNX inference -> 12D action
5. Compute final targets: CPG + residual * 0.05
6. Send to ESP32 via serial

## Safety Features

- 250ms watchdog timeout
- Joint angle limits (conservative)
- Load monitoring (>90% = emergency stop)
- Keyboard interrupt = safe stance

## Configuration

- `config/hardware.yaml` - Servo IDs, signs, limits
- `config/cpg.yaml` - CPG parameters (matches simulation)

## Testing

```bash
# Test IMU
python tests/test_imu.py

# Test ESP32 communication
python tests/test_esp32.py

# Test inference (no hardware)
python tests/test_inference.py
```

## Diagnostics

- Session CSVs now flush every row by default to survive abrupt resets.
- Controller exceptions write to `logs/controller_diagnostics.log` with throttling + USB state.
- To persist system logs across reboots, run `scripts/enable_persistent_journal.sh` on the Pi (requires sudo).
