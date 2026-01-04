# Harold Deployment - Raspberry Pi 5

Real-time inference for Harold quadruped robot (CPG + residual policy or pure RL depending on mode).

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
pip install -r deployment/requirements.txt

# Run controller (robot must be connected)
python inference/harold_controller.py
```

## Syncing Runtime from Desktop

The Pi runtime is a git checkout of this repo. Sync from the desktop before hardware tests:

```bash
# On the desktop after edits
git add -A
git commit -m "Describe changes"
git push

# On the Pi before tests
git status -sb     # must be clean
git pull --ff-only
```

Avoid editing runtime code on the Pi. If a hotfix is unavoidable, commit it on the Pi,
push, and then pull it on the desktop to keep both repos in sync.

## Components

Paths below are relative to the deployment root (typically `<repo-root>/deployment`, e.g. `/home/pi/harold/deployment`).

| Component | Description |
|-----------|-------------|
| `policy/harold_policy.onnx` | Exported neural network (observation -> 12D actions) |
| `policy/policy_metadata.json` | Normalization stats + joint config (if applicable) |
| `inference/harold_controller.py` | Main policy-rate control loop |
| `inference/cpg_generator.py` | CPG trajectory generator |
| `inference/observation_builder.py` | IMU + servo -> 50D observation |
| `inference/action_converter.py` | Policy output -> servo commands |
| `drivers/imu_reader_rpi5.py` | MPU6050 I2C driver |
| `drivers/esp32_serial.py` | ESP32 serial wrapper |

## Control Loop (policy rate)

1. Compute CPG base trajectory from time (if enabled)
2. Build observation from IMU + servo feedback (size depends on mode)
3. Run ONNX inference (normalization may be baked into the export; verify before adding preprocessing)
4. Compute final targets: CPG + residual * residual_scale (config)
5. Send to ESP32 via serial

## Safety Features

- Watchdog timeout (configured in firmware)
- Joint angle limits (conservative, configured in runtime)
- Load monitoring with emergency stop (threshold configured in runtime)
- Keyboard interrupt = safe stance

## Configuration

- `config/hardware.yaml` - Servo IDs, signs, limits
- `config/cpg.yaml` - CPG parameters (matches simulation)
- `config/stance.yaml` - Canonical ready stance (shared across hardware + sim)

Note: `policy/policy_metadata.json` captures training-time defaults; do not edit it for stance changes.

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
