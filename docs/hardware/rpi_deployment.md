# Harold Deployment - Raspberry Pi 5

Real-time inference for Harold quadruped robot (open-loop CPG or direct RL policy depending on mode).

## Architecture

```
RPi 5 (inference + CPG) <--USB Serial 115200--> ESP32 (servo control) <--1Mbps--> 12 Servos
         |
         | I2C
         v
     MPU6050 (IMU)
```

## Quick Start

Use system `python3` on the Pi (no virtualenv required).

```bash
# Install dependencies
python3 -m pip install -r deployment/requirements.txt

# Run controller (robot must be connected)
python3 -m inference.harold_controller
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
| `common/cpg_math.py` | Shared CPG math kernel (sim + hardware) |
| `inference/observation_builder.py` | IMU + servo -> 48D observation |
| `inference/action_converter.py` | Policy output -> servo commands |
| `drivers/imu_reader_rpi5.py` | MPU6050 I2C driver |
| `drivers/esp32_serial.py` | ESP32 serial wrapper |

## Control Loop (policy rate)

1. Compute CPG base trajectory from time (CPG-only mode)
2. Build 48D observation from IMU + servo feedback (policy mode)
3. Run ONNX inference (normalization may be baked into the export; verify before adding preprocessing)
4. Convert targets to hardware convention
5. Send to ESP32 via serial

In policy mode, `--command VX VY YAW` sets body-frame forward and left velocity in m/s and yaw rate in rad/s. The default is `0.3 0 0`. Use `--max-seconds` for a timed trial. When evaluating a staged export, pass both `--policy` and its matching `--metadata`.

## Safety Features

- Watchdog timeout (configured in firmware)
- The streaming firmware clamps physical joint angles after its sign conversion.
- The controller requests STOP at the critical servo load or temperature in `config/hardware.yaml`. STOP commands the ready stance with torque enabled.
- Policy inference requires a recent encoder packet and a valid IMU sample. A failed per-servo read appears as zero load, current, and temperature together. The controller stops before using its substituted position.
- Keyboard interrupt = safe stance

## Configuration

- `config/hardware.yaml` - Servo IDs, signs, limits
- `config/cpg.yaml` - CPG parameters (matches simulation)
- `config/stance.yaml` - Canonical ready stance (shared across hardware + sim)

Note: `policy/policy_metadata.json` captures training-time defaults; do not edit it for stance changes.

### Joint Sign Convention

- High-level shoulder semantics are the same on all four legs. Do not mirror FR/BR shoulders in policy space.
- Thigh and calf joints are sign-inverted between RL convention and hardware convention.
- Right-side servo mounting is handled separately by `config/hardware.yaml` `direction` values and the ESP32 firmware `DIR_TABLE`.
- Controller startup treats `config/hardware.yaml` as the source of truth and rejects exported metadata whose `joint_sign` disagrees with it.

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
- A 5 Hz row includes the sensor sample used for policy inference, or the current IMU sample during CPG playback. `imu_accel_g_*` is in g, `imu_gyro_*` is in rad/s, and `imu_proj_g_*` uses the Z-up IMU convention. Policy gravity uses the opposite sign. `imu_lin_vel_*` records the accelerometer-integrated estimate and `policy_lin_vel_b_m_s_*` records the actual body velocity input to the network.
- Controller exceptions write to `logs/controller_diagnostics.log` with throttling + USB state.
- To persist system logs across reboots, run `scripts/enable_persistent_journal.sh` on the Pi (requires sudo).
