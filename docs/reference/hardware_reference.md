# Hardware Reference (Robot + RPi)

This file is a technical reference for hardware operation. Use `AGENTS.md` for workflows and `docs/memory/CONTEXT.md` for current network details.

## Core hardware stack

- Raspberry Pi 5 runs policy inference and the 20 Hz control loop.
- ESP32 handles the servo bus control loop and safety watchdog.
- FeeTech STS3215 servos provide joint actuation.
- MPU6050 IMU provides inertial measurements.

For exact limits, servo specs, and safe ranges, see `docs/memory/HARDWARE_CONSTRAINTS.md`.

## Runtime layout (RPi)

Confirm repo and runtime paths in `docs/memory/CONTEXT.md` before running commands.
Runtime root is the repo’s `deployment/` directory (e.g. `/home/pi/harold/deployment`).

- Repo path (git checkout): see `docs/memory/CONTEXT.md`
- Runtime root: `<runtime-root>` (see `docs/memory/CONTEXT.md`)
- Main controller: `<runtime-root>/inference/harold_controller.py`
- Policy: `<runtime-root>/policy/harold_policy.onnx`
- Configs: `<runtime-root>/config/`
- Logs: `<runtime-root>/logs/`

## Syncing code to the Pi

The Pi runtime is a git checkout of this repo. Sync from the desktop with:

```bash
# On the desktop after edits
git add -A
git commit -m "Describe changes"
git push

# On the Pi before tests
git status -sb     # must be clean
git pull --ff-only
```

Do not edit runtime code directly on the Pi. If a hotfix is unavoidable, commit it on the Pi,
push, and then pull it on the desktop to keep both repos in sync.

## SSH access

Network addresses and credentials are tracked in `docs/memory/CONTEXT.md`.

```bash
ssh pi@<ip-or-host>
```

## Network configuration

Current IPs, hostnames, and WiFi details are maintained in `docs/memory/CONTEXT.md`.

## Service management (RPi)

```bash
sudo systemctl status harold
sudo systemctl restart harold
sudo systemctl stop harold
sudo systemctl start harold
sudo systemctl disable harold
sudo systemctl enable harold
```

## Manual controller run (RPi)

Use system `python3` on the Pi (no virtualenv required).

```bash
# Stop the service first
sudo systemctl stop harold

# Run controller manually
cd <runtime-root>
python3 -m inference.harold_controller
```

For CPG-only or scripted gait tests, see `AGENTS.md` and `docs/memory/NEXT_STEPS.md`.

## Firmware flashing (ESP32)

The ESP32 must run the streaming control firmware for the Pi to communicate with it.

```bash
# Install arduino-cli (one-time)
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh
export PATH=$PATH:/home/pi/bin
arduino-cli core install esp32:esp32

# Compile and upload (use your repo path on the Pi)
cd <repo-path>
arduino-cli compile --fqbn esp32:esp32:esp32 firmware/StreamingControl/HaroldStreamingControl
arduino-cli upload -p /dev/ttyUSB0 --fqbn esp32:esp32:esp32 firmware/StreamingControl/HaroldStreamingControl
```

## Required ESP32 firmware

- `firmware/StreamingControl/HaroldStreamingControl/` (required for Pi control)
- `firmware/scripted_gait_test_1/` (standalone scripted gait)

## Hardware connections

- ESP32 via USB (CP2102 adapter) -> `/dev/ttyUSB0` (verify on the Pi)
- IMU (MPU6050) via I2C bus 1, address `0x68`

## Troubleshooting

```bash
lsusb
ls /dev/ttyUSB*
dmesg | tail -20
```

If `/dev/ttyUSB0` permission errors occur:

```bash
sudo usermod -a -G dialout pi
```

For service logs and controller exceptions, see `<runtime-root>/logs/` and `docs/hardware/rpi_deployment.md`.

## Additional references

- Deployment runtime details: `docs/hardware/rpi_deployment.md`
- Servo calibration workflow: `docs/hardware/calibration_checklist.md`
- Servo datasheets and protocol: `docs/hardware/servos/`
- Hardware limits: `docs/memory/HARDWARE_CONSTRAINTS.md`
- ST3215 register map: `docs/hardware/servos/ST3215_memory_register_map_EN.xls`
- SCServo library (vendor drop): `firmware/third_party/SCServo`
