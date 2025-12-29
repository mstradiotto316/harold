#!/bin/bash
# Harold first-boot setup

set -e

echo "Installing Harold dependencies..."

# Enable I2C for IMU
sudo raspi-config nonint do_i2c 0

# Install Python packages
sudo apt update
sudo apt install -y python3-pip python3-smbus i2c-tools

# Install Python dependencies
pip3 install --break-system-packages onnxruntime pyserial smbus2 pyyaml numpy

# Create logs directory
mkdir -p /home/pi/harold/logs

# Mark setup complete
touch /home/pi/.harold_setup_complete

echo "Setup complete! Reboot to start Harold."
