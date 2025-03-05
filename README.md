# Harold Robot

This repository contains the code for the Harold quadruped robot.

## Repository Structure

- `firmware/`: Arduino code for the robot
  - `robot_controller/`: Main servo controller for the robot
  - `calibration/`: Servo calibration utility

- `policy/`: Neural network policy for robot control
  - `harold_policy.onnx`: ONNX model for the control policy
  - `action_config.pt`: Configuration for action scaling
  - `robot_controller.py`: Main script for deploying the policy on the robot

- `sensors/`: Code for interfacing with sensors
  - `imu_reader.py`: Class for MPU6050 IMU sensor integration

- `tools/`: Utility scripts
  - `verify_policy.py`: Verify the ONNX policy model
  - `test_imu.py`: Test the IMU sensor

- `logs/`: Log files for debugging and analysis

## Usage

To run the robot with the neural network policy:

```bash
python3 policy/robot_controller.py [--serial_port PORT] [--slow_mode] [--obs_log_file FILE]
```

## Hardware Requirements

- Arduino microcontroller
- MPU6050 IMU sensor
- 12 servo motors
