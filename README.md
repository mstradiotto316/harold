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

- `simulation_logs/`: Log files from simulation for playback testing

## Usage

### Running the Robot with Live IMU Data

To run the robot with live IMU sensor data:

```bash
python3 policy/robot_controller.py [--serial_port PORT]
```

### Running the Robot with Recorded Observation Logs

To run the robot using pre-recorded observation logs from simulation:

```bash
python3 policy/robot_controller_observations_playback.py [--obs_log_file PATH] [--serial_port PORT] [--loop]
```

Options:
- `--obs_log_file PATH`: Specify a custom observation log file (default: simulation_logs/observations.log)
- `--serial_port PORT`: Specify a custom serial port (default: /dev/ttyACM0)
- `--loop`: Continuously loop the observation log file

## Hardware Requirements

- Arduino microcontroller
- MPU6050 IMU sensor
- 12 servo motors

## Hardware Configuration

### Joint Configuration

The robot's joint configuration matches the simulation directly:

1. **Joint Order**:
   - 0: Front Left Shoulder
   - 1: Front Right Shoulder
   - 2: Back Left Shoulder
   - 3: Back Right Shoulder
   - 4: Front Left Thigh
   - 5: Front Right Thigh
   - 6: Back Left Thigh
   - 7: Back Right Thigh
   - 8: Front Left Knee
   - 9: Front Right Knee
   - 10: Back Left Knee
   - 11: Back Right Knee

2. **Servo Direction**:
   - Shoulders: Positive positions move upward from center, negative positions move downward
   - Thighs and Knees: Positive positions move toward the front of the robot, negative positions move toward the rear

This configuration ensures that the real robot's movements directly correspond to the simulation, with no need for joint remapping or direction inversions.
