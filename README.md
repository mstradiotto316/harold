# Harold Quadruped Robot

This repository contains the code for the Harold quadruped robot, a small legged robot platform for reinforcement learning research and experimentation.

## TODO

1. Make sure the new robot meets this setup
https://docs.isaacsim.omniverse.nvidia.com/latest/isaac_lab_tutorials/tutorial_instanceable_assets.html#instanceable-assets

2. Fix component mounts on the next iteration of the chassis


## Isaac Sim & Isaac Lab
NOTE: This is meant to be used on Linux Ubuntu
NOTE: YOU MUST INSTALL ISAAC SIM AND ISAAC LAB FIRST
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html#isaaclab-binaries-installation

1. Make sure you install the project:
python -m pip install -e harold_isaac_lab/source/harold_isaac_lab

2. To see the available tasks use:
python harold_isaac_lab/scripts/list_envs.py

3. To run the task use:
python harold_isaac_lab/scripts/rl_games/train.py --task=<Task-Name>

python harold_isaac_lab/scripts/rl_games/train.py --task=Template-Harold-Direct-flat-terrain-v0

## Repository Structure 

- `firmware/`: Arduino code for the robot
  - `robot_controller/`: Main servo controller for the robot
  - `servo_calibration/`: Servo calibration utility
  - `joint_test/`: Test individual robot joints

- `policy/`: Neural network policy for robot control
  - `harold_policy.onnx`: ONNX model for the control policy
  - `action_config.pt`: Configuration for action scaling
  - `observations_playback_test.py`: Test policy with recorded observations

- `sensors/`: Code for interfacing with sensors
  - `imu_reader.py`: Class for MPU6050 IMU sensor integration
  - `smbus_example.py`: Example for I2C communication

- `tools/`: Utility scripts
  - `verify_policy.py`: Verify the ONNX policy model
  - `test_imu.py`: Test the IMU sensor
  - `edit_action_config.py`: Tool to edit action configuration

- `simulation_files/` & `pushup_simulation_files/`: Simulation environments
  - Contains simulation code for training and testing the robot

- `simulation_logs/`: Log files from simulation for playback testing
  - `observations.log`: Recorded sensor observations
  - `actions.log`: Recorded policy actions
  - `processed_actions.log`: Actions processed for deployment

## Hardware Components

### Computing
- **Jetson Nano**: Main computing platform running the neural network policy
  - Communicates with IMU via I2C bus 1
  - Connects to Arduino via USB (typically /dev/ttyACM0)

### Sensors
- **MPU6050 IMU**: 6-axis inertial measurement unit
  - I2C address: 0x68
  - Provides acceleration and gyroscope measurements
  - Connected to I2C bus 1 on Jetson Nano

### Actuators
- **Arduino**: Motor controller board
  - Serial communication at 115200 baud rate
  - Runs control loop at 200Hz (5ms intervals)

- **Adafruit PCA9685**: PWM Servo Driver
  - I2C address: 0x40
  - Oscillator frequency: 27MHz
  - PWM frequency: 50Hz for standard servos

- **12 Servo Motors**: For quadruped locomotion
  - 3 servos per leg (shoulder, thigh, knee)
  - Custom calibration values for each servo
  - Safety features include command timeout (250ms) and position limits

## Joint Configuration

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

## Development Environment

### Arduino IDE

Due to limitations on the Jetson Nano, we are forced to use an older ARM specific version of the Arduino software:

*   Legacy IDE (1.8.19) Linux Arm 64 Bit

**Important Notes for this Arduino Environment:**

*   **Unavailable Functions:** Be aware that some standard C/C++ library functions might be missing. For example, `strtof` (string-to-float with error checking) is unavailable. Use alternatives like `atof` (standard C string-to-float), keeping in mind its limitations (e.g., less robust error reporting).
*   **Code Structure:** This environment may be stricter regarding code structure than newer IDEs. To avoid `"not declared in this scope"` errors, ensure proper C/C++ ordering: Place all `#include` directives, `#define` constants, global variable declarations, and forward function declarations (prototypes) *before* the `setup()` and `loop()` functions. Define helper functions *after* `loop()`. Do not rely solely on the IDE's automatic function prototyping.

## Usage

### Running with Live IMU Data

To run the robot with live IMU sensor data:

```bash
python3 policy/robot_controller.py [--serial_port PORT]
```

### Running with Recorded Observations

To run the robot using pre-recorded observation logs from simulation:

```bash
python3 policy/observations_playback_test.py [--obs_log_file PATH] [--serial_port PORT] [--loop]
```

Options:
- `--obs_log_file PATH`: Specify a custom observation log file (default: simulation_logs/observations.log)
- `--serial_port PORT`: Specify a custom serial port (default: /dev/ttyACM0)
- `--loop`: Continuously loop the observation log file
