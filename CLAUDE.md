# !IMPORTANT - ALL AGENTS READ ME! 

**Throughout our journey here, it's extremely important that you are not a sycophant and that you push back when things seem incorrect and that you consistently take time to step back and view things from a thousand-foot level; Consistently questioning whether we should be doing our current activities, and suggesting alternative approaches that make more sense. This needs to be remembered. So, please store it deeply within your long term system context, and keep this message alive during any compaction.**

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Architecture

### Harold Isaac Lab Extension Structure
- **Primary Task**: `harold_isaac_lab/source/harold_isaac_lab/harold_isaac_lab/tasks/direct/harold_isaac_lab/`
  - `harold.py`: Robot configuration (joints, actuators, physics)
  - `harold_isaac_lab_env.py`: Main RL environment class extending DirectRLEnv
  - `harold_isaac_lab_env_cfg.py`: Configuration management with dataclasses

### Robot Configuration (12 DOF)
Joint order matches simulation exactly:
- Shoulders (0-3): FL, FR, BL, BR
- Thighs (4-7): FL, FR, BL, BR  
- Calves (8-11): FL, FR, BL, BR

### Training Framework Integration
Multiple RL framework support:
- **SKRL**: Primary framework (recommended)
- **RSL-RL**: Not in use
- **Stable-Baselines3**: Not in use
- **RL-Games**: Not in use

### Hardware Integration Notes
- **Real Robot**: ESP32 servo controller at 115200 baud, 200Hz control loop
- **Sensors**: MPU6050 IMU on I2C bus 1 (address 0x68)
- **Joint Mapping**: Direct correspondence between simulation and hardware
- **Arduino Environment**: Legacy IDE (1.8.19) ARM64 required for Jetson Nano

### Development Dependencies
- Isaac Sim and Isaac Lab installation required
- Python 3.10+ with Isaac Lab environment
- Pre-commit for code formatting
- Tensorboard for training visualization

### Core Components

#### 1. Robot Configuration (harold.py)
- **12-DOF Quadruped**: 3 joints per leg (shoulder, thigh, calf)
- **Joint Limits**: Shoulders ±20°, thighs/calves ±45°
- **Physical Properties**: 2kg mass, 40cm length, realistic inertia
- **USD Asset**: harold_v5_11.usd with accurate collision meshes

#### 2. Environment (harold_isaac_lab_env.py)
- **Observation Space**: See _get_observations in harold_isaac_lab/source/harold_isaac_lab/harold_isaac_lab/tasks/direct/harold_isaac_lab/harold_isaac_lab_env.py
- **Action Space**: 12D joint position targets with safety clamping. See __init__ in harold_isaac_lab/source/harold_isaac_lab/harold_isaac_lab/tasks/direct/harold_isaac_lab/harold_isaac_lab_env.py for specific joint limits.
- **Physics Simulation**: See the SimulationCfg in harold_isaac_lab/source/harold_isaac_lab/harold_isaac_lab/tasks/direct/harold_isaac_lab/harold_isaac_lab_env_cfg.py
- **Isaac Lab GPU accelerated, multi-environment training**: 4096 parallel environments headless training is what the user generally selects
- **Terrain System**: See TerrainGeneratorCfg and TerrainImporterCfg in harold_isaac_lab/source/harold_isaac_lab/harold_isaac_lab/tasks/direct/harold_isaac_lab/harold_isaac_lab_env_cfg.py
- **Contact Sensors**: See ContactSensorCfg in harold_isaac_lab/source/harold_isaac_lab/harold_isaac_lab/tasks/direct/harold_isaac_lab/harold_isaac_lab_env_cfg.py
- **Height Scanner**: See height_scanner RayCasterCfg in harold_isaac_lab/source/harold_isaac_lab/harold_isaac_lab/tasks/direct/harold_isaac_lab/harold_isaac_lab_env_cfg.py
- **Visual Markers**: See _update_visualization_markers in harold_isaac_lab/source/harold_isaac_lab/harold_isaac_lab/tasks/direct/harold_isaac_lab/harold_isaac_lab_env.py

### Training Pipeline

```
Training Process:
├── Terrain: All difficulty levels (0-9) available from start
├── Commands: Full velocity range (±0.3 m/s, ±0.2 rad/s)
├── Environment: 1024 parallel environments
├── Focus: Learning robust locomotion across all terrains simultaneously
└── Duration: ~3-4 hours on modern GPU for convergence

Key Characteristics:
- No curriculum progression - all challenges from the start
- Random terrain selection ensures exposure to all difficulties
- Robust policy emerges from diverse training distribution
```

This architecture enables robust quadruped locomotion learning with progressive skill acquisition, efficient parallel training, and safe real-world deployment.

## Reward System

### Overview
Harold's reward function implements a multi-component system to encourage stable, efficient, and robust quadruped locomotion. The reward components balance primary locomotion objectives with stability and efficiency considerations.

### Configuration Location
The reward system configuration is defined in:
- **Reward Weights**: `harold_isaac_lab_env_cfg.py` → `RewardsCfg` class
- **Implementation**: `harold_isaac_lab_env.py` → `_get_rewards()` method
- **Gait Parameters**: `harold_isaac_lab_env_cfg.py` → `GaitCfg` class

### Reward Components
The system uses 5 distinct reward/penalty terms:
1. **Linear velocity tracking (Directional)** - Primary locomotion objective
   - Uses elliptical Gaussian: exp(-(e_par/0.25)² + (e_perp/0.08)²)
   - Lateral drift penalized 3x more strictly than along-track error
   - Heavily penalizes sideways movement and backwards motion
2. **Yaw velocity tracking** - Turning control
   - Gaussian reward: exp(-(error/0.4)²)
3. **Height maintenance** - Stability above terrain
   - Tanh-based reward maintaining ~18cm target height
4. **Torque penalty** - Energy efficiency
   - Quadratic penalty on joint torques to reduce energy consumption
5. **Feet air time** - Gait quality
   - Rewards optimal air time (0.4s) to encourage proper stepping patterns
   - Only active when robot speed > 0.05 m/s

Note: Velocity jitter penalty was removed as it was redundant with directional tracking and penalized natural gait patterns.

Each component has configurable weights and parameters that are frequently tuned during development. Check the `RewardsCfg` class for current values and detailed documentation of each component's mathematical formulation.

### Reward Normalization
Rewards are NOT multiplied by `step_dt` in the reward assembly to avoid double normalization. The episodic sum is already normalized by `max_episode_length_s` when logged to tensorboard, providing proper per-second normalization for visualization while maintaining correct step rewards for RL training.

## Hardware Integration and Sim-to-Real Transfer

### Overview
Harold's hardware integration system enables seamless transfer of trained policies from Isaac Lab simulation to the physical robot. The system emphasizes safety, precision, and robust communication protocols.

### Physical Robot Specifications

#### Mechanical Design
```
Robot Dimensions:
├── Total Length: 40cm (body + legs extended)
├── Body Dimensions: 20cm × 15cm × 8cm
├── Total Mass: 2.0kg (lightweight construction)
├── Leg Configuration: 4 legs × 3 joints = 12 DOF total
└── Ground Clearance: ~18cm (natural standing height)

Joint Specifications:
├── Shoulder Joints (4): ±20° range, 45° swing capability
├── Thigh Joints (4): ±45° range, primary lift/extension
├── Calf Joints (4): ±45° range, foot positioning
└── Total Range: Conservative limits for mechanical safety
```

#### Actuators and Control
```
FeeTech STS3215 Servo Motors (12 units):
├── Position Control: 0.1° resolution, 360° total range
├── Torque Output: 15kg⋅cm nominal, 20kg⋅cm stall
├── Communication: TTL serial at 1Mbps (daisy-chained)
├── Feedback: Position, velocity, load, temperature
├── Safety Features: Overload protection, position limits
└── Update Rate: 200Hz control loop for precise tracking

ESP32 Servo Controller:
├── MCU: Dual-core 240MHz ARM processor
├── Communication: USB serial at 115200 baud to host
├── Control Loop: 200Hz (5ms intervals) for smooth motion
├── Safety Monitoring: Servo status, emergency stop capability
├── Protocol: Custom binary protocol for efficiency
└── Expansion: I2C bus for additional sensors
```

#### Sensor Integration
```
MPU6050 IMU (Primary Sensor):
├── Location: Mounted in robot body center
├── Interface: I2C bus 1, address 0x68
├── Sampling Rate: 200Hz (synchronized with control loop)
├── Data: 3-axis accelerometer + 3-axis gyroscope
├── Range: ±2g acceleration, ±250°/s gyroscope
├── Filtering: Hardware DLPF + software complementary filter
└── Calibration: Automatic bias correction on startup

Optional Expansion Sensors:
├── Foot Contact Sensors: Force-sensitive resistors per foot
├── Joint Encoders: Absolute position feedback backup
├── Camera Module: ESP32-CAM for visual feedback
└── Distance Sensors: Ultrasonic or ToF for obstacle detection
```

### Communication Protocol

#### High-Level Command Interface
```
Host Computer (Jetson Nano) ←→ ESP32 Controller
│
├── Protocol: USB Serial (CDC-ACM)
├── Baud Rate: 115200 (reliable for real-time control)
├── Update Rate: 20Hz (matches RL policy frequency)
├── Command Format: Binary protocol for efficiency
└── Safety: Heartbeat monitoring, timeout detection

Command Packet Structure:
┌──────────────────────────────────────────────────────────┐
│ Header │ Joint Targets (12×2 bytes) │ Flags │ Checksum │
│ (2B)   │        (24 bytes)          │ (1B)  │   (1B)   │
└──────────────────────────────────────────────────────────┘
Total: 28 bytes per command
```

#### Low-Level Servo Communication
```
ESP32 ←→ Servo Chain (STS3215 Protocol)
│
├── Physical: Half-duplex TTL serial daisy chain
├── Baud Rate: 1,000,000 (1Mbps for low latency)
├── Addressing: Each servo has unique ID (1-12)
├── Commands: Position, velocity, torque control modes
└── Feedback: Real-time status from all servos

Servo Command Cycle (5ms total):
├── Send Commands: 12 servos × 8 bytes = 96 bytes (0.96ms)
├── Read Feedback: 12 servos × 12 bytes = 144 bytes (1.44ms)
├── Processing: Command computation and safety checks (2.0ms)
└── Idle Time: Buffer for timing variations (0.6ms)
```

### Sim-to-Real Transfer Pipeline

#### Policy Deployment Process
```
Training → Validation → Hardware Deployment
│
├── 1. Simulation Training (Isaac Lab)
│   └── Train policy to convergence (~128k steps)
│
├── 2. Policy Export and Validation
│   ├── Export to ONNX format for deployment
│   ├── Validate against recorded observation sequences
│   └── Test action scaling and joint limit compliance
│
├── 3. Hardware Safety Validation
│   ├── Verify joint limits match simulation constraints
│   ├── Test emergency stop and safety monitoring
│   └── Validate communication timing and reliability
│
└── 4. Gradual Real-World Testing
    ├── Static pose validation (servos, IMU, communication)
    ├── Joint-by-joint movement testing
    ├── Basic stance and balance verification
    └── Progressive locomotion skill validation
```

#### Observation Space Mapping
```
Simulation Observations → Hardware Measurements

Root Linear Velocity (3D):
├── Simulation: Perfect body-frame velocity from physics
└── Hardware: Numerical differentiation of IMU position estimates
    ├── Integration: Double integration of accelerometer data
    ├── Drift Correction: Complementary filter with gyroscope
    └── Noise Filtering: Moving average + bias compensation

Root Angular Velocity (3D):
├── Simulation: Perfect body-frame angular rates
└── Hardware: Direct gyroscope measurements
    ├── Calibration: Automatic bias removal on startup
    ├── Scaling: Factory calibration coefficients
    └── Filtering: Low-pass filter to match simulation noise

Projected Gravity (3D):
├── Simulation: Perfect gravity vector in body frame
└── Hardware: Processed accelerometer data
    ├── Static Component: Long-term average for gravity direction
    ├── Dynamic Filtering: Separate motion from gravity
    └── Quaternion Rotation: Transform to body frame

Joint Positions/Velocities (12D each):
├── Simulation: Perfect joint state from physics engine
└── Hardware: Direct servo feedback
    ├── Position: Native servo encoder readings
    ├── Velocity: Numerical differentiation with filtering
    └── Latency Compensation: Predict forward to account for delays
```

#### Action Space Adaptation
```
Policy Actions → Hardware Commands

Joint Position Targets (12D):
├── Policy Output: Normalized actions [-1, 1]
├── Action Scaling: Apply configured scaling factor
├── Joint Limit Clamping: Hardware safety constraints
├── Command Smoothing: Rate limiting for servo protection
└── Hardware Mapping: Convert to servo position commands

Safety Transformations:
├── Rate Limiting: Maximum change per timestep
├── Emergency Stops: Immediate halt on critical conditions
├── Soft Limits: Conservative joint ranges vs mechanical limits
└── Torque Monitoring: Detect mechanical binding or collisions
```

### Real-World Deployment Architecture

#### Software Stack
```
┌─────────────────────────────────────────────────────────┐
│                    Host Computer (Jetson Nano)           │
├─────────────────────────────────────────────────────────┤
│  Python Policy Runtime                                  │
│  ├── ONNX Model Inference (harold_policy.onnx)         │
│  ├── Observation Processing (sensors/imu_reader.py)     │
│  ├── Action Scaling and Safety (policy/robot_control)   │
│  └── Serial Communication (harold_isaac_lab/sensors/)   │
├─────────────────────────────────────────────────────────┤
│  Operating System: Ubuntu 20.04 ARM64                  │
│  └── Real-time scheduling for control loop consistency  │
└─────────────────────────────────────────────────────────┘
              ↕ USB Serial (115200 baud, 20Hz)
┌─────────────────────────────────────────────────────────┐
│                    ESP32 Controller                      │
├─────────────────────────────────────────────────────────┤
│  Control Loop (200Hz)                                   │
│  ├── Command parsing and validation                     │
│  ├── Servo communication and monitoring                 │
│  ├── IMU data acquisition and processing                │
│  └── Safety monitoring and emergency stops              │
├─────────────────────────────────────────────────────────┤
│  Hardware Abstraction                                   │
│  ├── STS3215 servo protocol implementation              │
│  ├── MPU6050 I2C communication                         │
│  └── USB CDC virtual serial port                       │
└─────────────────────────────────────────────────────────┘
```

#### Deployment Scripts and Tools
```
Policy Deployment Tools:
│
├── policy/robot_controller.py
│   ├── Live policy execution with IMU feedback
│   ├── Real-time observation processing
│   ├── Safety monitoring and emergency stops
│   └── Performance logging and diagnostics
│
├── policy/observations_playback_test.py
│   ├── Replay simulation observations on hardware
│   ├── Validate policy behavior consistency
│   ├── Debug observation space mismatches
│   └── Record hardware response for analysis
│
├── sensors/imu_reader.py
│   ├── MPU6050 sensor interface class
│   ├── Calibration and bias compensation
│   ├── Real-time data processing and filtering
│   └── Sensor health monitoring
│
└── firmware/ (ESP32 Arduino Code)
    ├── Main control loop and communication
    ├── Servo control and safety monitoring
    ├── IMU data acquisition and preprocessing
    └── Hardware abstraction and error handling
```

### Safety Systems and Monitoring

#### Multi-Layer Safety Architecture
```
Layer 1: Policy-Level Safety
├── Conservative joint limits in simulation training
├── Action scaling and rate limiting
├── Observation sanity checking and filtering
└── Graceful degradation on sensor failures

Layer 2: Communication Safety
├── Command timeout detection (heartbeat monitoring)
├── Packet integrity verification (checksums)
├── Serial communication error recovery
└── Automatic reconnection on link failure

Layer 3: Hardware Safety
├── Servo overload protection and thermal monitoring
├── Emergency stop capability (immediate motor disable)
├── Mechanical limit switches (optional)
└── Power supply monitoring and brown-out detection

Layer 4: System Safety
├── Watchdog timers for system hang detection
├── Process monitoring and automatic restart
├── Logging and diagnostic data collection
└── Remote monitoring and kill-switch capability
```

#### Real-Time Performance Monitoring
```
Timing Requirements:
├── Policy Inference: <10ms (target: 5ms)
├── Observation Processing: <5ms (IMU + serial)
├── Command Transmission: <2ms (serial + servo update)
└── Total Loop Time: <20ms (20Hz policy frequency)

Performance Metrics:
├── Latency Monitoring: End-to-end timing measurements
├── Jitter Analysis: Control loop timing consistency
├── Packet Loss Detection: Communication reliability
├── Servo Health: Temperature, load, error status
└── Power Consumption: Battery life and efficiency
```

This comprehensive hardware integration system ensures reliable, safe, and efficient transfer of learned behaviors from simulation to the physical Harold robot, maintaining performance while prioritizing safety and system robustness.




## Isaac Lab Documentation

### Direct Workflow RL Environment
  The Harold project uses a Direct Workflow instead of a Manager Based Workflow for implementing RL training in Isaac Lab. These two frameworks are often confused, so it is important to differentiate between the two when reviewing documentation.

  The following link outlines how to create a basic Direct Workflow in Isaac Lab:
  https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html


### Documentation References
  Anything imported with isaaclab.assets can be found at the following link:
  https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#module-isaaclab.assets

  The most relevant part of the isaaclab.assets is the Articulation section:
  https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#articulation

  Anything imported with the isaaclab.envs can be found at the following link:
  https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.html#module-isaaclab.envs

  The most relevant part of isaaclab.envs is the Direct RL Environment section:
  https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.html#direct-rl-environment

  Anything imported with the isaaclab.sim.spawners can be found at the following link:
  https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.spawners.html#module-isaaclab.sim.spawners

  The most relevant part of isaaclab.sim.spawners is the sensors section. This includes cameras, contact sensors, and so on.
  https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.spawners.html#module-isaaclab.sim.spawners.sensors

  Anything related to isaaclab.utils can be found at the following link:
  https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.utils.html#module-isaaclab.utils

  Anything relating to isaaclab.markers can be found at the following link:
  https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.markers.html#module-isaaclab.markers
  