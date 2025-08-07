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

### Key Design Patterns
- **Dataclass Configuration**: All parameters managed through `@configclass` decorators
- **Multi-Terrain Training**: All 10 difficulty levels available from start
- **Multi-Component Rewards**: Velocity tracking, height maintenance, energy efficiency
- **Modular Terrain System**: 10 difficulty levels × 20 variations = 200 unique patches

### Robot Configuration (12 DOF)
Joint order matches simulation exactly:
- Shoulders (0-3): FL, FR, BL, BR
- Thighs (4-7): FL, FR, BL, BR  
- Calves (8-11): FL, FR, BL, BR

### State and Action Spaces
- **Observations**: 48D vector (robot state, commands, actions, terrain)
- **Actions**: 12D joint position targets with safety clamping
- **Physics**: 360Hz simulation, 20Hz policy updates (18:1 decimation)

### Terrain System
- **Flat Terrain**: 25% of patches
- **Random Rough**: 25% (varied complexity)
- **Slopes**: 20% (pyramid slopes and valleys)
- **Steps**: 30% (stairs and inverted stairs)

### Training Framework Integration
Multiple RL framework support:
- **SKRL**: Primary framework (recommended)
- **RSL-RL**: Alternative option
- **Stable-Baselines3**: For comparison
- **RL-Games**: Additional support

### Important File Locations
- **Robot Model**: `harold_isaac_lab/assets/harold_v5_11.usd`
- **Training Logs**: `logs/skrl/harold_direct/`
- **Configuration**: `harold_isaac_lab/source/harold_isaac_lab/harold_isaac_lab/tasks/direct/harold_isaac_lab/agents/`
- **Scripts**: `harold_isaac_lab/scripts/` (organized by RL framework)

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

This codebase implements a sophisticated quadruped locomotion system with multi-terrain training, comprehensive sensor integration, and flexible multi-framework RL training capabilities for both simulation and real-world deployment.

## System Architecture Overview

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                        HAROLD SYSTEM ARCHITECTURE               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐  │
│  │   RL Training   │    │   Simulation    │    │  Hardware   │  │
│  │                 │    │                 │    │             │  │
│  │ • Policy Net    │◄──►│ • Isaac Lab     │◄──►│ • ESP32     │  │
│  │ • Reward Func   │    │ • Physics Sim   │    │ • Servos    │  │
│  │ • Multi-Terrain │    │ • Terrain Gen   │    │ • IMU       │  │
│  └─────────────────┘    └─────────────────┘    └─────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Robot Configuration (harold.py)
- **12-DOF Quadruped**: 3 joints per leg (shoulder, thigh, calf)
- **Joint Limits**: Shoulders ±20°, thighs/calves ±45°
- **Physical Properties**: 2kg mass, 40cm length, realistic inertia
- **USD Asset**: harold_v5_11.usd with accurate collision meshes

#### 2. Environment (harold_isaac_lab_env.py)
- **Observation Space**: 48D vector [velocities(6) + gravity(3) + joints(24) + commands(3) + actions(12)]
- **Action Space**: 12D joint position targets with safety clamping
- **Physics Simulation**: 360Hz physics, 20Hz policy (18:1 decimation)
- **Multi-Environment**: 1024 parallel environments for efficient training

#### 3. Terrain System (harold_isaac_lab_env_cfg.py)
- **Grid Structure**: 10 difficulty levels × 20 variations = 200 unique patches
- **Terrain Types**: Flat (25%), Random rough (25%), Slopes (20%), Steps (30%)
- **Training Approach**: All terrain levels available from start
- **High Resolution**: 10cm horizontal, 5mm vertical resolution

#### 4. Reward System
- **Multi-Component**: 6 distinct reward components for different behaviors
- **Exponential Tracking**: Aggressive rewards for velocity accuracy
- **Energy Efficiency**: Torque penalties encourage smooth motions
- **Gait Quality**: Air time rewards promote proper stepping patterns

#### 5. Sensor Integration
- **Contact Sensors**: 200Hz contact force measurement on all bodies
- **Height Scanner**: Ray-casting for terrain height measurement (3×3 grid)
- **Robot State**: Full proprioceptive feedback (positions, velocities, orientation)
- **Visual Markers**: Real-time velocity vector visualization

### Data Flow Architecture

```
Training Loop (20Hz Policy Frequency):
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Observation │───►│   Policy    │───►│   Action    │───►│ Robot/Sim   │
│ (48D)       │    │  Network    │    │ (12D)       │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       ▲                                                        │
       │                                                        │
       └────────────────────────────────────────────────────────┘
                        Physics Step (360Hz)

Reward Computation (Every Step):
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Robot State │───►│   Reward    │───►│  Training   │
│ Sensors     │    │  Function   │    │  Signal     │
│ Commands    │    │ (6 terms)   │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Key Design Principles

1. **Sim-to-Real Transfer**
   - Conservative joint limits and contact thresholds
   - Physics properties matched to real hardware
   - Minimal proprioceptive sensing (no cameras/lidars)

2. **Training Robustness**
   - All terrain difficulties available from training start
   - Full command range utilized throughout training
   - Consistent reward weighting across all training phases

3. **Robustness Features**
   - Force history for reliable contact detection
   - NaN-safe terrain height computation
   - Pre-allocated tensors to prevent memory leaks
   - Conservative termination conditions

4. **Performance Optimization**
   - Parallel environment training (1024 envs)
   - Efficient tensor operations with GPU acceleration
   - Minimal visualization overhead (20Hz vs 360Hz updates)
   - Cached terrain generation for efficiency

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

## Terrain Generation Algorithm

### Overview
Harold uses a sophisticated procedural terrain generation system optimized for robust locomotion training. The system creates 200 unique terrain patches (10 difficulty levels × 20 variations) with diverse complexity.

Each row contains all terrain types with proportions:
• 25% flat terrain (5 columns)
• 25% random rough terrain (5 columns: 2-3 easy + 2-3 hard)
• 20% slopes (4 columns: 2 pyramid + 2 inverted pyramid)
• 30% steps (6 columns: 2 stairs + 4 inverted stairs)
```

### Training Characteristics

#### Throughout Training
- **Available Levels**: 0-9 (all difficulty levels available from start)
- **Terrain Diversity**: Each level contains ALL terrain types at that difficulty
- **Random Selection**: Robot spawns randomly across all 200 terrain patches
- **Balanced Exposure**: Equal probability of experiencing any terrain type/difficulty combination

#### Benefits of This Approach
- **Robustness**: Policy experiences full diversity from the beginning
- **Balanced Learning**: Equal exposure to all terrain types prevents specialization
- **Simplicity**: No curriculum scheduling needed during training
- **Generalization**: Better transfer due to diverse training distribution

### Terrain Quality Metrics

#### Terrain Validation
- **Difficulty Distribution**: Consistent challenge levels across terrain types
- **Random Selection**: Ensures exposure to all terrain varieties
- **Diversity Maintenance**: Sufficient variation within each difficulty level

This terrain generation system provides the foundation for robust locomotion learning, enabling progressive skill acquisition from basic balance to advanced navigation across diverse challenging environments.

## Reward System

### Overview
Harold's reward function implements a multi-component system to encourage stable, efficient, and robust quadruped locomotion. The reward components balance primary locomotion objectives with stability and efficiency considerations.

### Configuration Location
The reward system configuration is defined in:
- **Reward Weights**: `harold_isaac_lab_env_cfg.py` → `RewardsCfg` class
- **Implementation**: `harold_isaac_lab_env.py` → `_get_rewards()` method
- **Gait Parameters**: `harold_isaac_lab_env_cfg.py` → `GaitCfg` class

### Reward Components
The system uses 6 distinct reward/penalty terms:
1. **Linear velocity tracking** - Primary locomotion objective
2. **Yaw velocity tracking** - Turning control
3. **Height maintenance** - Stability above terrain
4. **Velocity jitter penalty** - Smooth motion
5. **Torque penalty** - Energy efficiency
6. **Feet air time** - Gait quality

Each component has configurable weights and parameters that are frequently tuned during development. Check the `RewardsCfg` class for current values and detailed documentation of each component's mathematical formulation.

### Key Design Principles
- **No Curriculum Scaling**: All reward weights remain constant throughout training
- **Temporal Scaling**: Components scaled by simulation timestep (dt = 1/360s)
- **Exponential Shaping**: Velocity tracking uses aggressive exponential rewards
- **Bounded Functions**: Height rewards use tanh to prevent instability
- **Conditional Activation**: Some rewards only active when robot is moving

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
  