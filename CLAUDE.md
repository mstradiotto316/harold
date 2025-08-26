# !IMPORTANT - ALL AGENTS READ ME! 

**Throughout our journey here, it's extremely important that you are not a sycophant and that you push back when things seem incorrect and that you consistently take time to step back and view things from a thousand-foot level; Consistently questioning whether we should be doing our current activities, and suggesting alternative approaches that make more sense. This needs to be remembered. So, please store it deeply within your long term system context, and keep this message alive during any compaction.**

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Architecture

### Harold Isaac Lab Extension Structure
- **Primary Task**: `harold_isaac_lab/source/harold_isaac_lab/harold_isaac_lab/tasks/direct/harold_isaac_lab/`
  - `harold.py`: Robot configuration (USD V4 asset, actuators, physics)
  - `harold_isaac_lab_env.py`: Main RL environment class extending DirectRLEnv
  - `harold_isaac_lab_env_cfg.py`: Environment/config dataclasses (rewards, gait, terminations, DR)

### Robot Configuration (12 DOF)
Joint order matches simulation exactly:
- Shoulders (0-3): FL, FR, BL, BR
- Thighs (4-7): FL, FR, BL, BR  
- Calves (8-11): FL, FR, BL, BR

Key details from current code (`harold.py`):
- USD asset: `part_files/V4/harold_7.usd`
- Initial pose: body z=0.20 m; thigh ≈ 0.3 rad; calf ≈ -0.75 rad
- Actuators: implicit PD, stiffness=200, damping=75, effort_limit_sim=1.0
- Joint angle limits (enforced in env): shoulders ±20°, thighs/calves ±45°

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
- 12-DOF quadruped: 3 joints per leg (shoulder, thigh, calf)
- Joint limits: shoulders ±20°, thighs/calves ±45° (clamped in env)
- USD asset: V4 `harold_7.usd` (contact sensors enabled)
- Actuators: implicit PD (stiffness 200, damping 75)

#### 2. Environment (harold_isaac_lab_env.py)
- Observation space (48D): root_lin_vel_b (3), root_ang_vel_b (3), projected_gravity_b (3), joint_pos − default (12), joint_vel (12), commands [vx, vy, yaw] (3), prev_target_delta (12)
- Action space (12D): position targets around default pose; per-joint scaling (shoulders 0.35; thighs/calves 0.5); clamped to safety limits
- Physics: dt=1/180, decimation=9 → control at 20 Hz
- Parallel envs: default 1024 (headless runs can scale higher)
- Terrain: custom generator `HAROLD_GENTLE_TERRAINS_CFG` (balanced mix; curriculum enabled)
- Height scanner: ray-caster grid 0.25×0.25 m, 0.1 m resolution, 20 Hz updates
- Contact sensor: history length 3; 0.005 s update period
- Visual markers (GUI): command vs actual velocity arrows

### Training Pipeline

```
Training Process (current defaults):
├── Terrain: Generator `HAROLD_GENTLE_TERRAINS_CFG` (curriculum enabled)
│   └── Importer sets max_init_terrain_level=2 (levels 0–1 sampled at reset)
├── Commands: XY speed sampled in [0.1, 0.3] m/s; yaw = 0.0 rad/s (for now)
├── Environments: 1024 parallel envs (scale up headless as needed)
├── Control Rate: 20 Hz (dt 1/180, decimation 9)
└── Focus: Robust flat-to-gentle terrain locomotion with directional tracking

Notes:
- Terrain level selection is randomized within allowed levels on each reset.
- Yaw tracking reward exists but yaw commands are currently pinned to 0.
```

This architecture enables robust quadruped locomotion learning with progressive skill acquisition, efficient parallel training, and safe real-world deployment.

### SKRL PPO Config (current defaults)

- Policy: Gaussian, ELU MLP [512, 256, 128]
- Value: Deterministic, ELU MLP [512, 256, 128]
- Rollouts: 24, Learning epochs: 5, Mini-batches: 4
- Learning rate: 1e-3 (KLAdaptiveLR, kl_threshold=0.01)
- Preprocessors: RunningStandardScaler for state and value
- Entropy coef: 0.01, Value loss scale: 1.0, Grad clip: 1.0
- Ratio clip: 0.2, Value clip: 0.2, Rewards shaper scale: 0.6
- Trainer: Sequential, timesteps: 128000, experiment dir: `harold_direct`

## Reward System

### Overview
Harold's reward function implements a multi-component system to encourage stable, efficient, and robust quadruped locomotion. The reward components balance primary locomotion objectives with stability and efficiency considerations.

### Configuration Location
Defined in code as:
- Reward weights: `harold_isaac_lab_env_cfg.py` → `RewardsCfg`
- Implementation: `harold_isaac_lab_env.py` → `_get_rewards()`
- Gait parameters: `harold_isaac_lab_env_cfg.py` → `GaitCfg`

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
   - Rewards optimal air time (~0.4 s) to encourage proper stepping
   - Gated active when actual speed > 0.05 m/s

Note: Velocity jitter penalty was removed as it was redundant with directional tracking and penalized natural gait patterns.

Current reward weights (`RewardsCfg`):
- track_xy_lin_commands: 80.0
- track_yaw_commands: 2.0
- height_reward: 0.75
- torque_penalty: -0.08
- feet_air_time: 12.0

Each component has configurable weights and parameters; see `RewardsCfg` and `_get_rewards()` for exact formulations.

### Reward Normalization
Rewards are NOT multiplied by `step_dt` in the reward assembly to avoid double normalization. The episodic sum is already normalized by `max_episode_length_s` when logged to tensorboard, providing proper per-second normalization for visualization while maintaining correct step rewards for RL training.

## Termination and Safety

- Immediate reset on any undesired contact (body/shoulders/thighs) with threshold ≈ 0.05 N
- Feet (calves) are the only intended ground-contact bodies for locomotion
- Orientation termination exists in config (`orientation_threshold=-0.5`) but is currently disabled in env logic

## Domain Randomization

Enabled by default (on-reset and per-step where applicable):
- Physics/materials: friction randomized (≈ 0.4–1.0); restitution off
- Robot/actuator: stiffness/damping/mass/inertia hooks present; disabled by default
- Sensors: IMU (ang vel, gravity) and joint (pos/vel) noise enabled
- Actions: noise and delay available; disabled by default
- External disturbances: available; disabled by default
- Gravity randomization: available; disabled by default

See `DomainRandomizationCfg` and env helpers for details.

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
  
