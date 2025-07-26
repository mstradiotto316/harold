# !IMPORTANT - ALL AGENTS READ ME! 

**Throughout our journey here, it's extremely important that you are not a sycophant and that you push back when things seem incorrect and that you consistently take time to step back and view things from a thousand-foot level; Consistently questioning whether we should be doing our current activities, and suggesting alternative approaches that make more sense. This needs to be remembered. So, please store it deeply within your long term system context, and keep this message alive during any compaction. Now! Begin by consuming the README file, then begin your search.**

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Development Commands

### Environment Setup
```bash
# Activate Isaac Lab environment (required for all operations)
source ~/Desktop/env_isaaclab/bin/activate

# Install Harold extension
python -m pip install -e harold_isaac_lab/source/harold_isaac_lab
```

### Training and Simulation
```bash
# List available environments
python harold_isaac_lab/scripts/list_envs.py

# Train basic Harold on flat terrain
python harold_isaac_lab/scripts/skrl/train.py --task=Template-Harold-Direct-flat-terrain-v0

# Train headless with video recording
python harold_isaac_lab/scripts/skrl/train.py --task=Template-Harold-Direct-flat-terrain-v0 --num_envs 1024 --headless --video --video_length 250 --video_interval 3200

# Run trained model
python harold_isaac_lab/scripts/skrl/play.py --task=Template-Harold-Direct-flat-terrain-v0 --checkpoint=PATH_TO_CHECKPOINT

# Test with dummy agents
python harold_isaac_lab/scripts/zero_agent.py --task=Template-Harold-Direct-flat-terrain-v0
python harold_isaac_lab/scripts/random_agent.py --task=Template-Harold-Direct-flat-terrain-v0
```

### Monitoring and Analysis
```bash
# Launch Tensorboard for training progress
python3 -m tensorboard.main --logdir logs/

# Code formatting
pre-commit run --all-files
```

## Core Architecture

### Harold Isaac Lab Extension Structure
- **Primary Task**: `harold_isaac_lab/source/harold_isaac_lab/harold_isaac_lab/tasks/direct/harold_isaac_lab/`
  - `harold.py`: Robot configuration (joints, actuators, physics)
  - `harold_isaac_lab_env.py`: Main RL environment class extending DirectRLEnv
  - `harold_isaac_lab_env_cfg.py`: Configuration management with dataclasses

### Key Design Patterns
- **Dataclass Configuration**: All parameters managed through `@configclass` decorators
- **Curriculum Learning**: Progressive terrain difficulty scaling with α parameter
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

### Terrain Curriculum System
- **Flat Terrain**: 25% (difficulty levels 0-1)
- **Random Rough**: 25% (progressive complexity)
- **Pyramid Slopes**: 25% (climbing challenges)
- **Micro Steps**: 25% (precision locomotion)

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

This codebase implements a sophisticated quadruped locomotion system with robust curriculum learning, comprehensive sensor integration, and flexible multi-framework RL training capabilities for both simulation and real-world deployment.

## System Architecture Overview

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                        HAROLD SYSTEM ARCHITECTURE                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │   RL Training   │    │   Simulation    │    │  Hardware   │ │
│  │                 │    │                 │    │             │ │
│  │ • Policy Net    │◄──►│ • Isaac Lab     │◄──►│ • ESP32     │ │
│  │ • Reward Func   │    │ • Physics Sim   │    │ • Servos    │ │
│  │ • Curriculum    │    │ • Terrain Gen   │    │ • IMU       │ │
│  └─────────────────┘    └─────────────────┘    └─────────────┘ │
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
- **Terrain Types**: Flat (25%), Random rough (25%), Slopes (25%), Steps (25%)
- **Curriculum Integration**: Progressive difficulty scaling with training progress
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

2. **Curriculum Learning**
   - Progressive terrain difficulty (α: 0→1 over 128k steps)
   - Command magnitude scaling during early training
   - Reward component weighting based on curriculum progress

3. **Robustness Features**
   - Force history for reliable contact detection
   - NaN-safe terrain height computation
   - Pre-allocated tensors to prevent memory leaks
   - Conservative termination conditions

4. **Performance Optimization**
   - Parallel environment training (1024 envs)
   - Efficient tensor operations with GPU acceleration
   - Minimal visualization overhead (20Hz vs 360Hz updates)
   - Cached terrain generation with curriculum support

### Training Pipeline

```
Phase 1: Basic Stability (α = 0.0-0.3, 0-38k steps)
├── Terrain: Mostly flat with minimal obstacles
├── Commands: Small velocity targets (±0.1 m/s)
├── Focus: Standing balance and basic stepping
└── Duration: ~30-60 minutes

Phase 2: Locomotion Skills (α = 0.3-0.7, 38k-90k steps)  
├── Terrain: Mixed difficulty with slopes and rough patches
├── Commands: Medium velocity targets (±0.2 m/s)
├── Focus: Forward/backward walking, basic turning
└── Duration: ~1-2 hours

Phase 3: Advanced Behaviors (α = 0.7-1.0, 90k-128k steps)
├── Terrain: Full complexity including stairs and valleys
├── Commands: Full velocity range (±0.3 m/s, ±0.2 rad/s)
├── Focus: Complex terrain navigation, rapid direction changes
└── Duration: ~1 hour

Total Training Time: ~3-4 hours on modern GPU
```

This architecture enables robust quadruped locomotion learning with progressive skill acquisition, efficient parallel training, and safe real-world deployment.

## Terrain Generation Algorithm

### Overview
Harold uses a sophisticated procedural terrain generation system optimized for curriculum learning and locomotion training. The system creates 200 unique terrain patches (10 difficulty levels × 20 variations) with progressive complexity scaling.

### Terrain Grid Structure
```
Terrain Grid Layout (10 rows × 20 columns):
┌─────────────────────────────────────────────────────────────────┐
│ Level 0: Flat terrain + minimal noise (easiest)                │
├─────────────────────────────────────────────────────────────────┤
│ Level 1: Small random variations                               │
├─────────────────────────────────────────────────────────────────┤
│ Level 2: Gentle slopes + low noise                             │
├─────────────────────────────────────────────────────────────────┤
│ Level 3: Medium slopes + moderate noise                        │
├─────────────────────────────────────────────────────────────────┤
│ Level 4: Steeper terrain + higher noise                        │
├─────────────────────────────────────────────────────────────────┤
│ Level 5: Complex slopes + obstacles                            │
├─────────────────────────────────────────────────────────────────┤
│ Level 6: Mixed terrain types                                   │
├─────────────────────────────────────────────────────────────────┤
│ Level 7: Challenging slopes + steps                            │
├─────────────────────────────────────────────────────────────────┤
│ Level 8: Advanced obstacle courses                             │
├─────────────────────────────────────────────────────────────────┤
│ Level 9: Maximum difficulty (hardest)                          │
└─────────────────────────────────────────────────────────────────┘
  ◄────────────── 20 variations per level ──────────────►
```

### Terrain Type Distribution
The terrain generation system creates four distinct terrain categories:

#### 1. Flat Terrain (25% of patches)
- **Purpose**: Basic locomotion learning and curriculum initialization
- **Configuration**: Perfectly flat meshes with no height variation
- **Use Cases**: Standing balance, basic stepping patterns, policy initialization

#### 2. Random Rough Terrain (25% of patches)
- **Easy Random** (12.5%): Height noise 1-4cm, gentle irregularities
- **Hard Random** (12.5%): Height noise 1-8cm, challenging obstacles
- **Algorithm**: Uniform random height field generation with Perlin noise
- **Progression**: Noise magnitude scales with difficulty level

#### 3. Sloped Terrain (25% of patches)
- **Pyramid Slopes** (12.5%): Robot starts on peak, navigates downhill
- **Inverted Pyramids** (12.5%): Robot starts in valley, climbs uphill
- **Slope Range**: 10-40% grade (increasing with difficulty)
- **Platform Size**: 1m flat area at start position

#### 4. Stepped Terrain (25% of patches)
- **Pyramid Stairs** (12.5%): Discrete steps downward from platform
- **Inverted Pyramid Stairs** (12.5%): Climbing steps upward from pit
- **Step Heights**: 1-10cm (proportional to Harold's leg capability)
- **Step Width**: 30cm (comfortable for 40cm robot)

### Technical Specifications

#### Resolution Parameters
```yaml
Grid Configuration:
  size: [8.0m, 8.0m]              # Each terrain patch dimensions
  border_width: 20.0m             # Flat border around entire grid
  horizontal_scale: 0.1m          # 10cm horizontal resolution
  vertical_scale: 0.005m          # 5mm vertical resolution
  slope_threshold: 0.75           # Maximum traversable slope
```

#### Curriculum Integration
The terrain system integrates seamlessly with the curriculum learning pipeline:

```python
# Curriculum-based terrain selection during environment reset
terrain_level = int(α * max_terrain_level)  # α ∈ [0,1] curriculum progress
terrain_variation = random.randint(0, 19)   # Random variation within level
spawn_position = terrain_origins[terrain_level][terrain_variation]
```

### Terrain Generation Pipeline

#### Phase 1: Grid Initialization
1. **Memory Allocation**: Pre-allocate 10×20 terrain patch grid
2. **Border Creation**: Generate 20m flat border around entire grid
3. **Coordinate System**: Establish world coordinates for each patch

#### Phase 2: Procedural Generation
```
For each difficulty level (0-9):
  For each variation (0-19):
    1. Select terrain type based on proportion weights
    2. Generate base heightfield/mesh geometry
    3. Apply difficulty-appropriate parameters
    4. Add border transitions for seamless boundaries
    5. Compute collision meshes and visual materials
    6. Store in terrain_origins array for curriculum use
```

#### Phase 3: Physics Integration
1. **Collision Meshes**: Generate accurate collision geometry
2. **Material Properties**: Apply friction/restitution parameters
3. **Ray-casting Setup**: Configure height scanner compatibility
4. **Visual Rendering**: Apply height-based color schemes

### Curriculum Progression Examples

#### Early Training (α = 0.0-0.3)
- **Available Levels**: 0-2 (flat terrain, minimal noise)
- **Robot Behavior**: Learning basic balance and stepping
- **Terrain Features**: Smooth surfaces, gentle variations
- **Success Metrics**: Standing stability, simple forward motion

#### Mid Training (α = 0.3-0.7)
- **Available Levels**: 0-6 (introducing slopes and obstacles)
- **Robot Behavior**: Developing gait patterns, terrain adaptation
- **Terrain Features**: Moderate slopes, small steps, varied textures
- **Success Metrics**: Consistent locomotion, direction control

#### Advanced Training (α = 0.7-1.0)
- **Available Levels**: 0-9 (full complexity available)
- **Robot Behavior**: Mastering complex navigation, rapid adaptation
- **Terrain Features**: Steep slopes, large steps, combined obstacles
- **Success Metrics**: Robust locomotion, quick recovery, precise control

### Performance Optimizations

#### Caching System
- **Pre-generation**: All terrain patches generated once during initialization
- **Memory Efficiency**: Reuse terrain meshes across multiple training runs
- **Load Balancing**: Distribute generation across available CPU cores

#### GPU Acceleration
- **Batch Processing**: Generate multiple terrain patches simultaneously
- **Mesh Operations**: Use GPU-accelerated geometry processing
- **Memory Management**: Efficient GPU memory allocation for large terrains

### Terrain Quality Metrics

#### Traversability Analysis
- **Slope Validation**: Ensure all slopes within robot capability
- **Connectivity**: Verify paths exist between spawn and goal regions
- **Safety Margins**: Conservative limits prevent impossible situations

#### Curriculum Validation
- **Difficulty Monotonicity**: Higher levels consistently more challenging
- **Smooth Transitions**: Gradual difficulty increase prevents training collapse
- **Diversity Maintenance**: Sufficient variation within each difficulty level

This terrain generation system provides the foundation for robust locomotion learning, enabling progressive skill acquisition from basic balance to advanced navigation across diverse challenging environments.

## Reward System Mathematical Formulation

### Overview
Harold's reward function implements a sophisticated multi-component system designed to encourage stable, efficient, and robust quadruped locomotion. The system balances primary locomotion objectives with secondary stability and efficiency considerations.

### Total Reward Computation
```mathematical
R_total = Σ(w_i · r_i · s_i · dt)

Where:
- w_i = weight for component i (from configuration)
- r_i = normalized reward component i ∈ [0,1] or penalty ∈ [-∞,0]
- s_i = scaling factor (curriculum α or constant 1.0)
- dt = simulation timestep (1/360 seconds ≈ 0.00278)
```

### Component-by-Component Mathematical Analysis

#### 1. Linear Velocity Tracking (Primary Objective)
**Weight**: w₁ = 600 (highest priority)  
**Scaling**: α curriculum scaling  

```mathematical
Mathematical Formulation:
e_xy = Σ(v_cmd,xy - v_actual,xy)²     # Squared error in X,Y velocity
r₁ = exp(-e_xy² / σ₁²)               # Double exponential for aggressive shaping
R₁ = w₁ · r₁ · α · dt                # Final reward contribution

Parameters:
σ₁² = 0.0005                         # Very small normalization (aggressive)
v_cmd,xy ∈ [-0.3, 0.3] m/s          # Command velocity range
v_actual,xy = robot.root_lin_vel_b    # Measured body-frame velocity

Reward Characteristics:
- Exponential decay with squared error creates steep penalty curve
- Only very accurate tracking (error < 0.02 m/s) receives significant reward
- Curriculum scaling α reduces importance during early training
- Dominant reward component when tracking is accurate
```

#### 2. Yaw Velocity Tracking (Turning Control)
**Weight**: w₂ = 20 (medium priority)  
**Scaling**: α curriculum scaling  

```mathematical
Mathematical Formulation:
e_yaw = (ω_cmd,z - ω_actual,z)²       # Squared error in yaw rate
r₂ = exp(-e_yaw / σ₂²)               # Single exponential (less aggressive)
R₂ = w₂ · r₂ · α · dt                # Final reward contribution

Parameters:
σ₂² = 0.05                           # Moderate normalization
ω_cmd,z ∈ [-0.2, 0.2] rad/s         # Yaw rate command range
ω_actual,z = robot.root_ang_vel_b[2]  # Measured yaw rate

Reward Characteristics:
- Less aggressive than linear velocity (single exponential)
- Enables precise turning and orientation control
- Scaled by curriculum to focus on forward motion first
```

#### 3. Height Maintenance (Stability)
**Weight**: w₃ = 15 (low-medium priority)  
**Scaling**: No curriculum scaling (always active)

```mathematical
Mathematical Formulation:
h_actual = mean(scanner.pos_z - terrain_height)  # Ray-casting height measurement
e_height = |h_actual - h_target|                # Absolute height error
r₃ = tanh(3 · exp(-5 · e_height))              # Smooth bounded reward
R₃ = w₃ · r₃ · dt                              # Final reward contribution

Parameters:
h_target = 0.18 m                    # Target body height above terrain
h_actual ∈ [0.10, 0.30] m           # Typical height range during locomotion

Reward Characteristics:
- Smooth, bounded reward function prevents instability
- Maintains consistent ground clearance for obstacle avoidance
- Critical for preventing dragging behavior on rough terrain
- NaN-safe computation handles missing terrain data
```

#### 4. Velocity Jitter Penalty (Smooth Motion)
**Weight**: w₄ = -30 (medium penalty)  
**Scaling**: No curriculum scaling

```mathematical
Mathematical Formulation:
v_prev, v_curr = previous/current horizontal velocities
cos_θ = (v_prev · v_curr) / (|v_prev| · |v_curr|)  # Cosine similarity
θ = arccos(clamp(cos_θ, -1, 1))                   # Angle between velocities
jitter = θ · |v_cmd|                               # Scale by command magnitude
r₄ = -jitter                                       # Direct penalty
R₄ = w₄ · r₄ · dt                                 # Final penalty contribution

Filtering:
- Only computed when |v_prev| > ε and |v_curr| > ε (ε = 1e-3)
- Prevents numerical issues at low velocities
- Proportional to commanded speed (more penalty at higher speeds)

Penalty Characteristics:
- Encourages smooth, consistent motion patterns
- Prevents rapid direction changes and oscillatory behavior
- Stronger penalty when robot is commanded to move faster
```

#### 5. Torque Penalty (Energy Efficiency)
**Weight**: w₅ = -3 (low penalty)  
**Scaling**: α curriculum scaling (reduced during early training)

```mathematical
Mathematical Formulation:
τ = robot.applied_torque              # 12D joint torque vector
r₅ = -Σ(τᵢ²)                        # Quadratic penalty on all joints
R₅ = w₅ · r₅ · α · dt               # Final penalty contribution

Penalty Characteristics:
- Quadratic penalty encourages smooth, low-torque movements
- Curriculum scaling allows higher torques during learning phase
- Promotes energy-efficient gaits and smooth control
- Prevents aggressive actuator usage that could damage hardware
```

#### 6. Feet Air Time Reward (Gait Quality)
**Weight**: w₆ = 300 (high priority)  
**Scaling**: No curriculum scaling  

```mathematical
Mathematical Formulation:
For each foot i ∈ {FL, FR, BL, BR}:
  air_time_i = time since last ground contact
  first_contact_i = binary flag (foot just contacted ground)
  reward_i = (air_time_i - t_optimal) · first_contact_i

r₆ = Σ(reward_i) · active_gate        # Sum over all feet
R₆ = w₆ · r₆ · dt                     # Final reward contribution

Parameters:
t_optimal = 0.3 seconds               # Optimal air time for Harold's size
active_gate = |v_cmd| > 0.03 m/s     # Only reward when moving

Gait Quality Metrics:
- Promotes proper stepping patterns with appropriate swing phase duration
- Based on Anymal-C research scaled for smaller robot dynamics
- Prevents shuffling and sliding behaviors
- Only active during locomotion (not during standing)
```

### Reward Scaling and Normalization

#### Curriculum Integration
```mathematical
α(t) = min(policy_step / transition_steps, 1.0)

Where:
- policy_step = current training step
- transition_steps = 128,000 (configuration parameter)
- α ∈ [0, 1] represents curriculum progress

Components with curriculum scaling:
- Linear velocity tracking (primary objective)
- Yaw velocity tracking (turning control)  
- Torque penalty (energy efficiency)

Components without curriculum scaling:
- Height maintenance (always critical for stability)
- Velocity jitter penalty (always important for smoothness)
- Feet air time reward (gait quality independent of difficulty)
```

#### Temporal Scaling
All reward components are scaled by the simulation timestep dt = 1/360 ≈ 0.00278 seconds to ensure consistent magnitudes regardless of physics frequency.

### Expected Reward Magnitudes

#### During Optimal Performance
```
Component                  | Value Range      | Contribution
---------------------------|------------------|------------------
Linear Velocity Tracking  | 0.0 - 1.67      | Dominant (80%+)
Yaw Velocity Tracking     | 0.0 - 0.056     | Moderate (5-10%)
Height Maintenance         | 0.0 - 0.042     | Low (2-5%)
Velocity Jitter Penalty   | -0.25 - 0.0     | Variable penalty
Torque Penalty           | -0.5 - 0.0      | Small penalty
Feet Air Time Reward      | -2.5 - +2.5     | Moderate (10-15%)
```

#### Training Phase Analysis
- **Early Training (α < 0.3)**: Dominated by height maintenance and gait rewards
- **Mid Training (α ≈ 0.5)**: Balanced between velocity tracking and stability
- **Late Training (α → 1.0)**: Primarily velocity tracking with efficiency penalties

### Design Rationale

#### Hierarchical Importance
1. **Locomotion Objectives** (weights 600, 20): Core task performance
2. **Gait Quality** (weight 300): Proper stepping patterns for robustness  
3. **Stability** (weight 15): Maintain upright posture and height
4. **Efficiency** (weights -30, -3): Encourage smooth, low-energy movement

#### Mathematical Properties
- **Differentiability**: All components are smooth and differentiable for gradient-based learning
- **Boundedness**: Exponential and tanh functions prevent reward explosion
- **Scalability**: Curriculum and temporal scaling ensure consistent training dynamics
- **Robustness**: NaN-safe computations and filtering prevent numerical issues

This mathematically principled reward system enables stable, efficient quadruped locomotion learning while maintaining robustness to various terrain conditions and ensuring safe sim-to-real transfer.

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
  