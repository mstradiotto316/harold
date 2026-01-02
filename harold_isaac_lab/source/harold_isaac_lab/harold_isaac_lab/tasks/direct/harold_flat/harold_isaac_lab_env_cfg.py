from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.utils.noise import GaussianNoiseCfg
import math

from .harold import HAROLD_V4_CFG
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.trimesh import MeshPlaneTerrainCfg

# Flat terrain configuration for Harold's locomotion training
HAROLD_FLAT_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.001,
    slope_threshold=0.0,
    use_cache=False,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=1.0, size=(1.0, 1.0))
    },
    curriculum=False,
    color_scheme="height",
)


@configclass
class RewardsCfg:
    """Simplified reward structure following Isaac Lab reference pattern.

    Session 36: Pure RL for velocity-commanded walking.
    ~10 core terms for clean gradient signals, no CPG-specific rewards.

    Reference: isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py
    """

    # === TASK REWARDS (exponential kernels for smooth gradients) ===
    # Session 36d: Increased weights for stronger velocity incentive
    # With std=0.5, standing gives ~0.91 reward, walking gives ~1.0
    # Increasing weight amplifies this difference
    track_lin_vel_xy_weight: float = 5.0      # Primary: velocity tracking (was 1.5)
    track_lin_vel_xy_std: float = 0.25        # Steeper gradient (was 0.5)

    track_ang_vel_z_weight: float = 2.0       # Yaw rate tracking (was 0.75)
    track_ang_vel_z_std: float = 0.25         # Steeper gradient

    # === MOTION QUALITY PENALTIES ===
    # Session 36 fix: -2.0 caused -43800/ep, -0.05 caused -1149/ep
    # Harold produces higher body-frame vertical velocities than larger robots
    # Session 36h: -0.0005 still limits movement, trying -0.0001
    lin_vel_z_weight: float = -0.0001         # Allow more vertical movement
    ang_vel_xy_weight: float = -0.0001        # Also reduced for consistency

    # === SMOOTHNESS PENALTIES ===
    dof_torques_weight: float = -0.0001       # Smooth torques
    dof_acc_weight: float = -2.5e-7           # Smooth joint accelerations
    action_rate_weight: float = -0.01         # Smooth actions (was -0.05, reduced to allow movement)

    # === GAIT REWARDS ===
    # Session 36i: Increased 0.2 → 1.0 to force stepping behavior
    feet_air_time_weight: float = 1.0         # Strongly encourage stepping
    feet_air_time_threshold: float = 0.3      # Target air time (seconds)

    undesired_contacts_weight: float = -1.0   # Penalize body contact
    undesired_contacts_threshold: float = 1.0 # Force threshold (Newtons)

    # === STABILITY REWARD ===
    upright_weight: float = 2.0               # Stay upright (uses projected gravity)

    # === FORWARD MOTION BONUS ===
    # Session 36e: Direct reward for positive vx to bootstrap walking
    # Without this, policy converges to standing still (local minimum)
    # Session 36f-g: Weight sweep: 3.0→+0.01, 5.0→+0.001, 10.0→-0.017
    # 3.0 is optimal, trying longer training (4000 iter) to see if vx improves
    forward_motion_weight: float = 3.0        # Optimal weight (verified by sweep)


@configclass
class CommandCfg:
    """Configuration for velocity command sampling.

    Session 36: Pure RL with conservative velocity ranges.
    Commands are sampled per-episode at reset, from uniform distributions.
    """

    # Enable variable command sampling (True for pure RL)
    variable_commands: bool = True

    # Forward velocity range (m/s)
    # Session 36: Reverted to conservative for stability
    vx_min: float = 0.0
    vx_max: float = 0.3

    # Lateral velocity range (m/s)
    vy_min: float = -0.15
    vy_max: float = 0.15

    # Yaw rate range (rad/s) - about ±17 deg/s
    yaw_min: float = -0.30
    yaw_max: float = 0.30

    # Probability of sampling zero velocity (for stopping behavior)
    zero_velocity_prob: float = 0.02  # 2% standing training

    # Dynamic command updates during episode
    dynamic_commands: bool = True
    command_change_interval: float = 10.0  # seconds
    command_change_prob: float = 1.0


@configclass
class GaitCfg:
    """Gait parameters matched to the rough-terrain setup."""

    frequency: float = 2.0
    target_height: float = 0.275


@configclass
class ScriptedGaitCfg:
    """Configuration for scripted walking gait.

    Session 21 RESULT: SUCCESS - Open-loop scripted gait achieves vx=+0.141 m/s
    (141% of 0.1 m/s target!) with stiffness=1200.

    Session 22 RESULT: SUCCESS - Real robot walks forward with this gait.

    Session 34: BACKLASH-TOLERANT UPDATE - Hardware testing (Session 33) revealed
    ~30° servo backlash. Old amplitude (26°) was absorbed. New amplitude (50°)
    should exceed backlash with margin for actual foot lift.

    Use HAROLD_SCRIPTED_GAIT=1 to enable for physics validation.
    """

    # Master switch - enable via env var HAROLD_SCRIPTED_GAIT=1
    enabled: bool = False

    # Gait parameters - HARDWARE-VALIDATED from Session 36 RPi
    # These exact values produce smooth walking on real robot with feet lifting.
    # Source: firmware/scripted_gait_test_1/scripted_gait_test_1.ino
    #   BASE_STANCE_THIGH = -38.15°, BASE_SWING_THIGH = -30.65° (7.5° range)
    #   BASE_STANCE_CALF = 50°, BASE_SWING_CALF = 80° (30° range)
    frequency: float = 0.5   # Hz - Hardware-validated (slower = more stable)
    swing_thigh: float = 0.54    # -30.65° → +30.65° in sim coords
    stance_thigh: float = 0.67   # -38.15° → +38.15° in sim coords
    stance_calf: float = -0.87   # 50° → -50° in sim coords (extended)
    swing_calf: float = -1.3963  # 80° → -80° in sim coords (flexed)
    shoulder_amplitude: float = 0.0096  # 0.55 deg in hardware
    duty_cycle: float = 0.6


@configclass
class CPGCfg:
    """Central Pattern Generator configuration for residual learning.

    HARDWARE-VALIDATED from Session 36 RPi - these values produce actual
    walking on the real robot with feet lifting off the ground.

    ALIGNMENT APPROACH (Session 36):
    - CPG base trajectory matches hardware scripted gait
    - Low residual_scale keeps policy corrections small

    Architecture:
        target_joints = CPG_trajectory + policy_output * residual_scale

    With residual_scale=0.05, RL contributes small balance corrections
    while preserving the hardware-validated gait structure.

    Enable via: HAROLD_CPG=1
    """

    # Enable CPG-based action space
    enabled: bool = False  # Set True via env var HAROLD_CPG=1

    # Base gait parameters - HARDWARE-VALIDATED
    base_frequency: float = 0.5  # Hz - Hardware-validated (2 second cycle)
    duty_cycle: float = 0.6      # 60% stance, 40% swing

    # Trajectory parameters - HARDWARE-VALIDATED from Session 36 RPi
    # Source: firmware/scripted_gait_test_1/scripted_gait_test_1.ino
    # These produce real walking with feet actually lifting!
    swing_thigh: float = 0.54     # -30.65° in hardware → +30.65° in sim
    stance_thigh: float = 0.67    # -38.15° in hardware → +38.15° in sim
    stance_calf: float = -0.87    # 50° in hardware → -50° in sim (extended)
    swing_calf: float = -1.3963   # 80° in hardware → -80° in sim (flexed)
    shoulder_amplitude: float = 0.0096  # 0.55 deg in hardware

    # Residual scaling - LOW to preserve hardware gait
    # CPG provides timing/coordination, RL adds small balance corrections
    residual_scale: float = 0.05


@configclass
class TerminationCfg:
    """Episode termination thresholds (shared with rough terrain)."""

    base_contact_force_threshold: float = math.inf
    undesired_contact_force_threshold: float = math.inf
    orientation_threshold: float = -0.5
    # Height termination: terminate if base height < threshold
    # EXP-002: 10N contact alone wasn't enough - robot stayed low (height=1.76)
    # EXP-003-007: Height termination has issues - scanner returns bad values
    # EXP-008: Disable height termination, rely on height_reward=30.0 to incentivize
    # Spawn height is ~0.24m, elbow pose is ~0.15-0.18m
    height_threshold: float = 0.0
    # Warmup: skip height termination for first N steps after reset (sensor initialization)
    height_termination_warmup_steps: int = 20
    # Body contact termination: terminate if body/thigh/shoulder contact > threshold (N)
    # EXP-002: 10N kept body contact low (-0.04) but didn't prevent elbow pose
    # EXP-013: Root cause - elbow contact ~5N per point, below 10N threshold = undetected
    # Lowering to 3N should make elbow contact visible to the reward system
    body_contact_threshold: float = 3.0

    # Joint-angle termination: detect elbow pose via front leg joint angles
    # EXP-009: thigh>1.0, calf>-0.8 too loose - robot still found elbow pose (height=1.50)
    # EXP-010: Tighter thresholds still didn't work
    # EXP-011: Disable, use low_height_penalty instead
    elbow_pose_termination: bool = False
    front_thigh_threshold: float = 0.85   # Terminate if front thigh > 0.85 rad
    front_calf_threshold: float = -1.0    # AND front calf > -1.0 rad

@configclass
class BacklashCfg:
    """Explicit backlash hysteresis model for sim-to-real transfer.

    Session 37: Hardware testing (Session 33) revealed ~30° servo backlash on
    direction reversals. Previous approach of adding Gaussian noise (std=0.0175)
    is INCORRECT - backlash is hysteresis, not noise.

    This model tracks "engaged position" where gears are meshed. Commands can
    move within a dead zone without affecting output. Only when command exits
    the zone does output follow (with backlash offset).

    Physical behavior:
    - gap = command - engaged_position
    - if |gap| > half_backlash: output moves toward command
    - else: output stays where it was (motor in dead zone)

    This teaches policy to "overdrive" joints to compensate for dead zone.
    """

    # Enable explicit backlash hysteresis modeling
    # Session 37: Disabled to establish CPG baseline, then curriculum
    enable_backlash: bool = False

    # Backlash magnitude in radians
    # Session 33 hardware test: ~30° backlash observed on direction reversals
    # Session 37: Starting with 15° (0.26 rad) - easier for policy to learn
    # Can increase to 30° after policy learns to compensate for smaller backlash
    backlash_rad: float = 0.26  # 15 degrees

    # Per-joint backlash (optional - use if joints have different backlash)
    # If None, use backlash_rad for all joints
    per_joint_backlash: tuple | None = None

    # Randomize backlash magnitude per episode for robustness
    randomize_backlash: bool = False
    backlash_range: tuple = (0.4, 0.6)  # ±15% variation around 0.52


@configclass
class DomainRandomizationCfg:
    """Domain randomization configuration for sim-to-real transfer.

    To enable domain randomization in training, set:
    ```python
    cfg.domain_randomization.enable_randomization = True
    cfg.domain_randomization.randomize_on_reset = True  # Per-episode randomization
    cfg.domain_randomization.randomize_per_step = True   # Per-step noise
    ```

    Provides controlled variation in simulation parameters to improve policy
    robustness when deployed on physical hardware. Parameters are tuned
    specifically for Harold's lightweight construction and servo capabilities.

    Randomization Categories:
    - Physics: Material properties affecting contact dynamics
    - Robot: Mass, inertia, and actuator characteristics
    - Sensors: Observation noise modeling real sensor imperfections
    - Actions: Control delays and noise
    - External: Environmental disturbances

    All randomization can be toggled on/off for debugging and gradual
    introduction during training curriculum.
    """

    # === MASTER SWITCHES ===
    # EXP-090: FULL domain randomization made training HARDER - vx=0.0056 (MUCH WORSE)
    # Robot learned to stand still to cope with uncertainty
    # Session 28: Re-enabled for SENSOR NOISE ONLY to simulate gear backlash (~2°)
    # Session 37: Replaced noise with explicit hysteresis model (BacklashCfg)
    enable_randomization: bool = True   # Session 28: OPTIMAL for backlash robustness
    randomize_on_reset: bool = False
    randomize_per_step: bool = True     # Session 28: Per-step noise for backlash
    
    # === PHYSICS RANDOMIZATION ===
    # EXP-090: Domain randomization made training WORSE - DISABLED
    randomize_friction: bool = False  # DISABLED - caused vx=0.0056 (robot stood still)
    friction_range: tuple = (0.5, 0.9)        # Conservative range (was 0.4-1.0)
                                              # Base: 0.7, slight variation

    randomize_restitution: bool = False       # Randomize bounce characteristics
    restitution_range: tuple = (0.0, 0.2)     # Keep low for realistic ground contact

    # === ROBOT PROPERTIES RANDOMIZATION ===
    # EXP-090: Domain randomization made training WORSE - DISABLED
    randomize_mass: bool = False              # DISABLED - caused robot to stand still
    mass_range: tuple = (0.9, 1.1)            # ±10% mass variation (conservative)
                                              # Was ±15%, reduced for stability
    
    randomize_com: bool = False               # Randomize center of mass offsets
    com_offset_range: tuple = (-0.02, 0.02)   # ±2cm COM shift in X/Y/Z
                                              # Small shifts for balance variation
    
    randomize_inertia: bool = False           # Randomize rotational inertia
    inertia_range: tuple = (0.9, 1.1)         # ±10% inertia variation
    
    # === ACTUATOR RANDOMIZATION ===
    randomize_joint_stiffness: bool = False   # Vary joint PD controller stiffness
    stiffness_range: tuple = (150, 250)       # Base: 200, allows ±25% variation
                                              # Models servo response differences
    
    randomize_joint_damping: bool = False     # Vary joint PD controller damping
    damping_range: tuple = (50, 100)          # Base: 75, allows ±33% variation
                                              # Models servo damping characteristics
    
    randomize_effort_limit: bool = False      # Vary maximum joint torques
    effort_limit_range: tuple = (0.8, 1.0)    # 80-100% of nominal torque
                                              # Conservative to prevent servo damage
    
    randomize_joint_limits: bool = False      # Add small variations to joint limits
    joint_limit_noise: float = 0.02           # ±0.02 rad (~1.15°) variation
    
    # === SENSOR NOISE CONFIGURATION ===
    # IMU Noise (Body angular velocity and gravity projection)
    add_imu_noise: bool = True                # Add noise to IMU measurements
    imu_angular_velocity_noise: GaussianNoiseCfg = GaussianNoiseCfg(
        mean=0.0,
        std=0.01,                             # 0.01 rad/s noise (realistic for MPU6050)
        operation="add"
    )
    imu_gravity_noise: GaussianNoiseCfg = GaussianNoiseCfg(
        mean=0.0,
        std=0.05,                             # Small noise for gravity vector
        operation="add"
    )

    # Linear velocity noise (simulates IMU accelerometer integration noise)
    # Session 29: Hardware testing revealed lin_vel is computed via accelerometer
    # integration with 0.95 decay, resulting in noisy/drifting values
    add_lin_vel_noise: bool = True
    lin_vel_noise: GaussianNoiseCfg = GaussianNoiseCfg(
        mean=0.0,
        std=0.05,                             # 0.05 m/s noise (hardware shows ~5cm/s drift)
        operation="add"
    )
    # Velocity bias that persists per-episode (simulates calibration error)
    lin_vel_bias_std: float = 0.02            # ±2cm/s per-episode bias

    # Joint Sensor Noise
    # Session 28: Position noise simulates gear backlash (~1-3° in ST3215 servos)
    # - 2° (0.035 rad): STANDING, vx=0.007 - too much noise
    # - 1° (0.0175 rad): WALKING, vx=0.022 - OPTIMAL (31% better than baseline!)
    # Session 37: Re-enabling as explicit hysteresis didn't help
    add_joint_noise: bool = True              # Session 28: OPTIMAL for backlash robustness
    joint_position_noise: GaussianNoiseCfg = GaussianNoiseCfg(
        mean=0.0,
        std=0.0175,                           # Not used when disabled
        operation="add"
    )
    joint_velocity_noise: GaussianNoiseCfg = GaussianNoiseCfg(
        mean=0.0,
        std=0.05,                             # 0.05 rad/s velocity noise
        operation="add"
    )
    
    # === ACTION RANDOMIZATION ===
    # Session 29: Testing action noise for sim-to-real transfer
    # EXP-154: 0.5% (0.005) -> STANDING (vx=0.009), hurt training
    # EXP-155: 0.2% (0.002) -> STANDING (vx=0.009), still hurt training
    # CONCLUSION: Action noise hurts learning, disabled
    add_action_noise: bool = False             # DISABLED - hurts learning
    action_noise: GaussianNoiseCfg = GaussianNoiseCfg(
        mean=0.0,
        std=0.002,                            # Not used when disabled
        operation="add"
    )

    # EXP-156: Action delays hurt training -> DISABLED
    # Adding any action-side randomization hurts when observation noise already present
    add_action_delay: bool = False            # DISABLED - hurts learning
    action_delay_steps: tuple = (0, 1)        # Not used when disabled

    # === EXTERNAL DISTURBANCES ===
    # EXP-164: External perturbations FAILED - caused falling and backward drift
    # Even light forces (0.2-0.5N, 0.5% prob) broke training
    apply_external_forces: bool = False       # DISABLED - causes instability
    external_force_probability: float = 0.005 # 0.5% per step (conservative)
    external_force_range: tuple = (0.2, 0.5)  # Light forces (0.2-0.5 N)
    external_torque_range: tuple = (0.02, 0.1) # Light torques (0.02-0.1 Nm)
    
    # === TERRAIN RANDOMIZATION ===
    add_terrain_noise: bool = False           # Add height noise to terrain
    terrain_noise_magnitude: float = 0.01     # ±1cm height variations
    
    # === GRAVITY RANDOMIZATION ===
    randomize_gravity: bool = False               # Vary gravity magnitude and direction
    gravity_magnitude_range: tuple = (9.6, 10.0)  # 9.6-10.0 m/s² (small variation)
    gravity_angle_range: float = 0.05             # ±0.05 rad (~3°) tilt in gravity vector

@configclass
class HaroldIsaacLabEnvCfg(DirectRLEnvCfg):
    # env parameters
    episode_length_s = 30.0
    decimation = 9
    action_scale = 0.5  # Session 23: 0.7 was worse (vx=0.029, contact failing)

    # Space definitions
    # Session 36: Removed gait phase (no CPG), pure RL = 48D
    # Session 37: Restored for CPG mode = 50D (adds 2D gait phase)
    # Toggle based on HAROLD_CPG env var
    observation_space = 48  # Pure RL mode (no gait phase); use 50 for CPG
    action_space = 12
    state_space = 0

    # Action filtering (EMA low-pass)
    # Session 35: beta=0.40 is optimal (0.50 prevented walking)
    # Lower beta = more smoothing (60% carryover from previous action)
    action_filter_beta: float = 0.40

    # Observation clipping (matches deployment clip_obs=5.0)
    # Session 29: Hardware deployment clips normalized obs to ±5.0
    # Training without clipping causes policy to see larger ranges than deployment
    clip_observations: bool = True
    clip_observations_value: float = 5.0      # Match deployment clipping

    # Reward configuration
    rewards = RewardsCfg()

    # Gait configuration
    gait = GaitCfg()

    # Scripted gait configuration (Phase 1 - FAILED, kept for reference)
    scripted_gait = ScriptedGaitCfg()

    # CPG configuration (Phase 2 - structured action space)
    cpg = CPGCfg()

    # Command configuration (Phase 2 - variable velocity commands)
    commands = CommandCfg()

    # Termination configuration
    termination = TerminationCfg()

    # Domain randomization configuration
    domain_randomization = DomainRandomizationCfg()

    # Backlash hysteresis configuration (Session 37)
    backlash = BacklashCfg()

    # viewer configuration - follow single robot for clear screenshots
    viewer = ViewerCfg(
        eye=(1.0, 0.75, 0.45),    # Close-up view, elevated
        lookat=(0.0, 0.0, 0.12),  # Look at robot center
        origin_type="asset_root", # Follow robot's root position
        asset_name="robot",       # Matches prim_path ".../Robot"
        env_index=0,              # Track environment 0
    )

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt= 1 / 180, #dt= 1 / 360,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.7,
            dynamic_friction=0.7,
            restitution=0.0,
        ),
    )

    
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=HAROLD_FLAT_TERRAIN_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.15, 0.2, 0.25),  # Dark blue-gray for contrast with white robot
        ),
        debug_vis=False,
    )

    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        update_period=0.05,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.25, 0.25)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        drift_range=(0.0, 0.0),
    )

    

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=2.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = HAROLD_V4_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    harold_dof_names = [
        "fl_shoulder_joint", 
        "fr_shoulder_joint", 
        "bl_shoulder_joint", 
        "br_shoulder_joint",

        "fl_thigh_joint", 
        "fr_thigh_joint", 
        "bl_thigh_joint", 
        "br_thigh_joint",
        
        "fl_calf_joint", 
        "fr_calf_joint", 
        "bl_calf_joint", 
        "br_calf_joint"
    ]

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,            # 5ms update rate (much higher frequency than 0.05s)
        track_air_time=True             # Enabled for gait-based rewards (EXP-038)
    )

    # === Joint configuration (moved from env implementation) ===
    # Per-joint normalized action ranges (scaled later by action_scale)
    # Order: [shoulders(4), thighs(4), calves(4)]
    joint_range: tuple = (
        0.30, 0.30, 0.30, 0.30,
        0.90, 0.90, 0.90, 0.90,
        0.90, 0.90, 0.90, 0.90,
    )

    # === Joint configuration for push-up routine ===
    # Absolute joint angle limits in radians
    # Joint limits (Session 30: aligned with hardware safe limits)
    # Sign convention: thighs/calves are inverted between sim and hardware
    # Hardware limits from deployment/config/hardware.yaml
    # thigh hardware: [-55°, +5°] → sim: [-5°, +55°] = [-0.0873, +0.9599] rad
    # calf hardware: [-5°, +80°] → sim: [-80°, +5°] = [-1.3963, +0.0873] rad
    joint_angle_max: tuple = (
        0.4363, 0.4363, 0.4363, 0.4363,  # shoulders: ±25° (hardware safe limit)
        0.9599, 0.9599, 0.9599, 0.9599,  # thighs: sim +55° (hw -55°)
        0.0873, 0.0873, 0.0873, 0.0873   # calves: sim +5° (hw -5°)
    )
    joint_angle_min: tuple = (
        -0.4363, -0.4363, -0.4363, -0.4363,  # shoulders: ±25°
        -0.0873, -0.0873, -0.0873, -0.0873,  # thighs: sim -5° (hw +5°)
        -1.3963, -1.3963, -1.3963, -1.3963   # calves: sim -80° (hw +80°)
    )
