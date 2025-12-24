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
    """Reward structure for Phase-0: prioritize standing, then walking.

    EXP-036: Height-based termination to prevent elbow exploit.
    EXP-034/035 showed height_reward=25.0 alone was insufficient.
    Now terminating episodes when height < 0.165m (60% of target).
    """

    # Forward motion rewards - EXP-054: Test gait phase observation
    # Session 13 summary:
    # - EXP-050/051: Higher forward reward → SANITY_FAIL
    # - EXP-052/053: Slip factor experiments → no improvement
    # - EXP-054: Add gait phase (sin/cos) to observation space
    # Hypothesis: Phase signal helps policy coordinate leg movements
    progress_forward_pos: float = 40.0   # Keep optimal from EXP-032
    progress_forward_neg: float = 10.0   # Keep optimal from EXP-032
    standing_penalty: float = -5.0      # Best from EXP-021

    # Stability rewards - EXP-032 optimal values
    # EXP-049 showed reducing these causes instability (SANITY_FAIL)
    upright_reward: float = 10.0        # Keep upright incentive
    height_reward: float = 15.0         # Reverted from EXP-049's 8.0

    # Penalties
    torque_penalty: float = -0.005      # Gentle energy regularizer
    lat_vel_penalty: float = 12.0       # Penalize sideways skating
    yaw_rate_penalty: float = 1.0       # Dampen gratuitous spinning
    rear_support_bonus: float = 0.0     # Disabled - was encouraging standing

    height_tolerance: float = 0.02      # |height_error| tolerated before penalty (m)
    height_sigma: float = 0.045         # Controls falloff beyond tolerance (m)

    # EXP-011: Strong negative penalty for low height (instead of termination)
    # Punish being below minimum height to create gradient toward standing
    low_height_penalty: float = -50.0   # Strong negative reward
    low_height_threshold: float = 0.20  # Penalize if height < 0.20m (elbow pose ~0.15-0.18m)

    # EXP-044: Air time rewards DISABLED - they made vx worse in EXP-038-043
    # EXP-038: vx=-0.029, EXP-042: vx=-0.025, EXP-043: vx=-0.004 (all worse than baseline 0.029)
    # Hypothesis: Air time reward interferes with velocity reward gradient
    feet_air_time_reward: float = 0.0   # DISABLED - counterproductive
    optimal_air_time: float = 0.25      # Unused when weight=0

    # EXP-055/056/058: Contact-based diagonal gait reward with forward gating
    # Rewards alternating diagonal pair contacts (FL+BR vs FR+BL) when moving forward
    # EXP-055 (ungated): vx=+0.022 (worse - no direction bias)
    # EXP-056 (gated, weight=5.0): vx=+0.036 (BEST final, 24% better than baseline)
    # EXP-058 (gated, weight=10.0): peak vx=0.061 at 43%, regressed to 0.018
    diagonal_gait_reward: float = 10.0  # EXP-059: Test early stopping with higher weight
    gait_phase_tolerance: float = 0.3   # Phase window where contact is rewarded (0-1 scale)


@configclass
class GaitCfg:
    """Gait parameters matched to the rough-terrain setup."""

    frequency: float = 2.0
    target_height: float = 0.275


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
    enable_randomization: bool = False
    randomize_on_reset: bool = False
    randomize_per_step: bool = False
    
    # === PHYSICS RANDOMIZATION ===
    randomize_friction: bool = False  # Keep friction stable in Phase-0
    friction_range: tuple = (0.4, 1.0)        # Range for static/dynamic friction (match rough task)
                                              # Base: 0.7, Range allows slippery to grippy surfaces
    
    randomize_restitution: bool = False       # Randomize bounce characteristics
    restitution_range: tuple = (0.0, 0.2)     # Keep low for realistic ground contact
    
    # === ROBOT PROPERTIES RANDOMIZATION ===
    randomize_mass: bool = False              # Randomize body and link masses
    mass_range: tuple = (0.85, 1.15)          # ±15% mass variation (1.7-2.3kg total)
                                              # Conservative to prevent drastic dynamics changes
    
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
    
    # Joint Sensor Noise
    add_joint_noise: bool = True              # Add noise to joint measurements
    joint_position_noise: GaussianNoiseCfg = GaussianNoiseCfg(
        mean=0.0,
        std=0.005,                            # 0.005 rad (~0.3°) position noise
        operation="add"
    )
    joint_velocity_noise: GaussianNoiseCfg = GaussianNoiseCfg(
        mean=0.0,
        std=0.05,                             # 0.05 rad/s velocity noise
        operation="add"
    )
    
    # === ACTION RANDOMIZATION ===
    add_action_noise: bool = False             # Add noise to action commands
    action_noise: GaussianNoiseCfg = GaussianNoiseCfg(
        mean=0.0,
        std=0.01,                             # Small noise proportional to action scale
        operation="add"
    )
    
    add_action_delay: bool = False            # Simulate control delays
    action_delay_steps: tuple = (0, 2)        # 0-2 timestep random delay
                                              # Models USB/servo communication latency
    
    # === EXTERNAL DISTURBANCES ===
    apply_external_forces: bool = False       # Random pushes to robot body
    external_force_probability: float = 0.02  # 2% chance per step
    external_force_range: tuple = (0.5, 2.0)  # 0.5-2.0 N random forces
    external_torque_range: tuple = (0.1, 0.5) # 0.1-0.5 Nm random torques
    
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
    action_scale = 0.5

    # Space definitions
    # EXP-054: Added 2D gait phase (sin/cos) to help policy coordinate leg movements
    observation_space = 50  # Was 48, now includes gait phase
    action_space = 12
    state_space = 0

    # Action filtering (EMA low-pass)
    action_filter_beta: float = 0.18  # smoother actions to reduce jitter

    # Reward configuration
    rewards = RewardsCfg()
    
    # Gait configuration
    gait = GaitCfg()
    
    # Termination configuration
    termination = TerminationCfg()
    
    # Domain randomization configuration
    domain_randomization = DomainRandomizationCfg()

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
    joint_angle_max: tuple = (
        0.5236, 0.5236, 0.5236, 0.5236,  # shoulders ±30°
        1.5708, 1.5708, 1.5708, 1.5708,  # thighs ±90°
        1.5708, 1.5708, 1.5708, 1.5708   # calves ±90°
    )
    joint_angle_min: tuple = (
        -0.5236, -0.5236, -0.5236, -0.5236,
        -1.5708, -1.5708, -1.5708, -1.5708,
        -1.5708, -1.5708, -1.5708, -1.5708
    )
