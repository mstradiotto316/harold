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
    """Minimal reward structure for Phase-0 straight walking."""

    progress_forward: float = 60.0      # Pay only for forward progress
    upright_reward: float = 8.0         # Keep gravity vector aligned with body Z
    height_reward: float = 3.0          # Maintain nominal body height
    torque_penalty: float = -0.005      # Gentle energy regularizer
    lat_vel_penalty: float = 80.0       # Penalize sideways skating
    yaw_rate_penalty: float = 2.0       # Dampen gratuitous spinning

    height_tolerance: float = 0.02      # |height_error| tolerated before penalty (m)
    height_sigma: float = 0.05          # Controls falloff beyond tolerance (m)


@configclass
class GaitCfg:
    """Gait parameters matched to the rough-terrain setup."""

    frequency: float = 2.0
    target_height: float = 0.24


@configclass
class TerminationCfg:
    """Episode termination thresholds (shared with rough terrain)."""

    base_contact_force_threshold: float = math.inf
    undesired_contact_force_threshold: float = math.inf
    orientation_threshold: float = -0.5

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
    observation_space = 48
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

    # viewer configuration (match original flat task view)
    viewer = ViewerCfg(
        eye=(0.0, 10.0, 2.0),
        lookat=(0.0, 0.0, 0.0),
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
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
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
        track_air_time=False            # Disabled for minimal reward setup
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
