from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.envs.common import ViewerCfg

from .harold import HAROLD_V4_CFG
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.height_field import HfRandomUniformTerrainCfg, HfPyramidSlopedTerrainCfg, HfInvertedPyramidSlopedTerrainCfg
from isaaclab.terrains.trimesh import MeshPlaneTerrainCfg, MeshRandomGridTerrainCfg, MeshPyramidStairsTerrainCfg, MeshInvertedPyramidStairsTerrainCfg


# Custom terrain configuration for Harold - balanced mix of upward and downward terrain
HAROLD_GENTLE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,  # 10cm resolution
    vertical_scale=0.005,  # 5mm height resolution - much finer than default
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # Flat terrain for basic walking (25%)
        "flat": MeshPlaneTerrainCfg(proportion=0.25, size=(1.0, 1.0)),
        
        # Random rough terrain (25%)
        "easy_random": HfRandomUniformTerrainCfg(
            proportion=0.125,  # Split random terrain into two parts
            noise_range=(0.01, 0.04),  # Easy range: 0.5cm to 3cm noise
            noise_step=0.01,  
            border_width=0.25,
        ),
        "hard_random": HfRandomUniformTerrainCfg(
            proportion=0.125,  # Split random terrain into two parts  
            noise_range=(0.01, 0.08),  # Hard range: 8cm to 16cm noise
            noise_step=0.01,
            border_width=0.25,
        ),
        
        # Regular pyramid slopes - robot spawns on peak, goes downhill (15%)
        "tiny_slopes": HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.1, 0.3),  # % grade 0.1
            platform_width=1.0,
            border_width=0.25,
        ),
        
        # Inverted pyramid slopes - robot spawns in valley, must climb uphill (15%)
        "tiny_valleys": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.1, 0.3),  # % grade 0.1
            platform_width=1.0,
            border_width=0.25,
        ),
        
        # Regular pyramid stairs - robot spawns on top platform, goes down steps (10%)
        "micro_steps": MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.01, 0.1),  # In meters
            step_width=0.3,
            platform_width=1.0,
            border_width=0.25,
        ),
        
        # Inverted pyramid stairs - robot spawns in pit, must climb up steps (10%)
        "micro_pits": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.01, 0.1),  # In meters
            step_width=0.3,
            platform_width=1.0,
            border_width=0.25,
        )
    },
    curriculum=True,
    color_scheme="height",
)


@configclass
class RewardsCfg:
    """Reward function weights and scaling parameters.
    
    Carefully tuned reward weights for stable quadruped locomotion training.
    Each component addresses specific aspects of desired robot behavior:
    
    - Velocity tracking: Primary objective for locomotion control
    - Energy efficiency: Encourages smooth, low-power movements  
    - Gait quality: Promotes proper stepping patterns
    - Stability: Maintains upright posture and consistent height
        
    Weight Magnitudes:
    - track_xy_lin_commands: 600 (highest priority - locomotion objective)
    - feet_air_time: 300 (high priority - proper gait patterns)
    - velocity_jitter: -30 (medium penalty - smooth motion)
    - track_yaw_commands: 20 (medium priority - turning ability)
    - height_reward: 15 (low-medium priority - stability)
    - torque_penalty: -3 (low penalty - energy efficiency)
    """
    # === PRIMARY LOCOMOTION OBJECTIVES (Positive Rewards) ===
    track_xy_lin_commands: float = 300 #600   # Linear velocity tracking weight (HIGHEST PRIORITY)
                                        # Aggressive exponential reward: exp(-error²/0.0005)
                                        # Only high accuracy gets meaningful reward
                                        
    track_yaw_commands: float = 20      # Yaw velocity tracking weight (MEDIUM PRIORITY)  
                                       # Exponential reward: exp(-error²/0.05)
                                       # Enables turning and orientation control
                                       
    height_reward: float = 15           # Height maintenance reward (STABILITY)
                                       # Tanh-based: tanh(3*exp(-5*|height_error|))
                                       # Maintains ~18cm target height above terrain
                                       # Critical for stable locomotion
                                       
    feet_air_time: float = 800 #1500 #800 #1500 #3000   # Proper gait reward (HIGH PRIORITY)
                                       # Rewards 0.15s optimal air time per foot (fixed for Harold's scale)
                                       # Uses exponential reward curve to encourage stepping
                                       # Only active when moving (|v_cmd| > 0.03 m/s)
    
    # === SECONDARY OBJECTIVES AND PENALTIES (Negative Rewards) ===
    velocity_jitter: float = -30        # Smooth motion penalty (MEDIUM PENALTY)
                                       # Penalizes rapid velocity direction changes
                                       # Computes angle between consecutive velocity vectors
                                       # Scaled by commanded speed for proportional penalty
                                       
    torque_penalty: float = -1.5      # Energy efficiency penalty (LOW PENALTY)
                                       # Quadratic penalty: sum(torque²)
                                       # Encourages smooth, low-power movements


@configclass
class GaitCfg:
    """Gait and locomotion parameters for Harold quadruped.
    
    These parameters define the desired locomotion characteristics and are
    tuned specifically for Harold's physical dimensions and mass properties.
    
    Values are scaled appropriately for a small (40cm length, 2kg) quadruped
    compared to larger research platforms like ANYmal (70cm, 50kg) or Spot (110cm, 32kg).
    
    Key Scaling Relationships:
    - Gait frequency ∝ 1/√(leg_length) - smaller robots step faster
    - Target height ∝ leg_length - proportional to robot size
    """
    frequency: float = 2.0       # Hz - Desired gait frequency for proper stepping
                                # Harold: 2.0Hz (smaller robots step faster)
                                # ANYmal: 1.5Hz, Spot: 1.2Hz (larger robots slower)
                                # Scaling relationship: f ∝ 1/√(leg_length)
                                # Used in feet_air_time reward for optimal 0.15s air time
                                
    target_height: float = 0.18  # m - Desired body height above terrain surface
                                # Harold: 18cm (natural standing height)
                                # ANYmal: 40cm, Spot: 35cm (proportional to leg length)
                                # Critical for height_reward component calculation
                                # Ray-casting scanner measures actual height vs target


@configclass
class TerminationCfg:
    """Episode termination conditions and safety thresholds.
    
    Defines conditions that trigger environment resets to prevent damage
    to the physical robot and ensure training focuses on successful behaviors.
    
    Thresholds are scaled for Harold's lightweight (2kg) construction and
    are intentionally conservative to prevent hardware damage during 
    sim-to-real transfer.
    
    Force Scaling:
    - Contact forces scale with robot mass (2kg vs 30kg+ for larger robots)
    - Lower thresholds ensure safer operation for lightweight construction
    - Conservative limits prevent servo damage from over-torque conditions
    """
    # === CONTACT FORCE THRESHOLDS (Scaled for Harold's 2kg Mass) ===
    base_contact_force_threshold: float = 0.5       # Main body contact limit [N]
                                                   # Harold: 0.5N (2kg robot)
                                                   # ANYmal: ~5N (50kg robot) 
                                                   # Force scaling: F ∝ robot_mass
                                                   # Currently UNUSED - see undesired_contact instead
                                                   
    undesired_contact_force_threshold: float = 0.05 # Limb contact termination limit [N]
                                                    # Extremely sensitive threshold
                                                    # Applies to: body, shoulders, thighs
                                                    # Only feet (calves) should contact ground
                                                    # Conservative for hardware protection
                                                    # Triggers immediate environment reset
                                                    
    # === ORIENTATION TERMINATION ===
    orientation_threshold: float = -0.5             # Robot tilt termination limit
                                                   # projected_gravity_b[2] threshold
                                                   # -1.0 = perfectly upright
                                                   #  0.0 = completely horizontal  
                                                   # -0.5 = ~60° tilt before termination
                                                   # TODO: Currently disabled - may be too permissive
                                                   # Consider lowering to -0.7 for stricter control

@configclass
class HaroldIsaacLabEnvCfg(DirectRLEnvCfg):
    # env parameters
    episode_length_s = 30.0
    decimation = 18
    action_scale = 1.0
    
    # Space definitions
    observation_space = 48
    action_space = 12
    state_space = 0

    # Reward configuration
    rewards = RewardsCfg()
    
    # Gait configuration
    gait = GaitCfg()
    
    # Termination configuration
    termination = TerminationCfg()

    # viewer configuration
    viewer = ViewerCfg(
        eye     = (-20.0, -20.0, 2.0),   # camera XYZ in metres
        lookat = (0.0, 0.0, 0.0),  # aim at robot base
    )

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt= 1 / 360,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.7, #1.0,
            dynamic_friction=0.7, #1.0,
            restitution=0.0,
        ),
    )

    
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=HAROLD_GENTLE_TERRAINS_CFG,  # Use our custom gentle terrain
        max_init_terrain_level=9,  # Enable all terrain levels (0-9)
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
        history_length=3,               # Increased from 1 to 3 for proper contact force filtering
        update_period=0.005,            # 5ms update rate (much higher frequency than 0.05s)
        track_air_time=True             # Keep enabled for gait analysis and feet air time rewards
    )
