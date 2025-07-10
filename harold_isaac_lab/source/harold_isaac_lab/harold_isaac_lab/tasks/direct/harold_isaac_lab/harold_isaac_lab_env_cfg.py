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
        
        # Curriculum-aware random rough terrain (25%) - now properly scales with curriculum
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
            slope_range=(0.1, 0.4),  # % grade 0.1
            platform_width=1.0,
            border_width=0.25,
        ),
        
        # Inverted pyramid slopes - robot spawns in valley, must climb uphill (15%)
        "tiny_valleys": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.1, 0.4),  # % grade 0.1
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
    """Reward function weights and parameters."""
    # Reward weights
    track_xy_lin_commands: float = 200
    track_yaw_commands: float = 10 #50 (Yaw Experiment 1)
    velocity_jitter: float = -15  # Reduced from -35 to be less harsh on rough terrain
    height_reward: float = 15     # Reduced from 25 to prevent dominance over movement
    torque_penalty: float = -1.0  # Reduced from -2.0 to allow higher torques needed on rough terrain


@configclass
class GaitCfg:
    """Gait parameters."""
    frequency: float = 1.5  # Hz
    target_height: float = 0.40 #0.20  # TODO: I think something is wrong here, the height is offset by some weird amount


@configclass
class TerminationCfg:
    """Termination conditions."""
    # Contact thresholds
    contact_force_threshold: float = 0.1
    # Orientation threshold (robot tilted more than 60 degrees)
    orientation_threshold: float = -0.5 # TODO: This is not resulting in any terminations maybe it is too forgiving.


@configclass
class CurriculumCfg:
    """Curriculum learning parameters for transitioning from standing to walking."""
    # Number of global training steps over which to ramp from standing to full exploration
    phase_transition_steps: int = 64000 #16000


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

    # Curriculum configuration
    curriculum: CurriculumCfg = CurriculumCfg()

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
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=HAROLD_GENTLE_TERRAINS_CFG,  # Use our custom gentle terrain
        max_init_terrain_level=9,  # Enable all terrain levels (0-9) from the start for curriculum learning
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
        history_length=1,
        update_period=18 * (1 / 360),   # 0.05 s, one update per policy step
        track_air_time=True
    )
