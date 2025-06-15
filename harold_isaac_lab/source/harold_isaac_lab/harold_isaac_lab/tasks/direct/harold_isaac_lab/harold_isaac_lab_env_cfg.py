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


@configclass
class RewardsCfg:
    """Reward function weights and parameters."""
    # Reward weights
    track_xy_lin_commands: float = 200 #150.0 #100.0 #80.0
    #track_yaw_commands: float = 0.0  # -1.0
    velocity_jitter: float = -35 #-25 #-20.0 #-80.0
    height_reward: float = 25
    # Torque penalty (negative value): small penalty on joint effort
    torque_penalty: float = 0.0 #-0.003 # (EXPERIMENT 2) #-0.001 (EXPERIMENT 1)


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
    phase_transition_steps: int = 16000 #64000


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
        eye     = (1.0, 1.0, 0.75),   # camera XYZ in metres
        lookat = (0.0, 10.0, 0.20),  # aim at robot base
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
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
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
    
    

    
    """
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5, #9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    """
    

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
