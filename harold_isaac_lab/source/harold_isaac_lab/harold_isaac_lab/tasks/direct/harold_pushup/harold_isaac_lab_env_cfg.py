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

from .harold import HAROLD_V4_CFG
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.trimesh import MeshPlaneTerrainCfg

# Simple flat terrain for push-up playback (no curriculum or roughness)
PUSHUP_FLAT_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(4.0, 4.0),
    border_width=5.0,
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
class HaroldIsaacLabEnvCfg(DirectRLEnvCfg):
    # env parameters
    episode_length_s = 30.0
    # Control at 20 Hz (dt=1/180, decimation=9)
    decimation = 9
    action_scale = 1.0
    
    # Space definitions
    observation_space = 45
    action_space = 12
    state_space = 0

    # viewer configuration
    viewer = ViewerCfg(
        eye=(0.0, 1.0, 0.4),     # close-up view
        lookat=(0.0, 0.0, 0.3),  # center on robot body
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
        terrain_generator=PUSHUP_FLAT_TERRAIN_CFG,  # flat plane only
        max_init_terrain_level=1,
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

    # scene: single environment for playback comparison
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=True)

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
        update_period=0.005,
        track_air_time=True
    )
