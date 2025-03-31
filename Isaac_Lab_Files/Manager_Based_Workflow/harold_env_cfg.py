# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, patterns
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

import omni.isaac.lab_tasks.manager_based.harold.harold_flat_terrain.mdp as mdp

##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with the harold robot."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )

    # robot
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/matteo/Desktop/Harold_V5/harold_v5_11.usd",
            activate_contact_sensors=True,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=2,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
            )
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.20),
            rot=(1.0, 0.0, 0.0, 0.0), 
            joint_pos={
                'fl_shoulder_joint': 0.0,
                'fr_shoulder_joint': 0.0,
                'bl_shoulder_joint': 0.0,
                'br_shoulder_joint': 0.0,
                
                'fl_thigh_joint': 0.3,
                'fr_thigh_joint': 0.3,
                'bl_thigh_joint': 0.3,
                'br_thigh_joint': 0.3,

                'fl_knee_joint': -0.75,
                'fr_knee_joint': -0.75,
                'bl_knee_joint': -0.75,
                'br_knee_joint': -0.75,
            }
        ),
        actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=0.8, #0.92 is the quoted real world value
            stiffness=200.0,
            damping=100.0,
        ),
    },
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
    )

    body_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*body", history_length=3, update_period=0.005, track_air_time=False
    )

    shoulder_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*shoulder", history_length=3, update_period=0.005, track_air_time=False
    )

    thigh_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*thigh", history_length=3, update_period=0.005, track_air_time=False
    )

    foot_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*knee", history_length=3, update_period=0.005, track_air_time=True
    )

    # Height Scanner
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.25, 0.25)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        drift_range=(0.0, 0.0),
    )



##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    xy_commands = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        heading_command=True,  # Use heading-based control
        heading_control_stiffness=1.0,  # Default stiffness for heading control
        rel_standing_envs=0.1,  # 10% chance of standing still
        rel_heading_envs=1.0,  # All moving envs use heading-based control
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.25, 0.25),    # Forward/backward velocity in m/s
            lin_vel_y=(0.0, 0.0),    # Lateral velocity in m/s
            ang_vel_z=(0.0, 0.0),    # Yaw angular velocity in rad/s
            heading=(0, 0),    # Full range of heading angles in radians
        ),
        resampling_time_range=(4.0, 8.0),  # Change commands every 4-8 seconds
        debug_vis=True  # Enable visualization for debugging
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_positions = mdp.JointPositionActionCfg(
        use_default_offset=False,
        asset_name="robot",
        joint_names=[
            "fl_shoulder_joint", 
            "fr_shoulder_joint", 
            "bl_shoulder_joint", 
            "br_shoulder_joint",

            "fl_thigh_joint", 
            "fr_thigh_joint", 
            "bl_thigh_joint", 
            "br_thigh_joint",
            
            "fl_knee_joint", 
            "fr_knee_joint", 
            "bl_knee_joint", 
            "br_knee_joint"
        ],
        preserve_order=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        linear_velocity = ObsTerm(func=mdp.base_lin_vel)
        angular_velocity = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_positions = ObsTerm(func=mdp.joint_pos)
        joint_velocities = ObsTerm(func=mdp.joint_vel)
        commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "xy_commands"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
        

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    height_reward = RewTerm(
        func=mdp.scanner_height_l2, 
        weight=-1.0, 
        params={
            "target_height": 0.20,
            "scanner_cfg": SceneEntityCfg("height_scanner")
        }
    )

    lin_vel_tracking = RewTerm(
        func=mdp.track_lin_vel_xy_exp, 
        weight=2.0,
        params={
            "std": 0.25,  # Standard deviation for exponential reward
            "command_name": "xy_commands",
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

    joint_torque_reward = RewTerm(
        func=mdp.joint_torques_l2,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot")
        }
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Terminate if the episode length is exceeded (truncation)
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,  # This marks it as a truncation
    )

    # Terminate if we detect illegal contacts (true terminations)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"threshold": 0.2, "sensor_cfg": SceneEntityCfg("body_contact_sensor")}
    )
    shoulder_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"threshold": 0.2, "sensor_cfg": SceneEntityCfg("shoulder_contact_sensor")}
    )
    thigh_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"threshold": 0.2, "sensor_cfg": SceneEntityCfg("thigh_contact_sensor")}
    )
    

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class HaroldEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Harold robot walking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=1024, env_spacing=2.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 18
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 1 / 360.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0

