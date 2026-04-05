# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based Harold flat locomotion environment.

Adapts the Spot flat locomotion reward structure (manager-based) to Harold's
body/joint naming and physical scale. Uses the proven ManagerBasedRLEnv
architecture that produces walking for Spot.
"""

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

# Reuse Spot's reward and event functions (they work for any quadruped)
import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

# Harold-specific MDP components
from .mdp.actions import EMAJointPositionActionCfg
from .mdp.observations import zero_lin_vel

# Harold robot config
from .harold import HAROLD_V4_CFG


# Terrain: flat with some gentle roughness
HAROLD_FLAT_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.8),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.01, 0.02), noise_step=0.01, border_width=0.25
        ),
    },
)


@configclass
class HaroldActionsCfg:
    """Action specifications for the MDP.

    Uses EMA-filtered joint position action to match the deployment pipeline,
    where actions are smoothed before sending to hardware servos. This closes
    the train/deploy gap that causes "shuffling" behavior in sim.
    """

    joint_pos = EMAJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.2, use_default_offset=True,
        ema_beta=0.2,
    )


@configclass
class HaroldCommandsCfg:
    """Command specifications — scaled for Harold (2kg, ~0.15m legs)."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class HaroldObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group.

        Noise magnitudes match hardware sensors (MPU6050 IMU, ST3215 encoders)
        for sim-to-real robustness. enable_corruption=True applies noise per step.
        """

        # Velocity-blind: zeros instead of true velocity.
        # Hardware IMU (MPU6050) dead-reckons lin_vel via accelerometer integration
        # with 0.95 decay — produces noisy, drifting, physically unrealistic signal.
        # Training without velocity feedback is standard for low-cost quadrupeds.
        # The velocity tracking REWARD still uses true physics velocity.
        base_lin_vel = ObsTerm(
            func=zero_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")},
            # Increased from ±0.05 to ±0.08 to cover ST3215 midpoint drift (~5°=0.087 rad)
            noise=Unoise(n_min=-0.08, n_max=0.08),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class HaroldEventCfg:
    """Domain randomization — scaled for Harold's 2kg body."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # Wider range for sim-to-real: carpet (high) vs hardwood (low)
            "static_friction_range": (0.2, 1.2),
            "dynamic_friction_range": (0.2, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*body"),
            # URDF body=0.655 kg, total=1.68 kg. Real robot=2.0 kg.
            # Missing ~0.32 kg: servo internals, electronics, battery, wiring.
            # Range (0.12, 0.52) centers total at ~2.0 kg (range 1.80-2.20 kg).
            "mass_distribution_params": (0.12, 0.52),
            "operation": "add",
        },
    )

    # interval — random external forces (lateral pushes, roll torques)
    # Hardware testing showed lateral falls; train with sustained lateral forces.
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(6.0, 12.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*body"),
            "force_range": (-2.0, 2.0),   # ~1g lateral force on 2kg robot
            "torque_range": (-0.3, 0.3),  # Small roll/pitch torques
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.3, 0.3),
                "z": (-0.15, 0.15),
                "roll": (-0.4, 0.4),
                "pitch": (-0.4, 0.4),
                "yaw": (-0.7, 0.7),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=spot_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-2.5, 2.5),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # interval — velocity perturbations (stronger lateral, more frequent)
    # Hardware testing: robot fell sideways. Stronger Y perturbations force lateral recovery.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.6, 0.6)},
        },
    )


@configclass
class HaroldRewardsCfg:
    """Spot reward structure adapted for Harold.

    Uses the same reward functions but with Harold-scaled thresholds.
    """

    # -- task rewards
    air_time = RewardTermCfg(
        func=spot_mdp.air_time_reward,
        weight=14,
        params={
            "mode_time": 0.25,
            "velocity_threshold": 0.3,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*calf"),
        },
    )
    base_angular_velocity = RewardTermCfg(
        func=spot_mdp.base_angular_velocity_reward,
        weight=7,
        params={"std": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_linear_velocity = RewardTermCfg(
        func=spot_mdp.base_linear_velocity_reward,
        weight=14,
        params={"std": 1.0, "ramp_rate": 0.5, "ramp_at_vel": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    foot_clearance = RewardTermCfg(
        func=spot_mdp.foot_clearance_reward,
        weight=3.0,
        params={
            "std": 0.02,
            "tanh_mult": 2.0,
            "target_height": 0.07,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*calf"),
        },
    )
    gait = RewardTermCfg(
        func=spot_mdp.GaitReward,
        weight=20,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.3,
            "synced_feet_pair_names": (("fl_calf", "br_calf"), ("fr_calf", "bl_calf")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    # -- penalties
    action_smoothness = RewardTermCfg(func=spot_mdp.action_smoothness_penalty, weight=-1.0)
    air_time_variance = RewardTermCfg(
        func=spot_mdp.air_time_variance_penalty,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*calf")},
    )
    base_motion = RewardTermCfg(
        func=spot_mdp.base_motion_penalty, weight=-1.5, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    base_orientation = RewardTermCfg(
        func=spot_mdp.base_orientation_penalty, weight=-5.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    foot_slip = RewardTermCfg(
        func=spot_mdp.foot_slip_penalty,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*calf"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*calf"),
            "threshold": 1.0,
        },
    )
    joint_acc = RewardTermCfg(
        func=spot_mdp.joint_acceleration_penalty,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_pos = RewardTermCfg(
        func=spot_mdp.joint_position_penalty,
        weight=-0.4,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.15,
        },
    )
    joint_torques = RewardTermCfg(
        func=spot_mdp.joint_torques_penalty,
        weight=-0.002,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_vel = RewardTermCfg(
        func=spot_mdp.joint_velocity_penalty,
        weight=-1.0e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )


@configclass
class HaroldTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*body", ".*thigh", ".*shoulder"]),
            "threshold": 1.0,
        },
    )
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


@configclass
class HaroldCurriculumCfg:
    """No terrain curriculum for flat environment."""

    pass


@configclass
class HaroldFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Manager-based Harold flat locomotion environment.

    Inherits from LocomotionVelocityRoughEnvCfg (same as Spot) and overrides
    with Harold-specific settings.
    """

    observations: HaroldObservationsCfg = HaroldObservationsCfg()
    actions: HaroldActionsCfg = HaroldActionsCfg()
    commands: HaroldCommandsCfg = HaroldCommandsCfg()
    rewards: HaroldRewardsCfg = HaroldRewardsCfg()
    terminations: HaroldTerminationsCfg = HaroldTerminationsCfg()
    events: HaroldEventCfg = HaroldEventCfg()
    curriculum: HaroldCurriculumCfg = HaroldCurriculumCfg()

    viewer = ViewerCfg(eye=(1.5, 1.0, 0.6), lookat=(0.0, 0.0, 0.15), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        super().__post_init__()

        # Timing — 500Hz physics, 20Hz control (matches deployment CONTROL_RATE_HZ=20)
        self.decimation = 25
        self.episode_length_s = 20.0
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"

        # Contact sensor update
        self.scene.contact_forces.update_period = self.sim.dt

        # Swap robot to Harold
        self.scene.robot = HAROLD_V4_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Terrain — simple flat plane (minimize GPU memory)
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            debug_vis=False,
        )

        # No height scan
        self.scene.height_scanner = None
