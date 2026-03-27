# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for sim_flat_v2 — Harold robot with Spot reward structure.

Identical to sim_flat_v1 except the robot is Harold instead of Spot.
All rewards, commands, terrain, domain randomization, and training params are unchanged.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from .harold import HAROLD_V4_CFG

# ---------------------------------------------------------------------------
# Terrain (identical to sim_flat_v1: Spot reference cobblestone road)
# ---------------------------------------------------------------------------
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
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
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.05), noise_step=0.02, border_width=0.25
        ),
    },
)


# ---------------------------------------------------------------------------
# Config sub-classes (identical to sim_flat_v1)
# ---------------------------------------------------------------------------
@configclass
class RewardsCfg:
    """Spot flat locomotion reward weights — exact copy from reference."""

    # Task rewards
    air_time_weight: float = 5.0
    air_time_mode_time: float = 0.3
    air_time_velocity_threshold: float = 0.15

    base_linear_velocity_weight: float = 5.0
    base_linear_velocity_std: float = 1.0
    base_linear_velocity_ramp_rate: float = 0.5
    base_linear_velocity_ramp_at_vel: float = 1.0

    base_angular_velocity_weight: float = 5.0
    base_angular_velocity_std: float = 2.0

    foot_clearance_weight: float = 0.5
    foot_clearance_target_height: float = 0.1
    foot_clearance_std: float = 0.05
    foot_clearance_tanh_mult: float = 2.0

    gait_weight: float = 10.0
    gait_std: float = 0.1
    gait_max_err: float = 0.2
    gait_velocity_threshold: float = 0.15

    # Penalties (negative weights)
    action_smoothness_weight: float = -1.0

    air_time_variance_weight: float = -1.0

    base_motion_weight: float = -2.0

    base_orientation_weight: float = -3.0

    foot_slip_weight: float = -0.5
    foot_slip_threshold: float = 1.0

    joint_acceleration_weight: float = -1.0e-4  # shoulder+thigh joints only

    joint_position_weight: float = -0.7
    joint_position_stand_still_scale: float = 5.0
    joint_position_velocity_threshold: float = 0.15

    joint_torques_weight: float = -5.0e-4

    joint_velocity_weight: float = -1.0e-2  # shoulder+thigh joints only


@configclass
class CommandsCfg:
    """Velocity command sampling — Spot values (testing thresholds independently)."""

    vx_min: float = -2.0
    vx_max: float = 3.0
    vy_min: float = -1.5
    vy_max: float = 1.5
    yaw_min: float = -2.0
    yaw_max: float = 2.0
    resample_time: float = 10.0  # seconds between resampling
    standing_probability: float = 0.1  # fraction of envs with zero command


@configclass
class TerminationCfg:
    """Termination conditions — exact Spot values."""

    body_contact_threshold: float = 1.0  # N, on body + leg bodies
    terrain_out_of_bounds_distance: float = 3.0  # meters from terrain border


@configclass
class DomainRandomizationCfg:
    """Domain randomization — scaled for Harold (~2 kg, Spot was ~30 kg)."""

    # Startup (applied once)
    static_friction_range: tuple[float, float] = (0.3, 1.0)   # scale-independent, keep
    dynamic_friction_range: tuple[float, float] = (0.3, 0.8)  # scale-independent, keep
    base_mass_range: tuple[float, float] = (-0.2, 0.2)        # ~10% of 2 kg body (was ±2.5 for 30 kg Spot)

    # Reset (applied each episode)
    pose_x_range: tuple[float, float] = (-0.5, 0.5)           # keep (position is scale-independent)
    pose_y_range: tuple[float, float] = (-0.5, 0.5)           # keep
    pose_yaw_range: tuple[float, float] = (-3.14, 3.14)       # keep
    vel_x_range: tuple[float, float] = (-0.3, 0.3)            # ~5x reduction (was ±1.5)
    vel_y_range: tuple[float, float] = (-0.2, 0.2)            # ~5x reduction (was ±1.0)
    vel_z_range: tuple[float, float] = (-0.1, 0.1)            # ~5x reduction (was ±0.5)
    vel_roll_range: tuple[float, float] = (-0.3, 0.3)         # ~2x reduction (was ±0.7)
    vel_pitch_range: tuple[float, float] = (-0.3, 0.3)        # ~2x reduction (was ±0.7)
    vel_yaw_range: tuple[float, float] = (-0.4, 0.4)          # ~2x reduction (was ±1.0)
    joint_pos_range: tuple[float, float] = (-0.2, 0.2)        # keep (joint-level, not mass-dependent)
    joint_vel_range: tuple[float, float] = (-2.5, 2.5)        # keep (joint-level)

    # Interval (applied periodically)
    push_interval_range: tuple[float, float] = (10.0, 15.0)   # keep timing
    push_vel_range: tuple[float, float] = (-0.1, 0.1)         # ~5x reduction (was ±0.5)


# ---------------------------------------------------------------------------
# Main environment configuration
# ---------------------------------------------------------------------------
@configclass
class SimFlatV2EnvCfg(DirectRLEnvCfg):
    """Harold robot with Spot reward structure — direct env."""

    # --- Spaces ---
    observation_space = 48  # lin_vel(3) + ang_vel(3) + gravity(3) + cmd(3) + jpos(12) + jvel(12) + action(12)
    action_space = 12
    state_space = 0

    # --- Timing ---
    decimation = 10         # 50 Hz control
    episode_length_s = 20.0
    action_scale = 0.2      # Same as Spot reference

    # --- Sub-configs ---
    rewards: RewardsCfg = RewardsCfg()
    commands: CommandsCfg = CommandsCfg()
    termination: TerminationCfg = TerminationCfg()
    domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg()

    # --- Viewer (camera following the robot) ---
    viewer = ViewerCfg(
        eye=(1.2, 0.9, 0.5),      # Pulled back for full-body visibility
        lookat=(0.0, 0.0, 0.10),
        origin_type="asset_root",
        asset_name="robot",
        env_index=0,
    )

    # --- Simulation ---
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.002,           # 500 Hz physics
        render_interval=10,  # match decimation
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # --- Scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=True,
    )

    # --- Robot (HAROLD instead of Spot) ---
    robot = HAROLD_V4_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # --- Terrain ---
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=COBBLESTONE_ROAD_CFG,
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # --- Contact sensor ---
    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.002,  # match sim dt
        track_air_time=True,
    )

    # --- Lighting ---
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
