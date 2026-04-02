# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based Harold rough terrain locomotion environment.

Inherits everything from HaroldFlatEnvCfg (rewards, observations, actions,
domain randomization, sim-to-real stack) and replaces the flat plane with
a terrain generator for indoor robustness training.

Terrain mix is designed for Harold's real-world use case: indoor floors
with seams, carpet edges, cable bumps, door thresholds, and slight slopes.
No stairs or extreme terrain — Harold is 2kg with hobby servos.

No height scanner — Harold does "blind" locomotion, adapting reactively
through proprioception. This is more robust for sim-to-real than relying
on terrain sensing the robot doesn't have.
"""

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from harold_isaac_lab.tasks.manager_based.harold_flat.flat_env_cfg import HaroldFlatEnvCfg


# Indoor-realistic terrain for a 2kg quadruped
HAROLD_ROUGH_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        # 60% flat — normal indoor floor (EXP-843: gentler mix)
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.6),
        # 25% gentle rough (1-2cm) — carpet texture, floor irregularities
        "gentle_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25,
            noise_range=(0.01, 0.02),
            noise_step=0.005,
            border_width=0.25,
        ),
        # 10% moderate rough (2-4cm) — cable bumps, carpet edges, tile seams
        "moderate_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10,
            noise_range=(0.02, 0.04),
            noise_step=0.005,
            border_width=0.25,
        ),
        # 5% gentle slopes (5-10 degrees) — door thresholds, slight ramps
        "gentle_slopes": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.09, 0.18),  # ~5-10 degrees (tan)
            platform_width=1.5,
            border_width=0.25,
        ),
    },
)


@configclass
class HaroldRoughEnvCfg(HaroldFlatEnvCfg):
    """Harold rough terrain environment.

    Inherits ALL settings from HaroldFlatEnvCfg:
    - 20 Hz control (matches hardware)
    - Velocity-blind observation (MPU6050 unusable)
    - EMA action filtering (beta=0.2)
    - Servo velocity limit (4.29 rad/s)
    - Corrected body mass (+0.32 kg)
    - All reward weights and domain randomization

    Only changes: terrain (flat plane → generated terrain with curriculum).
    """

    def __post_init__(self):
        super().__post_init__()

        # Replace flat plane with generated rough terrain
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=HAROLD_ROUGH_TERRAIN_CFG,
            max_init_terrain_level=None,  # Sample all difficulties from start
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            debug_vis=False,
        )
