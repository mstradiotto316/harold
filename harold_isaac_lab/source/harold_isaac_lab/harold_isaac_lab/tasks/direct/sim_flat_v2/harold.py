"""Harold robot asset for sim_flat_v2.

Same robot as harold_flat but with fixed actuator params (no env var overrides)
for reproducibility in controlled experiments.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from pathlib import Path

from harold_isaac_lab.common.stance import load_ready_pose_dict

# Determine the project root directory
try:
    import harold_isaac_lab
    HAROLD_ROOT = Path(harold_isaac_lab.__file__).parent.parent.parent.parent.parent
except Exception:
    print("ERROR: Could not find harold_isaac_lab package.")
    exit(1)

USD_FILE_PATH = HAROLD_ROOT / "part_files" / "V4" / "harold_8.usd"

if not USD_FILE_PATH.exists():
    raise FileNotFoundError(
        f"USD file not found at: {USD_FILE_PATH}\n"
        f"Please ensure the part_files directory is at the project root."
    )

HAROLD_V4_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_FILE_PATH),
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
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.20),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=load_ready_pose_dict(),
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=2.8,   # FeeTech ST3215 servo max (2.94 Nm @ 12V, 95%)
            stiffness=40.0,         # Proportional to effort limit (Spot ratio: Kp/effort ≈ 1.33)
            damping=0.5,            # Allows max speed (4.71 rad/s) without saturating torque budget
        ),
    },
)
"""Configuration for the Harold V4 robot."""
