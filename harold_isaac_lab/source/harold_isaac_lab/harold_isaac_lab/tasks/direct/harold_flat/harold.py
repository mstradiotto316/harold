"""Harold flat-terrain robot asset.

Training and monitoring should use `scripts/harold.py` (see `AGENTS.md`).
"""


# Isaac Lab Imports
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg
import os
from pathlib import Path

from harold_isaac_lab.common.stance import load_ready_pose_dict

# Allow quick actuator sweeps without code changes.
def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        print(f"WARNING: Invalid {name}='{value}', using default {default}.")
        return default

# Determine the project root directory
try:
    import harold_isaac_lab
    # Go up 5 levels from the package __init__.py to reach project root
    HAROLD_ROOT = Path(harold_isaac_lab.__file__).parent.parent.parent.parent.parent
except:
    # Method 2: Print error and exit since harold_isaac_lab package not found
    print("ERROR: Could not find harold_isaac_lab package environment variable.")
    print("Please ensure harold_isaac_lab is installed.")
    exit(1)

# Construct the USD file path
USD_FILE_PATH = HAROLD_ROOT / "part_files" / "V4" / "harold_8.usd"

# Validate that the file exists
if not USD_FILE_PATH.exists():
    raise FileNotFoundError(
        f"USD file not found at: {USD_FILE_PATH}\n"
        f"Please ensure the part_files directory is at the project root.\n"
        f"You can also set HAROLD_PROJECT_ROOT environment variable to the project root directory."
    )

ACTUATOR_EFFORT_LIMIT = _env_float("HAROLD_ACTUATOR_EFFORT_LIMIT", 2.8)
ACTUATOR_STIFFNESS = _env_float("HAROLD_ACTUATOR_STIFFNESS", 400.0)
ACTUATOR_DAMPING = _env_float("HAROLD_ACTUATOR_DAMPING", 150.0)

# robot
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
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=2
        )
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Spawn height slightly above target standing height
        pos=(0.0, 0.0, 0.30),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=load_ready_pose_dict()
    ),

    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            # FeeTech ST3215 servo: max 2.94 Nm @ 12V.
            # Defaults target a stable baseline; override via HAROLD_ACTUATOR_* for sweeps.
            effort_limit_sim=ACTUATOR_EFFORT_LIMIT,
            stiffness=ACTUATOR_STIFFNESS,
            damping=ACTUATOR_DAMPING,
        ),
    },
)
