"""
# WHEN USING SKRL TRAINING LIBRARY
Train model:
python harold_isaac_lab/scripts/skrl/train.py --task=Template-Harold-Direct-rough-terrain-v0 --num_envs 1024

Train model in headless mode with video recording:
python harold_isaac_lab/scripts/skrl/train.py --task=Template-Harold-Direct-rough-terrain-v0 --num_envs 4096 --headless --video --video_length 250 --video_interval 6400

Resume training from checkpoint:
python harold_isaac_lab/scripts/skrl/train.py --task=Template-Harold-Direct-rough-terrain-v0 --num_envs 4096 --checkpoint=/home/matteo/Desktop/code_projects/harold/logs/skrl/harold_direct/terrain_10/checkpoints/best_agent.pt --headless --video --video_length 250 --video_interval 6400

Play back from checkpoint:
python harold_isaac_lab/scripts/skrl/play.py --task=Template-Harold-Direct-rough-terrain-v0 --num_envs 16 --checkpoint=/home/matteo/Desktop/code_projects/harold/logs/skrl/harold_direct/terrain_17/checkpoints/best_agent.pt 


Start Tensorboard:
source ~/Desktop/env_isaaclab/bin/activate
python3 -m tensorboard.main --logdir logs/skrl/harold_direct/ --bind_all

Export trained model (MY CUSTOM VERSION) Note: You may need to run this with sudo for it to get write priveleges)
./isaaclab.sh -p source/standalone/environments/run_harold_v3.py --task Isaac-Harold-Direct-v3 --num_envs 1 --use_last_checkpoint --export_policy
"""

"""
# To edit USD files:
usdedit Harold_V4_STABLE_V2.usd
"""

# Isaac Lab Imports
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg
import os
from pathlib import Path

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
USD_FILE_PATH = HAROLD_ROOT / "part_files" / "V4" / "harold_7.usd"

# Validate that the file exists
if not USD_FILE_PATH.exists():
    raise FileNotFoundError(
        f"USD file not found at: {USD_FILE_PATH}\n"
        f"Please ensure the part_files directory is at the project root.\n"
        f"You can also set HAROLD_PROJECT_ROOT environment variable to the project root directory."
    )

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

            'fl_calf_joint': -0.75,
            'fr_calf_joint': -0.75,
            'bl_calf_joint': -0.75,
            'br_calf_joint': -0.75,
        }
    ),

    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=1.0,
            stiffness=200.0, #100.0, #200.0,
            damping=75.0,
        ),
    },
)

