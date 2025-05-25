"""
# WHEN USING SKRL TRAINING LIBRARY
Train model:
./isaaclab.sh -p source/standalone/workflows/skrl/train.py --task Isaac-Harold-Direct-v3 --num_envs 1024

Train model in headless mode with video recording:
./isaaclab.sh -p source/standalone/workflows/skrl/train.py --task Isaac-Harold-Direct-v3 --num_envs 1024 --headless --video --video_length 500 --video_interval 3000

Start Tensorboard:
python3 -m tensorboard.main --logdir /home/matteo/Desktop/harold/logs/skrl/harold_direct/

Play back trained model (BUILT IN ISAAC LAB VERSION)
./isaaclab.sh -p source/standalone/workflows/skrl/play.py --task Isaac-Harold-Direct-v3 --num_envs 1 --use_last_checkpoint

Play back trained model (MY CUSTOM VERSION)
./isaaclab.sh -p source/standalone/environments/run_harold_v3.py --task Isaac-Harold-Direct-v3 --num_envs 1 --use_last_checkpoint

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


# robot
HAROLD_V4_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/matteo/Desktop/harold/part_files/V4/harold_7.usd",
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
        pos=(0.0, 0.0, 0.18),
        rot=(1.0, 0.0, 0.0, 0.0), 
        joint_pos={
            'fl_shoulder_joint': 0.0,
            'fr_shoulder_joint': 0.0,
            'bl_shoulder_joint': 0.0,
            'br_shoulder_joint': 0.0,
            
            'fl_thigh_joint': 0.4,
            'fr_thigh_joint': 0.4,
            'bl_thigh_joint': 0.4,
            'br_thigh_joint': 0.4,

            'fl_calf_joint': -0.8,
            'fr_calf_joint': -0.8,
            'bl_calf_joint': -0.8,
            'br_calf_joint': -0.8,
        }
    ),

    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=1.5,
            stiffness=200.0,
            damping=100.0,
        ),
    },
)

